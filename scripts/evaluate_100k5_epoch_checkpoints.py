#!/usr/bin/env python3
"""Evaluate existing 100k x 5 batch8 epoch checkpoints, without training.

The evaluator reuses the common 10k-per-regime data, Ng=16 mask protocol,
15 dB direct-CFR observation noise, and seed schedule used by the existing
Fig.8-style evaluator. It intentionally does not run Fig.9 or Fig.11.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "runs/current_valid_baseline/training_convergence_100k_5ep_20260916_final_controlled/config_used.json"
CHECKPOINT_DIR = ROOT / "runs/current_valid_baseline/training_convergence_100k_5ep_20260916_final_controlled"
COMMON = ROOT / "runs/current_valid_baseline/diversity_ablation/reproducibility_20260914/common_eval"
REGIMES = [
    ("ID-Easy 20 ns", "ID-Easy_20_ns.npz"),
    ("ID-Hard 80 ns", "ID-Hard_80_ns.npz"),
    ("OOD-Near 120 ns", "OOD-Near_120_ns.npz"),
    ("OOD-Far 1 ms", "OOD-Far_1_ms.npz"),
]


def auc(negative: np.ndarray, positive: np.ndarray) -> float:
    negative = np.asarray(negative, dtype=np.float64)
    positive = np.asarray(positive, dtype=np.float64)
    if not np.isfinite(negative).all() or not np.isfinite(positive).all():
        return float("nan")
    values = np.concatenate([negative, positive])
    ranks = rankdata(values, method="average")
    n0 = len(negative)
    n1 = len(positive)
    return float((ranks[n0:].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def finite_stats(values: list[float]) -> dict[str, float | int | None]:
    a = np.asarray(values, dtype=np.float64)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return {"count": int(a.size), "finite_count": 0, "nonfinite_count": int(a.size), "min": None, "median": None, "max": None}
    return {
        "count": int(a.size),
        "finite_count": int(finite.size),
        "nonfinite_count": int(a.size - finite.size),
        "min": float(finite.min()),
        "median": float(np.median(finite)),
        "max": float(finite.max()),
    }


def tensor_stats(tensor: torch.Tensor) -> dict[str, float | int | None]:
    return finite_stats(tensor.detach().float().reshape(-1).cpu().numpy().tolist())


def load_model(cfg: dict, checkpoint: Path, device: torch.device):
    from scripts.diagnose_predictor import _make_model

    model = _make_model(cfg, device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    state = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state)
    model.eval()
    return model


def escaped_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def evaluate_checkpoint(model, cfg: dict, device: torch.device, samples_per_regime: int, batch_size: int):
    from scripts.train_predictor import set_seeds
    from src.training.data import build_noisy_sparse_input, uniform_grouping_mask

    raw_values: dict[str, list[torch.Tensor]] = {"gamma": [], "psi": [], "kappa": [], "nu": []}
    hooks = []
    for name in raw_values:
        module = getattr(model, f"{name}_head")
        hooks.append(module.register_forward_hook(lambda _module, _inputs, output, key=name: raw_values[key].append(output.detach())))

    sample_rows: list[dict] = []
    regime_summary: dict[str, dict] = {}
    for regime_index, (label, filename) in enumerate(REGIMES):
        cfr = np.load(COMMON / filename, allow_pickle=False)["cfr"][:samples_per_regime]
        regime_rows: list[dict] = []
        stage_values = {key: [] for key in ("raw_gamma", "raw_psi", "raw_kappa", "raw_nu", "gamma", "psi", "kappa", "nu", "nu_minus_2049", "aleatoric", "epistemic", "nmse_omitted", "epi_score")}
        stage_nonfinite = {key: 0 for key in stage_values}
        first_failure = None
        for start in range(0, len(cfr), batch_size):
            batch_np = cfr[start : start + batch_size]
            c = torch.from_numpy(batch_np).to(device)
            mask = uniform_grouping_mask(len(batch_np), 1024, 16, device)
            set_seeds(20262000 + regime_index * 100000 + start)
            raw_values.update({key: [] for key in raw_values})
            x, target, _ = build_noisy_sparse_input(c, mask, 15.0)
            out = model(x)
            omitted = (1.0 - mask).bool()

            tensors = {
                "raw_gamma": raw_values["gamma"][-1],
                "raw_psi": raw_values["psi"][-1],
                "raw_kappa": raw_values["kappa"][-1],
                "raw_nu": raw_values["nu"][-1],
                "gamma": out.gamma,
                "psi": out.psi,
                "kappa": out.kappa,
                "nu": out.nu,
                "nu_minus_2049": out.nu_expanded - 2049.0,
                "aleatoric": out.aleatoric,
                "epistemic": out.epistemic,
            }
            score_map = out.epistemic.reshape(len(batch_np), 2, 4, 1024).sum(dim=1).mean(dim=1)
            score = (score_map * omitted).sum(dim=1) / omitted.sum(dim=1).clamp_min(1.0)
            error = (out.predicted - target).square()
            omitted_float = omitted[:, None, :].to(error.dtype)
            numerator = (error * omitted_float).sum(dim=(1, 2))
            denominator = (target.square() * omitted_float).sum(dim=(1, 2)).clamp_min(1e-12)
            nmse = 10.0 * torch.log10((numerator / denominator).clamp_min(1e-12))
            tensors["nmse_omitted"] = nmse.reshape(-1)
            tensors["epi_score"] = score.reshape(-1)
            nmse_np = nmse.detach().float().cpu().numpy()
            score_np = score.detach().float().cpu().numpy()
            finite_output_np = (
                torch.isfinite(out.gamma).reshape(len(batch_np), -1).all(dim=1)
                & torch.isfinite(out.psi).reshape(len(batch_np), -1).all(dim=1)
                & torch.isfinite(out.kappa).reshape(len(batch_np), -1).all(dim=1)
                & torch.isfinite(out.nu).reshape(len(batch_np), -1).all(dim=1)
            ).cpu().numpy()
            finite_uncertainty_np = (
                torch.isfinite(out.aleatoric).reshape(len(batch_np), -1).all(dim=1)
                & torch.isfinite(out.epistemic).reshape(len(batch_np), -1).all(dim=1)
            ).cpu().numpy()
            for key, tensor in tensors.items():
                values = tensor.detach().float().reshape(-1).cpu().numpy()
                stage_values[key].extend(values.tolist())
                count = int((~np.isfinite(values)).sum())
                stage_nonfinite[key] += count
                if first_failure is None and count:
                    first_failure = {"stage": key, "batch_start": start, "nonfinite_count": count}
            for i in range(len(batch_np)):
                regime_rows.append({
                    "regime": label,
                    "sample": start + i,
                    "nmse_omitted_db": float(nmse_np[i]),
                    "epistemic_score": float(score_np[i]),
                    "finite_output": bool(finite_output_np[i]),
                    "finite_uncertainty": bool(finite_uncertainty_np[i]),
                })
            del c, mask, x, target, out, tensors
        sample_rows.extend(regime_rows)
        regime_summary[label] = {
            "samples": len(regime_rows),
            "nmse_omitted_db": finite_stats(stage_values["nmse_omitted"]),
            "parameter_stats": {key: finite_stats(stage_values[key]) for key in ("kappa", "nu", "nu_minus_2049", "psi")},
            "uncertainty_stats": {key: finite_stats(stage_values[key]) for key in ("aleatoric", "epistemic", "epi_score")},
            "raw_head_stats": {key: finite_stats(stage_values[key]) for key in ("raw_gamma", "raw_psi", "raw_kappa", "raw_nu")},
            "nonfinite_counts_by_stage": stage_nonfinite,
            "first_nonfinite": first_failure,
        }
    for hook in hooks:
        hook.remove()

    by_regime = {label: np.asarray([row["epistemic_score"] for row in sample_rows if row["regime"] == label], dtype=np.float64) for label, _ in REGIMES}
    id_scores = np.concatenate([by_regime["ID-Easy 20 ns"], by_regime["ID-Hard 80 ns"]])
    near_scores = by_regime["OOD-Near 120 ns"]
    far_scores = by_regime["OOD-Far 1 ms"]
    finite_id = np.isfinite(id_scores)
    finite_near = np.isfinite(near_scores)
    finite_far = np.isfinite(far_scores)
    auroc = {
        "near": auc(id_scores, near_scores),
        "far": auc(id_scores, far_scores),
        "pooled": auc(id_scores, np.concatenate([near_scores, far_scores])),
        "near_finite_only_diagnostic": auc(id_scores[finite_id], near_scores[finite_near]),
        "far_finite_only_diagnostic": auc(id_scores[finite_id], far_scores[finite_far]),
    }
    return sample_rows, regime_summary, auroc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples-per-regime", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    out = ROOT / args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT))
    from scripts.train_predictor import load_config

    cfg = load_config(CONFIG)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" or "GB10" not in torch.cuda.get_device_name(0):
        raise RuntimeError("epoch evaluation requires NVIDIA GB10 on cuda:0")
    checkpoints = sorted(CHECKPOINT_DIR.glob("uacp_predictor_100k_5ep_epoch_*.pt"))
    if len(checkpoints) != 5:
        raise FileNotFoundError(f"expected five epoch checkpoints, found {len(checkpoints)}")

    all_summary = []
    all_auc = []
    all_samples = []
    all_results = {}
    started = time.perf_counter()
    for checkpoint in checkpoints:
        epoch = int(checkpoint.stem.rsplit("_", 1)[1])
        print(json.dumps({"epoch_start": epoch, "checkpoint": str(checkpoint.relative_to(ROOT))}), flush=True)
        model = load_model(cfg, checkpoint, device)
        rows, summaries, aurocs = evaluate_checkpoint(model, cfg, device, args.samples_per_regime, args.batch_size)
        for row in rows:
            row["epoch"] = epoch
        all_samples.extend(rows)
        for label, summary in summaries.items():
            all_summary.append({"epoch": epoch, "regime": label, "summary_json": json.dumps(summary, allow_nan=False)})
        all_auc.append({"epoch": epoch, **aurocs})
        all_results[str(epoch)] = {
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "checkpoint_size_bytes": checkpoint.stat().st_size,
            "regime_summary": summaries,
            "fig8_auroc": aurocs,
        }
        print(json.dumps({"epoch_complete": epoch, "fig8_auroc": aurocs, "first_nonfinite": {k: v["first_nonfinite"] for k, v in summaries.items()}}, allow_nan=True), flush=True)
        del model
        torch.cuda.empty_cache()

    escaped_rows(out / "sample_scores.csv", all_samples)
    escaped_rows(out / "epoch_regime_summary.csv", all_summary)
    escaped_rows(out / "epoch_fig8_auroc.csv", all_auc)
    result = {
        "experiment": "existing 100k x 5 batch8 FP32 epoch checkpoint evaluation",
        "training_performed": False,
        "device": "cuda:0",
        "gpu": torch.cuda.get_device_name(0),
        "config": str(CONFIG.relative_to(ROOT)),
        "common_eval_dir": str(COMMON.relative_to(ROOT)),
        "samples_per_regime": args.samples_per_regime,
        "batch_size": args.batch_size,
        "mask_protocol": "uniform_grouping_mask Ng=16",
        "observation": "direct CFR AWGN 15 dB",
        "pooling_assumption": "IMPLEMENTATION-ASSUMPTION: sample-level equal-count ID=(20,80) and OOD=(120,1ms) AUROC pooling",
        "uncertainty_assumption": "diagonal-Psi pair-scalar Eq.12/13 score",
        "elapsed_s": time.perf_counter() - started,
        "epochs": all_results,
    }
    (out / "epoch_evaluation_results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out.relative_to(ROOT)), "elapsed_s": result["elapsed_s"], "epochs": list(all_results)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
