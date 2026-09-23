#!/usr/bin/env python3
"""Experiment 1: ID-only Ng-conditioned epistemic thresholds.

This is a frozen-checkpoint diagnostic.  It keeps only scalar scores per
sample (the small score vectors are needed for exact quantiles/AUROC) and
never retains model predictions or feature tensors.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
K = 1024
CHECKPOINT = ROOT / "runs/current_valid_baseline/paper_sample_alignment_100k_1ep_fig8_20260915/uacp_predictor_step4a.pt"
DATA_DIR = ROOT / "runs/baseline_reproduction/step2_delay_sweep_repro/generated_data"
REGIMES = [("20 ns", 20), ("80 ns", 80), ("120 ns", 120), ("1 ms", "1_ms")]
NGS = [4, 8, 16, 32, 64, 128, 1]


def summarize_scores(values: list[float]) -> dict[str, float | int]:
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return {"count": 0, "mean": float("nan"), "median": float("nan"),
                "q90": float("nan"), "q95": float("nan"), "q99": float("nan")}
    return {"count": int(a.size), "mean": float(a.mean()), "median": float(np.quantile(a, .5)),
            "q90": float(np.quantile(a, .9)), "q95": float(np.quantile(a, .95)),
            "q99": float(np.quantile(a, .99))}


def threshold_metrics(values: list[float], threshold: float) -> dict[str, float | int]:
    if not values or not np.isfinite(threshold):
        return {"rate": float("nan"), "count": 0, "total": len(values)}
    count = int(np.count_nonzero(np.asarray(values) > threshold))
    return {"rate": float(count / len(values)), "count": count, "total": len(values)}


def auc_from_scores(negative: list[float], positive: list[float]) -> float:
    """Exact tie-aware rank AUC without retaining tensors or importing sklearn."""
    x = np.asarray(negative, dtype=np.float64)
    y = np.asarray(positive, dtype=np.float64)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return float("nan")
    combined = np.concatenate([x, y])
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty(combined.size, dtype=np.float64)
    sorted_values = combined[order]
    i = 0
    while i < sorted_values.size:
        j = i + 1
        while j < sorted_values.size and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return float((ranks[x.size:] .sum() - y.size * (y.size + 1) / 2.0) / (x.size * y.size))


def rss_gib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024 ** 2)


def make_mask(batch: int, ng: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros((batch, K), dtype=torch.float32, device=device)
    mask[:, ::ng] = 1.0
    return mask


def load_model(cfg: dict, device: torch.device):
    from scripts.diagnose_predictor import _make_model
    model = _make_model(cfg, device)
    payload = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    state = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
    model.load_state_dict(state)
    model.eval()
    return model


def evaluate_scores(model, device: torch.device, cfr: np.ndarray, ng: int, seed: int,
                    batch_size: int) -> list[float]:
    if ng == 1:
        return []
    from scripts.train_predictor import set_seeds
    from src.training.data import build_noisy_sparse_input
    scores: list[float] = []
    for start in range(0, len(cfr), batch_size):
        set_seeds(seed + start)
        batch = torch.from_numpy(cfr[start:start + batch_size]).to(device)
        mask = make_mask(batch.shape[0], ng, device)
        x, _, _ = build_noisy_sparse_input(batch, mask, 15.0)
        with torch.inference_mode():
            out = model(x)
            ale_map = out.psi / (out.nu_expanded - 2 * K - 1.0)
            epi_map = ale_map / out.kappa_expanded
            # Eq. (12): pair-wise trace map; Eq. (13): omitted-subcarrier mean.
            pair_epi = epi_map.reshape(epi_map.shape[0], 2, 4, K).permute(0, 2, 1, 3).sum(dim=2).mean(dim=1)
            omitted = 1.0 - mask
            denom = omitted.sum(dim=1)
            batch_scores = ((pair_epi * omitted).sum(dim=1) / denom).detach().cpu().numpy()
        scores.extend(float(v) for v in batch_scores if np.isfinite(v))
        del batch, mask, x, out, ale_map, epi_map, pair_epi, omitted, denom, batch_scores
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return scores


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--samples-per-regime", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()
    out = ROOT / args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT))
    from scripts.train_predictor import load_config
    cfg = load_config("configs/current_valid_baseline_100k1_seed_20260819.json")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    start = time.perf_counter()
    model = load_model(cfg, device)
    all_scores: dict[str, dict[int, list[float]]] = {name: {} for name, _ in REGIMES}
    for ng in NGS:
        for ri, (name, suffix) in enumerate(REGIMES):
            filename = f"test_delay_{suffix}.npz" if suffix == "1_ms" else f"test_delay_{suffix}_ns.npz"
            path = DATA_DIR / filename
            with np.load(path) as data:
                cfr = np.asarray(data["cfr"][:args.samples_per_regime], dtype=np.complex64)
            all_scores[name][ng] = evaluate_scores(model, device, cfr, ng, args.seed + ri * 10000 + ng, args.batch_size)
            del cfr
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    summary_rows = []
    threshold_rows = []
    for ng in NGS:
        id_scores = all_scores["20 ns"].get(ng, []) + all_scores["80 ns"].get(ng, [])
        tau = float(np.quantile(id_scores, .99)) if id_scores else float("nan")
        status = "training-supported" if ng in {4, 8, 16, 32} else ("mask-extrapolation" if ng in {64, 128} else "full-feedback-diagnostic")
        id_metrics = threshold_metrics(id_scores, tau)
        near = threshold_metrics(all_scores["120 ns"].get(ng, []), tau)
        far = threshold_metrics(all_scores["1 ms"].get(ng, []), tau)
        threshold_rows.append({"ng": ng, "support_status": status, "tau_ng_id_q99": tau,
                               "id_false_trigger_rate": id_metrics["rate"], "id_trigger_count": id_metrics["count"],
                               "120ns_trigger_rate": near["rate"], "120ns_trigger_count": near["count"],
                               "1ms_trigger_rate": far["rate"], "1ms_trigger_count": far["count"],
                               "near_auroc": auc_from_scores(id_scores, all_scores["120 ns"].get(ng, [])),
                               "far_auroc": auc_from_scores(id_scores, all_scores["1 ms"].get(ng, []))})
        for name, _ in REGIMES:
            row = {"ng": ng, "regime": name, "support_status": status}
            row.update(summarize_scores(all_scores[name].get(ng, [])))
            summary_rows.append(row)

    write_csv(out / "ng_threshold_summary.csv", threshold_rows)
    write_csv(out / "score_summary.csv", summary_rows)
    results = {"experiment": "Experiment 1: Ng-conditioned Epistemic Threshold",
               "hypothesis": "global threshold may mix incompatible Ng mask regimes",
               "checkpoint": str(CHECKPOINT.relative_to(ROOT)), "no_training": True,
               "data": {"source": str(DATA_DIR.relative_to(ROOT)), "samples_per_regime": args.samples_per_regime,
                        "regimes": [name for name, _ in REGIMES], "snr_db": 15.0},
               "ngs": NGS, "training_supported_ng": [4, 8, 16, 32],
               "formula": "U_epi Eq.(8) mapped by Eq.(12), averaged over omitted subcarriers per Eq.(13); tau_Ng=ID(20ns+80ns) q99",
               "implementation_assumptions": ["existing 200-sample delay-sweep files reused", "Ng=64/128 are mask-pattern extrapolation; Ng=1 has no omitted score", "current diagonal-Psi model and corrected antenna-pair mapping"],
               "peak_rss_gib": rss_gib(), "peak_gpu_allocated_mib": (torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None),
               "elapsed_s": time.perf_counter() - start,
               "output": str(out.relative_to(ROOT))}
    (out / "results.json").write_text(json.dumps(results, indent=2, allow_nan=True) + "\n")
    print(json.dumps({"output": str(out), "device": str(device), "gpu": gpu_name, "peak_rss_gib": results["peak_rss_gib"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
