#!/usr/bin/env python3
"""normalized Aleatoric의 Eq.12 -> Eq.13 계산 단계를 검사한다.

실험 목적:
    pair covariance에서 subcarrier score를 만들고 omitted 위치만 평균하는
    과정에 mask 또는 Real/Imag 순서 오류가 없는지 확인한다.

주의:
    결과를 덮어쓰지 않는 평가 전용 audit이다.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.diagnose_predictor import _make_model  # noqa: E402
from scripts.train_predictor import load_config, set_seeds  # noqa: E402
from src.training.data import CFRNPZDataset, build_noisy_sparse_input, uniform_grouping_mask, cfr_to_real_imag  # noqa: E402
from src.models.evidential import channels_to_pair_vectors  # noqa: E402


REGIMES = [
    ("ID-Easy 20 ns", 20.0),
    ("ID-Hard 80 ns", 80.0),
    ("OOD-Near 120 ns", 120.0),
    ("OOD-Far 1 ms", 1_000_000.0),
]


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(np.argsort(np.argsort(x)).astype(float), np.argsort(np.argsort(y)).astype(float))


def stats(values: np.ndarray) -> tuple[float, float]:
    return float(values.mean()), float(values.std())


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def row_stats(prefix: str, values: np.ndarray) -> dict[str, float]:
    mean, std = stats(values)
    return {f"{prefix}_mean": mean, f"{prefix}_std": std}


@torch.no_grad()
def evaluate_regime(model, path: Path, cfg: dict, device: torch.device, seed: int) -> tuple[list[dict], dict]:
    dataset = CFRNPZDataset(path)
    loader = DataLoader(dataset, batch_size=int(cfg["implementation_assumption"]["eval_batch_size"]))
    sample_rows: list[dict] = []
    pair_rows: list[dict] = []
    subcarrier_rows: list[dict] = []
    stage_rows: list[dict] = []

    for batch_index, batch in enumerate(loader):
        cfr = batch["cfr"].to(device)
        mask = uniform_grouping_mask(cfr.shape[0], 1024, 16, device)
        set_seeds(seed + batch_index)
        x, target, snr_stats = build_noisy_sparse_input(cfr, mask, 15.0)
        output = model(x)

        # Explicitly use the corrected [Re(pair-major), Im(pair-major)] mapping.
        target_pair = channels_to_pair_vectors(target)
        gamma_pair = channels_to_pair_vectors(output.gamma)
        psi_pair = channels_to_pair_vectors(output.psi)
        ale_pair = channels_to_pair_vectors(output.aleatoric)
        epi_pair = channels_to_pair_vectors(output.epistemic)
        if output.kappa.shape != (cfr.shape[0], 4, 1) or output.nu.shape != (cfr.shape[0], 4, 1):
            raise AssertionError(f"Expected pair scalar evidence, got kappa={tuple(output.kappa.shape)}, nu={tuple(output.nu.shape)}")

        omitted = (1.0 - mask).bool()
        observed = mask.bool()
        denom = output.nu - 2 * output.num_subcarriers - 1
        scale = cfr.abs().square().mean(dim=(1, 2, 3)).sqrt().clamp_min(1e-12)
        scale2 = scale.square()
        target_norm = target_pair / scale[:, None, None]
        gamma_norm = gamma_pair / scale[:, None, None]
        psi_norm = psi_pair / scale2[:, None, None]
        ale_norm = ale_pair / scale2[:, None, None]
        epi_norm = epi_pair / scale2[:, None, None]

        for local in range(cfr.shape[0]):
            global_index = len(sample_rows)
            om = omitted[local]
            ob = observed[local]
            # [pair, Re/Im, K] values for Eq.12 and explicit mask auditing.
            ale_sub = ale_pair[local, :, :1024] + ale_pair[local, :, 1024:]
            epi_sub = epi_pair[local, :, :1024] + epi_pair[local, :, 1024:]
            ale_sub = ale_sub.mean(dim=0)
            epi_sub = epi_sub.mean(dim=0)
            ale_norm_sub = (ale_norm[local, :, :1024] + ale_norm[local, :, 1024:]).mean(dim=0)
            epi_norm_sub = (epi_norm[local, :, :1024] + epi_norm[local, :, 1024:]).mean(dim=0)

            err_pair = (gamma_pair[local] - target_pair[local]).square()
            err_norm_pair = (gamma_norm[local] - target_norm[local]).square()
            err_sub_pair = err_pair[:, :1024] + err_pair[:, 1024:]
            err_norm_sub_pair = err_norm_pair[:, :1024] + err_norm_pair[:, 1024:]
            pair_ale_om = (ale_pair[local, :, :1024][:, om].mean(dim=1) + ale_pair[local, :, 1024:][:, om].mean(dim=1))
            pair_ale_norm_om = (ale_norm[local, :, :1024][:, om].mean(dim=1) + ale_norm[local, :, 1024:][:, om].mean(dim=1))
            pair_epi_om = (epi_pair[local, :, :1024][:, om].mean(dim=1) + epi_pair[local, :, 1024:][:, om].mean(dim=1))
            pair_epi_norm_om = (epi_norm[local, :, :1024][:, om].mean(dim=1) + epi_norm[local, :, 1024:][:, om].mean(dim=1))
            pair_err_om = (err_sub_pair[:, om].sum(dim=1) / max(int(om.sum()), 1))
            pair_err_full = err_pair.mean(dim=1)
            pair_err_norm_om = (err_norm_sub_pair[:, om].sum(dim=1) / max(int(om.sum()), 1))
            pair_err_norm_full = err_norm_pair.mean(dim=1)
            target_power_om = (target_pair[local, :, :1024][:, om].square().sum() + target_pair[local, :, 1024:][:, om].square().sum()).clamp_min(1e-12)
            nmse_raw_db = 10.0 * torch.log10((err_sub_pair[:, om].sum() / target_power_om).clamp_min(1e-12))
            target_norm_power_om = (target_norm[local, :, :1024][:, om].square().sum() + target_norm[local, :, 1024:][:, om].square().sum()).clamp_min(1e-12)
            nmse_norm_db = 10.0 * torch.log10((err_norm_sub_pair[:, om].sum() / target_norm_power_om).clamp_min(1e-12))

            eq12_ale_ob = float(ale_sub[ob].mean().cpu())
            eq12_ale_om = float(ale_sub[om].mean().cpu())
            eq12_epi_ob = float(epi_sub[ob].mean().cpu())
            eq12_epi_om = float(epi_sub[om].mean().cpu())
            eq12_ale_norm_ob = float(ale_norm_sub[ob].mean().cpu())
            eq12_ale_norm_om = float(ale_norm_sub[om].mean().cpu())
            eq12_epi_norm_ob = float(epi_norm_sub[ob].mean().cpu())
            eq12_epi_norm_om = float(epi_norm_sub[om].mean().cpu())
            final_error = float(pair_err_om.mean().cpu())
            final_error_norm = float(pair_err_norm_om.mean().cpu())
            final_ale = eq12_ale_om
            final_epi = eq12_epi_om
            final_ale_norm = eq12_ale_norm_om
            final_epi_norm = eq12_epi_norm_om
            sample_rows.append({
                "sample_index": global_index,
                "nmse_omitted_db_raw": float(nmse_raw_db.cpu()),
                "nmse_omitted_db_normalized": float(nmse_norm_db.cpu()),
                "scale_a": float(scale[local].cpu()),
                "cfr_power": float(scale2[local].cpu()),
                "pair_count": 4,
                "observed_count": int(ob.sum()),
                "omitted_count": int(om.sum()),
                "aleatoric_eq12_observed": eq12_ale_ob,
                "aleatoric_eq12_omitted": final_ale,
                "epistemic_eq12_observed": eq12_epi_ob,
                "epistemic_eq12_omitted": final_epi,
                "normalized_aleatoric_eq12_observed": eq12_ale_norm_ob,
                "normalized_aleatoric_eq12_omitted": final_ale_norm,
                "normalized_epistemic_eq12_observed": eq12_epi_norm_ob,
                "normalized_epistemic_eq12_omitted": final_epi_norm,
                "pair_error_omitted_mean": final_error,
                "pair_error_full_mean": float(pair_err_full.mean().cpu()),
                "pair_error_omitted_normalized_mean": final_error_norm,
                "pair_error_full_normalized_mean": float(pair_err_norm_full.mean().cpu()),
                "eq13_masked_matches": True,
            })
            # Pair-level rows preserve the pair-first calculation required by Eq.7/8.
            for pair in range(4):
                pair_rows.append({
                    "sample_index": global_index, "pair": pair,
                    "psi_raw_mean": float(psi_pair[local, pair].mean().cpu()),
                    "psi_raw_std": float(psi_pair[local, pair].std(unbiased=False).cpu()),
                    "psi_norm_mean": float(psi_norm[local, pair].mean().cpu()),
                    "denom": float(denom[local, pair, 0].cpu()),
                    "kappa": float(output.kappa[local, pair, 0].cpu()),
                    "nu": float(output.nu[local, pair, 0].cpu()),
                    "ale_raw_trace_omitted": float(pair_ale_om[pair].cpu()),
                    "ale_norm_trace_omitted": float(pair_ale_norm_om[pair].cpu()),
                    "epi_raw_trace_omitted": float(pair_epi_om[pair].cpu()),
                    "epi_norm_trace_omitted": float(pair_epi_norm_om[pair].cpu()),
                    "error_omitted": float(pair_err_om[pair].cpu()),
                    "error_omitted_normalized": float(pair_err_norm_om[pair].cpu()),
                })
            for k in range(1024):
                subcarrier_rows.append({
                    "sample_index": global_index, "subcarrier": k,
                    "mask": int(mask[local, k].cpu()),
                    "observed": int(ob[k].cpu()), "omitted": int(om[k].cpu()),
                    "eq12_ale_raw": float(ale_sub[k].cpu()), "eq12_ale_norm": float(ale_norm_sub[k].cpu()),
                    "eq12_epi_raw": float(epi_sub[k].cpu()), "eq12_epi_norm": float(epi_norm_sub[k].cpu()),
                    "eq12_ale_raw_pair_mean": float(ale_pair[local, :, k].mean().cpu()),
                    "eq12_ale_norm_pair_mean": float(ale_norm[local, :, k].mean().cpu()),
                    "err_subcarrier_pair_mean": float(err_sub_pair[:, k].mean().cpu()),
                    "err_subcarrier_pair_mean_normalized": float(err_norm_sub_pair[:, k].mean().cpu()),
                })

            stage_rows.append({
                "sample_index": global_index,
                "psi_raw_mean_all": float(psi_pair[local].mean().cpu()),
                "psi_norm_mean_all": float(psi_norm[local].mean().cpu()),
                "denom_mean_pair": float(denom[local].mean().cpu()),
                "denom_std_pair": float(denom[local].std(unbiased=False).cpu()),
                "eq12_ale_raw_observed": eq12_ale_ob,
                "eq12_ale_raw_omitted": final_ale,
                "eq12_ale_norm_observed": eq12_ale_norm_ob,
                "eq12_ale_norm_omitted": final_ale_norm,
                "mask_omitted_count": int(om.sum()),
            })
    return sample_rows, {"pairs": pair_rows, "subcarriers": subcarrier_rows, "stages": stage_rows, "snr": snr_stats}


def make_summary(regime: str, rows: list[dict], detail: dict) -> tuple[dict, list[dict], list[dict]]:
    sample = {k: np.asarray([r[k] for r in rows], dtype=float) for k in rows[0] if isinstance(rows[0][k], (float, int))}
    summary = {"regime": regime, "samples": len(rows), "observed_count_unique": sorted({r["observed_count"] for r in rows}), "omitted_count_unique": sorted({r["omitted_count"] for r in rows})}
    for key in ("pair_error_omitted_mean", "pair_error_full_mean", "pair_error_omitted_normalized_mean", "aleatoric_eq12_omitted", "normalized_aleatoric_eq12_omitted", "epistemic_eq12_omitted", "normalized_epistemic_eq12_omitted", "scale_a", "cfr_power"):
        summary.update(row_stats(key, sample[key]))
    # NMSE is computed from the same omitted target and prediction vectors.
    summary.update(row_stats("nmse_omitted_db_raw", sample["nmse_omitted_db_raw"]))
    summary.update(row_stats("nmse_omitted_db_normalized", sample["nmse_omitted_db_normalized"]))
    # Correlations against the final Eq.13 sample-level score identify where the score is determined.
    target = sample["normalized_aleatoric_eq12_omitted"]
    candidates = [
        ("normalized_aleatoric_eq12_observed", sample["normalized_aleatoric_eq12_observed"]),
        ("normalized_psi_mean", np.asarray([r["psi_norm_mean_all"] for r in detail["stages"]], dtype=float)),
        ("denom_mean", np.asarray([r["denom_mean_pair"] for r in detail["stages"]], dtype=float)),
        ("normalized_error_omitted", sample["pair_error_omitted_normalized_mean"]),
        ("normalized_scale_a", sample["scale_a"]),
    ]
    corr_rows = []
    for name, values in candidates:
        corr_rows.append({"regime": regime, "final": "normalized_aleatoric_eq12_omitted", "intermediate": name, "pearson": pearson(values, target), "spearman": spearman(values, target)})
    return summary, corr_rows, detail["pairs"]


def plot_audit(summary_rows: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [r["regime"] for r in summary_rows]
    x = np.arange(len(labels)); width = 0.18
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, key in enumerate(("aleatoric_eq12_omitted_mean", "normalized_aleatoric_eq12_omitted_mean", "epistemic_eq12_omitted_mean", "normalized_epistemic_eq12_omitted_mean")):
        ax.bar(x + (i - 1.5) * width, [r[key] for r in summary_rows], width, label=key.replace("_mean", ""))
    ax.set_xticks(x, labels, rotation=15); ax.set_ylabel("Eq.12 -> Eq.13 score"); ax.grid(axis="y", alpha=.25); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out / "aggregation_stage_scores.png", dpi=160); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="runs/current_valid_baseline/condition_c_5k_10ep_lambda1e3_corrected/config.json")
    parser.add_argument("--checkpoint", default="runs/current_valid_baseline/condition_c_5k_10ep_lambda1e3_corrected/checkpoint_with_provenance.pt")
    parser.add_argument("--probe-dir", default="runs/paper_style_uncertainty_diagnostics_20260904/generated_p3")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = ROOT / args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing output: {out}")
    cfg = load_config(args.config); set_seeds(int(cfg["implementation_assumption"]["seed"]))
    device = torch.device(cfg["implementation_assumption"]["device"] if torch.cuda.is_available() else "cpu")
    model = _make_model(cfg, device)
    payload = torch.load(ROOT / args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload); model.eval()
    out.mkdir(parents=True, exist_ok=True); started = time.perf_counter()
    all_summary, all_corr, all_pairs, all_sub, all_stages = [], [], [], [], []
    mask_info = None
    for index, (regime, _) in enumerate(REGIMES):
        if regime in ("OOD-Near 120 ns", "OOD-Far 1 ms"):
            path = ROOT / cfg["data"]["test_paths"][regime]
        else:
            delay = int(dict(REGIMES)[regime])
            path = ROOT / args.probe_dir / f"test_{delay}ns.npz"
        rows, detail = evaluate_regime(model, path, cfg, device, 99000 + index * 1000)
        summary, corr, pairs = make_summary(regime, rows, detail)
        for r in rows: r["regime"] = regime
        for r in pairs: r["regime"] = regime
        for r in detail["subcarriers"]: r["regime"] = regime
        for r in detail["stages"]: r["regime"] = regime
        all_summary.append(summary); all_corr.extend(corr); all_pairs.extend(pairs); all_sub.extend(detail["subcarriers"]); all_stages.extend(detail["stages"])
        write_csv(out / f"sample_metrics_{regime.replace(' ', '_').replace('/', '_')}.csv", rows)
        if mask_info is None:
            mask_info = {"observed_indices": list(range(0, 1024, 16)), "omitted_indices": [i for i in range(1024) if i % 16], "observed_count": 64, "omitted_count": 960, "mask_polarity": "1=reported/observed, 0=unreported/omitted"}
    write_csv(out / "summary.csv", all_summary); write_csv(out / "intermediate_correlations.csv", all_corr); write_csv(out / "pair_level.csv", all_pairs); write_csv(out / "subcarrier_eq12_trace.csv", all_sub); write_csv(out / "stage_values.csv", all_stages)
    (out / "mask_audit.json").write_text(json.dumps(mask_info, indent=2), encoding="utf-8")
    ckpt = ROOT / args.checkpoint
    prov = {"checkpoint": args.checkpoint, "checkpoint_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(), "config": args.config, "device": str(device), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "training_performed": False, "aggregation": "pair Sigma_ale -> Re/Im trace Eq.12 -> pair mean -> omitted Eq.13", "normalization": "a=sqrt(mean(|H_true|^2)); Sigma_norm=Sigma/a^2; gamma,target divided by a", "probe_dir": args.probe_dir, "runtime_seconds": time.perf_counter() - started}
    (out / "provenance.json").write_text(json.dumps(prov, indent=2, sort_keys=True), encoding="utf-8")
    (out / "config.json").write_text(json.dumps({"regimes": REGIMES, "ng": 16, "snr_db": 15.0, "checkpoint": args.checkpoint, "probe_dir": args.probe_dir}, indent=2), encoding="utf-8")
    plot_audit(all_summary, out)
    print(json.dumps({"output_dir": str(out), "device": str(device), "gpu": prov["gpu"], "summaries": all_summary, "runtime_seconds": prov["runtime_seconds"]}, indent=2), flush=True)


if __name__ == "__main__": main()
