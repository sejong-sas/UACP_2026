#!/usr/bin/env python3
"""Streaming Fig.8/Fig.9 evaluation for the completed 100k×5 batch-400 run."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
REGIMES = [("ID-Easy 20 ns", "ID-Easy_20_ns.npz"), ("ID-Hard 80 ns", "ID-Hard_80_ns.npz"),
           ("OOD-Near 120 ns", "OOD-Near_120_ns.npz"), ("OOD-Far 1 ms", "OOD-Far_1_ms.npz")]
# Include the high-coverage anchors requested for the paper-style Fig.9 report.
NOMINALS = np.concatenate([np.arange(0.1, 1.0, 0.1), np.array([0.95, 0.99])])


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def pairwise_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="runs/current_valid_baseline/paper_sample_alignment_100k_1ep_fig8_20260915/config.json")
    parser.add_argument("--common-eval-dir", default="runs/current_valid_baseline/diversity_ablation/reproducibility_20260914/common_eval")
    parser.add_argument("--samples-per-regime", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    out = ROOT / args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT))
    from scripts.diagnose_predictor import _make_model
    from scripts.train_predictor import load_config, set_seeds
    from src.training.data import build_noisy_sparse_input, uniform_grouping_mask
    from src.training.metrics import nmse_all_db, nmse_omitted_db

    cfg = load_config(ROOT / args.config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for this evaluation")
    model = _make_model(cfg, device)
    payload = torch.load(ROOT / args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload)
    model.eval()
    common = ROOT / args.common_eval_dir
    sample_rows: list[dict] = []
    coverage_counts = {(regime, float(nominal)): [0, 0] for regime, _ in REGIMES for nominal in NOMINALS}
    regime_scores: dict[str, list[float]] = {regime: [] for regime, _ in REGIMES}
    regime_nmse: dict[str, list[float]] = {regime: [] for regime, _ in REGIMES}
    regime_ale: dict[str, list[float]] = {regime: [] for regime, _ in REGIMES}
    finite_count = 0
    nonfinite_count = 0
    min_nu = float("inf")
    min_df = float("inf")
    max_nu = float("-inf")
    set_seeds(20262000)
    with torch.inference_mode():
        for ri, (regime, filename) in enumerate(REGIMES):
            cfr = np.load(common / filename)["cfr"]
            count = min(args.samples_per_regime, len(cfr))
            for start in range(0, count, args.batch_size):
                c = torch.from_numpy(cfr[start:start + args.batch_size]).to(device)
                n = c.shape[0]
                set_seeds(20262000 + ri * 100000 + start)
                mask = uniform_grouping_mask(n, 1024, 16, device)
                x, target, _ = build_noisy_sparse_input(c, mask, 15.0)
                output = model(x)
                omitted = (1.0 - mask).bool()
                error_sq = (output.predicted - target).square()
                omitted_float = omitted[:, None, :].to(error_sq.dtype)
                nmse = (10.0 * torch.log10(
                    (error_sq.mul(omitted_float).sum((1, 2)) /
                     target.square().mul(omitted_float).sum((1, 2)).clamp_min(1e-12)).clamp_min(1e-12)
                )).detach().cpu().numpy()
                nmse_all = (10.0 * torch.log10(
                    (error_sq.sum((1, 2)) / target.square().sum((1, 2)).clamp_min(1e-12)).clamp_min(1e-12)
                )).detach().cpu().numpy()
                ale = output.aleatoric.reshape(n, 2, 4, 1024).sum(1).mean(1)
                epi = output.epistemic.reshape(n, 2, 4, 1024).sum(1).mean(1)
                om_count = omitted.sum(1).clamp_min(1)
                ale_sample = ((ale * omitted).sum(1) / om_count).cpu().numpy()
                epi_sample = ((epi * omitted).sum(1) / om_count).cpu().numpy()
                nu_values = output.nu_expanded.detach()
                df_values = nu_values - 2 * 1024 + 1
                min_nu = min(min_nu, float(nu_values.min().cpu()))
                max_nu = max(max_nu, float(nu_values.max().cpu()))
                min_df = min(min_df, float(df_values.min().cpu()))
                finite = torch.isfinite(nu_values) & torch.isfinite(output.psi) & torch.isfinite(output.kappa_expanded)
                finite_count += int(finite.sum().cpu())
                nonfinite_count += int((~finite).sum().cpu())
                regime_scores[regime].extend(epi_sample.tolist())
                regime_nmse[regime].extend(nmse.tolist())
                regime_ale[regime].extend(ale_sample.tolist())
                for i in range(n):
                    sample_rows.append({"regime": regime, "sample": start + i,
                                        "nmse_all_db": float(nmse_all[i]), "nmse_omitted_db": float(nmse[i]),
                                        "aleatoric": float(ale_sample[i]), "epistemic": float(epi_sample[i])})

                # Exact marginal Student-t coverage, vectorized at pair-level df.
                # No component-wise scipy ppf call is made: nu/kappa are pair scalars.
                scale_sq = ((output.kappa_expanded + 1.0) / (output.kappa_expanded * df_values)) * output.psi
                for nominal in NOMINALS:
                    q_pair = student_t.ppf((1.0 + float(nominal)) / 2.0,
                                           output.nu.detach().cpu().numpy() - 2 * 1024 + 1)
                    q = torch.from_numpy(q_pair).to(device=device, dtype=scale_sq.dtype)
                    q = q.repeat_interleave(2, dim=1).reshape(n, 8, 1).expand_as(scale_sq)
                    half = q * torch.sqrt(scale_sq.clamp_min(1e-12))
                    covered = (target - output.gamma).abs() <= half
                    keep = omitted[:, None, :].expand_as(covered)
                    coverage_counts[(regime, float(nominal))][0] += int(covered[keep].sum().cpu())
                    coverage_counts[(regime, float(nominal))][1] += int(keep.sum().cpu())

    summary = []
    for regime, _ in REGIMES:
        values = np.asarray(regime_scores[regime])
        summary.append({"regime": regime, "samples": len(values),
                        "nmse_all_db_mean": float(np.mean([r["nmse_all_db"] for r in sample_rows if r["regime"] == regime])),
                        "nmse_omitted_db_mean": float(np.mean(regime_nmse[regime])),
                        "nmse_omitted_db_median": float(np.median(regime_nmse[regime])),
                        "aleatoric_mean": float(np.mean(regime_ale[regime])),
                        "epistemic_mean": float(values.mean()), "epistemic_median": float(np.median(values)),
                        "epistemic_std": float(values.std()), "epistemic_p05": float(np.quantile(values, .05)),
                        "epistemic_p95": float(np.quantile(values, .95))})
    id_scores = np.asarray(regime_scores["ID-Easy 20 ns"] + regime_scores["ID-Hard 80 ns"])
    near_scores = np.asarray(regime_scores["OOD-Near 120 ns"])
    far_scores = np.asarray(regime_scores["OOD-Far 1 ms"])
    auc = [{"metric": "id_vs_near", "auroc": pairwise_auc(np.r_[id_scores, near_scores], np.r_[np.zeros(len(id_scores)), np.ones(len(near_scores))])},
           {"metric": "id_vs_far", "auroc": pairwise_auc(np.r_[id_scores, far_scores], np.r_[np.zeros(len(id_scores)), np.ones(len(far_scores))])},
           {"metric": "pooled_id_vs_ood", "auroc": pairwise_auc(np.r_[id_scores, near_scores, far_scores], np.r_[np.zeros(len(id_scores)), np.ones(len(near_scores) + len(far_scores))])},
           {"metric": "id_hard_vs_near", "auroc": pairwise_auc(np.r_[np.asarray(regime_scores["ID-Hard 80 ns"]), near_scores], np.r_[np.zeros(len(regime_scores["ID-Hard 80 ns"])), np.ones(len(near_scores))])}]
    calibration = []
    for regime, _ in REGIMES:
        for nominal in NOMINALS:
            covered, total = coverage_counts[(regime, float(nominal))]
            calibration.append({"regime": regime, "pool": "ID" if regime.startswith("ID") else "OOD",
                                "nominal": float(nominal), "empirical": covered / max(total, 1), "count": total})
    for pool, regimes in (("ID", ("ID-Easy 20 ns", "ID-Hard 80 ns")), ("OOD", ("OOD-Near 120 ns", "OOD-Far 1 ms")),
                          ("Near", ("OOD-Near 120 ns",)), ("Far", ("OOD-Far 1 ms",))):
        for nominal in NOMINALS:
            z = [r for r in calibration if r["regime"] in regimes and r["nominal"] == float(nominal)]
            total = sum(r["count"] for r in z)
            calibration.append({"regime": "pooled", "pool": pool, "nominal": float(nominal),
                                "empirical": sum(r["empirical"] * r["count"] for r in z) / max(total, 1), "count": total})
    ce = [{"pool": pool, "mae": float(np.mean([abs(r["empirical"] - r["nominal"]) for r in calibration if r["pool"] == pool and r["regime"] == "pooled"]))}
          for pool in ("ID", "OOD", "Near", "Far")]
    write_rows(out / "regime_summary.csv", summary); write_rows(out / "fig8_auroc.csv", auc)
    write_rows(out / "fig9_calibration.csv", calibration); write_rows(out / "fig9_calibration_error.csv", ce); write_rows(out / "per_sample.csv", sample_rows)

    plt.figure(figsize=(8, 5))
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    for (regime, _), color in zip(REGIMES, colors):
        values = np.asarray(regime_scores[regime]); db = 10 * np.log10(np.maximum(values, 1e-12))
        plt.hist(db, bins=45, density=True, histtype="step", linewidth=1.8, color=color, label=regime)
    plt.xlabel("Epistemic uncertainty (dB)"); plt.ylabel("Density"); plt.grid(alpha=.25); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(out / "fig8_epistemic_distribution.png", dpi=240); plt.savefig(out / "fig8_epistemic_distribution.pdf"); plt.close()
    plt.figure(figsize=(6, 5)); plt.plot([0, 1], [0, 1], "k--", label="Ideal")
    for pool, color, marker in (("ID", "#1f77b4", "^"), ("OOD", "#ff7f0e", "o")):
        z = [r for r in calibration if r["pool"] == pool and r["regime"] == "pooled"]
        plt.plot([r["nominal"] for r in z], [r["empirical"] for r in z], marker + "-", color=color,
                 label=f"{pool} (MAE={next(x['mae'] for x in ce if x['pool'] == pool):.4f})")
    plt.xlabel("Nominal coverage"); plt.ylabel("Empirical coverage"); plt.xlim(0, 1); plt.ylim(0, 1); plt.grid(alpha=.25); plt.legend(); plt.tight_layout()
    plt.savefig(out / "fig9_calibration.png", dpi=240); plt.savefig(out / "fig9_calibration.pdf"); plt.close()
    result = {"checkpoint": args.checkpoint, "device": str(device), "gpu": torch.cuda.get_device_name(0),
              "samples_per_regime": args.samples_per_regime, "batch_size": args.batch_size,
              "regime_summary": summary, "fig8_auroc": auc, "fig9_calibration_error": ce,
              "coverage_anchor_nominals": [0.5, 0.8, 0.9, 0.95, 0.99],
              "nu_boundary": {"min_nu": min_nu, "max_nu": max_nu, "min_df": min_df,
                              "finite_parameter_components": finite_count, "nonfinite_parameter_components": nonfinite_count,
                              "nu_boundary_value": 2049.0, "df_boundary_value": 1.0},
              "assumptions": ["IMPLEMENTATION-ASSUMPTION: diagonal-Psi approximation",
                              "IMPLEMENTATION-ASSUMPTION: Eq.(5) component-wise marginal Student-t intervals",
                              "IMPLEMENTATION-ASSUMPTION: ID=(20,80), OOD=(120,1ms) equal-count pooling",
                              "Student-t ppf vectorized over pair-scalar df; no per-element ppf materialization",
                              "Coverage pooled over omitted real/imag components"]}
    write_json(out / "results.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
