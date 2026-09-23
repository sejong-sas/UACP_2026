#!/usr/bin/env python3
"""Summarize epoch3 parameter and κ-gradient trajectories across delay."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
GRAD = ROOT / "runs/current_valid_baseline/epoch3_kappa_delay_sweep_diagnosis_20260920/per_sample_gradients.csv"
METRIC = ROOT / "runs/current_valid_baseline/epoch3_balanced_mask_exposure_comparison_20260920/per_sample.csv"
OUT = ROOT / "runs/current_valid_baseline/epoch3_kappa_delay_sweep_diagnosis_20260920"
NGS = [16, 32]
DELAYS = [10, 20, 40, 60, 80, 100, 120]


def stats(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "median": float(np.median(values)), "q05": float(np.quantile(values, .05)), "q50": float(np.quantile(values, .50)), "q95": float(np.quantile(values, .95)), "q99": float(np.quantile(values, .99)), "max": float(values.max())}


def auc(negative, positive):
    ranks = rankdata(np.concatenate([negative, positive]), method="average")
    return float((ranks[len(negative):].sum() - len(positive) * (len(positive) + 1) / 2) / (len(negative) * len(positive)))


def overlap(left, right):
    left = left[left > 0]
    right = right[right > 0]
    edges = np.linspace(np.log10(np.concatenate([left, right])).min(), np.log10(np.concatenate([left, right])).max(), 80 + 1)
    lh, _ = np.histogram(np.log10(left), bins=edges, density=True)
    rh, _ = np.histogram(np.log10(right), bins=edges, density=True)
    return float(np.sum(np.minimum(lh, rh) * np.diff(edges)))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    gradient_rows = list(csv.DictReader(GRAD.open(newline="", encoding="utf-8")))
    metric_rows = list(csv.DictReader(METRIC.open(newline="", encoding="utf-8")))
    baseline = [row for row in metric_rows if row["model"] == "baseline_epoch3"]
    summary_rows = []
    for ng in NGS:
        for delay in DELAYS:
            gs = [row for row in gradient_rows if int(row["ng"]) == ng and int(row["delay_ns"]) == delay]
            ms = [row for row in baseline if int(row["ng"]) == ng and int(float(row["delay_ns"])) == delay]
            row = {"ng": ng, "delay_ns": delay}
            for metric in ["nmse", "psi", "kappa", "nu_margin", "aleatoric", "epistemic"]:
                for key, value in stats([float(x[metric]) for x in gs]).items():
                    row[f"{metric}_{key}"] = value
            for loss_name in ["nll", "reg", "total"]:
                for grad_name in ["kappa_grad_mean", "kappa_grad_abs_mean", "kappa_grad_q05", "kappa_grad_q95", "raw_kappa_grad_mean", "raw_kappa_grad_abs_mean", "lower_kappa_fraction", "higher_kappa_fraction"]:
                    values = [float(x[f"{loss_name}_{grad_name}"]) for x in gs]
                    row[f"{loss_name}_{grad_name}_mean"] = float(np.mean(values))
                    row[f"{loss_name}_{grad_name}_median"] = float(np.median(values))
            row["metric_finite_count"] = int(sum(all(np.isfinite(float(x[m])) for m in ["nmse", "psi", "kappa", "nu_margin", "aleatoric", "epistemic"]) for x in ms))
            summary_rows.append(row)
    write_csv(OUT / "delay_trajectory_summary.csv", summary_rows)

    comparison_rows = []
    for ng in NGS:
        for delay in [20, 80, 100, 120]:
            row = next(x for x in summary_rows if x["ng"] == ng and x["delay_ns"] == delay)
            comparison_rows.append({"ng": ng, "delay_ns": delay, "kappa_median": row["kappa_median"], "kappa_q05": row["kappa_q05"], "kappa_q95": row["kappa_q95"], "epistemic_median": row["epistemic_median"], "epistemic_q99": row["epistemic_q99"], "nu_margin_median": row["nu_margin_median"], "psi_median": row["psi_median"], "total_kappa_gradient_median": row["total_kappa_grad_mean_median"], "total_lower_kappa_fraction": row["total_lower_kappa_fraction_mean"]})
    write_csv(OUT / "comparison_20_80_100_120.csv", comparison_rows)

    boundary_rows = []
    for ng in NGS:
        s100 = next(x for x in summary_rows if x["ng"] == ng and x["delay_ns"] == 100)
        s120 = next(x for x in summary_rows if x["ng"] == ng and x["delay_ns"] == 120)
        m100 = np.array([float(x["epistemic"]) for x in gradient_rows if int(x["ng"]) == ng and int(x["delay_ns"]) == 100])
        m120 = np.array([float(x["epistemic"]) for x in gradient_rows if int(x["ng"]) == ng and int(x["delay_ns"]) == 120])
        k100 = np.array([float(x["kappa"]) for x in gradient_rows if int(x["ng"]) == ng and int(x["delay_ns"]) == 100])
        k120 = np.array([float(x["kappa"]) for x in gradient_rows if int(x["ng"]) == ng and int(x["delay_ns"]) == 120])
        boundary_rows.append({"ng": ng, "kappa_120_over_100": s120["kappa_median"] / s100["kappa_median"], "nu_margin_120_over_100": s120["nu_margin_median"] / s100["nu_margin_median"], "psi_120_over_100": s120["psi_median"] / s100["psi_median"], "aleatoric_120_over_100": s120["aleatoric_median"] / s100["aleatoric_median"], "epistemic_120_over_100": s120["epistemic_median"] / s100["epistemic_median"], "total_grad_abs_120_over_100": s120["total_kappa_grad_abs_mean_mean"] / s100["total_kappa_grad_abs_mean_mean"], "total_lower_fraction_100": s100["total_lower_kappa_fraction_mean"], "total_lower_fraction_120": s120["total_lower_kappa_fraction_mean"], "kappa_overlap_100_vs_120": overlap(k100, k120), "epistemic_overlap_100_vs_120": overlap(m100, m120), "kappa_auroc_100_vs_120": auc(k100, k120), "epistemic_auroc_100_vs_120": auc(m100, m120)})
    write_csv(OUT / "boundary_100_vs_120.csv", boundary_rows)
    manifest = {"gradient_input": str(GRAD.relative_to(ROOT)), "metric_input": str(METRIC.relative_to(ROOT)), "id_delays_ns": [10, 20, 40, 60, 80, 100], "ood_delay_ns": 120, "checkpoint_only": True, "120ns_gradient_note": "Diagnostic gradient computed by evaluating the checkpoint on OOD data; it is not a gradient received during training.", "no_training": True, "no_loss_or_model_changes": True}
    (OUT / "trajectory_analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
