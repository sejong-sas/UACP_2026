#!/usr/bin/env python3
"""Diagnostic-only post-hoc Ng scalar κ alignment; no training or controller changes."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
FACTORIAL = ROOT / "runs/current_valid_baseline/epoch3_ng_delay_factorial_diagnosis_20260920/per_sample.csv"
TEN_NS = ROOT / "runs/current_valid_baseline/epoch3_id_vs_120_epistemic_20260919/per_sample.csv"
TRACE = ROOT / "runs/current_valid_baseline/epoch3_restricted_ng_fig11_20260919/dynamic_trace.csv"
OUT = ROOT / "runs/current_valid_baseline/epoch3_ng_scalar_kappa_calibration_20260920"
CORRECTED_OUT = ROOT / "runs/current_valid_baseline/epoch3_ng_scalar_kappa_calibration_20260920_corrected"
NGS = [4, 8, 16, 32]
ID_DELAYS = [10, 20, 40, 60, 80, 100]
SPLIT_POINT = 100


def read_rows(path, model=None):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out = []
    for row in rows:
        if model is not None and row.get("model") != model:
            continue
        if "delay_ns" not in row or "ng" not in row or "sample" not in row:
            continue
        converted = {key: value for key, value in row.items()}
        converted["delay_ns"] = int(float(row["delay_ns"]))
        converted["ng"] = int(row["ng"])
        converted["sample"] = int(row["sample"])
        for key in ["kappa", "aleatoric", "epistemic", "nmse", "psi", "nu_margin"]:
            converted[key] = float(row[key])
        out.append(converted)
    return out


def auc(negative, positive):
    ranks = rankdata(np.concatenate([negative, positive]), method="average")
    return float((ranks[len(negative):].sum() - len(positive) * (len(positive) + 1) / 2) / (len(negative) * len(positive)))


def stats(values):
    values = np.asarray(values, dtype=float)
    return {"count": int(len(values)), "mean": float(values.mean()), "median": float(np.median(values)), "q05": float(np.quantile(values, .05)), "q95": float(np.quantile(values, .95)), "q99": float(np.quantile(values, .99)), "max": float(values.max())}


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    CORRECTED_OUT.mkdir(parents=True, exist_ok=True)
    rows = [row for row in read_rows(FACTORIAL) if row["delay_ns"] in [20, 40, 60, 80, 100, 120]]
    rows += [row for row in read_rows(TEN_NS) if row["delay_ns"] == 10]
    rows = [row for row in rows if row["ng"] in NGS]
    calibration = [row for row in rows if row["delay_ns"] in ID_DELAYS and row["sample"] < SPLIT_POINT]
    heldout_id = [row for row in rows if row["delay_ns"] in ID_DELAYS and row["sample"] >= SPLIT_POINT]
    heldout_ood = [row for row in rows if row["delay_ns"] == 120 and row["sample"] >= SPLIT_POINT]
    m_ng = {ng: float(np.median(np.log([row["kappa"] for row in calibration if row["ng"] == ng]))) for ng in NGS}
    m_ref = float(np.median(np.log([row["kappa"] for row in calibration])))
    factors = {ng: float(np.exp(m_ref - m_ng[ng])) for ng in NGS}
    for row in rows:
        row["kappa_cal"] = factors[row["ng"]] * row["kappa"]
        # Correct κ-only post-hoc transform. The persisted ``aleatoric`` field
        # uses a different pair/component aggregation than persisted
        # ``epistemic`` in the source artifacts, so recomputing aleatoric/kappa
        # would introduce an extra factor. Preserve original Epistemic exactly.
        row["epistemic_cal"] = row["epistemic"] / factors[row["ng"]]
        row["expected_epi_ratio"] = 1.0 / factors[row["ng"]]
        row["actual_epi_ratio"] = row["epistemic_cal"] / row["epistemic"]

    tau_before = float(np.quantile([row["epistemic"] for row in calibration], .99))
    tau_after = float(np.quantile([row["epistemic_cal"] for row in calibration], .99))
    scale_rows = []
    for ng in NGS:
        cal = [r for r in calibration if r["ng"] == ng]
        ev = [r for r in heldout_id if r["ng"] == ng]
        scale_rows.append({"ng": ng, "calibration_n": len(cal), "m_ng_log_kappa": m_ng[ng], "m_ref_log_kappa": m_ref, "c_ng": factors[ng], "kappa_median_before": np.median([r["kappa"] for r in ev]), "kappa_median_after": np.median([r["kappa_cal"] for r in ev]), "epistemic_median_before": np.median([r["epistemic"] for r in ev]), "epistemic_median_after": np.median([r["epistemic_cal"] for r in ev]), "epistemic_q95_before": np.quantile([r["epistemic"] for r in ev], .95), "epistemic_q95_after": np.quantile([r["epistemic_cal"] for r in ev], .95), "epistemic_q99_before": np.quantile([r["epistemic"] for r in ev], .99), "epistemic_q99_after": np.quantile([r["epistemic_cal"] for r in ev], .99)})
    write_csv(CORRECTED_OUT / "scale_alignment.csv", scale_rows)

    threshold_rows = []
    for ng in NGS:
        id_rows = [r for r in heldout_id if r["ng"] == ng]
        ood_rows = [r for r in heldout_ood if r["ng"] == ng]
        threshold_rows.append({"ng": ng, "id_fpr_before": np.mean([r["epistemic"] > tau_before for r in id_rows]), "id_fpr_after": np.mean([r["epistemic_cal"] > tau_after for r in id_rows]), "ood_tpr_before": np.mean([r["epistemic"] > tau_before for r in ood_rows]), "ood_tpr_after": np.mean([r["epistemic_cal"] > tau_after for r in ood_rows]), "near_auroc_before": auc(np.array([r["epistemic"] for r in id_rows]), np.array([r["epistemic"] for r in ood_rows])), "near_auroc_after": auc(np.array([r["epistemic_cal"] for r in id_rows]), np.array([r["epistemic_cal"] for r in ood_rows]))})
    write_csv(CORRECTED_OUT / "global_threshold_results.csv", threshold_rows)

    # Offline threshold crossing on the existing restricted-controller trace only.
    offline_rows = []
    if TRACE.is_file():
        with TRACE.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                ng = int(row["current_ng"])
                if ng not in factors:
                    continue
                epi = float(row["epistemic"])
                offline_rows.append({**row, "tau_before": tau_before, "tau_after": tau_after, "epistemic_cal": epi / factors[ng], "trigger_before": bool(np.isfinite(epi) and epi > tau_before), "trigger_after": bool(np.isfinite(epi) and epi / factors[ng] > tau_after), "diagnostic_only": True})
    if offline_rows:
        write_csv(CORRECTED_OUT / "offline_fig11_threshold_crossings.csv", offline_rows)
        offline_summary = []
        for regime in sorted(set(row["regime"] for row in offline_rows), key=str):
            part = [row for row in offline_rows if row["regime"] == regime]
            offline_summary.append({"regime": regime, "trigger_before": sum(row["trigger_before"] for row in part), "trigger_after": sum(row["trigger_after"] for row in part), "diagnostic_only": True})
        write_csv(CORRECTED_OUT / "offline_fig11_summary.csv", offline_summary)

    summary = {"calibration_split": "sample<100", "heldout_split": "sample>=100", "id_delays_ns": ID_DELAYS, "ood_reference_ns": 120, "m_ref_log_kappa": m_ref, "c_ng": factors, "tau_before": tau_before, "tau_after": tau_after, "calibration_n_per_ng": {str(ng): sum(row["ng"] == ng for row in calibration) for ng in NGS}, "heldout_id_n_per_ng": {str(ng): sum(row["ng"] == ng for row in heldout_id) for ng in NGS}, "heldout_ood_n_per_ng": {str(ng): sum(row["ng"] == ng for row in heldout_ood) for ng in NGS}, "implementation_assumption": "IMPLEMENTATION-ASSUMPTION / DIAGNOSTIC ONLY: positive scalar Ng calibration fit from ID calibration split; no controller or model change; 120 ns excluded from fitting.", "within_ng_ranking_invariance": "positive scaling preserves within-Ng ordering; AUROC should remain unchanged up to floating-point effects."}
    summary["output_kind"] = "CORRECTED κ-ONLY DIAGNOSTIC"
    summary["calibration_formula"] = "kappa_cal=c_Ng*kappa; aleatoric_cal=aleatoric_original; epistemic_cal=epistemic_original/c_Ng"
    summary["source_artifact_note"] = "Original artifact aleatoric field is not aggregation-compatible with persisted epistemic; corrected result preserves persisted epistemic and applies only 1/c_Ng."
    (CORRECTED_OUT / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    (CORRECTED_OUT / "analysis_manifest.json").write_text(json.dumps({"inputs": [str(FACTORIAL.relative_to(ROOT)), str(TEN_NS.relative_to(ROOT)), str(TRACE.relative_to(ROOT))], "checkpoint_only": True, "no_training": True, "no_optimizer_step": True, "gpu_note": "Input artifacts were generated with NVIDIA GB10/cuda:0; this post-hoc aggregation did not run inference.", "calibration_formula": "m_Ng=median(log(kappa)); m_ref=pooled ID median(log(kappa)); c_Ng=exp(m_ref-m_Ng); kappa_cal=c_Ng*kappa; aleatoric_cal=aleatoric_original; epistemic_cal=epistemic_original/c_Ng", "threshold_definition": "pooled ID calibration q99 before/after", "ratio_sanity": "actual_epi_ratio must equal 1/c_Ng sample-wise"}, indent=2) + "\n")

    ratio_rows = []
    for ng in NGS:
        vals = [r["actual_epi_ratio"] for r in rows if r["ng"] == ng and np.isfinite(r["actual_epi_ratio"])]
        ratio_rows.append({"ng": ng, "expected_ratio_1_over_c": 1.0 / factors[ng], "actual_ratio_min": min(vals), "actual_ratio_median": float(np.median(vals)), "actual_ratio_max": max(vals), "max_abs_error": max(abs(v - 1.0 / factors[ng]) for v in vals), "finite_sample_count": len(vals)})
    write_csv(CORRECTED_OUT / "samplewise_ratio_sanity.csv", ratio_rows)


if __name__ == "__main__":
    main()
