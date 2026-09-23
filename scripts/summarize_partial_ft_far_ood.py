#!/usr/bin/env python3
"""Create fixed comparison tables for the Far-OOD diagnosis and confirmation run."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/current_valid_baseline/partial_ft_far_ood_diagnosis_20260921"


def rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write(path, data):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)


def auc(pos, neg):
    scores = np.r_[neg, pos]; labels = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
    order = np.argsort(scores, kind="mergesort"); ranks = np.empty(len(scores), float)
    ordered = scores[order]; start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and ordered[end] == ordered[start]: end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0; start = end
    pr = ranks[labels == 1]
    return float((pr.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    old_p = rows(ROOT / "runs/current_valid_baseline/partial_ft_20260921_sparse_evaluation/primary_performance_epoch3.csv")
    old_u = rows(ROOT / "runs/current_valid_baseline/partial_ft_20260921_sparse_evaluation/uncertainty_epoch3.csv")
    p25 = rows(ROOT / "runs/current_valid_baseline/partial_ft_20260921_sparse_25pct_evaluation/summary.csv")
    cf = rows(ROOT / "runs/current_valid_baseline/partial_ft_far_ood_confirmatory_last4_20260921_evaluation/summary.csv")
    values = {}
    for r in old_p:
        m = {"pre": "Pre", "partial": "3.16%", "full": "Full"}[r["model"]]
        values[(m, r["ng"], r["regime"], "nmse")] = float(r["nmse_omitted_db"])
    for r in old_u:
        m = {"pre": "Pre", "partial": "3.16%", "full": "Full"}[r["model"]]
        values[(m, r["ng"], r["regime"], "epi")] = float(r["epistemic"])
    for r in p25:
        values[("25%", r["ng"], r["regime"], "nmse")] = float(r["nmse_omitted_db"])
        values[("25%", r["ng"], r["regime"], "epi")] = float(r["epistemic"])
    for r in cf:
        values[("Confirmatory", r["ng"], r["regime"], "nmse")] = float(r["nmse_omitted_db"])
        values[("Confirmatory", r["ng"], r["regime"], "epi")] = float(r["epistemic"])
    result = []
    for ng in (16, 32):
        full_imp = values[("Pre", str(ng), "120ns", "nmse")] - values[("Full", str(ng), "120ns", "nmse")]
        for model in ("3.16%", "25%", "Full", "Confirmatory"):
            imp = values[("Pre", str(ng), "120ns", "nmse")] - values[(model, str(ng), "120ns", "nmse")]
            result.append({"ng": ng, "model": model,
                           "nmse_improvement_120_db": imp,
                           "retention_vs_full_percent": 100 * imp / full_imp,
                           "epistemic_change_120": values[(model, str(ng), "120ns", "epi")] - values[("Pre", str(ng), "120ns", "epi")],
                           "nmse_change_20": values[(model, str(ng), "20ns", "nmse")] - values[("Pre", str(ng), "20ns", "nmse")],
                           "nmse_change_80": values[(model, str(ng), "80ns", "nmse")] - values[("Pre", str(ng), "80ns", "nmse")],
                           "epi_1ms_pre": values[("Pre", str(ng), "1ms", "epi")],
                           "epi_1ms_post": values[(model, str(ng), "1ms", "epi")],
                           "epi_1ms_to_120_post": values[(model, str(ng), "1ms", "epi")] / values[(model, str(ng), "120ns", "epi")],
                           "nmse_120_post": values[(model, str(ng), "120ns", "nmse")]})
    write(OUT / "confirmatory_result.csv", result)
    # Confirmatory AUROC uses the same 1,000 samples and evaluator order.
    sample = rows(ROOT / "runs/current_valid_baseline/partial_ft_far_ood_confirmatory_last4_20260921_evaluation/sample_metrics.csv")
    auc_rows = []
    for ng in (16, 32):
        idv = np.asarray([float(r["epistemic"]) for r in sample if int(r["ng"]) == ng and r["regime"] in ("20ns", "80ns")])
        for regime in ("120ns", "1ms"):
            ood = np.asarray([float(r["epistemic"]) for r in sample if int(r["ng"]) == ng and r["regime"] == regime])
            auc_rows.append({"model": "Confirmatory", "ng": ng, "comparison": f"ID(20+80) vs {regime}",
                             "id_mean": float(idv.mean()), "ood_mean": float(ood.mean()),
                             "id_median": float(np.median(idv)), "ood_median": float(np.median(ood)),
                             "auroc": auc(ood, idv)})
    write(OUT / "confirmatory_auroc.csv", auc_rows)
    (OUT / "comparison_summary.json").write_text(json.dumps({"models": ["Pre", "3.16%", "25%", "Full", "Confirmatory"], "primary_endpoint": "epoch3", "no_hyperparameter_tuning": True}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
