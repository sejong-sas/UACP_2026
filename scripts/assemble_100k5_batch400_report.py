#!/usr/bin/env python3
"""Assemble a non-destructive comparison table for the three 100k baselines."""
from __future__ import annotations
import csv, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/current_valid_baseline/batch400_100k5ep_20260918_bf16_final"

def rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))

def f(x):
    try: return float(x)
    except (TypeError, ValueError): return None

def coverage(cal, pool, nominal):
    for r in cal:
        if r["pool"] == pool and r["regime"] == "pooled" and abs(float(r["nominal"]) - nominal) < 1e-8:
            return f(r["empirical"])
    return None

def fig11(path):
    rr = rows(path)
    return {r["regime"]: {"mean_ng": f(r["mean_ng"]), "nmse_all_db": f(r["mean_nmse_all_db"])} for r in rr}

cur = OUT / "evaluation_streaming_v5"
cur_reg = {r["regime"]: r for r in rows(cur / "regime_summary.csv")}
cur_auc = {r["metric"]: f(r["auroc"]) for r in rows(cur / "fig8_auroc.csv")}
cur_ce = {r["pool"]: f(r["mae"]) for r in rows(cur / "fig9_calibration_error.csv")}
cur_cal = rows(cur / "fig9_calibration.csv")
old_cmp = rows(ROOT / "runs/current_valid_baseline/training_convergence_100k_5ep_20260916_final_controlled/matched_fig8_fig9_comparison/comparison_100k1_vs_100k5.csv")
old_eval_1 = {r["regime"]: r for r in rows(ROOT / "runs/current_valid_baseline/paper_sample_alignment_100k_1ep_fig8_20260915/evaluation_table.csv")}
old_eval_5 = {r["regime"]: r for r in rows(ROOT / "runs/current_valid_baseline/training_convergence_100k_5ep_20260916_final_controlled/evaluation_table.csv")}

def old_row(i, label):
    c = old_cmp[i]
    ev = old_eval_1 if i == 0 else old_eval_5
    prefix = "ID-Easy 20 ns"
    return {
        "model": label, "unique_cfr": 100000, "epochs": 1 if i == 0 else 5,
        "batch": 8, "nmse_20_db": f(ev[prefix]["nmse_all_db"]),
        "nmse_80_db": f(ev["ID-Hard 80 ns"]["nmse_all_db"]),
        "nmse_120_db": f(ev["OOD-Near 120 ns"]["nmse_all_db"]),
        "nmse_1ms_db": f(ev["OOD-Far 1 ms"]["nmse_all_db"]),
        "auc_id_near": f(c["ID_vs_Near_AUROC"]), "auc_80_vs_120": f(c["80_vs_120_AUROC"]),
        "auc_id_far": f(c["ID_vs_Far_AUROC"]), "auc_pooled": f(c["ID_vs_pooled_AUROC"]),
        "fig9_id_mae": f(c["ID_CE"]), "fig9_near_mae": f(c["Near_CE"]),
        "fig9_far_mae": f(c["Far_CE"]), "fig9_ood_mae": f(c["OOD_pooled_CE"]),
        "coverage_ood_0.9": f(c["Near_cov_0.9"]), "coverage_far_0.9": f(c["Far_cov_0.9"]),
        "nu_status": "nonfinite/inf in Far" if i == 1 else "finite status from prior evaluation not re-audited",
    }

def current_row():
    def nm(reg, key="nmse_all_db_mean"): return f(cur_reg[reg][key])
    return {
        "model": "100k×5 batch400 BF16", "unique_cfr": 100000, "epochs": 5, "batch": 400,
        "nmse_20_db": nm("ID-Easy 20 ns"), "nmse_80_db": nm("ID-Hard 80 ns"),
        "nmse_120_db": nm("OOD-Near 120 ns"), "nmse_1ms_db": nm("OOD-Far 1 ms"),
        "auc_id_near": cur_auc["id_vs_near"], "auc_80_vs_120": cur_auc["id_hard_vs_near"],
        "auc_id_far": cur_auc["id_vs_far"], "auc_pooled": cur_auc["pooled_id_vs_ood"],
        "fig9_id_mae": cur_ce["ID"], "fig9_near_mae": cur_ce["Near"],
        "fig9_far_mae": cur_ce["Far"], "fig9_ood_mae": cur_ce["OOD"],
        "coverage_ood_0.9": coverage(cur_cal, "OOD", .9), "coverage_far_0.9": coverage(cur_cal, "Far", .9),
        "coverage_ood_0.5": coverage(cur_cal, "OOD", .5), "coverage_ood_0.8": coverage(cur_cal, "OOD", .8),
        "coverage_ood_0.95": coverage(cur_cal, "OOD", .95), "coverage_ood_0.99": coverage(cur_cal, "OOD", .99),
        "nu_status": "finite; min nu=2051.598, nonfinite=0",
    }

comparison = [old_row(0, "100k×1 batch8"), old_row(1, "100k×5 batch8"), current_row()]
fields = list(dict.fromkeys(k for row in comparison for k in row))
with (OUT / "comparison_100k1_100k5batch8_100k5batch400.csv").open("w", newline="") as h:
    w = csv.DictWriter(h, fieldnames=fields); w.writeheader(); w.writerows(comparison)
payload = {"comparison": comparison, "fig11_lite": {
    "100k×1 batch8": fig11(OUT / "fig11_comparison/100k1_batch8/regime_summary.csv"),
    "100k×5 batch8": fig11(OUT / "fig11_comparison/100k5_batch8/regime_summary.csv"),
    "100k×5 batch400": fig11(OUT / "fig11_lite/regime_summary.csv"),
}, "evaluation_samples_per_regime": 10000,
"assumptions": ["IMPLEMENTATION-ASSUMPTION: prior batch8 Fig.8/Fig.9 values use the preserved matched comparison artifact",
                 "IMPLEMENTATION-ASSUMPTION: Fig.11-lite uses the existing delay-sweep samples and fixed ID-only controller threshold"]}
(OUT / "comparison_100k1_100k5batch8_100k5batch400.json").write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")
print(json.dumps(payload, indent=2, allow_nan=True))
