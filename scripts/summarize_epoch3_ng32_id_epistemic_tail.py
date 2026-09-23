#!/usr/bin/env python3
"""Post-process the existing Ng32 tail diagnostic; no inference or training."""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/current_valid_baseline/epoch3_ng32_id_epistemic_tail_diagnosis_20260920"

def read_rows():
    with (OUT / "per_sample.csv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def values(rows, key): return np.asarray([float(r[key]) for r in rows], dtype=float)

def main():
    rows = read_rows()
    id32 = [r for r in rows if r["ng"] == "32" and int(r["delay_ns"]) <= 100]
    groups = ["normal_0_50", "upper_q95_99", "extreme_q99_plus"]
    keys = ["cfr_mag_mean", "cfr_mag_std", "cfr_adjacent_diff_mean", "cfr_peak_to_mean", "nmse", "kappa", "nu_margin", "psi", "aleatoric", "epistemic"]
    feature_rows = []
    for g in groups:
        part = [r for r in id32 if r["group"] == g]
        for key in keys:
            x = values(part, key)
            feature_rows.append({"group": g, "metric": key, "n": len(x), "mean": x.mean(), "median": np.median(x), "q05": np.quantile(x,.05), "q95": np.quantile(x,.95)})
    write(OUT / "group_feature_summary.csv", feature_rows)

    pairs = []
    with (OUT / "error_matched_extreme_normal.csv").open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if "extreme_sample" not in r: continue
            pairs.append(r)
    effect = {}
    if pairs:
        for name, e, n in [("kappa", "kappa_extreme", "kappa_normal"), ("nu_margin", "nu_margin_extreme", "nu_margin_normal"), ("aleatoric", "aleatoric_extreme", "aleatoric_normal"), ("epistemic", "epistemic_extreme", "epistemic_normal")]:
            d = np.asarray([float(x[e]) - float(x[n]) for x in pairs])
            effect[name] = {"n": len(d), "mean_difference_extreme_minus_normal": float(d.mean()), "median_difference": float(np.median(d)), "extreme_greater_fraction": float(np.mean(d > 0))}
        effect["nmse_abs_difference"] = {"n": len(pairs), "median": float(np.median([float(x["nmse_abs_diff"]) for x in pairs]))}
    (OUT / "error_matched_effects.json").write_text(json.dumps(effect, indent=2) + "\n")

if __name__ == "__main__": main()
