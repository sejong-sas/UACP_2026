#!/usr/bin/env python3
"""Post-process existing Ng16/32 tail samples; diagnostic only, no inference."""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "runs/current_valid_baseline/epoch3_ng32_id_epistemic_tail_diagnosis_20260920/per_sample.csv"
OUT = ROOT / "runs/current_valid_baseline/epoch3_cfr_feature_tail_diagnosis_20260920"
ID_DELAYS = {20, 40, 60, 80, 100}
FEATURES = ["cfr_mag_mean", "cfr_mag_std", "rms_magnitude", "cfr_adjacent_diff_mean", "cfr_peak_to_mean"]
TARGETS = ["kappa", "aleatoric", "epistemic", "nmse"]

def read_rows():
    with IN.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ["delay_ns", "ng", "sample"] + FEATURES[0:2] + ["cfr_adjacent_diff_mean", "cfr_peak_to_mean"] + TARGETS:
            if k in r: r[k] = float(r[k])
        r["rms_magnitude"] = float(np.sqrt(r["cfr_mag_mean"] ** 2 + r["cfr_mag_std"] ** 2))
    return rows

def write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields: fields.append(key)
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(rows)

def summary(rows, group, keys):
    part = [r for r in rows if r["group"] == group]
    out = {"group": group, "n": len(part)}
    for k in keys:
        x = np.asarray([r[k] for r in part], float)
        out[k+"_median"] = float(np.median(x)); out[k+"_mean"] = float(x.mean())
    return out

def matched(tail, normal, feature_names, caliper_fraction=0.25):
    # Greedy no-reuse nearest-neighbor matching after scale normalization.
    allx = np.asarray([[r[f] for f in feature_names] for r in tail + normal], float)
    scale = np.nanpercentile(allx, 75, axis=0) - np.nanpercentile(allx, 25, axis=0)
    scale[scale <= 1e-12] = 1.0
    caliper = float(np.sqrt(len(feature_names))) * caliper_fraction
    available = set(range(len(normal))); pairs = []
    for t in sorted(tail, key=lambda r: r["epistemic"], reverse=True):
        if not available: break
        tx = np.asarray([t[f] for f in feature_names], float)
        distances = {i: float(np.linalg.norm((tx - np.asarray([normal[i][f] for f in feature_names])) / (scale * caliper_fraction))) for i in available}
        j = min(distances, key=distances.get)
        if distances[j] > caliper:
            continue
        available.remove(j); n = normal[j]
        row = {"tail_sample": int(t["sample"]), "normal_sample": int(n["sample"]), "match_features": "+".join(feature_names)}
        for f in feature_names: row[f+"_abs_diff"] = abs(t[f] - n[f])
        for k in TARGETS: row[k+"_tail"] = t[k]; row[k+"_normal"] = n[k]; row[k+"_diff"] = t[k] - n[k]
        pairs.append(row)
    return pairs

def pair_summary(pairs, label):
    out = {"matching": label, "n": len(pairs)}
    if not pairs: return out
    for k in TARGETS:
        d = np.asarray([p[k+"_diff"] for p in pairs])
        out[k+"_diff_median"] = float(np.median(d)); out[k+"_tail_greater_rate"] = float(np.mean(d > 0))
    for k in FEATURES:
        key = k + "_abs_diff"
        if key in pairs[0]: out[key+"_median"] = float(np.median([p[key] for p in pairs]))
    return out

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = read_rows()
    id32 = [r for r in rows if r["ng"] == 32 and r["delay_ns"] in ID_DELAYS]
    id16 = [r for r in rows if r["ng"] == 16 and r["delay_ns"] in ID_DELAYS]
    # RMS is derived algebraically from the stored mean/std; no normalization is applied.
    for r in rows: r["rms_magnitude"] = float(np.sqrt(r["cfr_mag_mean"]**2 + r["cfr_mag_std"]**2))
    write(OUT / "per_sample_with_rms.csv", rows)

    corr_rows = []
    for ng, part in [(16, id16), (32, id32)]:
        for feature in FEATURES:
            for target in TARGETS:
                corr_rows.append({"ng": ng, "feature": feature, "target": target, "n": len(part), "spearman": float(spearmanr([r[feature] for r in part], [r[target] for r in part]).statistic)})
    write(OUT / "spearman_correlations.csv", corr_rows)

    normal = [r for r in id32 if r["group"] == "normal_0_50"]
    tail = [r for r in id32 if r["group"] == "extreme_q99_plus"]
    write(OUT / "amplitude_matched_pairs.csv", matched(tail, normal, ["rms_magnitude"]))
    write(OUT / "frequency_matched_pairs.csv", matched(tail, normal, ["cfr_adjacent_diff_mean"]))
    write(OUT / "amplitude_frequency_error_matched_pairs.csv", matched(tail, normal, ["rms_magnitude", "cfr_adjacent_diff_mean", "nmse"]))
    pair_rows = []
    for label, path in [("amplitude", OUT/"amplitude_matched_pairs.csv"), ("frequency", OUT/"frequency_matched_pairs.csv"), ("amplitude_frequency_nmse", OUT/"amplitude_frequency_error_matched_pairs.csv")]:
        with path.open(newline="", encoding="utf-8") as f: pp = list(csv.DictReader(f))
        for p in pp:
            for k in TARGETS + FEATURES:
                if k+"_diff" in p: p[k+"_diff"] = float(p[k+"_diff"])
                if k+"_abs_diff" in p: p[k+"_abs_diff"] = float(p[k+"_abs_diff"])
        pair_rows.append(pair_summary(pp, label))
    write(OUT / "matched_summary.csv", pair_rows)

    # 3x3 quantile bins for two primary features, q99 is fixed from the previous artifact.
    for r in id32: r["amp_bin"] = int(min(2, np.searchsorted(np.quantile([x["rms_magnitude"] for x in id32], [1/3, 2/3]), r["rms_magnitude"])))
    for r in id32: r["freq_bin"] = int(min(2, np.searchsorted(np.quantile([x["cfr_adjacent_diff_mean"] for x in id32], [1/3, 2/3]), r["cfr_adjacent_diff_mean"])))
    bin_rows = []
    for a in range(3):
        for q in range(3):
            part = [r for r in id32 if r["amp_bin"] == a and r["freq_bin"] == q]
            if not part: continue
            threshold = np.quantile([r["epistemic"] for r in id32], .99)
            bin_rows.append({"amplitude_bin": a, "frequency_bin": q, "n": len(part), "q99_count": sum(r["epistemic"] >= threshold for r in part), "q99_rate": float(np.mean([r["epistemic"] >= threshold for r in part])), "kappa_median": float(np.median([r["kappa"] for r in part])), "epistemic_median": float(np.median([r["epistemic"] for r in part]))})
    write(OUT / "tail_rate_2d_bins.csv", bin_rows)

    # Standardized OLS partial-effect diagnostic, not causal inference.
    x = np.asarray([[1, r["rms_magnitude"], r["cfr_adjacent_diff_mean"], r["nmse"], r["delay_ns"]] for r in id32], float)
    y = np.log(np.asarray([r["kappa"] for r in id32], float))
    for j in range(1, x.shape[1]):
        s = x[:,j].std(); x[:,j] = (x[:,j]-x[:,j].mean()) / (s if s else 1)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]; pred = x @ beta
    (OUT / "partial_effect_diagnostic.json").write_text(json.dumps({"model": "log(kappa) ~ rms_magnitude + adjacent_variation + nmse + delay_ns", "interpretation": "conditional association / partial-effect diagnostic, not causal proof", "n": len(id32), "intercept": float(beta[0]), "standardized_coefficients": {"rms_magnitude": float(beta[1]), "adjacent_variation": float(beta[2]), "nmse": float(beta[3]), "delay_ns": float(beta[4])}, "r2": float(1 - ((y-pred)**2).sum()/((y-y.mean())**2).sum())}, indent=2) + "\n")

    # q99 reference summaries for Ng16, Ng32 ID, and Ng32 OOD.
    ref = []
    for label, part, ng in [("Ng16_ID_q99", id16, 16), ("Ng32_ID_q99", id32, 32), ("Ng32_120ns", [r for r in rows if r["ng"] == 32 and r["delay_ns"] == 120], 32)]:
        threshold = np.quantile([r["epistemic"] for r in part], .99) if label != "Ng16_ID_q99" else np.quantile([r["epistemic"] for r in part], .99)
        tailpart = [r for r in part if r["epistemic"] >= threshold] if "q99" in label else part
        ref.append({"label": label, "ng": ng, "n": len(tailpart), **{k+"_median": float(np.median([r[k] for r in tailpart])) for k in ["rms_magnitude", "cfr_mag_mean", "cfr_mag_std", "cfr_adjacent_diff_mean", "cfr_peak_to_mean", "kappa", "aleatoric", "epistemic", "nmse"]}})
    write(OUT / "reference_summary.csv", ref)
    (OUT / "analysis_manifest.json").write_text(json.dumps({"input": str(IN.relative_to(ROOT)), "checkpoint": "epoch3 baseline (inherited from input artifact)", "no_training": True, "no_normalization_applied": True, "normalization_note": "Current dataset generation uses normalize_channel=false; this is an IMPLEMENTATION-ASSUMPTION, not a paper-public setting.", "matching": "No-reuse greedy nearest-neighbor; amplitude=rms derived from stored mean/std; frequency=adjacent-subcarrier difference; joint adds NMSE.", "matching_caliper": "0.25 pooled-IQR per feature, Euclidean for multi-feature matching; pairs outside caliper rejected.", "tail_definition": "Ng32 ID q99 threshold inherited from prior artifact; not changed after seeing these results.", "ood_use": "120 ns reference only; not used for fitting or thresholds."}, indent=2) + "\n")

if __name__ == "__main__": main()
