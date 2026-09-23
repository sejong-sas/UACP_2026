#!/usr/bin/env python3
"""Matched baseline-vs-balanced parameter causality analysis; no inference/training."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "runs/current_valid_baseline/epoch3_balanced_mask_exposure_comparison_20260920/per_sample.csv"
OUT = ROOT / "runs/current_valid_baseline/epoch3_balanced_mask_parameter_causality_20260920"
MODELS = ["baseline_epoch3", "balanced_epoch3"]
NGS = [16, 32]
REGIMES = [20, 80, 120]
METRICS = ["nmse", "psi", "kappa", "nu_margin", "aleatoric", "epistemic"]


def quantile(values, probability):
    return float(np.quantile(np.asarray(values, dtype=float), probability))


def stats(values):
    values = np.asarray(values, dtype=float)
    return {"count": int(values.size), "mean": float(np.mean(values)), "median": float(np.median(values)), "q05": quantile(values, .05), "q50": quantile(values, .50), "q95": quantile(values, .95), "q99": quantile(values, .99), "max": float(np.max(values))}


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else float("nan")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    with INPUT.open(newline="", encoding="utf-8") as handle:
        rows.extend(csv.DictReader(handle))
    grouped = defaultdict(dict)
    for row in rows:
        key = (row["model"], int(row["ng"]), int(float(row["delay_ns"])), int(row["sample"]))
        grouped[key] = {metric: float(row[metric]) for metric in METRICS}

    distributions = []
    for ng in NGS:
        for model in MODELS:
            for delay in REGIMES:
                values = {metric: [grouped[(model, ng, delay, sample)][metric] for sample in range(200)] for metric in METRICS}
                row = {"model": model, "ng": ng, "delay_ns": delay}
                for metric in METRICS:
                    for name, value in stats(values[metric]).items():
                        row[f"{metric}_{name}"] = value
                distributions.append(row)
    write_csv(OUT / "distribution_summary.csv", distributions)
    (OUT / "distribution_summary.json").write_text(json.dumps(distributions, indent=2) + "\n")

    ratio_rows = []
    for ng in NGS:
        for model in MODELS:
            by_delay = {delay: next(row for row in distributions if row["model"] == model and row["ng"] == ng and row["delay_ns"] == delay) for delay in REGIMES}
            row = {"model": model, "ng": ng}
            for metric in ["psi", "kappa", "nu_margin", "aleatoric", "epistemic"]:
                row[f"{metric}_120_over_80"] = ratio(by_delay[120][f"{metric}_median"], by_delay[80][f"{metric}_median"])
                row[f"{metric}_120_minus_80"] = by_delay[120][f"{metric}_median"] - by_delay[80][f"{metric}_median"]
            ratio_rows.append(row)
    write_csv(OUT / "ratio_120_over_80.csv", ratio_rows)

    matched = []
    for ng in NGS:
        for delay in REGIMES:
            for sample in range(200):
                base = grouped[("baseline_epoch3", ng, delay, sample)]
                bal = grouped[("balanced_epoch3", ng, delay, sample)]
                row = {"ng": ng, "delay_ns": delay, "sample": sample}
                for metric in METRICS:
                    row[f"baseline_{metric}"] = base[metric]
                    row[f"balanced_{metric}"] = bal[metric]
                    row[f"delta_{metric}"] = bal[metric] - base[metric]
                    row[f"ratio_{metric}"] = ratio(bal[metric], base[metric])
                matched.append(row)
    write_csv(OUT / "matched_sample_comparison.csv", matched)

    # Descriptive groups are fixed from baseline distributions; they are not controller thresholds.
    group_rows = []
    group_meta = {}
    for ng in NGS:
        id_epi = np.array([grouped[("baseline_epoch3", ng, delay, sample)]["epistemic"] for delay in [20, 80] for sample in range(200)])
        ood_epi = np.array([grouped[("baseline_epoch3", ng, 120, sample)]["epistemic"] for sample in range(200)])
        id_q95 = float(np.quantile(id_epi, .95))
        id_q99 = float(np.quantile(id_epi, .99))
        group_meta[str(ng)] = {"id_q95": id_q95, "id_q99": id_q99, "group_definition": "A=baseline ID (20/80) >= q95; B=baseline 120 below baseline ID q99; C=largest 10 negative balanced-baseline OOD epistemic deltas among baseline OOD upper quartile"}
        for delay in [20, 80]:
            for sample in range(200):
                b = grouped[("baseline_epoch3", ng, delay, sample)]
                a = grouped[("balanced_epoch3", ng, delay, sample)]
                if b["epistemic"] >= id_q95:
                    group_rows.append({"group": "A_ID_upper_tail", "ng": ng, "delay_ns": delay, "sample": sample, "baseline_epistemic": b["epistemic"], "balanced_epistemic": a["epistemic"], "delta_epistemic": a["epistemic"] - b["epistemic"], "delta_aleatoric": a["aleatoric"] - b["aleatoric"], "delta_kappa": a["kappa"] - b["kappa"], "delta_nu_margin": a["nu_margin"] - b["nu_margin"], "delta_psi": a["psi"] - b["psi"]})
        for sample in range(200):
            b = grouped[("baseline_epoch3", ng, 120, sample)]
            a = grouped[("balanced_epoch3", ng, 120, sample)]
            if b["epistemic"] < id_q99:
                group_rows.append({"group": "B_ood_below_baseline_id_q99", "ng": ng, "delay_ns": 120, "sample": sample, "baseline_epistemic": b["epistemic"], "balanced_epistemic": a["epistemic"], "delta_epistemic": a["epistemic"] - b["epistemic"], "delta_aleatoric": a["aleatoric"] - b["aleatoric"], "delta_kappa": a["kappa"] - b["kappa"], "delta_nu_margin": a["nu_margin"] - b["nu_margin"], "delta_psi": a["psi"] - b["psi"]})
        candidates = []
        for sample in range(200):
            b = grouped[("baseline_epoch3", ng, 120, sample)]
            a = grouped[("balanced_epoch3", ng, 120, sample)]
            candidates.append((a["epistemic"] - b["epistemic"], sample, b, a))
        baseline_q75 = float(np.quantile(ood_epi, .75))
        for delta, sample, b, a in sorted([x for x in candidates if x[2]["epistemic"] >= baseline_q75])[:10]:
            group_rows.append({"group": "C_ood_largest_negative_delta", "ng": ng, "delay_ns": 120, "sample": sample, "baseline_epistemic": b["epistemic"], "balanced_epistemic": a["epistemic"], "delta_epistemic": delta, "delta_aleatoric": a["aleatoric"] - b["aleatoric"], "delta_kappa": a["kappa"] - b["kappa"], "delta_nu_margin": a["nu_margin"] - b["nu_margin"], "delta_psi": a["psi"] - b["psi"]})
    write_csv(OUT / "matched_groups.csv", group_rows)
    (OUT / "group_definitions.json").write_text(json.dumps(group_meta, indent=2) + "\n")

    # Aggregate matched deltas by condition to expose which parameter moves with epistemic.
    aggregate = []
    for ng in NGS:
        for delay in REGIMES:
            subset = [row for row in matched if row["ng"] == ng and row["delay_ns"] == delay]
            out = {"ng": ng, "delay_ns": delay}
            for metric in ["aleatoric", "kappa", "nu_margin", "psi", "epistemic"]:
                values = np.array([row[f"delta_{metric}"] for row in subset])
                out[f"delta_{metric}_mean"] = float(values.mean())
                out[f"delta_{metric}_median"] = float(np.median(values))
                out[f"balanced_over_baseline_{metric}_median"] = float(np.median([row[f"ratio_{metric}"] for row in subset]))
            aggregate.append(out)
    write_csv(OUT / "matched_delta_summary.csv", aggregate)

    manifest = {"input": str(INPUT.relative_to(ROOT)), "models": MODELS, "ngs": NGS, "regimes_ns": REGIMES, "samples_per_condition": 200, "matched_protocol": "same existing per_sample artifact: same CFR sample, Ng, periodic mask, seed-derived noise protocol", "no_training": True, "no_postprocessing": True}
    (OUT / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
