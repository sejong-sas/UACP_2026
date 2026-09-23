#!/usr/bin/env python3
"""Factorial Ng×delay diagnosis for the existing epoch3 checkpoint only."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "runs/current_valid_baseline/training_convergence_100k_5ep_20260916_final_controlled/uacp_predictor_100k_5ep_epoch_3.pt"
DATA = ROOT / "runs/baseline_reproduction/step2_delay_sweep_repro/generated_data"
OUT = ROOT / "runs/current_valid_baseline/epoch3_ng_delay_factorial_diagnosis_20260920"
NGS = [4, 8, 16, 32]
DELAYS = [20, 40, 60, 80, 100, 120]
ID_DELAYS = [20, 40, 60, 80, 100]
SAMPLES = 200
BATCH_SIZE = 64
SEED = 20262000


def stats(values):
    values = np.asarray(values, dtype=float)
    return {"count": int(values.size), "mean": float(values.mean()), "median": float(np.median(values)), "q05": float(np.quantile(values, .05)), "q50": float(np.quantile(values, .50)), "q95": float(np.quantile(values, .95)), "q99": float(np.quantile(values, .99)), "max": float(values.max())}


def auc(negative, positive):
    ranks = rankdata(np.concatenate([negative, positive]), method="average")
    return float((ranks[len(negative):].sum() - len(positive) * (len(positive) + 1) / 2) / (len(negative) * len(positive)))


def overlap(left, right):
    left = left[left > 0]
    right = right[right > 0]
    edges = np.linspace(np.log10(np.concatenate([left, right])).min(), np.log10(np.concatenate([left, right])).max(), 81)
    lh, _ = np.histogram(np.log10(left), bins=edges, density=True)
    rh, _ = np.histogram(np.log10(right), bins=edges, density=True)
    return float(np.sum(np.minimum(lh, rh) * np.diff(edges)))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    import sys
    sys.path.insert(0, str(ROOT))
    import scripts.analyze_epoch3_id_vs_120_epistemic as canonical
    from scripts.diagnose_predictor import _make_model
    from scripts.train_predictor import load_config

    device = torch.device("cuda:0")
    if not torch.cuda.is_available() or "GB10" not in torch.cuda.get_device_name(0):
        raise RuntimeError("NVIDIA GB10 / cuda:0 is required")
    config = load_config(ROOT / "configs/current_valid_baseline_100k1_seed_20260819.json")
    model = _make_model(config, device)
    state = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    state = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
    model.load_state_dict(state)
    model.eval()

    records = []
    arrays = {}
    for delay_index, delay in enumerate(DELAYS):
        with np.load(DATA / f"test_delay_{delay}_ns.npz") as archive:
            cfr = np.asarray(archive["cfr"][:SAMPLES], dtype=np.complex64)
        for ng in NGS:
            pieces = []
            for start in range(0, SAMPLES, BATCH_SIZE):
                pieces.append(canonical.eval_batch(model, cfr[start:start + BATCH_SIZE], ng, SEED + delay_index * 100000 + ng + start, device))
            values = {key: np.concatenate([part[key] for part in pieces]) for key in pieces[0]}
            arrays[(ng, delay)] = values
            for sample in range(SAMPLES):
                records.append({"ng": ng, "delay_ns": delay, "sample": sample, **{key: float(value[sample]) for key, value in values.items()}})
    write_csv(OUT / "per_sample.csv", records)

    summary_rows = []
    for delay in DELAYS:
        for ng in NGS:
            row = {"ng": ng, "delay_ns": delay}
            for metric in ["nmse", "psi", "kappa", "nu_margin", "aleatoric", "epistemic"]:
                for key, value in stats(arrays[(ng, delay)][metric]).items():
                    row[f"{metric}_{key}"] = value
            summary_rows.append(row)
    write_csv(OUT / "summary.csv", summary_rows)

    sparsity_rows = []
    for delay in DELAYS:
        a, b, c, d = [arrays[(ng, delay)] for ng in NGS]
        sparsity_rows.append({"delay_ns": delay, "kappa_32_over_4": np.median(d["kappa"]) / np.median(a["kappa"]), "kappa_32_over_16": np.median(d["kappa"]) / np.median(c["kappa"]), "epistemic_32_over_16": np.median(d["epistemic"]) / np.median(c["epistemic"]), "nmse_32_minus_4_db": np.median(d["nmse"]) - np.median(a["nmse"]), "kappa_4_median": np.median(a["kappa"]), "kappa_8_median": np.median(b["kappa"]), "kappa_16_median": np.median(c["kappa"]), "kappa_32_median": np.median(d["kappa"])})
    write_csv(OUT / "sparsity_effect.csv", sparsity_rows)

    difficulty_rows = []
    for ng in NGS:
        first, last = arrays[(ng, 20)], arrays[(ng, 100)]
        difficulty_rows.append({"ng": ng, "kappa_100_over_20": np.median(last["kappa"]) / np.median(first["kappa"]), "epistemic_100_over_20": np.median(last["epistemic"]) / np.median(first["epistemic"]), "aleatoric_100_over_20": np.median(last["aleatoric"]) / np.median(first["aleatoric"]), "nmse_100_minus_20_db": np.median(last["nmse"]) - np.median(first["nmse"]), "kappa_20_median": np.median(first["kappa"]), "kappa_100_median": np.median(last["kappa"])})
    write_csv(OUT / "difficulty_effect.csv", difficulty_rows)

    # Two-factor decomposition on log median kappa; descriptive only.
    matrix = np.array([[np.log(next(x for x in summary_rows if x["ng"] == ng and x["delay_ns"] == delay)["kappa_median"]) for delay in ID_DELAYS] for ng in NGS])
    grand = matrix.mean(); row_means = matrix.mean(axis=1); col_means = matrix.mean(axis=0)
    total_ss = float(((matrix - grand) ** 2).sum())
    ng_ss = float(len(ID_DELAYS) * ((row_means - grand) ** 2).sum())
    delay_ss = float(len(NGS) * ((col_means - grand) ** 2).sum())
    interaction = matrix - row_means[:, None] - col_means[None, :] + grand
    interaction_ss = float((interaction ** 2).sum())
    effects = {"scale": "log(median kappa)", "ng_levels": NGS, "delay_levels_ns": ID_DELAYS, "total_ss": total_ss, "ng_main_ss": ng_ss, "delay_main_ss": delay_ss, "interaction_ss": interaction_ss, "ng_main_fraction": ng_ss / total_ss, "delay_main_fraction": delay_ss / total_ss, "interaction_fraction": interaction_ss / total_ss, "residual_fraction": max(0.0, 1.0 - (ng_ss + delay_ss + interaction_ss) / total_ss)}
    (OUT / "two_factor_effects.json").write_text(json.dumps(effects, indent=2) + "\n")

    correlation_rows = []
    for ng in NGS:
        errors = np.concatenate([arrays[(ng, delay)]["nmse"] for delay in ID_DELAYS])
        kappas = np.concatenate([arrays[(ng, delay)]["kappa"] for delay in ID_DELAYS])
        epistemic = np.concatenate([arrays[(ng, delay)]["epistemic"] for delay in ID_DELAYS])
        correlation_rows.append({"ng": ng, "spearman_nmse_vs_kappa": float(spearmanr(errors, kappas).statistic), "spearman_nmse_vs_epistemic": float(spearmanr(errors, epistemic).statistic), "n": len(errors)})
    write_csv(OUT / "reconstruction_uncertainty_correlations.csv", correlation_rows)

    boundary_rows = []
    for ng in NGS:
        a, b = arrays[(ng, 100)], arrays[(ng, 120)]
        boundary_rows.append({"ng": ng, "kappa_120_over_100": np.median(b["kappa"]) / np.median(a["kappa"]), "epistemic_120_over_100": np.median(b["epistemic"]) / np.median(a["epistemic"]), "aleatoric_120_over_100": np.median(b["aleatoric"]) / np.median(a["aleatoric"]), "nmse_120_minus_100_db": np.median(b["nmse"]) - np.median(a["nmse"]), "kappa_overlap": overlap(a["kappa"], b["kappa"]), "epistemic_overlap": overlap(a["epistemic"], b["epistemic"]), "kappa_auroc": auc(a["kappa"], b["kappa"]), "epistemic_auroc": auc(a["epistemic"], b["epistemic"])})
    write_csv(OUT / "boundary_100_vs_120.csv", boundary_rows)

    # Heatmaps and reconstruction-error scatter summaries.
    for metric, title in [("kappa", "κ median"), ("epistemic", "Epistemic median")]:
        matrix = np.array([[next(x for x in summary_rows if x["ng"] == ng and x["delay_ns"] == delay)[f"{metric}_median"] for ng in NGS] for delay in DELAYS])
        fig, ax = plt.subplots(figsize=(7, 4.5)); image = ax.imshow(matrix, aspect="auto"); ax.set_xticks(range(len(NGS)), NGS); ax.set_yticks(range(len(DELAYS)), DELAYS); ax.set_xlabel("Ng"); ax.set_ylabel("Delay spread (ns)"); ax.set_title(title); fig.colorbar(image, ax=ax); fig.tight_layout(); fig.savefig(OUT / f"{metric}_heatmap.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ng, color in zip(NGS, ["C0", "C1", "C2", "C3"]):
        errors = np.concatenate([arrays[(ng, delay)]["nmse"] for delay in ID_DELAYS]); kappas = np.concatenate([arrays[(ng, delay)]["kappa"] for delay in ID_DELAYS]); epis = np.concatenate([arrays[(ng, delay)]["epistemic"] for delay in ID_DELAYS]); axes[0].scatter(errors, kappas, s=4, alpha=.25, color=color, label=f"Ng={ng}"); axes[1].scatter(errors, epis, s=4, alpha=.25, color=color, label=f"Ng={ng}")
    axes[0].set_xlabel("NMSE (dB)"); axes[0].set_ylabel("κ"); axes[1].set_xlabel("NMSE (dB)"); axes[1].set_ylabel("Epistemic"); axes[0].legend(); axes[1].legend(); fig.tight_layout(); fig.savefig(OUT / "reconstruction_vs_uncertainty.png", dpi=180); plt.close(fig)

    manifest = {"checkpoint": str(CHECKPOINT.relative_to(ROOT)), "device": str(device), "gpu": torch.cuda.get_device_name(0), "ngs": NGS, "id_delays_ns": ID_DELAYS, "ood_reference_delay_ns": 120, "samples_per_grid_point": SAMPLES, "batch_size": BATCH_SIZE, "protocol": "canonical eval_batch; direct-CFR AWGN 15 dB; periodic mask; same sample indices and seed policy per Ng×delay", "no_training": True, "120ns_gradient_not_used": True, "two_factor_scale": "log median kappa, descriptive only"}
    (OUT / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
