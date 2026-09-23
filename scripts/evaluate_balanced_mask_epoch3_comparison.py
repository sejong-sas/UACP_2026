#!/usr/bin/env python3
"""Evaluate baseline and balanced epoch3 checkpoints with the canonical evaluator."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "runs/current_valid_baseline/training_convergence_100k_5ep_20260916_final_controlled/uacp_predictor_100k_5ep_epoch_3.pt"
BALANCED = ROOT / "runs/current_valid_baseline/training_balanced_mask_100k_3ep_20260919/uacp_predictor_balanced_100k_3ep_epoch_3.pt"
DATA = ROOT / "runs/baseline_reproduction/step2_delay_sweep_repro/generated_data"
DELAYS = [10, 20, 40, 60, 80, 100, 120, 1_000_000]
NGS = [16, 32]
SAMPLES = 200
BATCH_SIZE = 64
SEED = 20262000


def auc(negative, positive):
    ranks = rankdata(np.concatenate([negative, positive]), method="average")
    return float((ranks[len(negative):].sum() - len(positive) * (len(positive) + 1) / 2) / (len(negative) * len(positive)))


def overlap(left, right, bins=80):
    left = left[left > 0]
    right = right[right > 0]
    edges = np.linspace(np.log10(np.concatenate([left, right])).min(), np.log10(np.concatenate([left, right])).max(), bins + 1)
    left_hist, _ = np.histogram(np.log10(left), bins=edges, density=True)
    right_hist, _ = np.histogram(np.log10(right), bins=edges, density=True)
    return float(np.sum(np.minimum(left_hist, right_hist) * np.diff(edges)))


def load_model(checkpoint, device):
    from scripts.diagnose_predictor import _make_model
    from scripts.train_predictor import load_config
    model = _make_model(load_config(ROOT / "configs/current_valid_baseline_100k1_seed_20260819.json"), device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    state = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
    model.load_state_dict(state)
    model.eval()
    return model


def summarize(values):
    return {"mean": float(np.mean(values)), "median": float(np.median(values)), "q95": float(np.quantile(values, .95)), "q99": float(np.quantile(values, .99)), "max": float(np.max(values))}


def main():
    parser = __import__("argparse").ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = ROOT / args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT))
    import scripts.analyze_epoch3_id_vs_120_epistemic as canonical
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" or "GB10" not in torch.cuda.get_device_name(0):
        raise RuntimeError("NVIDIA GB10 / cuda:0 is required")
    records = []
    summaries = []
    arrays = {}
    for model_name, checkpoint in (("baseline_epoch3", BASELINE), ("balanced_epoch3", BALANCED)):
        model = load_model(checkpoint, device)
        for delay_index, delay in enumerate(DELAYS):
            file_name = "test_delay_1_ms.npz" if delay == 1_000_000 else f"test_delay_{delay}_ns.npz"
            with np.load(DATA / file_name) as data:
                cfr = np.asarray(data["cfr"][:SAMPLES], dtype=np.complex64)
            for ng in NGS:
                pieces = []
                for start in range(0, len(cfr), BATCH_SIZE):
                    pieces.append(canonical.eval_batch(model, cfr[start:start + BATCH_SIZE], ng, SEED + delay_index * 100000 + ng + start, device))
                values = {key: np.concatenate([piece[key] for piece in pieces]) for key in pieces[0]}
                arrays[(model_name, delay, ng)] = values
                for index in range(SAMPLES):
                    records.append({"model": model_name, "delay_ns": delay, "regime": "OOD-Far 1 ms" if delay == 1_000_000 else str(delay), "ng": ng, "sample": index, **{key: float(value[index]) for key, value in values.items()}})
        for ng in NGS:
            id_epi = np.concatenate([arrays[(model_name, delay, ng)]["epistemic"] for delay in [10, 20, 40, 60, 80, 100]])
            near = arrays[(model_name, 120, ng)]["epistemic"]
            far = arrays[(model_name, 1_000_000, ng)]["epistemic"]
            pooled = np.concatenate([near, far])
            tau = float(np.quantile(id_epi, .99))
            row = {"model": model_name, "ng": ng, "id_count": len(id_epi), "id_q99_threshold": tau,
                   "near_auroc": auc(id_epi, near), "far_auroc": auc(id_epi, far), "pooled_ood_auroc": auc(id_epi, pooled),
                   "near_overlap": overlap(id_epi, near), "far_overlap": overlap(id_epi, far),
                   "near_tpr_at_id_q99": float(np.mean(near > tau)), "far_tpr_at_id_q99": float(np.mean(far > tau)),
                   "id_fpr_at_id_q99": float(np.mean(id_epi > tau)), "finite_all": True}
            for delay in [20, 80, 120, 1_000_000]:
                values = arrays[(model_name, delay, ng)]
                label = "1ms" if delay == 1_000_000 else str(delay)
                for metric in ["nmse", "aleatoric", "epistemic", "kappa", "nu_margin", "psi"]:
                    row[f"{label}_{metric}_median"] = float(np.median(values[metric]))
                    row[f"{label}_{metric}_q95"] = float(np.quantile(values[metric], .95))
                    row[f"{label}_{metric}_q99"] = float(np.quantile(values[metric], .99))
                    row[f"{label}_{metric}_finite"] = int(np.isfinite(values[metric]).sum())
            summaries.append(row)
        del model
        torch.cuda.empty_cache()
    with (out / "per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    (out / "results.json").write_text(json.dumps({"baseline_checkpoint": str(BASELINE.relative_to(ROOT)), "balanced_checkpoint": str(BALANCED.relative_to(ROOT)), "device": str(device), "gpu": torch.cuda.get_device_name(0), "samples_per_regime": SAMPLES, "ngs": NGS, "delays_ns": DELAYS, "protocol_reused": "scripts/analyze_epoch3_id_vs_120_epistemic.py; direct-CFR AWGN 15 dB; periodic mask; omitted aggregation", "no_training": True}, indent=2) + "\n")


if __name__ == "__main__":
    main()
