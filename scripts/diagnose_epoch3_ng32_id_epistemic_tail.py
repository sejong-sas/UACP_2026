#!/usr/bin/env python3
"""Diagnostic-only decomposition of the Ng32 ID Epistemic upper tail."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CHECKPOINT = ROOT / "runs/current_valid_baseline/training_convergence_100k_5ep_20260916_final_controlled/uacp_predictor_100k_5ep_epoch_3.pt"
DATA = ROOT / "runs/baseline_reproduction/step2_delay_sweep_repro/generated_data"
OUT = ROOT / "runs/current_valid_baseline/epoch3_ng32_id_epistemic_tail_diagnosis_20260920"
NGS = [16, 32]
ID_DELAYS = [20, 40, 60, 80, 100]
OOD_DELAY = 120
K = 1024
SAMPLES = 200
BATCH_SIZE = 32
SEED = 20262000


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stats(values: list[float] | np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    return {"n": int(x.size), "mean": float(x.mean()), "median": float(np.median(x)),
            "q05": float(np.quantile(x, .05)), "q95": float(np.quantile(x, .95)),
            "q99": float(np.quantile(x, .99)), "max": float(x.max())}


def load_model(device):
    from scripts.diagnose_predictor import _make_model
    from scripts.train_predictor import load_config
    model = _make_model(load_config(ROOT / "configs/current_valid_baseline_100k1_seed_20260819.json"), device)
    state = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    state = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
    model.load_state_dict(state)
    model.eval()
    return model


def eval_batch(model, cfr_np, ng, seed, device):
    from scripts.train_predictor import set_seeds
    from src.training.data import build_noisy_sparse_input
    set_seeds(seed)
    cfr = torch.from_numpy(cfr_np).to(device)
    mask = torch.zeros((len(cfr), K), device=device)
    mask[:, ::ng] = 1.0  # canonical evaluator: fixed offset 0
    x, target, _ = build_noisy_sparse_input(cfr, mask, 15.0)
    with torch.inference_mode():
        h = model.input_projection(x)
        for block in model.residual_blocks:
            h = block(h)
        pooled = model.pool(h)
        raw_kappa = model.kappa_head(pooled)
        raw_nu = model.nu_head(pooled)
        out = model(x)
        ale_map = out.psi / (out.nu_expanded - 2 * K - 1.0)
        epi_map = ale_map / out.kappa_expanded
        omitted = 1.0 - mask
        omitted_count = omitted.sum(1).clamp_min(1.0)
        # Match canonical evaluator's pair/component aggregation.
        pa = ale_map.reshape(len(cfr), 2, 4, K).permute(0, 2, 1, 3).sum(2).mean(1)
        pe = epi_map.reshape(len(cfr), 2, 4, K).permute(0, 2, 1, 3).sum(2).mean(1)
        ale = (pa * omitted).sum(1) / omitted_count
        epi = (pe * omitted).sum(1) / omitted_count
        nmse = 10.0 * torch.log10((((out.gamma - target).square() * omitted[:, None, :]).sum((1, 2)) /
                                    (target.square() * omitted[:, None, :]).sum((1, 2)).clamp_min(1e-12)).clamp_min(1e-12))
        om = omitted[:, None, :].expand_as(ale_map)
        params = {
            "kappa": (out.kappa_expanded * om).sum((1, 2)) / om.sum((1, 2)).clamp_min(1),
            "nu_margin": ((out.nu_expanded - (2 * K + 1.0)) * om).sum((1, 2)) / om.sum((1, 2)).clamp_min(1),
            "psi": (out.psi * om).sum((1, 2)) / om.sum((1, 2)).clamp_min(1),
        }
        raw_kappa_mean = raw_kappa.mean((1, 2))
        raw_nu_mean = raw_nu.mean((1, 2))
        mag = cfr.abs()
        adjacent = (cfr[:, 1:] - cfr[:, :-1]).abs()
        features = {
            "cfr_mag_mean": mag.mean((1, 2, 3)),
            "cfr_mag_std": mag.std((1, 2, 3), unbiased=False),
            "cfr_adjacent_diff_mean": adjacent.mean((1, 2, 3)),
            "cfr_peak_to_mean": mag.amax((1, 2, 3)) / mag.mean((1, 2, 3)).clamp_min(1e-12),
        }
    result = {"nmse": nmse, "aleatoric": ale, "epistemic": epi,
              **params, "raw_kappa": raw_kappa_mean, "raw_nu": raw_nu_mean, **features}
    return {key: value.detach().cpu().numpy() for key, value in result.items()}


def group_name(epi, q50, q95, q99):
    if epi >= q99:
        return "extreme_q99_plus"
    if epi >= q95:
        return "upper_q95_99"
    if epi <= q50:
        return "normal_0_50"
    return "middle_50_95"


def group_summary(rows, group):
    part = [r for r in rows if r["group"] == group]
    result = {"group": group, "n": len(part)}
    for key in ["nmse", "kappa", "nu_margin", "psi", "aleatoric", "epistemic", "raw_kappa", "raw_nu"]:
        s = stats([r[key] for r in part])
        for q in ["mean", "median", "q05", "q95"]:
            result[f"{key}_{q}"] = s[q]
    return result


def nearest_match(extreme, normal):
    available = set(range(len(normal)))
    pairs = []
    for e in sorted(extreme, key=lambda r: r["nmse"]):
        if not available:
            break
        j = min(available, key=lambda idx: abs(normal[idx]["nmse"] - e["nmse"]))
        available.remove(j)
        pairs.append((e, normal[j]))
    return pairs


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    if not torch.cuda.is_available() or "GB10" not in torch.cuda.get_device_name(0):
        raise RuntimeError("NVIDIA GB10 / cuda:0 is required")
    model = load_model(device)
    rows = []
    arrays = {}
    for delay_index, delay in enumerate(ID_DELAYS + [OOD_DELAY]):
        with np.load(DATA / f"test_delay_{delay}_ns.npz") as archive:
            cfr = np.asarray(archive["cfr"][:SAMPLES], dtype=np.complex64)
        for ng in NGS:
            pieces = []
            for start in range(0, SAMPLES, BATCH_SIZE):
                pieces.append(eval_batch(model, cfr[start:start + BATCH_SIZE], ng, SEED + delay_index * 100000 + ng + start, device))
            values = {key: np.concatenate([piece[key] for piece in pieces]) for key in pieces[0]}
            arrays[(ng, delay)] = values
            for sample in range(SAMPLES):
                rows.append({"sample": sample, "delay_ns": delay, "ng": ng, "offset": 0,
                             "mask_id": f"periodic_offset0_ng{ng}",
                             "observed_indices_rule": f"0::{ng}",
                             **{key: float(value[sample]) for key, value in values.items()}})

    id32 = [r for r in rows if r["ng"] == 32 and r["delay_ns"] in ID_DELAYS]
    q50, q95, q99 = np.quantile([r["epistemic"] for r in id32], [0.50, 0.95, 0.99])
    for r in rows:
        r["group"] = group_name(r["epistemic"], q50, q95, q99) if r["ng"] == 32 and r["delay_ns"] in ID_DELAYS else ("ood_reference" if r["delay_ns"] == OOD_DELAY else "ng16_reference")
    write_csv(OUT / "per_sample.csv", rows)
    (OUT / "tail_thresholds.json").write_text(json.dumps({"ng32_id_p50": float(q50), "ng32_id_p95": float(q95), "ng32_id_p99": float(q99), "samples_per_regime": SAMPLES, "seed": SEED, "mask_offset": 0}, indent=2) + "\n")

    write_csv(OUT / "group_summary.csv", [group_summary(id32, g) for g in ["normal_0_50", "upper_q95_99", "extreme_q99_plus"]])
    delay_rows = []
    for delay in ID_DELAYS:
        part = [r for r in id32 if r["delay_ns"] == delay]
        delay_rows.append({"delay_ns": delay, "total_samples": len(part), "q95_plus_count": sum(r["epistemic"] >= q95 for r in part), "q95_plus_rate": np.mean([r["epistemic"] >= q95 for r in part]), "q99_count": sum(r["epistemic"] >= q99 for r in part), "q99_rate": np.mean([r["epistemic"] >= q99 for r in part])})
    write_csv(OUT / "delay_concentration.csv", delay_rows)

    offset_rows = []
    for ng in NGS:
        part = [r for r in rows if r["ng"] == ng and r["delay_ns"] in ID_DELAYS]
        offset_rows.append({"ng": ng, "offset": 0, "samples": len(part), "q95_count": sum(r["epistemic"] >= np.quantile([x["epistemic"] for x in part], .95) for r in part), "q99_count": sum(r["epistemic"] >= np.quantile([x["epistemic"] for x in part], .99) for r in part), "q99_rate": .01, "kappa_median": np.median([r["kappa"] for r in part]), "epistemic_median": np.median([r["epistemic"] for r in part]), "nmse_median": np.median([r["nmse"] for r in part])})
    write_csv(OUT / "offset_concentration.csv", offset_rows)

    # Error-matched extreme vs normal, within delay first.
    match_rows = []
    for delay in ID_DELAYS:
        extreme = [r for r in id32 if r["delay_ns"] == delay and r["group"] == "extreme_q99_plus"]
        normal = [r for r in id32 if r["delay_ns"] == delay and r["group"] == "normal_0_50"]
        for e, n in nearest_match(extreme, normal):
            match_rows.append({"delay_ns": delay, "extreme_sample": e["sample"], "normal_sample": n["sample"], "nmse_extreme": e["nmse"], "nmse_normal": n["nmse"], "nmse_abs_diff": abs(e["nmse"] - n["nmse"]), "kappa_extreme": e["kappa"], "kappa_normal": n["kappa"], "nu_margin_extreme": e["nu_margin"], "nu_margin_normal": n["nu_margin"], "aleatoric_extreme": e["aleatoric"], "aleatoric_normal": n["aleatoric"], "epistemic_extreme": e["epistemic"], "epistemic_normal": n["epistemic"]})
    write_csv(OUT / "error_matched_extreme_normal.csv", match_rows or [{"delay_ns": "none", "note": "No same-delay pairs"}])
    match_summary = []
    if match_rows:
        for key in ["nmse_abs_diff", "kappa_extreme", "kappa_normal", "nu_margin_extreme", "nu_margin_normal", "aleatoric_extreme", "aleatoric_normal", "epistemic_extreme", "epistemic_normal"]:
            match_summary.append({"metric": key, **stats([r[key] for r in match_rows])})
    write_csv(OUT / "error_matched_summary.csv", match_summary or [{"metric": "none", "n": 0}])

    ng_ref = []
    for ng in NGS:
        part = [r for r in rows if r["ng"] == ng and r["delay_ns"] in ID_DELAYS]
        threshold = np.quantile([r["epistemic"] for r in part], .99)
        tail = [r for r in part if r["epistemic"] >= threshold]
        counts = {d: sum(r["delay_ns"] == d for r in tail) for d in ID_DELAYS}
        ng_ref.append({"ng": ng, "tail_n": len(tail), "kappa_median": np.median([r["kappa"] for r in tail]), "nu_margin_median": np.median([r["nu_margin"] for r in tail]), "aleatoric_median": np.median([r["aleatoric"] for r in tail]), "epistemic_median": np.median([r["epistemic"] for r in tail]), "nmse_median": np.median([r["nmse"] for r in tail]), "dominant_delay": max(counts, key=counts.get), "dominant_delay_count": max(counts.values())})
    ood32 = [r for r in rows if r["ng"] == 32 and r["delay_ns"] == OOD_DELAY]
    ng_ref.append({"ng": "32_ood120", "tail_n": len(ood32), "kappa_median": np.median([r["kappa"] for r in ood32]), "nu_margin_median": np.median([r["nu_margin"] for r in ood32]), "aleatoric_median": np.median([r["aleatoric"] for r in ood32]), "epistemic_median": np.median([r["epistemic"] for r in ood32]), "nmse_median": np.median([r["nmse"] for r in ood32]), "dominant_delay": 120, "dominant_delay_count": len(ood32)})
    write_csv(OUT / "ng16_ng32_ood_reference.csv", ng_ref)

    # Simple plots, diagnostic only.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ng, color in [(16, "tab:blue"), (32, "tab:orange")]:
        vals = [np.median([r["epistemic"] for r in rows if r["ng"] == ng and r["delay_ns"] == d]) for d in ID_DELAYS + [OOD_DELAY]]
        axes[0].plot(ID_DELAYS + [OOD_DELAY], vals, "o-", label=f"Ng={ng}", color=color)
        axes[1].scatter([r["nmse"] for r in rows if r["ng"] == ng and r["delay_ns"] in ID_DELAYS], [r["kappa"] for r in rows if r["ng"] == ng and r["delay_ns"] in ID_DELAYS], s=5, alpha=.25, label=f"Ng={ng}", color=color)
    axes[0].set_xlabel("delay (ns)"); axes[0].set_ylabel("Epistemic median"); axes[0].legend(); axes[0].grid(alpha=.25)
    axes[1].set_xlabel("NMSE (dB)"); axes[1].set_ylabel("κ"); axes[1].legend(); axes[1].grid(alpha=.25)
    fig.tight_layout(); fig.savefig(OUT / "tail_diagnostic_plots.png", dpi=180); plt.close(fig)

    manifest = {"checkpoint": str(CHECKPOINT.relative_to(ROOT)), "device": str(device), "gpu": torch.cuda.get_device_name(0), "ngs": NGS, "id_delays_ns": ID_DELAYS, "ood_reference_ns": OOD_DELAY, "samples_per_regime": SAMPLES, "seed": SEED, "no_training": True, "no_optimizer_step": True, "mask_protocol": "canonical fixed periodic mask mask[:,::Ng], offset=0; observed indices are 0,Ng,2Ng,...", "offset_audit": "Only offset 0 exists in canonical evaluator; no offset rate comparison is identifiable without changing protocol.", "raw_parameter_note": "raw_kappa/raw_nu captured from the same model heads before softplus admissibility transform; aggregated over pair scalars.", "tail_definition": "Ng32 pooled ID 20-100 ns Epistemic: normal <= p50, upper p95-p99, extreme >= p99", "diagnostic_only": True}
    (OUT / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
