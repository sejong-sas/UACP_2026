#!/usr/bin/env python3
"""STEP 1-3 diagnosis for the epoch-3 Fig.11 false fallback.

No threshold or controller is changed here.  Existing Eq.(12)/(13) scoring and
the existing global ID-only calibration are reproduced on the frozen epoch-3
checkpoint, then standalone/transition traces and Ng-conditioned ID summaries
are written for inspection.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
K = 1024
CHECKPOINT = ROOT / "runs/current_valid_baseline/training_convergence_100k_5ep_20260916_final_controlled/uacp_predictor_100k_5ep_epoch_3.pt"
DATA_DIR = ROOT / "runs/baseline_reproduction/step2_delay_sweep_repro/generated_data"
NGS = [4, 8, 16, 32, 64, 128]
ID_DELAYS = [10, 20, 40, 60, 80, 100]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def load_model(device):
    from scripts.diagnose_predictor import _make_model
    from scripts.train_predictor import load_config
    model = _make_model(load_config("configs/current_valid_baseline_100k1_seed_20260819.json"), device)
    payload = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    state = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
    model.load_state_dict(state); model.eval(); return model


def cfr_for(delay: int, count: int) -> np.ndarray:
    path = DATA_DIR / f"test_delay_{delay}_ns.npz"
    with np.load(path) as d: return np.asarray(d["cfr"][:count], dtype=np.complex64)


def score_batch(model, batch_np: np.ndarray, ng: int, seed: int, device: torch.device) -> dict[str, np.ndarray]:
    from scripts.train_predictor import set_seeds
    from src.training.data import build_noisy_sparse_input
    set_seeds(seed)
    batch = torch.from_numpy(batch_np).to(device)
    mask = torch.zeros((len(batch), K), dtype=torch.float32, device=device); mask[:, ::ng] = 1.0
    x, target, _ = build_noisy_sparse_input(batch, mask, 15.0)
    with torch.inference_mode():
        out = model(x); ale_map = out.psi / (out.nu_expanded - 2 * K - 1.0); epi_map = ale_map / out.kappa_expanded
        pair_ale = ale_map.reshape(len(batch), 2, 4, K).permute(0, 2, 1, 3).sum(2).mean(1)
        pair_epi = epi_map.reshape(len(batch), 2, 4, K).permute(0, 2, 1, 3).sum(2).mean(1)
        omitted = 1.0 - mask; count = omitted.sum(1)
        ale = (pair_ale * omitted).sum(1) / count.clamp_min(1.0); epi = (pair_epi * omitted).sum(1) / count.clamp_min(1.0)
        num = (((out.gamma - target).square() * omitted[:, None, :]).sum((1, 2)))
        den = ((target.square() * omitted[:, None, :]).sum((1, 2))).clamp_min(1e-12)
        nmse = 10 * torch.log10((num / den).clamp_min(1e-12))
    result = {"ale": ale.cpu().numpy(), "epi": epi.cpu().numpy(), "nmse": nmse.cpu().numpy(), "omitted_count": count.cpu().numpy()}
    del batch, mask, x, target, out, ale_map, epi_map, pair_ale, pair_epi, omitted, count
    if device.type == "cuda": torch.cuda.empty_cache()
    return result


def scores_for(model, delay: int, ng: int, count: int, seed: int, device: torch.device, batch_size: int) -> dict[str, np.ndarray]:
    cfr = cfr_for(delay, count); chunks = []
    for start in range(0, len(cfr), batch_size): chunks.append(score_batch(model, cfr[start:start + batch_size], ng, seed + start, device))
    del cfr; gc.collect()
    return {key: np.concatenate([c[key] for c in chunks]) for key in chunks[0]}


def global_threshold(model, device, seed: int, count: int, batch_size: int) -> tuple[float, dict]:
    # Reproduce scripts/fig11_dynamic_runtime_validation.py exactly: batch 8,
    # 40 ID samples/regime, and its per-batch seed schedule.
    values = []; by_ng = {}
    for ng in [128, 64, 32, 16, 8, 4]:
        parts = []
        for regime_index, delay in enumerate((20, 80)):
            cfr = cfr_for(delay, count); local = []
            for start in range(0, len(cfr), 8):
                # The existing Fig.11 calibration stores one batch-mean score
                # per calibration batch, not one score per sample.
                local.append(float(np.mean(score_batch(model, cfr[start:start + 8], ng, seed + 500000 + regime_index * 10000 + start, device)["epi"])))
            parts.extend(local)
        by_ng[str(ng)] = {"count": len(parts), "q99": float(np.quantile(parts, .99)), "max": float(np.max(parts))}; values.extend(parts)
    return float(np.quantile(values, .99)), {"method": "existing global ID-only q99 across 20/80 ns and candidate Ng", "count": len(values), "by_ng": by_ng}


def trace(model, device, delay_order: list[int], count: int, seed: int, threshold: float, batch_size: int, label: str) -> list[dict]:
    from scripts.train_predictor import set_seeds
    # Existing controller's aleatoric target/hysteresis source.
    prior = json.loads((ROOT / "runs/current_valid_baseline/paper_sample_alignment_100k_1ep_fig8_20260915/final_results.json").read_text())
    probes = prior["epoch_probe_results"]["1"]
    ale_target = (probes["ID-Easy 20 ns"]["aleatoric_paper_eq12_eq13"] + probes["ID-Hard 80 ns"]["aleatoric_paper_eq12_eq13"]) / 2.0; ale_delta = .10 * ale_target
    from scripts.fig11_dynamic_runtime_validation import CANDIDATE_NGS
    current = 128; rows = []; step = 0
    for delay in delay_order:
        cfr = cfr_for(delay, count)
        for i in range(count):
            set_seeds(seed + step); s = score_batch(model, cfr[i:i + 1], current, seed + step, device)
            epi = float(s["epi"][0]); ale = float(s["ale"][0]); nmse = float(s["nmse"][0]); omitted = int(s["omitted_count"][0]); ratio = epi / threshold if np.isfinite(epi) and threshold else float("nan")
            if np.isfinite(epi) and epi > threshold:
                next_ng, trigger, reason = 1, True, "epistemic_threshold"
            elif current == 1 or not np.isfinite(ale):
                next_ng, trigger, reason = current, False, "hold_no_omitted_set"
            else:
                idx = CANDIDATE_NGS.index(current)
                if ale > ale_target + ale_delta: next_ng, reason = CANDIDATE_NGS[min(idx + 1, len(CANDIDATE_NGS) - 1)], "aleatoric_denser"
                elif ale < ale_target - ale_delta: next_ng, reason = CANDIDATE_NGS[max(idx - 1, 0)], "aleatoric_sparser"
                else: next_ng, reason = current, "hysteresis_hold"
                trigger = False
            rows.append({"sequence": label, "time_step": step, "delay_ns": delay, "regime": f"{delay} ns", "previous_ng": current, "current_ng": current, "next_ng": next_ng, "epistemic": epi, "threshold": threshold, "epi_threshold_ratio": ratio, "aleatoric": ale, "nmse_db": nmse, "omitted_count": omitted, "ood_trigger": trigger, "decision_reason": reason})
            current = int(next_ng); step += 1
        del cfr
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--output-dir", required=True); ap.add_argument("--samples", type=int, default=200); ap.add_argument("--batch-size", type=int, default=64); ap.add_argument("--sequence-steps", type=int, default=40); ap.add_argument("--seed", type=int, default=20262000)
    a = ap.parse_args(); out = ROOT / a.output_dir
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True); sys.path.insert(0, str(ROOT)); device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available(): torch.cuda.set_device(0); torch.cuda.reset_peak_memory_stats()
    model = load_model(device); threshold, provenance = global_threshold(model, device, a.seed, 40, a.batch_size)
    standalone = trace(model, device, [10], a.sequence_steps, a.seed + 100000, threshold, a.batch_size, "10ns_standalone")
    transition = trace(model, device, [80, 10], a.sequence_steps, a.seed + 200000, threshold, a.batch_size, "80_to_10_transition")
    canonical = trace(model, device, [20, 80, 10, 40, 60, 120], a.sequence_steps, a.seed, threshold, a.batch_size, "canonical_sequence")
    write_csv(out / "step1_standalone_transition_trace.csv", standalone + transition)
    write_csv(out / "step1_canonical_trace.csv", canonical)
    score_rows = []; score_cache = {}; raw_score_rows = []
    for delay in ID_DELAYS:
        score_cache[delay] = {}
        for ng in NGS:
            s = scores_for(model, delay, ng, a.samples, a.seed + delay * 10000 + ng, device, a.batch_size); score_cache[delay][ng] = s
            epi = s["epi"]; score_rows.append({"delay_ns": delay, "ng": ng, "count": len(epi), "mean": float(np.mean(epi)), "median": float(np.median(epi)), "q95": float(np.quantile(epi,.95)), "q99": float(np.quantile(epi,.99)), "max": float(np.max(epi)), "global_threshold": threshold, "global_exceed_rate": float(np.mean(epi > threshold)), "global_exceed_count": int(np.sum(epi > threshold))})
            raw_score_rows.extend({"delay_ns": delay, "ng": ng, "sample": i, "epistemic": float(v)} for i, v in enumerate(epi))
    write_csv(out / "step3_id_ng_distribution.csv", score_rows)
    write_csv(out / "step3_id_ng_scores.csv", raw_score_rows)
    fig, ax = plt.subplots(figsize=(10, 5)); ax.plot([r["time_step"] for r in canonical], [r["epistemic"] for r in canonical], label="Epistemic"); ax.axhline(threshold, color="r", ls="--", label="global threshold"); ax.set_xlabel("time step"); ax.set_ylabel("Eq.(13) epistemic"); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(out / "step1_canonical_epistemic_trace.png", dpi=180); plt.close(fig)
    result = {"checkpoint": str(CHECKPOINT), "device": str(device), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "samples_step3": a.samples, "global_threshold": threshold, "global_threshold_provenance": provenance, "step1_standalone_trigger_count": sum(r["ood_trigger"] for r in standalone), "step1_transition_trigger_count": sum(r["ood_trigger"] for r in transition), "canonical_trigger_count": sum(r["ood_trigger"] for r in canonical), "canonical_trigger_rows": [r for r in canonical if r["ood_trigger"]], "implementation_assumptions": ["existing global ID-only q99 threshold and hysteresis source reused", "direct-CFR AWGN 15 dB and existing delay-sweep files", "Ng=1 is full-feedback/empty omitted set; omitted metrics are N/A", "Ng-conditioned threshold is not applied in this STEP 1-3 run"], "no_training": True}
    (out / "results.json").write_text(json.dumps(result, indent=2, allow_nan=True))


if __name__ == "__main__": main()
