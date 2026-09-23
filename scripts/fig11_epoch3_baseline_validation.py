#!/usr/bin/env python3
"""Figure-11 dynamic validation using the frozen epoch-3 checkpoint.

The controller functions and threshold calibration are imported unchanged from
the repository implementation.  No model update or checkpoint write occurs.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "runs/current_valid_baseline/training_convergence_100k_5ep_20260916_final_controlled/uacp_predictor_100k_5ep_epoch_3.pt"
SEQUENCE_DIR = ROOT / "runs/baseline_reproduction/step2_delay_sweep_repro/generated_data"
REGIME_ORDER = [("20 ns", 20.0), ("80 ns", 80.0), ("10 ns", 10.0), ("40 ns", 40.0), ("60 ns", 60.0), ("120 ns", 120.0)]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--output-dir", required=True); ap.add_argument("--segment-length", type=int, default=40); ap.add_argument("--runtime-repeats", type=int, default=100); ap.add_argument("--eval-seed", type=int, default=20262000)
    args = ap.parse_args(); out = ROOT / args.output_dir
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True); sys.path.insert(0, str(ROOT))
    import scripts.fig11_dynamic_runtime_validation as canonical
    from scripts.train_predictor import load_config, set_seeds
    from src.training.data import build_noisy_sparse_input
    canonical.CHECKPOINT = CHECKPOINT; canonical.SEQUENCE_DIR = SEQUENCE_DIR
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available(): torch.cuda.set_device(0); torch.cuda.reset_peak_memory_stats()
    cfg = load_config("configs/current_valid_baseline_100k1_seed_20260819.json")
    model, _ = canonical.load_model(cfg, device)
    threshold, threshold_provenance = canonical.calibrate_epi_threshold(model, device, args.eval_seed)
    # Preserve the existing controller target source and hysteresis rule.
    prior = json.loads((ROOT / "runs/current_valid_baseline/paper_sample_alignment_100k_1ep_fig8_20260915/final_results.json").read_text())
    probes = prior["epoch_probe_results"]["1"]
    ale_target = (probes["ID-Easy 20 ns"]["aleatoric_paper_eq12_eq13"] + probes["ID-Hard 80 ns"]["aleatoric_paper_eq12_eq13"]) / 2.0
    ale_delta = 0.10 * ale_target
    sequence = []
    for regime, delay in REGIME_ORDER:
        path = SEQUENCE_DIR / ("test_delay_1_ms.npz" if delay == 1_000_000 else f"test_delay_{int(delay)}_ns.npz")
        with np.load(path) as data: cfr = data["cfr"][:args.segment_length]
        sequence.extend((regime, delay, sample) for sample in cfr)
    trace = []; current_ng = 128; previous = None
    for step, (regime, delay, cfr_np) in enumerate(sequence):
        if regime != previous: previous = regime
        set_seeds(args.eval_seed + step)
        cfr = torch.from_numpy(cfr_np[None]).to(device); mask = canonical.make_mask(1, current_ng, device)
        x, target, _ = build_noisy_sparse_input(cfr, mask, 15.0)
        with torch.inference_mode():
            output = model(x); ale_map = output.psi / (output.nu_expanded - 2 * canonical.K - 1.0); epi_map = ale_map / output.kappa_expanded
            ale_score, epi_score = canonical.aggregate_omitted_scores(ale_map, epi_map, mask); nmse = canonical.nmse_omitted_db(output.gamma, target, mask)
        decision = canonical.select_next_ng(current_ng, ale_score, epi_score, threshold, ale_target, ale_delta)
        trace.append({"time_step": step, "regime": regime, "delay_spread_ns": delay, "ood": delay > 100, "current_ng": current_ng, "predicted_next_ng": decision["next_ng"], "aleatoric": ale_score, "epistemic": epi_score, "epistemic_threshold": threshold, "nmse_db": nmse, "full_feedback_fallback": decision["full_feedback_fallback"], "adaptation_trigger": decision["adaptation_trigger"], "decision_reason": decision["decision_reason"]})
        current_ng = int(decision["next_ng"])
    # Inference/controller timing only; no adaptation optimizer step is run.
    sample = torch.from_numpy(sequence[0][2][None]).to(device); mask = canonical.make_mask(1, 16, device); x, _, _ = build_noisy_sparse_input(sample, mask, 15.0)
    with torch.inference_mode(): warm = model(x); ale = warm.psi / (warm.nu_expanded - 2 * canonical.K - 1.0); epi = ale / warm.kappa_expanded
    stages = {"predictor_forward": [], "uncertainty_aggregation": [], "controller_decision": [], "end_to_end_inference_controller": []}
    for _ in range(args.runtime_repeats):
        if device.type == "cuda": torch.cuda.synchronize(device)
        t = time.perf_counter(); z = model(x)
        if device.type == "cuda": torch.cuda.synchronize(device)
        stages["predictor_forward"].append((time.perf_counter()-t)*1000)
        if device.type == "cuda": torch.cuda.synchronize(device)
        t = time.perf_counter(); canonical.aggregate_omitted_scores(ale, epi, mask)
        if device.type == "cuda": torch.cuda.synchronize(device)
        stages["uncertainty_aggregation"].append((time.perf_counter()-t)*1000)
        t = time.perf_counter(); canonical.select_next_ng(16, .01, .001, threshold, ale_target, ale_delta); stages["controller_decision"].append((time.perf_counter()-t)*1000)
        if device.type == "cuda": torch.cuda.synchronize(device)
        t = time.perf_counter(); zz = model(x); canonical.aggregate_omitted_scores(ale, epi, mask); canonical.select_next_ng(16, .01, .001, threshold, ale_target, ale_delta)
        if device.type == "cuda": torch.cuda.synchronize(device)
        stages["end_to_end_inference_controller"].append((time.perf_counter()-t)*1000)
    timing = [{"stage": k, "iterations": len(v), "mean_ms": float(np.mean(v)), "p50_ms": float(np.quantile(v,.5)), "p95_ms": float(np.quantile(v,.95))} for k,v in stages.items()]
    write_csv(out / "dynamic_trace.csv", trace); write_csv(out / "runtime_summary.csv", timing)
    regime_summary = []
    for regime, delay in REGIME_ORDER:
        sub = [r for r in trace if r["regime"] == regime]
        regime_summary.append({"regime": regime, "delay_spread_ns": delay, "mean_current_ng": float(np.mean([r["current_ng"] for r in sub])), "mean_nmse_db": float(np.mean([r["nmse_db"] for r in sub])), "mean_aleatoric": float(np.mean([r["aleatoric"] for r in sub])), "mean_epistemic": float(np.mean([r["epistemic"] for r in sub])), "fallback_count": sum(bool(r["full_feedback_fallback"]) for r in sub), "trigger_count": sum(bool(r["adaptation_trigger"]) for r in sub)})
    write_csv(out / "regime_summary.csv", regime_summary)
    xaxis = [r["time_step"] for r in trace]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].step(xaxis, [r["current_ng"] for r in trace], where="post")
    axes[0].set_ylabel("Ng"); axes[0].grid(alpha=.25)
    axes[1].plot(xaxis, [r["nmse_db"] for r in trace])
    axes[1].set_ylabel("NMSE [dB]"); axes[1].set_xlabel("Time step"); axes[1].grid(alpha=.25)
    fig.tight_layout(); fig.savefig(out / "fig11_style.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(xaxis, [r["aleatoric"] for r in trace], label="Aleatoric")
    ax.plot(xaxis, [r["epistemic"] for r in trace], label="Epistemic")
    ax.axhline(threshold, color="r", linestyle="--", label="U_epi threshold")
    ax.set_xlabel("Time step"); ax.set_ylabel("Eq.(13) score"); ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(out / "uncertainty_diagnostic.png", dpi=180); plt.close(fig)
    config = {"checkpoint": str(CHECKPOINT), "sequence_source": str(SEQUENCE_DIR), "device": str(device), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "regime_order": REGIME_ORDER, "segment_length": args.segment_length, "candidate_ngs": canonical.CANDIDATE_NGS, "initial_ng": 128, "epistemic_threshold": threshold, "threshold_provenance": threshold_provenance, "ale_target": ale_target, "ale_delta": ale_delta, "IMPLEMENTATION_ASSUMPTION": ["controller threshold is existing 99th-percentile ID-only calibration", "hysteresis delta is existing 10% of aleatoric target", "existing delay-sweep samples stand in for a time sequence", "EVM/BER unavailable because no repository PHY/precoding/postcoding path"], "no_training": True}
    (out / "config_used.json").write_text(json.dumps(config, indent=2)); (out / "results.json").write_text(json.dumps({"checkpoint": str(CHECKPOINT), "dynamic_steps": len(trace), "fallback_count": sum(bool(r["full_feedback_fallback"]) for r in trace), "trigger_count": sum(bool(r["adaptation_trigger"]) for r in trace), "ber_evm_available": False, "no_training": True, "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device)/2**20 if torch.cuda.is_available() else None}, indent=2))


if __name__ == "__main__": main()
