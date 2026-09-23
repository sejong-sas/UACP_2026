#!/usr/bin/env python3
"""Figure 11-style dynamic validation and runtime benchmark for the frozen baseline."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import resource
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
K = 1024
REGIME_ORDER = [("20 ns", 20.0), ("80 ns", 80.0), ("10 ns", 10.0),
                ("40 ns", 40.0), ("60 ns", 60.0), ("120 ns", 120.0)]
CANDIDATE_NGS = [128, 64, 32, 16, 8, 4, 1]
CHECKPOINT = ROOT / "runs/current_valid_baseline/paper_sample_alignment_100k_1ep_fig8_20260915/uacp_predictor_step4a.pt"
SEQUENCE_DIR = ROOT / "runs/baseline_reproduction/step2_delay_sweep_repro/generated_data"


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rss_gib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024 ** 2)


def aggregate_omitted_scores(ale_map: torch.Tensor, epi_map: torch.Tensor, mask: torch.Tensor) -> tuple[float, float]:
    """Eq.(12) map followed by Eq.(13), averaged over the batch."""
    if ale_map.ndim != 3 or ale_map.shape[1] != 8:
        raise ValueError("uncertainty maps must have shape [B, 8, K]")
    num_subcarriers = ale_map.shape[-1]
    pair_ale = ale_map.reshape(ale_map.shape[0], 2, 4, num_subcarriers).permute(0, 2, 1, 3).sum(dim=2).mean(dim=1)
    pair_epi = epi_map.reshape(epi_map.shape[0], 2, 4, num_subcarriers).permute(0, 2, 1, 3).sum(dim=2).mean(dim=1)
    omitted = 1.0 - mask
    count = omitted.sum(dim=1).clamp_min(1.0)
    ale = (pair_ale * omitted).sum(dim=1) / count
    epi = (pair_epi * omitted).sum(dim=1) / count
    if bool((omitted.sum(dim=1) == 0).any()):
        ale = torch.where(omitted.sum(dim=1) == 0, torch.full_like(ale, float("nan")), ale)
        epi = torch.where(omitted.sum(dim=1) == 0, torch.full_like(epi, float("nan")), epi)
    return float(ale.mean().detach().cpu()), float(epi.mean().detach().cpu())


def select_next_ng(current_ng: int, ale_score: float, epi_score: float, epi_threshold: float,
                   ale_target: float, ale_delta: float) -> dict[str, int | bool | str]:
    """Apply Algorithm 1 causally: return the action for the next round."""
    if np.isfinite(epi_score) and epi_score > epi_threshold:
        return {"next_ng": 1, "adaptation_trigger": True, "full_feedback_fallback": True,
                "decision_reason": "epistemic_threshold"}
    if current_ng == 1 or not np.isfinite(ale_score):
        return {"next_ng": current_ng, "adaptation_trigger": False, "full_feedback_fallback": False,
                "decision_reason": "hold_no_omitted_set"}
    index = CANDIDATE_NGS.index(current_ng)
    if ale_score > ale_target + ale_delta:
        return {"next_ng": CANDIDATE_NGS[min(index + 1, len(CANDIDATE_NGS) - 1)],
                "adaptation_trigger": False, "full_feedback_fallback": False, "decision_reason": "aleatoric_denser"}
    if ale_score < ale_target - ale_delta:
        return {"next_ng": CANDIDATE_NGS[max(index - 1, 0)],
                "adaptation_trigger": False, "full_feedback_fallback": False, "decision_reason": "aleatoric_sparser"}
    return {"next_ng": current_ng, "adaptation_trigger": False, "full_feedback_fallback": False,
            "decision_reason": "hysteresis_hold"}


def make_mask(batch_size: int, ng: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros((batch_size, K), dtype=torch.float32, device=device)
    mask[:, ::ng] = 1.0
    return mask


def nmse_omitted_db(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    omitted = (1.0 - mask)[:, None, :].expand_as(target)
    denominator = (target.square() * omitted).sum()
    if float(denominator.detach().cpu()) == 0.0 or int(omitted.sum()) == 0:
        return float("nan")
    value = ((prediction - target).square() * omitted).sum() / denominator.clamp_min(1e-12)
    return float((10.0 * torch.log10(value.clamp_min(1e-12))).detach().cpu())


def load_model(cfg: dict, device: torch.device):
    from scripts.diagnose_predictor import _make_model
    model = _make_model(cfg, device)
    payload = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    state = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
    model.load_state_dict(state)
    model.eval()
    return model, state


def calibrate_epi_threshold(model, device: torch.device, eval_seed: int, samples_per_regime: int = 40) -> tuple[float, dict]:
    """Calibrate only from existing ID 20/80 ns data across candidate masks."""
    from scripts.train_predictor import set_seeds
    from src.training.data import build_noisy_sparse_input
    values = []
    by_candidate = {}
    for regime_index, delay in enumerate((20, 80)):
        path = SEQUENCE_DIR / f"test_delay_{delay}_ns.npz"
        with np.load(path) as data:
            cfr = data["cfr"][:samples_per_regime]
        for ng in CANDIDATE_NGS[:-1]:
            current = []
            for start in range(0, len(cfr), 8):
                set_seeds(eval_seed + 500000 + regime_index * 10000 + start)
                batch = torch.from_numpy(cfr[start:start + 8]).to(device)
                mask = make_mask(batch.shape[0], ng, device)
                x, _, _ = build_noisy_sparse_input(batch, mask, 15.0)
                with torch.inference_mode():
                    output = model(x)
                    ale_map = output.psi / (output.nu_expanded - 2 * K - 1.0)
                    epi_map = ale_map / output.kappa_expanded
                    _, score = aggregate_omitted_scores(ale_map, epi_map, mask)
                if np.isfinite(score):
                    current.append(score)
                del batch, mask, x, output, ale_map, epi_map
            values.extend(current)
            by_candidate[f"{delay}ns_Ng{ng}"] = {"mean": float(np.mean(current)), "max": float(np.max(current)), "count": len(current)}
    threshold = float(np.quantile(np.asarray(values), .99))
    return threshold, {"method": "99th percentile of ID-only Eq.(13) scores across candidate Ng", "scores": by_candidate, "count": len(values)}


def cuda_timed(fn, repeats: int, device: torch.device) -> list[float]:
    values = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        values.append((time.perf_counter() - start) * 1000.0)
    return values


def timing_rows(stage_values: dict[str, list[float]]) -> list[dict]:
    rows = []
    for stage, values in stage_values.items():
        a = np.asarray(values, dtype=np.float64)
        rows.append({"stage": stage, "iterations": len(a), "mean_ms": float(a.mean()),
                     "p50_ms": float(np.quantile(a, .50)), "p95_ms": float(np.quantile(a, .95)),
                     "p99_ms": float(np.quantile(a, .99))})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--segment-length", type=int, default=40)
    parser.add_argument("--runtime-repeats", type=int, default=200)
    parser.add_argument("--adaptation-repeats", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-seed", type=int, default=20262000)
    args = parser.parse_args()
    out = ROOT / args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT))
    from scripts.train_predictor import load_config, set_seeds
    from src.models.evidential import evidential_loss
    from src.training.data import build_noisy_sparse_input

    cfg = load_config("configs/current_valid_baseline_100k1_seed_20260819.json")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    start_info = {"torch_cuda_available": bool(torch.cuda.is_available()), "device": str(device), "gpu": gpu_name,
                  "rss_gib": rss_gib()}
    print(json.dumps({"start": start_info}), flush=True)
    model, state = load_model(cfg, device)

    # Candidate-wide ID-only scores define the fixed threshold; dynamic data are never used to tune it.
    prior = json.loads((ROOT / "runs/current_valid_baseline/paper_sample_alignment_100k_1ep_fig8_20260915/final_results.json").read_text())
    probes = prior["epoch_probe_results"]["1"]
    epi_threshold, threshold_provenance = calibrate_epi_threshold(model, device, args.eval_seed)
    ale_target = (probes["ID-Easy 20 ns"]["aleatoric_paper_eq12_eq13"] + probes["ID-Hard 80 ns"]["aleatoric_paper_eq12_eq13"]) / 2.0
    ale_delta = 0.10 * ale_target

    # Use existing delay-sweep CFR files. No temporal generator exists in the repository.
    sequence = []
    for regime, delay in REGIME_ORDER:
        path = SEQUENCE_DIR / ("test_delay_1_ms.npz" if delay == 1_000_000 else f"test_delay_{int(delay)}_ns.npz")
        with np.load(path) as data:
            cfr = data["cfr"][:args.segment_length]
        sequence.extend((regime, delay, sample) for sample in cfr)

    trace = []
    current_ng = 128
    previous_regime = None
    for step, (regime, delay, cfr_np) in enumerate(sequence):
        if regime != previous_regime:
            previous_regime = regime
            print(json.dumps({"regime_start": regime, "time_step": step}), flush=True)
        set_seeds(args.eval_seed + step)
        cfr = torch.from_numpy(cfr_np[None]).to(device)
        mask = make_mask(1, current_ng, device)
        x, target, _ = build_noisy_sparse_input(cfr, mask, 15.0)
        with torch.inference_mode():
            output = model(x)
            ale_map = output.psi / (output.nu_expanded - 2 * K - 1.0)
            epi_map = ale_map / output.kappa_expanded
            ale_score, epi_score = aggregate_omitted_scores(ale_map, epi_map, mask)
            nmse = nmse_omitted_db(output.gamma, target, mask)
        decision = select_next_ng(current_ng, ale_score, epi_score, epi_threshold, ale_target, ale_delta)
        trace.append({"time_step": step, "delay_spread_ns": delay, "regime": regime,
                      "ood": bool(delay > 100.0), "current_ng": current_ng,
                      "reported_subcarriers": int(round(K / current_ng)),
                      "aleatoric_score_eq13": ale_score, "epistemic_score_eq13": epi_score,
                      "epistemic_threshold": epi_threshold, "predicted_next_ng": decision["next_ng"],
                      "full_feedback_fallback": decision["full_feedback_fallback"],
                      "adaptation_trigger": decision["adaptation_trigger"], "decision_reason": decision["decision_reason"],
                      "nmse_omitted_db": nmse})
        current_ng = int(decision["next_ng"])
        del cfr, mask, x, target, output, ale_map, epi_map
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Representative batch-one online timing; outputs are not retained between iterations.
    set_seeds(args.eval_seed + 999999)
    sample = torch.from_numpy(sequence[0][2][None]).to(device)
    timing_mask = make_mask(1, 16, device)
    timing_x, timing_target, _ = build_noisy_sparse_input(sample, timing_mask, 15.0)
    with torch.inference_mode():
        warm_output = model(timing_x)
    warm_ale = warm_output.psi / (warm_output.nu_expanded - 2 * K - 1.0)
    warm_epi = warm_ale / warm_output.kappa_expanded
    for _ in range(10):
        with torch.inference_mode():
            z = model(timing_x)
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        forward_ms = cuda_timed(lambda: model(timing_x), args.runtime_repeats, device)
        uncertainty_ms = cuda_timed(lambda: (warm_output.psi / (warm_output.nu_expanded - 2 * K - 1.0), warm_output.kappa_expanded), args.runtime_repeats, device)
        aggregation_ms = cuda_timed(lambda: aggregate_omitted_scores(warm_ale, warm_epi, timing_mask), args.runtime_repeats, device)
        controller_ms = cuda_timed(lambda: select_next_ng(16, 0.01, 0.001, epi_threshold, ale_target, ale_delta), args.runtime_repeats, device)
        online_ms = cuda_timed(lambda: (model(timing_x), aggregate_omitted_scores(warm_ale, warm_epi, timing_mask), select_next_ng(16, 0.01, 0.001, epi_threshold, ale_target, ale_delta)), args.runtime_repeats, device)
        end_to_end_ms = cuda_timed(lambda: (torch.from_numpy(sequence[0][2][None]).to(device), model(timing_x), aggregate_omitted_scores(warm_ale, warm_epi, timing_mask), select_next_ng(16, 0.01, 0.001, epi_threshold, ale_target, ale_delta)), args.runtime_repeats, device)
    runtime_rows = timing_rows({"predictor_forward": forward_ms, "evidential_uncertainty": uncertainty_ms,
                                "uncertainty_aggregation": aggregation_ms, "controller_decision": controller_ms,
                                "total_online_control": online_ms, "end_to_end_software": end_to_end_ms})

    # Separate model instance: one full-model optimizer update only, never saved to the canonical path.
    from scripts.diagnose_predictor import _make_model
    adapt_model = _make_model(cfg, device)
    adapt_model.load_state_dict(state)
    adapt_model.train()
    optimizer = torch.optim.Adam(adapt_model.parameters(), lr=1e-4)
    batch_np = np.stack([sequence[i][2] for i in range(args.batch_size)], axis=0)
    set_seeds(args.eval_seed + 888888)
    batch_cfr = torch.from_numpy(batch_np).to(device)
    batch_mask = make_mask(args.batch_size, 16, device)
    batch_x, batch_target, _ = build_noisy_sparse_input(batch_cfr, batch_mask, 15.0)
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        update_output = adapt_model(batch_x)
        update_loss = evidential_loss(update_output, batch_target, 1e-3, nll_mode="diagonal_multivariate", reg_mode="pair")["total"]
        update_loss.backward(); optimizer.step()
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats(device)
    adapt_values = {"forward": [], "backward": [], "optimizer_step": [], "total_single_update": []}
    for _ in range(args.adaptation_repeats):
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda": torch.cuda.synchronize(device)
        total_start = time.perf_counter(); forward_start = time.perf_counter()
        update_output = adapt_model(batch_x)
        update_loss = evidential_loss(update_output, batch_target, 1e-3, nll_mode="diagonal_multivariate", reg_mode="pair")["total"]
        if device.type == "cuda": torch.cuda.synchronize(device)
        adapt_values["forward"].append((time.perf_counter() - forward_start) * 1000.0)
        backward_start = time.perf_counter(); update_loss.backward()
        if device.type == "cuda": torch.cuda.synchronize(device)
        adapt_values["backward"].append((time.perf_counter() - backward_start) * 1000.0)
        step_start = time.perf_counter(); optimizer.step()
        if device.type == "cuda": torch.cuda.synchronize(device)
        adapt_values["optimizer_step"].append((time.perf_counter() - step_start) * 1000.0)
        adapt_values["total_single_update"].append((time.perf_counter() - total_start) * 1000.0)
    adaptation_rows = timing_rows(adapt_values)
    peak_gpu = {"allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20,
                "reserved_mb": torch.cuda.max_memory_reserved(device) / 2**20} if torch.cuda.is_available() else None

    write_csv(out / "dynamic_trace.csv", trace)
    write_csv(out / "runtime_summary.csv", runtime_rows)
    write_csv(out / "adaptation_runtime.csv", adaptation_rows)
    labels = [row["regime"] for row in trace]
    xaxis = [row["time_step"] for row in trace]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].step(xaxis, [row["current_ng"] for row in trace], where="post"); axes[0].set_ylabel("Ng"); axes[0].grid(alpha=.25)
    axes[1].plot(xaxis, [row["nmse_omitted_db"] for row in trace]); axes[1].set_ylabel("NMSE [dB]"); axes[1].set_xlabel("Time Step"); axes[1].grid(alpha=.25)
    fig.tight_layout(); fig.savefig(out / "fig11_style.png", dpi=180); plt.close(fig)
    plt.figure(figsize=(10, 5)); plt.plot(xaxis, [row["aleatoric_score_eq13"] for row in trace], label="Aleatoric"); plt.plot(xaxis, [row["epistemic_score_eq13"] for row in trace], label="Epistemic"); plt.axhline(epi_threshold, color="r", linestyle="--", label="U_epi threshold");
    for i in range(1, len(trace)):
        if labels[i] != labels[i - 1]: plt.axvline(i, color="k", alpha=.2)
    plt.xlabel("Time Step"); plt.ylabel("Eq.(13) score"); plt.grid(alpha=.25); plt.legend(); plt.tight_layout(); plt.savefig(out / "uncertainty_diagnostic.png", dpi=180); plt.close()

    end_info = {"torch_cuda_available": bool(torch.cuda.is_available()), "device": str(device), "gpu": gpu_name,
                "peak_rss_gib": rss_gib(), "peak_gpu_memory": peak_gpu}
    config_used = {"checkpoint": str(CHECKPOINT), "sequence_source": str(SEQUENCE_DIR),
                   "paper_settings": {"Nr": 2, "Nt": 2, "K": K, "carrier_ghz": 3.5, "subcarrier_spacing_khz": 30, "snr_db": 15},
                   "regime_order": [{"regime": r, "delay_spread_ns": d} for r, d in REGIME_ORDER],
                   "segment_length": args.segment_length, "candidate_ngs": CANDIDATE_NGS,
                   "initial_ng": 128, "epi_threshold": epi_threshold, "epi_threshold_provenance": threshold_provenance,
                   "ale_target": ale_target, "ale_delta": ale_delta,
                   "temporal_assumption": "existing time-ordered Sionna CFR sweep samples; no temporal-correlated generator found",
                   "ber_evm": "unavailable: no repository PHY/precoding/postcoding BER/EVM pipeline",
                   "adaptation": "trigger flag and next-round full feedback only; no online fine-tuning"}
    (out / "config_used.json").write_text(json.dumps(config_used, indent=2))
    result = {"experiment": "Figure 11-style dynamic validation using current reproduction baseline",
              "checkpoint": str(CHECKPOINT), "dynamic_steps": len(trace), "no_training": True,
              "forbidden_runs_not_executed": ["Partial Fine-Tuning", "Full-Psi", "new pretrained-model training"],
              "start": start_info, "end": end_info,
              "adaptation_microbenchmark": "full-model single optimizer update on a separate in-memory model; not an end-to-end adaptation estimate",
              "ber_evm_available": False,
              "trigger_count": sum(bool(r["adaptation_trigger"]) for r in trace),
              "full_feedback_count": sum(bool(r["full_feedback_fallback"]) for r in trace),
              "regime_first_actions": {r: next(row for row in trace if row["regime"] == r) for r, _ in REGIME_ORDER}}
    (out / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"end": end_info, "dynamic_steps": len(trace), "trigger_count": result["trigger_count"]}), flush=True)


if __name__ == "__main__":
    main()
