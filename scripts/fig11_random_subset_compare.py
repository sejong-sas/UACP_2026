#!/usr/bin/env python3
"""Compare Fig.11-lite behavior for the two frozen random-subset checkpoints."""
from __future__ import annotations

import csv
import gc
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
K = 1024
REGIMES = [("20 ns", 20), ("80 ns", 80), ("10 ns", 10), ("40 ns", 40), ("60 ns", 60), ("120 ns", 120)]
NGS = [1, 4, 8, 16, 32, 64, 128]
NONFULL_NGS = [4, 8, 16, 32, 64, 128]
SEQUENCE_DIR = ROOT / "runs/baseline_reproduction/step2_delay_sweep_repro/generated_data"
CHECKPOINTS = {
    "batch8_random": "runs/current_valid_baseline/random_subset_mask_100k1ep_batch8_bf16_20260918/random_subset_mask_100k_1ep_batch8_bf16.pt",
    "batch400_random": "runs/current_valid_baseline/random_subset_mask_100k2ep_batch400_bf16_20260918/random_subset_mask_100k_2ep_batch400_bf16.pt",
}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def random_subset_mask(batch_size: int, ng: int, device: torch.device) -> torch.Tensor:
    if ng == 1:
        return torch.ones((batch_size, K), dtype=torch.float32, device=device)
    count = K // ng
    indices = torch.rand((batch_size, K), device=device).topk(count, dim=1).indices
    mask = torch.zeros((batch_size, K), dtype=torch.float32, device=device)
    mask.scatter_(1, indices, 1.0)
    return mask


def aggregate(ale_map: torch.Tensor, epi_map: torch.Tensor, mask: torch.Tensor) -> tuple[float, float]:
    pair_ale = ale_map.reshape(ale_map.shape[0], 2, 4, K).permute(0, 2, 1, 3).sum(2).mean(1)
    pair_epi = epi_map.reshape(epi_map.shape[0], 2, 4, K).permute(0, 2, 1, 3).sum(2).mean(1)
    omitted = 1.0 - mask
    count = omitted.sum(1).clamp_min(1.0)
    ale = (pair_ale * omitted).sum(1) / count
    epi = (pair_epi * omitted).sum(1) / count
    if bool((omitted.sum(1) == 0).any()):
        ale = torch.where(omitted.sum(1) == 0, torch.full_like(ale, float("nan")), ale)
        epi = torch.where(omitted.sum(1) == 0, torch.full_like(epi, float("nan")), epi)
    return float(ale.mean().detach().cpu()), float(epi.mean().detach().cpu())


def select_next(current: int, ale: float, epi: float, epi_threshold: float,
                ale_target: float, ale_delta: float) -> dict:
    if np.isfinite(epi) and epi > epi_threshold:
        return {"next_ng": 1, "trigger": True, "reason": "epistemic_threshold"}
    if current == 1 or not np.isfinite(ale):
        return {"next_ng": current, "trigger": False, "reason": "hold_no_omitted_set"}
    # Ordered sparse-to-dense movement: lower Aleatoric -> Ng increase;
    # higher Aleatoric -> Ng decrease. One step only, with hysteresis.
    ordered = [1, 4, 8, 16, 32, 64, 128]
    i = ordered.index(current)
    if ale < ale_target - ale_delta:
        nxt = ordered[min(i + 1, len(ordered) - 1)]
        return {"next_ng": nxt, "trigger": False, "reason": "aleatoric_low_sparser"}
    if ale > ale_target + ale_delta:
        nxt = ordered[max(i - 1, 0)]
        return {"next_ng": nxt, "trigger": False, "reason": "aleatoric_high_denser"}
    return {"next_ng": current, "trigger": False, "reason": "hysteresis_hold"}


def score_sample(model, cfr_np: np.ndarray, ng: int, device: torch.device):
    from src.training.data import build_noisy_sparse_input
    cfr = torch.from_numpy(cfr_np[None]).to(device)
    mask = random_subset_mask(1, ng, device)
    x, target, _ = build_noisy_sparse_input(cfr, mask, 15.0)
    with torch.inference_mode():
        output = model(x)
        ale_map = output.psi.float() / (output.nu_expanded.float() - 2 * K - 1.0)
        epi_map = ale_map / output.kappa_expanded.float()
        ale, epi = aggregate(ale_map, epi_map, mask)
        error = (output.gamma.float() - target.float()).square()
        nmse_all = float((10 * torch.log10((error.sum() / target.float().square().sum()).clamp_min(1e-12))).cpu())
        omitted = (1.0 - mask)[:, None, :].expand_as(target)
        omitted_den = (target.float().square() * omitted).sum()
        nmse_om = float("nan") if float(omitted_den.cpu()) == 0 else float((10 * torch.log10((error * omitted).sum() / omitted_den.clamp_min(1e-12))).cpu())
        finite = bool(torch.isfinite(output.nu_expanded).all().cpu())
    del cfr, mask, x, target, output, ale_map, epi_map
    return nmse_all, nmse_om, ale, epi, finite


def calibrate(model, device: torch.device, seed: int, samples: int = 40) -> tuple[float, float, dict]:
    from scripts.train_predictor import set_seeds
    epi_values, ale_values = [], []
    by_ng = {}
    for ri, delay in enumerate((20, 80)):
        with np.load(SEQUENCE_DIR / f"test_delay_{delay}_ns.npz") as z:
            cfr = z["cfr"][:samples]
        for ng in NONFULL_NGS:
            av, ev = [], []
            for i, cfr_np in enumerate(cfr):
                set_seeds(seed + 100000 + ri * 10000 + ng * 100 + i)
                _, _, a, e, finite = score_sample(model, cfr_np, ng, device)
                if finite and np.isfinite(a) and np.isfinite(e):
                    av.append(a); ev.append(e)
            epi_values.extend(ev)
            if ng == 16:
                ale_values.extend(av)
            by_ng[f"{delay}ns_Ng{ng}"] = {"aleatoric_mean": float(np.mean(av)), "epistemic_mean": float(np.mean(ev)), "count": len(ev)}
    epi_threshold = float(np.quantile(epi_values, .99))
    ale_target = float(np.mean(ale_values))
    return epi_threshold, ale_target, {"method": "ID-only 99th percentile epistemic; ID Ng16 Aleatoric midpoint proxy", "count_epi": len(epi_values), "count_ale": len(ale_values), "by_ng": by_ng}


def evaluate_model(name: str, checkpoint: str, out: Path, device: torch.device, seed: int, segment_length: int = 40) -> dict:
    from scripts.diagnose_predictor import _make_model
    from scripts.train_predictor import load_config, set_seeds
    cfg = load_config("runs/current_valid_baseline/paper_sample_alignment_100k_1ep_fig8_20260915/config.json")
    model = _make_model(cfg, device)
    payload = torch.load(ROOT / checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload)
    model.eval()
    epi_threshold, ale_target, calibration = calibrate(model, device, seed)
    ale_delta = .10 * ale_target
    sequence = []
    for regime, delay in REGIMES:
        fn = f"test_delay_{delay}_ns.npz"
        with np.load(SEQUENCE_DIR / fn) as z:
            sequence.extend((regime, delay, x) for x in z["cfr"][:segment_length])
    trace, current = [], 16
    start = time.perf_counter()
    for step, (regime, delay, cfr_np) in enumerate(sequence):
        set_seeds(seed + step)
        nmse_all, nmse_om, ale, epi, finite = score_sample(model, cfr_np, current, device)
        decision = select_next(current, ale, epi, epi_threshold, ale_target, ale_delta)
        trace.append({"time_step": step, "regime": regime, "delay_spread_ns": delay,
                      "current_ng": current, "predicted_next_ng": decision["next_ng"],
                      "observed_subcarriers": K // current if current > 1 else K,
                      "all_nmse_db": nmse_all, "omitted_nmse_db": nmse_om,
                      "aleatoric": ale, "epistemic": epi,
                      "epistemic_threshold": epi_threshold, "aleatoric_target": ale_target,
                      "aleatoric_hysteresis": ale_delta, "controller_action": decision["reason"],
                      "epistemic_trigger": decision["trigger"], "nu_finite": finite})
        current = decision["next_ng"]
    model_out = out / name
    model_out.mkdir(parents=True, exist_ok=True)
    write_csv(model_out / "dynamic_trace.csv", trace)
    regime_rows = []
    for regime, delay in REGIMES:
        rows = [r for r in trace if r["regime"] == regime]
        regime_rows.append({"regime": regime, "delay_spread_ns": delay, "steps": len(rows),
                            "mean_ng": float(np.mean([r["current_ng"] for r in rows])),
                            "median_ng": float(np.median([r["current_ng"] for r in rows])),
                            "mean_all_nmse_db": float(np.mean([r["all_nmse_db"] for r in rows])),
                            "mean_omitted_nmse_db": float(np.nanmean([r["omitted_nmse_db"] for r in rows])),
                            "epistemic_trigger_count": sum(bool(r["epistemic_trigger"]) for r in rows),
                            "nu_nonfinite_samples": sum(not bool(r["nu_finite"]) for r in rows)})
    write_csv(model_out / "regime_summary.csv", regime_rows)
    config = {"model": name, "checkpoint": checkpoint, "sequence": REGIMES,
              "segment_length": segment_length, "candidate_ngs": NGS, "initial_ng": 16,
              "mask_placement": "random subset without replacement", "snr_db": 15.0,
              "threshold_calibration": calibration, "epi_threshold": epi_threshold,
              "ale_target": ale_target, "ale_delta": ale_delta,
              "implementation_assumptions": ["ID-only calibration; OOD never used for threshold", "10% Aleatoric hysteresis", "existing delay-sweep files used as ordered dynamic sequence; no temporal-correlated generator", "Ng/NMSE only; EVM/BER unavailable"]}
    (model_out / "config_used.json").write_text(json.dumps(config, indent=2) + "\n")
    result = {"model": name, "checkpoint": checkpoint, "runtime_seconds": time.perf_counter() - start,
              "steps": len(trace), "trigger_count": sum(bool(r["epistemic_trigger"]) for r in trace),
              "nonfinite_nu_samples": sum(not bool(r["nu_finite"]) for r in trace),
              "epi_threshold": epi_threshold, "ale_target": ale_target, "regime_summary": regime_rows}
    (model_out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    del model; gc.collect(); torch.cuda.empty_cache()
    return result


def plot_model(trace_path: Path, output: Path, title: str) -> None:
    rows = list(csv.DictReader(trace_path.open()))
    x = np.arange(len(rows)); boundaries = [0]
    for i in range(1, len(rows)):
        if rows[i]["regime"] != rows[i - 1]["regime"]: boundaries.append(i)
    boundaries.append(len(rows))
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax[0].step(x, [float(r["current_ng"]) for r in rows], where="post", color="#1f77b4")
    ax[0].set_ylabel("Ng"); ax[0].set_yscale("symlog", linthresh=1); ax[0].set_ylim(.8, 150); ax[0].set_yticks([1,4,8,16,32,64,128]); ax[0].grid(alpha=.25)
    ax[1].plot(x, [float(r["all_nmse_db"]) for r in rows], color="#d62728"); ax[1].set_ylabel("NMSE [dB]"); ax[1].set_xlabel("Time step"); ax[1].grid(alpha=.25)
    for i, (left, right) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        for a in ax: a.axvline(left, color="k", alpha=.25)
        ax[0].text((left + right - 1) / 2, 135, rows[left]["regime"], ha="center", va="top", fontsize=9)
    ax[0].axvline(boundaries[-1], color="k", alpha=.25); fig.suptitle(title); fig.tight_layout(); fig.savefig(output, dpi=220); fig.savefig(output.with_suffix(".pdf")); plt.close(fig)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--output-dir", required=True); ap.add_argument("--segment-length", type=int, default=40); args = ap.parse_args()
    out = ROOT / args.output_dir
    if out.exists() and any(out.iterdir()): raise FileExistsError(f"Refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    print(json.dumps({"device": str(device), "gpu": torch.cuda.get_device_name(0), "seed": 20262000}), flush=True)
    results = []
    for i, (name, checkpoint) in enumerate(CHECKPOINTS.items()):
        results.append(evaluate_model(name, checkpoint, out, device, 20262000, args.segment_length))
        plot_model(out / name / "dynamic_trace.csv", out / f"fig11_{name}_ng_nmse.png", name)
    # Combined 2x2 plot uses exactly the same axes and regime boundaries.
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex="col")
    all_rows = {}
    for col, name in enumerate(CHECKPOINTS):
        rows = list(csv.DictReader((out / name / "dynamic_trace.csv").open())); all_rows[name] = rows
        x = np.arange(len(rows)); bounds = [0] + [i for i in range(1, len(rows)) if rows[i]["regime"] != rows[i - 1]["regime"]] + [len(rows)]
        axes[0, col].step(x, [float(r["current_ng"]) for r in rows], where="post"); axes[0, col].set_title(name); axes[0, col].set_yscale("symlog", linthresh=1); axes[0, col].set_ylim(.8, 150); axes[0, col].set_yticks([1,4,8,16,32,64,128]); axes[0, col].grid(alpha=.25)
        axes[1, col].plot(x, [float(r["all_nmse_db"]) for r in rows]); axes[1, col].set_ylim(-20, 2); axes[1, col].grid(alpha=.25); axes[1, col].set_xlabel("Time step")
        for left in bounds[:-1]:
            for row in range(2): axes[row, col].axvline(left, color="k", alpha=.2)
    axes[0,0].set_ylabel("Ng"); axes[1,0].set_ylabel("NMSE [dB]")
    fig.suptitle("Fig.11-lite random-subset comparison (identical axes)"); fig.tight_layout(); fig.savefig(out / "fig11_random_subset_comparison.png", dpi=240); fig.savefig(out / "fig11_random_subset_comparison.pdf"); plt.close(fig)
    (out / "comparison_results.json").write_text(json.dumps({"models": results, "no_training": True, "same_dynamic_seed": 20262000}, indent=2) + "\n")


if __name__ == "__main__": main()
