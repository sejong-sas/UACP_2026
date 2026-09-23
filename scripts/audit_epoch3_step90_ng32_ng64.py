#!/usr/bin/env python3
"""Frozen epoch-3 diagnosis: matched 10 ns step-90 CFR at Ng=32 vs Ng=64."""
from __future__ import annotations

import argparse
import csv
import hashlib
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
DATA = ROOT / "runs/baseline_reproduction/step2_delay_sweep_repro/generated_data/test_delay_10_ns.npz"
EVAL_SEED = 20262000
STEP90 = 90


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def load_model(device):
    from scripts.diagnose_predictor import _make_model
    from scripts.train_predictor import load_config
    m = _make_model(load_config("configs/current_valid_baseline_100k1_seed_20260819.json"), device)
    p = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    state = p["model_state_dict"] if isinstance(p, dict) and "model_state_dict" in p else p
    m.load_state_dict(state); m.eval(); return m


def matched_input(cfr_np: np.ndarray, ng: int, seed: int, device: torch.device):
    """Replicate add_complex_awgn/build_sparse_input with matched RNG draws."""
    from scripts.train_predictor import set_seeds
    from src.training.data import cfr_to_real_imag
    set_seeds(seed)
    cfr = torch.from_numpy(cfr_np[None]).to(device)
    mask = torch.zeros((1, K), dtype=torch.float32, device=device); mask[:, ::ng] = 1.0
    mask_complex = mask[:, :, None, None].to(cfr.real.dtype)
    count = mask_complex.expand_as(cfr.real).sum(dim=(1, 2, 3)).clamp_min(1.0)
    signal_power = (cfr.abs().square() * mask_complex).sum(dim=(1, 2, 3)) / count
    noise_power = signal_power / (10.0 ** (15.0 / 10.0))
    base_real = torch.randn_like(cfr.real); base_imag = torch.randn_like(cfr.real)
    noise = torch.complex(base_real, base_imag) * torch.sqrt(noise_power[:, None, None, None] / 2.0)
    observed = cfr + noise * mask_complex
    target = cfr_to_real_imag(cfr)
    sparse = cfr_to_real_imag(observed) * mask[:, None, :]
    x = torch.cat([sparse, mask[:, None, :].to(sparse.dtype)], dim=1)
    return cfr, x, target, mask, {"base_real_sha256": hashlib.sha256(base_real.detach().cpu().numpy().tobytes()).hexdigest(), "base_imag_sha256": hashlib.sha256(base_imag.detach().cpu().numpy().tobytes()).hexdigest(), "signal_power": float(signal_power.cpu()), "noise_power": float(noise_power.cpu())}


def raw_and_metrics(model, cfr_np, ng, seed, device):
    cfr, x, target, mask, noise_meta = matched_input(cfr_np, ng, seed, device)
    with torch.inference_mode():
        h = model.input_projection(x)
        for block in model.residual_blocks: h = block(h)
        raw_gamma = model.gamma_head(h); raw_psi = model.psi_head(h)
        pooled = model.pool(h); raw_kappa = model.kappa_head(pooled); raw_nu = model.nu_head(pooled)
        from src.models.evidential import constrain_pair_scalar_evidential
        out = constrain_pair_scalar_evidential(raw_gamma, raw_psi, raw_kappa, raw_nu, num_subcarriers=K)
        kappa = out.kappa_expanded; nu_margin = out.nu_expanded - (2 * K + 1.0); psi = out.psi
        ale = psi / nu_margin; epi = ale / kappa
        omitted = 1.0 - mask
        pair_ale = ale.reshape(1, 2, 4, K).permute(0, 2, 1, 3).sum(2).mean(1)
        pair_epi = epi.reshape(1, 2, 4, K).permute(0, 2, 1, 3).sum(2).mean(1)
        ale_score = (pair_ale * omitted).sum(1) / omitted.sum(1).clamp_min(1.0)
        epi_score = (pair_epi * omitted).sum(1) / omitted.sum(1).clamp_min(1.0)
        nmse = 10.0 * torch.log10((((out.gamma - target).square() * omitted[:, None, :]).sum() / (target.square() * omitted[:, None, :]).sum().clamp_min(1e-12)).clamp_min(1e-12))
    def stat(t, selected=None):
        z = t if selected is None else t[selected]
        return float(z.mean().cpu()), float(z.min().cpu()), float(z.max().cpu())
    omitted_bool = omitted.bool().expand_as(psi)
    tensors = {"raw_gamma": raw_gamma, "raw_psi": raw_psi, "raw_kappa": raw_kappa, "raw_nu": raw_nu, "gamma": out.gamma, "psi": psi, "kappa": kappa, "nu_margin": nu_margin, "aleatoric": ale, "epistemic": epi}
    row = {"ng": ng, "seed": seed, "observed_count": int(mask.sum()), "omitted_count": int(omitted.sum()), "nmse_db": float(nmse.cpu()), "aleatoric_score": float(ale_score.cpu()), "epistemic_score": float(epi_score.cpu()), "kappa_mean_omitted": stat(kappa, omitted_bool)[0], "nu_margin_mean_omitted": stat(nu_margin, omitted_bool)[0], "psi_mean_omitted": stat(psi, omitted_bool)[0], "raw_finite": all(bool(torch.isfinite(v).all()) for v in tensors.values()), "final_finite": bool(torch.isfinite(ale).all() and torch.isfinite(epi).all()), **noise_meta}
    for name, tensor in tensors.items():
        row[f"{name}_min"] = float(tensor.min().cpu()); row[f"{name}_median"] = float(tensor.median().cpu()); row[f"{name}_max"] = float(tensor.max().cpu()); row[f"{name}_nonfinite"] = int((~torch.isfinite(tensor)).sum().cpu())
    del cfr, x, target, mask, out, h
    if device.type == "cuda": torch.cuda.empty_cache()
    return row


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--output-dir", required=True); ap.add_argument("--matched-samples", type=int, default=40)
    a = ap.parse_args(); out = ROOT / a.output_dir
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True); sys.path.insert(0, str(ROOT))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available(): torch.cuda.set_device(0); torch.cuda.reset_peak_memory_stats()
    model = load_model(device)
    with np.load(DATA) as d: cfr = np.asarray(d["cfr"], dtype=np.complex64)
    step90_index = 10; step90_cfr = cfr[step90_index]
    step90_seed = EVAL_SEED + STEP90
    step_rows = [raw_and_metrics(model, step90_cfr, ng, step90_seed, device) for ng in (32, 64)]
    write_csv(out / "step90_ng32_ng64.csv", step_rows)
    matched = []
    for i in range(a.matched_samples):
        # These are the same 10 ns samples/seeds used by canonical steps 80..119.
        seed = EVAL_SEED + 80 + i
        for ng in (32, 64):
            row = raw_and_metrics(model, cfr[i], ng, seed, device); row.update({"sample_index": i, "canonical_step": 80 + i}); matched.append(row)
    write_csv(out / "matched_10ns_ng32_ng64.csv", matched)
    dist = []
    for ng in (32, 64):
        z = np.asarray([float(r["epistemic_score"]) for r in matched if int(r["ng"]) == ng])
        dist.append({"ng": ng, "count": len(z), "median": float(np.median(z)), "q95": float(np.quantile(z,.95)), "q99": float(np.quantile(z,.99)), "max": float(np.max(z)), "extreme_gt_52_45695_rate": float(np.mean(z > 52.456952552795315)), "extreme_gt_1000_rate": float(np.mean(z > 1000.0)), "finite_count": int(np.isfinite(z).sum())})
    write_csv(out / "matched_distribution_summary.csv", dist)
    fig, ax = plt.subplots(figsize=(8,5));
    for ng in (32,64):
        z=np.asarray([float(r["epistemic_score"]) for r in matched if int(r["ng"])==ng]); ax.hist(z[np.isfinite(z)],bins=40,alpha=.5,label=f"Ng={ng}")
    ax.set_yscale("log"); ax.set_xlabel("Epistemic Eq.(13)"); ax.set_ylabel("Count (log)"); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(out/"matched_epistemic_histogram.png",dpi=180); plt.close(fig)
    (out / "results.json").write_text(json.dumps({"checkpoint": str(CHECKPOINT), "dataset": str(DATA), "step90_dataset_index": step90_index, "step90_canonical_step": STEP90, "step90_seed": step90_seed, "matched_samples": a.matched_samples, "training_mask_factors": [4,8,16,32], "evaluation_masks": [32,64], "device": str(device), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "same_base_noise": True, "noise_caveat": "same standard-normal RNG tensors were reused; pipeline signal/noise power is recomputed from each mask's observed CFR power", "implementation_assumptions": ["direct-CFR AWGN at 15 dB", "Eq.(12)/(13) omitted-subcarrier aggregation", "Ng=64 is outside epoch3 training mask factors"], "no_training": True, "step90_rows": step_rows, "distribution": dist}, indent=2, allow_nan=True))


if __name__ == "__main__": main()
