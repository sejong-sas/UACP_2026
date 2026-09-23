#!/usr/bin/env python3
"""Static Fig.8/Fig.9 validation for the frozen 100k x 5 epoch-3 checkpoint.

This evaluator deliberately uses the existing common_eval files and the
repository's canonical Eq.(12)/(13) and Student-t interval definitions.  It
does not train, tune, or write to any existing run.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import gaussian_kde, rankdata, t as student_t

ROOT = Path(__file__).resolve().parents[1]
K = 1024
REGIMES = ["ID-Easy 20 ns", "ID-Hard 80 ns", "OOD-Near 120 ns", "OOD-Far 1 ms"]
NOMINALS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
COMMON = ROOT / "runs/current_valid_baseline/diversity_ablation/reproducibility_20260914/common_eval"
CHECKPOINT = ROOT / "runs/current_valid_baseline/training_convergence_100k_5ep_20260916_final_controlled/uacp_predictor_100k_5ep_epoch_3.pt"


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def auc(negative: np.ndarray, positive: np.ndarray) -> float:
    values = np.concatenate([negative, positive])
    ranks = rankdata(values, method="average")
    return float((ranks[len(negative):].sum() - len(positive) * (len(positive) + 1) / 2) /
                 (len(negative) * len(positive)))


def read_cfr(label: str, count: int) -> np.ndarray:
    path = COMMON / (label.replace(" ", "_").replace("/", "_") + ".npz")
    with np.load(path) as data:
        cfr = np.asarray(data["cfr"][:count], dtype=np.complex64)
    if len(cfr) != count:
        raise RuntimeError(f"{label}: expected {count}, found {len(cfr)}")
    return cfr


def load_model(device: torch.device):
    from scripts.diagnose_predictor import _make_model
    from scripts.train_predictor import load_config
    cfg = load_config("configs/current_valid_baseline_100k1_seed_20260819.json")
    model = _make_model(cfg, device)
    payload = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    state = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
    model.load_state_dict(state); model.eval()
    return model


def sample_stats(values: np.ndarray) -> dict:
    finite = values[np.isfinite(values)]
    return {"count": int(values.size), "finite_count": int(finite.size),
            "nonfinite_count": int(values.size - finite.size),
            "min": float(np.min(finite)) if finite.size else math.nan,
            "median": float(np.median(finite)) if finite.size else math.nan,
            "max": float(np.max(finite)) if finite.size else math.nan,
            "mean": float(np.mean(finite)) if finite.size else math.nan}


def uncertainty_score(epi: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pair = epi.reshape(epi.shape[0], 2, 4, K).permute(0, 2, 1, 3).sum(dim=2).mean(dim=1)
    omitted = 1.0 - mask
    return (pair * omitted).sum(dim=1) / omitted.sum(dim=1).clamp_min(1.0)


def nmse_db(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    omitted = (1.0 - mask)[:, None, :].expand_as(target)
    num = ((pred - target).square() * omitted).sum(dim=(1, 2))
    den = (target.square() * omitted).sum(dim=(1, 2)).clamp_min(1e-12)
    return 10.0 * torch.log10((num / den).clamp_min(1e-12))


def evaluate(model, device: torch.device, count: int, batch_size: int):
    from scripts.train_predictor import set_seeds
    from src.training.data import build_noisy_sparse_input, uniform_grouping_mask
    rows = []; scores = {}; arrays = {}
    for ri, label in enumerate(REGIMES):
        cfr = read_cfr(label, count); all_rows = []
        for start in range(0, count, batch_size):
            batch = torch.from_numpy(cfr[start:start + batch_size]).to(device)
            mask = uniform_grouping_mask(len(batch), K, 16, device)
            set_seeds(20262000 + ri * 100000 + start)
            x, target, _ = build_noisy_sparse_input(batch, mask, 15.0)
            with torch.inference_mode():
                out = model(x)
                raw_gamma = out.gamma; raw_kappa = out.kappa_expanded; raw_psi = out.psi
                raw_nu = out.nu_expanded
                nu_margin = raw_nu - (2 * K + 1.0)
                ale = raw_psi / nu_margin
                epi = ale / raw_kappa
                score = uncertainty_score(epi, mask)
                nmse = nmse_db(raw_gamma, target, mask)
                # The model output is already the admissibility transform; retain
                # raw head finiteness via the transformed tensors and derived stages.
                tensors = {"kappa": raw_kappa, "nu_margin": nu_margin, "psi": raw_psi,
                           "aleatoric": ale, "epistemic": epi, "score": score}
                for i in range(len(batch)):
                    item = {"regime": label, "sample": start + i, "nmse_db": float(nmse[i].cpu()),
                            "aleatoric": float(ale[i].mean().cpu()), "epistemic": float(epi[i].mean().cpu()),
                            "fig8_score": float(score[i].cpu())}
                    for key, value in tensors.items():
                        item[f"{key}_nonfinite"] = int((~torch.isfinite(value[i])).sum().cpu())
                    all_rows.append(item)
                batch_np = {"nmse": nmse.cpu().numpy(), "ale": ale.mean(dim=tuple(range(1, ale.ndim))).cpu().numpy(),
                            "epi": epi.mean(dim=tuple(range(1, epi.ndim))).cpu().numpy(), "score": score.cpu().numpy(),
                            "kappa": raw_kappa.mean(dim=tuple(range(1, raw_kappa.ndim))).cpu().numpy(),
                            "nu_margin": nu_margin.mean(dim=tuple(range(1, nu_margin.ndim))).cpu().numpy(),
                            "psi": raw_psi.mean(dim=tuple(range(1, raw_psi.ndim))).cpu().numpy(),
                            "aleatoric": ale.mean(dim=tuple(range(1, ale.ndim))).cpu().numpy(),
                            "epistemic": epi.mean(dim=tuple(range(1, epi.ndim))).cpu().numpy()}
                for key, value in tensors.items():
                    batch_np[f"{key}_nonfinite"] = (~torch.isfinite(value)).reshape(value.shape[0], -1).sum(dim=1).cpu().numpy()
                for key, value in batch_np.items():
                    arrays.setdefault(key, {}).setdefault(label, []).extend(value.tolist())
            del batch, mask, x, target, out
        rows.extend(all_rows); scores[label] = np.asarray(arrays["score"][label], dtype=float)
        del cfr; gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()
    return rows, arrays, scores


def calibration(model, device: torch.device, count: int, batch_size: int):
    from scripts.train_predictor import set_seeds
    from src.training.data import build_noisy_sparse_input, uniform_grouping_mask
    counts = {(r, p, n): [0, 0] for r in REGIMES for p in ("all_subcarrier", "observed_only", "omitted_only") for n in NOMINALS}
    probs = np.asarray([(1.0 + n) / 2.0 for n in NOMINALS], dtype=float)[:, None, None, None]
    for ri, label in enumerate(REGIMES):
        cfr = read_cfr(label, count)
        for start in range(0, count, batch_size):
            batch = torch.from_numpy(cfr[start:start + batch_size]).to(device)
            mask = uniform_grouping_mask(len(batch), K, 16, device)
            set_seeds(20262000 + ri * 100000 + start)
            x, target, _ = build_noisy_sparse_input(batch, mask, 15.0)
            with torch.inference_mode():
                out = model(x)
                df = out.nu - 2 * K + 1.0
                scale_sq = ((out.kappa_expanded + 1.0) / (out.kappa_expanded * out.nu_expanded.sub(2 * K).add(1.0))) * out.psi
                scale = torch.sqrt(scale_sq.clamp_min(1e-12))
                q_np = student_t.ppf(probs, df.detach().cpu().numpy()[None, ...])
                q = torch.as_tensor(q_np, dtype=scale.dtype, device=device)
                q = torch.cat((q.expand(-1, -1, -1, K), q.expand(-1, -1, -1, K)), dim=2)
                covered = (target - out.gamma).abs().unsqueeze(0) <= q * scale.unsqueeze(0)
                keep = torch.stack((torch.ones_like(mask), mask, 1.0 - mask), dim=0).bool()
                covered_counts = (covered.unsqueeze(1) & keep[None, :, :, None, :]).sum(dim=(2, 3, 4)).cpu().numpy()
                component_counts = keep[:, :, None, :].expand(-1, -1, 8, -1).sum(dim=(1, 2, 3)).cpu().numpy()
                for ni, nominal in enumerate(NOMINALS):
                    for pi, protocol in enumerate(("all_subcarrier", "observed_only", "omitted_only")):
                        counts[(label, protocol, nominal)][0] += int(covered_counts[ni, pi])
                        counts[(label, protocol, nominal)][1] += int(component_counts[pi])
            del batch, mask, x, target, out, df, scale_sq, scale
        del cfr; gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()
    pools = {"ID": REGIMES[:2], "OOD-pooled": REGIMES[2:], "OOD-Near": [REGIMES[2]], "OOD-Far": [REGIMES[3]]}
    rows = []
    for protocol in ("all_subcarrier", "observed_only", "omitted_only"):
        for pool, regimes in pools.items():
            errors = []
            for nominal in NOMINALS:
                covered = sum(counts[(r, protocol, nominal)][0] for r in regimes)
                total = sum(counts[(r, protocol, nominal)][1] for r in regimes)
                empirical = covered / total
                errors.append(abs(empirical - nominal))
                rows.append({"protocol": protocol, "pool": pool, "nominal": nominal,
                             "covered": covered, "component_count": total, "empirical": empirical,
                             "abs_error": abs(empirical - nominal), "calibration_error_mae": ""})
            ce = float(np.mean(errors))
            for row in rows:
                if row["protocol"] == protocol and row["pool"] == pool: row["calibration_error_mae"] = ce
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--output-dir", required=True)
    ap.add_argument("--samples-per-regime", type=int, default=10000); ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args(); out = ROOT / args.output_dir
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True); sys.path.insert(0, str(ROOT))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats()
    model = load_model(device); start = time.perf_counter()
    rows, arrays, scores = evaluate(model, device, args.samples_per_regime, args.batch_size)
    write_csv(out / "static_per_sample.csv", rows)
    stats_rows = []
    for label in REGIMES:
        row = {"regime": label}
        for key in ("nmse", "ale", "epi", "score", "kappa", "nu_margin", "psi", "aleatoric", "epistemic"):
            if key not in arrays or label not in arrays[key]: continue
            s = sample_stats(np.asarray(arrays[key][label], dtype=float))
            for field in ("min", "median", "max", "mean", "nonfinite_count"): row[f"{key}_{field}"] = s[field]
        stats_rows.append(row)
    write_csv(out / "fig8_regime_summary.csv", stats_rows)
    id_pool = np.concatenate([scores[REGIMES[0]], scores[REGIMES[1]]]); near = scores[REGIMES[2]]; far = scores[REGIMES[3]]
    auc_rows = [{"comparison": "ID-pooled_vs_OOD-Near", "auroc": auc(id_pool, near)},
                {"comparison": "ID-pooled_vs_OOD-Far", "auroc": auc(id_pool, far)},
                {"comparison": "ID-pooled_vs_OOD-pooled", "auroc": auc(id_pool, np.concatenate([near, far]))}]
    write_csv(out / "fig8_auroc.csv", auc_rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    for label in REGIMES:
        values = scores[label]; grid = np.linspace(float(np.nanpercentile(values, 0.2)), float(np.nanpercentile(values, 99.8)), 400)
        try: density = gaussian_kde(values[np.isfinite(values)])(grid); ax.plot(grid, density, label=label)
        except Exception: ax.hist(values, bins=80, density=True, alpha=.35, label=label)
    ax.set_xlabel("Epistemic uncertainty (Eq. 13)"); ax.set_ylabel("Density"); ax.legend(fontsize=8); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(out / "fig8_epistemic_density.png", dpi=180); plt.close(fig)
    cal_rows = calibration(model, device, args.samples_per_regime, args.batch_size); write_csv(out / "fig9_calibration_curve.csv", cal_rows)
    ce_rows = []
    for protocol in ("omitted_only",):
        for pool in ("ID", "OOD-pooled", "OOD-Near", "OOD-Far"):
            subset = [r for r in cal_rows if r["protocol"] == protocol and r["pool"] == pool]
            ce_rows.append({"protocol": protocol, "pool": pool, "calibration_error_mae": subset[0]["calibration_error_mae"]})
    write_csv(out / "fig9_calibration_error.csv", ce_rows)
    fig, ax = plt.subplots(figsize=(6, 6)); ax.plot([0, 1], [0, 1], "k--", label="ideal")
    for pool in ("ID", "OOD-pooled", "OOD-Near", "OOD-Far"):
        sub = [r for r in cal_rows if r["protocol"] == "omitted_only" and r["pool"] == pool]; ax.plot([r["nominal"] for r in sub], [r["empirical"] for r in sub], marker="o", ms=3, label=pool)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Nominal coverage", ylabel="Empirical coverage"); ax.set_aspect("equal", adjustable="box"); ax.grid(alpha=.25); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out / "fig9_calibration_curve.png", dpi=180); plt.close(fig)
    summary = {"checkpoint": str(CHECKPOINT), "common_eval": str(COMMON), "samples_per_regime": args.samples_per_regime, "batch_size": args.batch_size, "ng": 16, "snr_db": 15.0, "device": str(device), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "elapsed_s": time.perf_counter() - start, "no_training": True, "implementation_assumptions": ["direct-CFR sample-wise AWGN at 15 dB", "uniform grouping mask Ng=16", "Student-t nominal quantile mapping used by canonical evaluator", "Eq.(12)/(13) sample aggregation"], "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20 if torch.cuda.is_available() else None}
    (out / "validation_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
