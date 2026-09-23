#!/usr/bin/env python3
"""Experiment 2: frozen-backbone feature-space OOD diagnostic."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
K = 1024
CHECKPOINT = ROOT / "runs/current_valid_baseline/paper_sample_alignment_100k_1ep_fig8_20260915/uacp_predictor_step4a.pt"
DATA_DIR = ROOT / "runs/baseline_reproduction/step2_delay_sweep_repro/generated_data"
REGIMES = [("20 ns", "20"), ("80 ns", "80"), ("120 ns", "120"), ("1 ms", "1_ms")]


def diagonal_mahalanobis(values: np.ndarray, mean: np.ndarray, variance: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(((values - mean) ** 2 / np.maximum(variance, 1e-8)).sum(axis=1), 0.0))


def rank_corr(x: list[float] | np.ndarray, y: list[float] | np.ndarray) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    xr, yr = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(xr, yr)[0, 1])


def pearson(x, y) -> float:
    if len(x) < 2:
        return float("nan")
    return float(np.corrcoef(np.asarray(x), np.asarray(y))[0, 1])


def auc(negative, positive) -> float:
    x = np.asarray(negative, float); y = np.asarray(positive, float)
    if not len(x) or not len(y): return float("nan")
    return float((sum((v > u) + .5 * (v == u) for v in y for u in x)) / (len(x) * len(y)))


def summary(values) -> dict:
    a = np.asarray(values, float)
    return {"count": int(len(a)), "mean": float(a.mean()), "median": float(np.quantile(a, .5)),
            "q95": float(np.quantile(a, .95))} if len(a) else {"count": 0, "mean": float("nan"), "median": float("nan"), "q95": float("nan")}


def rss_gib():
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024**2


def mask(batch, ng, device):
    m = torch.zeros((batch, K), device=device)
    m[:, ::ng] = 1.0
    return m


def load_model(cfg, device):
    from scripts.diagnose_predictor import _make_model
    model = _make_model(cfg, device)
    payload = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    state = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
    model.load_state_dict(state); model.eval()
    return model


def collect(model, device, cfr, seed, batch_size):
    from scripts.train_predictor import set_seeds
    from src.training.data import build_noisy_sparse_input
    features, scores = [], []
    captured = []
    handle = model.residual_blocks[-1].register_forward_hook(lambda _, __, out: captured.append(out))
    try:
        for start in range(0, len(cfr), batch_size):
            set_seeds(seed + start)
            b = torch.from_numpy(cfr[start:start + batch_size]).to(device)
            m = mask(b.shape[0], 16, device)  # fixed training-supported mask; assumption for feature diagnostic
            x, _, _ = build_noisy_sparse_input(b, m, 15.0)
            with torch.inference_mode():
                out = model(x)
                h = captured.pop()
                # h is the shared representation immediately before evidential heads.
                f = h.mean(dim=-1).detach().cpu().numpy()
                ale = out.psi / (out.nu_expanded - 2 * K - 1.0)
                epi = ale / out.kappa_expanded
                pair = epi.reshape(epi.shape[0], 2, 4, K).permute(0, 2, 1, 3).sum(dim=2).mean(dim=1)
                omitted = 1.0 - m
                s = ((pair * omitted).sum(dim=1) / omitted.sum(dim=1)).detach().cpu().numpy()
            features.extend(f.astype(np.float32, copy=False))
            scores.extend(float(v) for v in s)
            del b, m, x, out, h, f, ale, epi, pair, omitted, s
    finally:
        handle.remove()
    gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache()
    return np.asarray(features, np.float32), np.asarray(scores, np.float64)


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--output-dir", required=True); p.add_argument("--batch-size", type=int, default=8); p.add_argument("--samples-per-regime", type=int, default=200); p.add_argument("--seed", type=int, default=20260917); a = p.parse_args()
    out = ROOT / a.output_dir
    if out.exists() and any(out.iterdir()): raise FileExistsError(f"refusing to overwrite non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=True); sys.path.insert(0, str(ROOT))
    from scripts.train_predictor import load_config
    cfg = load_config("configs/current_valid_baseline_100k1_seed_20260819.json")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    model = load_model(cfg, device); collected = {}
    for i, (name, suffix) in enumerate(REGIMES):
        filename = f"test_delay_{suffix}.npz" if suffix == "1_ms" else f"test_delay_{suffix}_ns.npz"
        with np.load(DATA_DIR / filename) as data: cfr = np.asarray(data["cfr"][:a.samples_per_regime], np.complex64)
        collected[name] = collect(model, device, cfr, a.seed + i * 10000, a.batch_size); del cfr; gc.collect()
    id_features = np.concatenate([collected["20 ns"][0], collected["80 ns"][0]])
    id_mean = id_features.mean(axis=0); id_var = id_features.var(axis=0) + 1e-6
    id_maha = diagonal_mahalanobis(id_features, id_mean, id_var)
    rows = []; corr_rows = []
    for name, _ in REGIMES:
        f, u = collected[name]; d = diagonal_mahalanobis(f, id_mean, id_var)
        r = {"regime": name}; r.update(summary(d)); rows.append(r)
        corr_rows.append({"regime": name, "pearson_mahalanobis_vs_u_epi": pearson(d, u), "spearman_mahalanobis_vs_u_epi": rank_corr(d, u)})
    near_d = diagonal_mahalanobis(collected["120 ns"][0], id_mean, id_var); far_d = diagonal_mahalanobis(collected["1 ms"][0], id_mean, id_var)
    result = {"experiment": "Experiment 2: Backbone Feature-space OOD Detection", "no_training": True,
              "checkpoint": str(CHECKPOINT.relative_to(ROOT)), "feature_layer": "output of residual_blocks[-1], immediately before evidential heads",
              "feature_transform": "mean pool across frequency: [B,192,1024] -> [B,192]",
              "covariance": "ID-only diagonal covariance with variance floor 1e-6 (IMPLEMENTATION-ASSUMPTION)",
              "mask": "fixed Ng=16, training-supported, to avoid mask confounding (IMPLEMENTATION-ASSUMPTION)",
              "source": str(DATA_DIR.relative_to(ROOT)), "samples_per_regime": a.samples_per_regime,
              "summary": rows, "correlation": corr_rows,
              "near_auroc": auc(id_maha, near_d), "far_auroc": auc(id_maha, far_d),
              "u_epi_near_auroc": auc(np.concatenate([collected["20 ns"][1], collected["80 ns"][1]]), collected["120 ns"][1]),
              "u_epi_far_auroc": auc(np.concatenate([collected["20 ns"][1], collected["80 ns"][1]]), collected["1 ms"][1]),
              "device": str(device), "gpu": gpu, "peak_rss_gib": rss_gib(),
              "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None}
    write_csv(out / "mahalanobis_summary.csv", rows); write_csv(out / "correlation_summary.csv", corr_rows)
    (out / "results.json").write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    print(json.dumps({"output": str(out), "device": str(device), "gpu": gpu, "near_auroc": result["near_auroc"], "far_auroc": result["far_auroc"], "peak_rss_gib": result["peak_rss_gib"]}, indent=2), flush=True)


if __name__ == "__main__": main()
