#!/usr/bin/env python3
"""sample과 antenna pair 단위에서 Psi가 error를 따라가는지 확인한다.

실험 목적:
    CFR power와 reconstruction error에 대한 learned Psi의 관계를 보고,
    20 ns와 80 ns의 차이가 evidential parameter에 전달되는지 확인한다.

주의:
    power matching은 비교용일 뿐이며 checkpoint는 수정하지 않는다.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.diagnose_predictor import _make_model  # noqa: E402
from scripts.train_predictor import load_config, set_seeds  # noqa: E402
from src.models.evidential import channels_to_pair_vectors  # noqa: E402
from src.training.data import CFRNPZDataset, build_noisy_sparse_input, uniform_grouping_mask  # noqa: E402

REGIMES = [("ID-Easy 20 ns", 20, "test_20ns.npz"), ("ID-Hard 80 ns", 80, "test_80ns.npz")]


def corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0, 0.0
    p = float(np.corrcoef(x, y)[0, 1])
    sx = np.argsort(np.argsort(x)).astype(float)
    sy = np.argsort(np.argsort(y)).astype(float)
    return p, float(np.corrcoef(sx, sy)[0, 1])


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


@torch.no_grad()
def evaluate(model, path: Path, cfg: dict, device: torch.device, seed: int, regime: str) -> list[dict]:
    loader = DataLoader(CFRNPZDataset(path), batch_size=int(cfg["implementation_assumption"]["eval_batch_size"]))
    rows = []
    for batch_index, batch in enumerate(loader):
        cfr = batch["cfr"].to(device)
        mask = uniform_grouping_mask(cfr.shape[0], 1024, 16, device)
        set_seeds(seed + batch_index)
        x, target, _ = build_noisy_sparse_input(cfr, mask, 15.0)
        output = model(x)
        target_pair = channels_to_pair_vectors(target)
        gamma_pair = channels_to_pair_vectors(output.gamma)
        psi_pair = channels_to_pair_vectors(output.psi)
        ale_pair = channels_to_pair_vectors(output.aleatoric)
        omitted = (1.0 - mask).bool()
        scale = cfr.abs().square().mean(dim=(1, 2, 3)).sqrt().clamp_min(1e-12)
        scale2 = scale.square()
        gamma_norm = gamma_pair / scale[:, None, None]
        target_norm = target_pair / scale[:, None, None]
        psi_norm = psi_pair / scale2[:, None, None]
        ale_norm = ale_pair / scale2[:, None, None]
        error = (gamma_pair - target_pair).square()
        error_norm = (gamma_norm - target_norm).square()
        for i in range(cfr.shape[0]):
            om = omitted[i]
            # All quantities below are per sample and per antenna pair.
            raw_error = (error[i, :, :1024][:, om].mean(dim=1) + error[i, :, 1024:][:, om].mean(dim=1))
            norm_error = (error_norm[i, :, :1024][:, om].mean(dim=1) + error_norm[i, :, 1024:][:, om].mean(dim=1))
            raw_psi = psi_pair[i].mean(dim=1)
            norm_psi = psi_norm[i].mean(dim=1)
            raw_ale = ale_pair[i, :, :1024][:, om].mean(dim=1) + ale_pair[i, :, 1024:][:, om].mean(dim=1)
            norm_ale = ale_norm[i, :, :1024][:, om].mean(dim=1) + ale_norm[i, :, 1024:][:, om].mean(dim=1)
            for pair in range(4):
                rows.append({
                    "regime": regime, "sample_index": len({r["regime"] + str(r["sample_index"]) for r in rows}), "pair": pair,
                    "cfr_power": float(scale2[i].cpu()), "cfr_rms": float(scale[i].cpu()),
                    "absolute_error_mse": float(raw_error[pair].cpu()), "normalized_error_mse": float(norm_error[pair].cpu()),
                    "raw_psi_scale": float(raw_psi[pair].cpu()), "normalized_psi_scale": float(norm_psi[pair].cpu()),
                    "raw_aleatoric": float(raw_ale[pair].cpu()), "normalized_aleatoric": float(norm_ale[pair].cpu()),
                    "kappa": float(output.kappa[i, pair, 0].cpu()), "nu": float(output.nu[i, pair, 0].cpu()),
                })
    # Correct sample index after pair rows were emitted.
    for idx, row in enumerate(rows):
        row["sample_index"] = idx // 4
    return rows


def make_corr(rows: list[dict]) -> list[dict]:
    out = []
    for regime in sorted({r["regime"] for r in rows}):
        rr = [r for r in rows if r["regime"] == regime]
        pairs = [
            ("raw_psi_scale", "cfr_power"), ("raw_psi_scale", "absolute_error_mse"),
            ("normalized_psi_scale", "normalized_error_mse"), ("normalized_psi_scale", "cfr_power"),
        ]
        for left, right in pairs:
            p, s = corr(np.array([r[left] for r in rr]), np.array([r[right] for r in rr]))
            out.append({"regime": regime, "x": left, "y": right, "pearson": p, "spearman": s, "observations": len(rr)})
    return out


def make_summary(rows: list[dict]) -> list[dict]:
    output = []
    for regime in sorted({r["regime"] for r in rows}):
        rr = [r for r in rows if r["regime"] == regime]
        row = {"regime": regime, "samples": len({r["sample_index"] for r in rr}), "sample_pairs": len(rr)}
        for key in ("cfr_power", "absolute_error_mse", "normalized_error_mse", "raw_psi_scale", "normalized_psi_scale", "raw_aleatoric", "normalized_aleatoric", "kappa", "nu"):
            values = np.array([r[key] for r in rr], dtype=float); row[key + "_mean"] = float(values.mean()); row[key + "_std"] = float(values.std())
        output.append(row)
    return output


def make_power_bins(rows: list[dict], bins: int = 5) -> list[dict]:
    powers = np.array([r["cfr_power"] for r in rows], dtype=float)
    edges = np.quantile(powers, np.linspace(0, 1, bins + 1)); edges[0] -= 1e-12; edges[-1] += 1e-12
    for r in rows:
        r["power_bin"] = int(np.searchsorted(edges, r["cfr_power"], side="right") - 1)
        r["power_bin"] = min(max(r["power_bin"], 0), bins - 1)
    out = []
    for b in range(bins):
        for regime in ("ID-Easy 20 ns", "ID-Hard 80 ns"):
            rr = [r for r in rows if r["power_bin"] == b and r["regime"] == regime]
            if not rr: continue
            out.append({"power_bin": b, "regime": regime, "count_pairs": len(rr), "power_min": float(min(r["cfr_power"] for r in rr)), "power_max": float(max(r["cfr_power"] for r in rr)), **{k + "_mean": float(np.mean([r[k] for r in rr])) for k in ("cfr_power", "normalized_error_mse", "normalized_psi_scale", "normalized_aleatoric")}})
    return out


def plot(rows: list[dict], bins: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (x, y, title) in zip(axes.flat, [("cfr_power", "raw_psi_scale", "Raw Psi vs CFR power"), ("absolute_error_mse", "raw_psi_scale", "Raw Psi vs absolute error"), ("normalized_error_mse", "normalized_psi_scale", "Normalized Psi vs normalized error"), ("cfr_power", "normalized_psi_scale", "Normalized Psi vs CFR power")]):
        for regime, color in (("ID-Easy 20 ns", "tab:blue"), ("ID-Hard 80 ns", "tab:orange")):
            rr = [r for r in rows if r["regime"] == regime]; ax.scatter([r[x] for r in rr], [r[y] for r in rr], s=7, alpha=.25, label=regime, color=color)
        ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(title); ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "psi_error_alignment_scatter.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, key, title in zip(axes, ("normalized_error_mse_mean", "normalized_psi_scale_mean", "normalized_aleatoric_mean"), ("Normalized error", "Normalized Psi", "Normalized Aleatoric")):
        for regime, color in (("ID-Easy 20 ns", "tab:blue"), ("ID-Hard 80 ns", "tab:orange")):
            rr = [r for r in bins if r["regime"] == regime]; ax.plot([r["power_bin"] for r in rr], [r[key] for r in rr], "o-", label=regime, color=color)
        ax.set_xlabel("Common CFR-power quantile bin"); ax.set_title(title); ax.grid(alpha=.2)
    axes[0].set_ylabel("Mean value"); axes[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(out / "power_matched_bins.png", dpi=160); plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--config", default="runs/current_valid_baseline/condition_c_5k_10ep_lambda1e3_corrected/config.json"); p.add_argument("--checkpoint", default="runs/current_valid_baseline/condition_c_5k_10ep_lambda1e3_corrected/checkpoint_with_provenance.pt"); p.add_argument("--probe-dir", default="runs/paper_style_uncertainty_diagnostics_20260904/generated_p3"); p.add_argument("--output-dir", required=True); args = p.parse_args()
    out = ROOT / args.output_dir
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    cfg = load_config(args.config); set_seeds(int(cfg["implementation_assumption"]["seed"])); device = torch.device(cfg["implementation_assumption"]["device"] if torch.cuda.is_available() else "cpu")
    model = _make_model(cfg, device); payload = torch.load(ROOT / args.checkpoint, map_location=device, weights_only=False); model.load_state_dict(payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload); model.eval()
    started = time.perf_counter(); rows = []
    for index, (regime, delay, filename) in enumerate(REGIMES):
        rows.extend(evaluate(model, ROOT / args.probe_dir / filename, cfg, device, 99000 + index * 1000, regime))
    bins = make_power_bins(rows); summary = make_summary(rows); correlations = make_corr(rows)
    out.mkdir(parents=True, exist_ok=True); write_csv(out / "sample_pair_metrics.csv", rows); write_csv(out / "power_matched_bins.csv", bins); write_csv(out / "summary.csv", summary); write_csv(out / "correlations.csv", correlations); plot(rows, bins, out)
    ck = ROOT / args.checkpoint; prov = {"checkpoint": args.checkpoint, "checkpoint_sha256": hashlib.sha256(ck.read_bytes()).hexdigest(), "config": args.config, "probe_dir": args.probe_dir, "device": str(device), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "training_performed": False, "mask": "Ng=16, 64 observed (mask=1), 960 omitted (mask=0)", "pair_mapping": "channels (0,4),(1,5),(2,6),(3,7)", "normalization": "a=sqrt(mean(|H_true|^2)); Psi and Sigma variance-derived quantities divided by a^2", "runtime_seconds": time.perf_counter() - started}
    (out / "provenance.json").write_text(json.dumps(prov, indent=2, sort_keys=True)); (out / "config.json").write_text(json.dumps({"regimes": [r[:2] for r in REGIMES], "power_bins": 5, "snr_db": 15.0, "ng": 16, "training_performed": False}, indent=2)); (out / "results.json").write_text(json.dumps({"summary": summary, "correlations": correlations, "power_matched_bins": bins, "provenance": prov}, indent=2, sort_keys=True)); print(json.dumps({"output_dir": str(out), "device": str(device), "gpu": prov["gpu"], "summary": summary, "correlations": correlations}, indent=2), flush=True)


if __name__ == "__main__": main()
