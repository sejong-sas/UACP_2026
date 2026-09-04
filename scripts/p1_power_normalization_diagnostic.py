#!/usr/bin/env python3
"""CFR power 차이가 uncertainty 진단을 섞는지 확인한다.

실험 목적:
    sample별 CFR RMS power를 제거한 뒤 prediction error와 uncertainty의
    관계가 바뀌는지 비교한다.

주의:
    CFR은 amplitude로 나누고 covariance 기반 uncertainty는 variance이므로
    power scale의 제곱으로 나눈다. checkpoint와 weight는 읽기만 한다.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.diagnose_predictor import _make_model  # noqa: E402
from scripts.train_predictor import load_config, set_seeds  # noqa: E402
from src.training.data import CFRNPZDataset, build_noisy_sparse_input, uniform_grouping_mask  # noqa: E402
from src.training.uncertainty import paper_subcarrier_uncertainty_map  # noqa: E402

REGIMES = [("ID-Easy 20 ns", 20.0), ("ID-Hard 80 ns", 80.0), ("OOD-Near 120 ns", 120.0), ("OOD-Far 1 ms", 1_000_000.0)]


def corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0, 0.0
    pearson = float(np.corrcoef(x, y)[0, 1])
    xr = np.argsort(np.argsort(x)).astype(float)
    yr = np.argsort(np.argsort(y)).astype(float)
    spearman = float(np.corrcoef(xr, yr)[0, 1])
    return pearson, spearman


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


@torch.no_grad()
def evaluate_regime(model, path: str, cfg: dict, device: torch.device, seed: int) -> list[dict]:
    loader = DataLoader(CFRNPZDataset(path), batch_size=int(cfg["implementation_assumption"]["eval_batch_size"]))
    rows = []
    for batch_index, batch in enumerate(loader):
        cfr = batch["cfr"].to(device)
        mask = uniform_grouping_mask(cfr.shape[0], 1024, 16, device)
        set_seeds(seed + batch_index)
        x, target, _ = build_noisy_sparse_input(cfr, mask, 15.0)
        output = model(x)
        omitted = 1.0 - mask
        error_sq = (output.predicted - target).square()
        omitted_channels = omitted[:, None, :]
        abs_mse = (error_sq * omitted_channels).sum(dim=(1, 2)) / (omitted_channels.sum(dim=(1, 2)).clamp_min(1.0))
        target_power = cfr.abs().square().mean(dim=(1, 2, 3))
        nmse = 10.0 * torch.log10(((error_sq * omitted_channels).sum(dim=(1, 2)) / (target.square() * omitted_channels).sum(dim=(1, 2)).clamp_min(1e-12)).clamp_min(1e-12))
        ale_map = paper_subcarrier_uncertainty_map(output.aleatoric)
        epi_map = paper_subcarrier_uncertainty_map(output.epistemic)
        ale = (ale_map * omitted).sum(dim=1) / omitted.sum(dim=1).clamp_min(1.0)
        epi = (epi_map * omitted).sum(dim=1) / omitted.sum(dim=1).clamp_min(1.0)
        scale_sq = target_power.clamp_min(1e-12)
        norm_target = target / torch.sqrt(scale_sq)[:, None, None]
        norm_pred = output.predicted / torch.sqrt(scale_sq)[:, None, None]
        norm_error_sq = (norm_pred - norm_target).square()
        norm_abs_mse = (norm_error_sq * omitted_channels).sum(dim=(1, 2)) / omitted_channels.sum(dim=(1, 2)).clamp_min(1.0)
        norm_ale = ale / scale_sq
        norm_epi = epi / scale_sq
        norm_nmse = 10.0 * torch.log10(((norm_error_sq * omitted_channels).sum(dim=(1, 2)) / (norm_target.square() * omitted_channels).sum(dim=(1, 2)).clamp_min(1e-12)).clamp_min(1e-12))
        for i in range(cfr.shape[0]):
            rows.append({"sample_index": len(rows), "true_cfr_power": float(target_power[i].cpu()), "rms_scale": float(torch.sqrt(scale_sq[i]).cpu()), "absolute_prediction_mse": float(abs_mse[i].cpu()), "existing_nmse_db": float(nmse[i].cpu()), "total_uncertainty": float((ale[i] + epi[i]).cpu()), "normalized_absolute_prediction_mse": float(norm_abs_mse[i].cpu()), "normalized_nmse_db": float(norm_nmse[i].cpu()), "normalized_total_uncertainty": float((norm_ale[i] + norm_epi[i]).cpu()), "aleatoric": float(ale[i].cpu()), "epistemic": float(epi[i].cpu()), "normalized_aleatoric": float(norm_ale[i].cpu()), "normalized_epistemic": float(norm_epi[i].cpu())})
    return rows


def plot_scatter(rows: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for col, (regime, _) in enumerate(REGIMES):
        values = [r for r in rows if r["regime"] == regime]
        for ax, ux, ey, title in ((axes[0, col], "total_uncertainty", "existing_nmse_db", "Before"), (axes[1, col], "normalized_total_uncertainty", "normalized_nmse_db", "After power normalization")):
            ax.scatter([r[ux] for r in values], [r[ey] for r in values], s=7, alpha=.35)
            ax.set_title(f"{title}\n{regime}"); ax.set_xlabel("Total uncertainty"); ax.set_ylabel("NMSE (dB)"); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out / "p1_before_after_scatter.png", dpi=160); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="runs/current_valid_baseline/condition_c_5k_10ep_lambda1e3_corrected/config.json"); parser.add_argument("--checkpoint", default="runs/current_valid_baseline/condition_c_5k_10ep_lambda1e3_corrected/checkpoint_with_provenance.pt"); parser.add_argument("--output-dir", required=True); args = parser.parse_args()
    out = ROOT / args.output_dir
    if out.exists() and any(out.iterdir()): raise FileExistsError(f"Refusing to overwrite existing output: {out}")
    cfg = load_config(args.config); set_seeds(int(cfg["implementation_assumption"]["seed"])); device = torch.device(cfg["implementation_assumption"]["device"] if torch.cuda.is_available() else "cpu")
    model = _make_model(cfg, device); payload = torch.load(ROOT / args.checkpoint, map_location=device, weights_only=False); model.load_state_dict(payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload); model.eval()
    rows = []
    for index, (regime, _) in enumerate(REGIMES):
        for row in evaluate_regime(model, cfg["data"]["test_paths"][regime], cfg, device, 97000 + index * 1000): row["regime"] = regime; rows.append(row)
    summaries = []
    for regime, _ in REGIMES:
        values = [r for r in rows if r["regime"] == regime]
        arrays = {key: np.array([r[key] for r in values]) for key in ("true_cfr_power", "absolute_prediction_mse", "existing_nmse_db", "total_uncertainty", "normalized_absolute_prediction_mse", "normalized_nmse_db", "normalized_total_uncertainty")}
        row = {"regime": regime, "samples": len(values)}
        for prefix, ux, ey in (("before", "total_uncertainty", "existing_nmse_db"), ("after", "normalized_total_uncertainty", "normalized_nmse_db")):
            p, s = corr(arrays[ux], arrays[ey]); row[f"{prefix}_u_nmse_pearson"] = p; row[f"{prefix}_u_nmse_spearman"] = s
        for ux, ey, label in (("total_uncertainty", "true_cfr_power", "u_power"), ("total_uncertainty", "absolute_prediction_mse", "u_abs_mse"), ("total_uncertainty", "existing_nmse_db", "u_nmse"), ("normalized_total_uncertainty", "normalized_absolute_prediction_mse", "norm_u_abs_mse"), ("normalized_total_uncertainty", "normalized_nmse_db", "norm_u_nmse")):
            p, s = corr(arrays[ux], arrays[ey]); row[f"{label}_pearson"] = p; row[f"{label}_spearman"] = s
        for key, array in arrays.items(): row[f"{key}_mean"] = float(array.mean()); row[f"{key}_std"] = float(array.std())
        summaries.append(row)
    out.mkdir(parents=True, exist_ok=True); write_csv(out / "sample_metrics.csv", rows); write_csv(out / "summary.csv", summaries); plot_scatter(rows, out)
    provenance = {"checkpoint": args.checkpoint, "checkpoint_sha256": hashlib.sha256((ROOT / args.checkpoint).read_bytes()).hexdigest(), "device": str(device), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "seed": cfg["implementation_assumption"]["seed"], "training_performed": False, "normalization": "H_norm=H/a, gamma_norm=gamma/a, Sigma_norm=Sigma/a^2, a=sqrt(mean(|H|^2))"}
    (out / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True)); (out / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True)); (out / "results.json").write_text(json.dumps({"summaries": summaries, "provenance": provenance}, indent=2, sort_keys=True))
    print(json.dumps({"output_dir": str(out), "device": str(device), "gpu": provenance["gpu"], "summaries": summaries}, indent=2), flush=True)


if __name__ == "__main__": main()
