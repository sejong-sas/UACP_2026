#!/usr/bin/env python3
"""현재 valid checkpoint로 논문 Figure 5~8 목적의 평가를 수행한다.

실험 목적:
    uncertainty가 실제 reconstruction error의 proxy인지, ID hardness와
    OOD를 어느 정도 구분하는지 학습 없이 확인한다.

입력과 출력:
    checkpoint와 20/80/120 ns 및 1 ms CFR을 읽어 sample별 NMSE,
    Aleatoric/Epistemic, correlation, risk-coverage 결과를 저장한다.

주의:
    평가 전용 파일이며 checkpoint를 수정하지 않는다.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.diagnose_predictor import _make_model  # noqa: E402
from scripts.train_predictor import load_config, set_seeds  # noqa: E402
from src.channel.sionna_channel import generate_cfr_for_delay_spreads, load_sectioned_config  # noqa: E402
from src.training.data import CFRNPZDataset, build_noisy_sparse_input, uniform_grouping_mask  # noqa: E402
from src.training.uncertainty import paper_subcarrier_uncertainty_map  # noqa: E402

REGIMES = [("ID-Easy 20 ns", 20.0), ("ID-Hard 80 ns", 80.0), ("OOD-Near 120 ns", 120.0), ("OOD-Far 1 ms", 1_000_000.0)]


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(np.argsort(np.argsort(x)).astype(float), np.argsort(np.argsort(y)).astype(float))


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if not len(positives) or not len(negatives):
        return float("nan")
    comparisons = (positives[:, None] > negatives[None, :]).mean()
    ties = (positives[:, None] == negatives[None, :]).mean()
    return float(comparisons + 0.5 * ties)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_p1(rows: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, regime in zip(axes.flat, [r[0] for r in REGIMES]):
        values = [r for r in rows if r["regime"] == regime]
        ax.scatter([r["total_uncertainty"] for r in values], [r["nmse_omitted_db"] for r in values], s=8, alpha=0.35)
        ax.set_title(regime)
        ax.set_xlabel("Total uncertainty")
        ax.set_ylabel("Omitted NMSE (dB)")
        ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(out / "p1_total_uncertainty_vs_nmse.png", dpi=160); plt.close(fig)


def plot_p2(curves: dict[str, list[dict]], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for regime, values in curves.items():
        ax.plot([r["retained_fraction"] for r in values], [r["mean_nmse_omitted_db"] for r in values], marker="o", label=regime)
    ax.set_xlabel("Retained low-uncertainty fraction"); ax.set_ylabel("Mean omitted NMSE (dB)"); ax.grid(alpha=0.25); ax.legend(); fig.tight_layout(); fig.savefig(out / "p2_risk_coverage.png", dpi=160); plt.close(fig)


def plot_p3(rows: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = [r["delay_spread_ns"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(8, 5)); ax2 = ax1.twinx()
    ax1.plot(x, [r["aleatoric_mean"] for r in rows], "o-", label="Aleatoric")
    ax1.plot(x, [r["epistemic_mean"] for r in rows], "s-", label="Epistemic")
    ax2.plot(x, [r["nmse_omitted_mean_db"] for r in rows], "^-", color="black", label="NMSE")
    ax1.set_xlabel("RMS delay spread (ns)"); ax1.set_ylabel("Uncertainty"); ax2.set_ylabel("Omitted NMSE (dB)"); ax1.grid(alpha=0.25)
    lines, labels = ax1.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels(); ax1.legend(lines + lines2, labels + labels2); fig.tight_layout(); fig.savefig(out / "p3_delay_spread_uncertainty_nmse.png", dpi=160); plt.close(fig)


def plot_p4(distributions: dict[str, np.ndarray], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, values in distributions.items():
        ax.hist(values, bins=30, alpha=0.4, density=True, label=label)
    ax.set_xlabel("Epistemic uncertainty"); ax.set_ylabel("Density"); ax.grid(alpha=0.25); ax.legend(); fig.tight_layout(); fig.savefig(out / "p4_epistemic_distributions.png", dpi=160); plt.close(fig)


@torch.no_grad()
def evaluate_samples(model, dataset_path: str, cfg: dict, device: torch.device, seed: int) -> list[dict]:
    dataset = CFRNPZDataset(dataset_path)
    loader = DataLoader(dataset, batch_size=int(cfg["implementation_assumption"]["eval_batch_size"]))
    rows = []
    for batch_index, batch in enumerate(loader):
        cfr = batch["cfr"].to(device)
        mask = uniform_grouping_mask(cfr.shape[0], 1024, 16, device)
        set_seeds(seed + batch_index)
        x, target, _ = build_noisy_sparse_input(cfr, mask, 15.0)
        output = model(x)
        error = (output.predicted - target).square()
        omitted = (1.0 - mask)[:, None, :]
        nmse = 10 * torch.log10((error.mul(omitted).sum(dim=(1, 2)) / target.square().mul(omitted).sum(dim=(1, 2)).clamp_min(1e-12)).clamp_min(1e-12))
        ale_map = paper_subcarrier_uncertainty_map(output.aleatoric)
        epi_map = paper_subcarrier_uncertainty_map(output.epistemic)
        ale = (ale_map * (1.0 - mask)).sum(dim=1) / (1.0 - mask).sum(dim=1).clamp_min(1.0)
        epi = (epi_map * (1.0 - mask)).sum(dim=1) / (1.0 - mask).sum(dim=1).clamp_min(1.0)
        for i in range(cfr.shape[0]):
            rows.append({"sample_index": len(rows), "nmse_omitted_db": float(nmse[i].cpu()), "aleatoric": float(ale[i].cpu()), "epistemic": float(epi[i].cpu()), "total_uncertainty": float((ale[i] + epi[i]).cpu())})
    return rows


def run_p1_p2(model, cfg: dict, device: torch.device, out: Path) -> tuple[list[dict], dict[str, list[dict]]]:
    all_rows, curves = [], {}
    for index, (regime, _) in enumerate(REGIMES):
        rows = evaluate_samples(model, cfg["data"]["test_paths"][regime], cfg, device, 97000 + index * 1000)
        for row in rows:
            row["regime"] = regime
            all_rows.append(row)
        ordered = sorted(rows, key=lambda r: r["total_uncertainty"])
        values = []
        for fraction in [i / 10 for i in range(1, 11)]:
            selected = ordered[: max(1, math.ceil(len(ordered) * fraction))]
            values.append({"regime": regime, "retained_fraction": fraction, "selected_count": len(selected), "mean_nmse_omitted_db": float(np.mean([r["nmse_omitted_db"] for r in selected])), "mean_total_uncertainty": float(np.mean([r["total_uncertainty"] for r in selected]))})
        curves[regime] = values
    summary = []
    for regime, _ in REGIMES:
        values = [r for r in all_rows if r["regime"] == regime]
        errors = np.array([r["nmse_omitted_db"] for r in values]); uncertainty = np.array([r["total_uncertainty"] for r in values])
        summary.append({"regime": regime, "samples": len(values), "mean_nmse_omitted_db": float(errors.mean()), "std_nmse_omitted_db": float(errors.std()), "mean_total_uncertainty": float(uncertainty.mean()), "std_total_uncertainty": float(uncertainty.std()), "pearson": pearson(uncertainty, errors), "spearman": spearman(uncertainty, errors), "low30_nmse_omitted_db": curves[regime][2]["mean_nmse_omitted_db"], "all_nmse_omitted_db": curves[regime][-1]["mean_nmse_omitted_db"]})
    write_csv(out / "p1_sample_metrics.csv", all_rows); write_csv(out / "p1_summary.csv", summary); write_csv(out / "p2_risk_coverage.csv", [r for values in curves.values() for r in values]); plot_p1(all_rows, out); plot_p2(curves, out)
    return all_rows, curves


def run_p3_p4(model, cfg: dict, device: torch.device, out: Path) -> tuple[list[dict], dict]:
    channel_cfg = load_sectioned_config("configs/dataset_prototype.json")
    delays = [10.0, 20.0, 40.0, 60.0, 80.0, 100.0]
    p3_rows, distributions = [], {}
    for index, delay in enumerate(delays):
        cfr_np = generate_cfr_for_delay_spreads(channel_cfg, np.full(200, delay, dtype=np.float32), seed=98000 + index * 1000)
        temp = out / "generated_p3"; temp.mkdir(exist_ok=True)
        path = temp / f"test_{int(delay)}ns.npz"
        np.savez(path, cfr=cfr_np, delay_spread_ns=np.full(200, delay, dtype=np.float32), regime_label=np.full(200, "ID-delay-sweep"), metadata_json=np.array(json.dumps({"delay_spread_ns": delay})))
        rows = evaluate_samples(model, str(path), cfg, device, 99000 + index * 1000)
        errors = np.array([r["nmse_omitted_db"] for r in rows]); ale = np.array([r["aleatoric"] for r in rows]); epi = np.array([r["epistemic"] for r in rows])
        p3_rows.append({"delay_spread_ns": delay, "samples": len(rows), "nmse_omitted_mean_db": float(errors.mean()), "nmse_omitted_std_db": float(errors.std()), "aleatoric_mean": float(ale.mean()), "aleatoric_std": float(ale.std()), "epistemic_mean": float(epi.mean()), "epistemic_std": float(epi.std()), "nmse_ci95_halfwidth_db": float(1.96 * errors.std() / math.sqrt(len(errors))), "aleatoric_ci95_halfwidth": float(1.96 * ale.std() / math.sqrt(len(ale))), "epistemic_ci95_halfwidth": float(1.96 * epi.std() / math.sqrt(len(epi)))})
    for index, (regime, _) in enumerate(REGIMES):
        rows = evaluate_samples(model, cfg["data"]["test_paths"][regime], cfg, device, 99500 + index * 1000)
        distributions[regime] = np.array([r["epistemic"] for r in rows])
    id_scores = np.concatenate([distributions[REGIMES[0][0]], distributions[REGIMES[1][0]]])
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(distributions[REGIMES[2][0]]) + len(distributions[REGIMES[3][0]]))])
    scores = np.concatenate([id_scores, distributions[REGIMES[2][0]], distributions[REGIMES[3][0]]])
    aurocs = {"overall_ood": auroc(scores, labels), "near_ood": auroc(np.concatenate([id_scores, distributions[REGIMES[2][0]]]), np.concatenate([np.zeros(len(id_scores)), np.ones(len(distributions[REGIMES[2][0]]))])), "far_ood": auroc(np.concatenate([id_scores, distributions[REGIMES[3][0]]]), np.concatenate([np.zeros(len(id_scores)), np.ones(len(distributions[REGIMES[3][0]]))]))}
    p4_rows = []
    for regime, values in distributions.items():
        p4_rows.append({"regime": regime, "samples": len(values), "mean": float(values.mean()), "median": float(np.median(values)), "std": float(values.std()), "p05": float(np.quantile(values, .05)), "p25": float(np.quantile(values, .25)), "p50": float(np.quantile(values, .50)), "p75": float(np.quantile(values, .75)), "p95": float(np.quantile(values, .95))})
    write_csv(out / "p3_delay_sweep.csv", p3_rows); write_csv(out / "p4_epistemic_distribution.csv", p4_rows); (out / "p4_auroc.json").write_text(json.dumps(aurocs, indent=2)); plot_p3(p3_rows, out); plot_p4(distributions, out)
    return p3_rows, aurocs


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="runs/current_valid_baseline/condition_c_5k_10ep_lambda1e3_corrected/config.json"); parser.add_argument("--checkpoint", default="runs/current_valid_baseline/condition_c_5k_10ep_lambda1e3_corrected/checkpoint_with_provenance.pt"); parser.add_argument("--output-dir", required=True); args = parser.parse_args()
    out = ROOT / args.output_dir
    if out.exists() and any(out.iterdir()): raise FileExistsError(f"Refusing to overwrite existing output: {out}")
    cfg = load_config(args.config); set_seeds(int(cfg["implementation_assumption"]["seed"])); device = torch.device(cfg["implementation_assumption"]["device"] if torch.cuda.is_available() else "cpu")
    model = _make_model(cfg, device); payload = torch.load(ROOT / args.checkpoint, map_location=device, weights_only=False); model.load_state_dict(payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload); model.eval()
    out.mkdir(parents=True, exist_ok=True); start = time.perf_counter(); p1_rows, curves = run_p1_p2(model, cfg, device, out); p3_rows, aurocs = run_p3_p4(model, cfg, device, out); elapsed = time.perf_counter() - start
    provenance = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "device": str(device), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "checkpoint": str(args.checkpoint), "checkpoint_sha256": hashlib.sha256((ROOT / args.checkpoint).read_bytes()).hexdigest(), "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), "seed": cfg["implementation_assumption"]["seed"], "runtime_seconds": elapsed, "training_performed": False}
    (out / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True)); (out / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True)); (out / "raw_metrics.csv").write_text((out / "p1_sample_metrics.csv").read_text()); (out / "summary.csv").write_text((out / "p1_summary.csv").read_text())
    summary = {"p1": [{k: v for k, v in r.items() if k != "samples" or True} for r in json.loads(json.dumps([]))], "p3": p3_rows, "p4_auroc": aurocs, "runtime_seconds": elapsed, "device": str(device), "gpu": provenance["gpu"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2)); print(json.dumps(provenance, indent=2), flush=True)


if __name__ == "__main__":
    main()
