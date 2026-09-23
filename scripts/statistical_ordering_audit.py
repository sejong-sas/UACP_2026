#!/usr/bin/env python3
"""Large independent-test audit for the current-valid Aleatoric ordering."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.diagnose_predictor import _make_model  # noqa: E402
from scripts.train_predictor import load_config, set_seeds  # noqa: E402
from src.channel.sionna_channel import generate_cfr_batch, load_sectioned_config  # noqa: E402
from src.training.data import build_noisy_sparse_input, cfr_to_real_imag, uniform_grouping_mask  # noqa: E402
from src.training.metrics import nmse_all_db, nmse_omitted_db  # noqa: E402
from src.training.uncertainty import paper_omitted_uncertainty_score  # noqa: E402


def bootstrap_delta(a20: np.ndarray, a80: np.ndarray, seed: int, draws: int = 5000) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    values = np.empty(draws, dtype=np.float64)
    for i in range(draws):
        values[i] = rng.choice(a80, a80.size).mean() - rng.choice(a20, a20.size).mean()
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
        "draws": draws,
        "seed": seed,
    }


@torch.no_grad()
def evaluate_regime(model, channel_cfg: dict, delay_ns: float, count: int, batch_size: int,
                    device: torch.device, seed: int) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for start in range(0, count, batch_size):
        n = min(batch_size, count - start)
        _, cfr = generate_cfr_batch(channel_cfg, delay_ns, n, seed + start)
        cfr = cfr.to(device)
        mask = uniform_grouping_mask(n, 1024, 16, device)
        set_seeds(seed + 100_000 + start)
        x, target, snr_stats = build_noisy_sparse_input(cfr, mask, 15.0)
        output = model(x)
        omitted = (1.0 - mask)[:, None, :]
        omitted_per_sample = (1.0 - mask).sum(dim=1).clamp_min(1.0)
        ale_map = output.aleatoric.reshape(n, 2, 4, 1024).sum(dim=2).mean(dim=1)
        epi_map = output.epistemic.reshape(n, 2, 4, 1024).sum(dim=2).mean(dim=1)
        ale = ((ale_map * (1.0 - mask)).sum(dim=1) / omitted_per_sample).cpu().numpy()
        epi = ((epi_map * (1.0 - mask)).sum(dim=1) / omitted_per_sample).cpu().numpy()
        pred = output.predicted
        target_real = target
        error = pred - target_real
        omitted_bool = omitted.bool().expand_as(error)
        error_sq = error.square()[omitted_bool].reshape(n, -1)
        target_sq = target_real.square()[omitted_bool].reshape(n, -1)
        nmse_om = 10.0 * torch.log10((error_sq.sum(dim=1) / target_sq.sum(dim=1).clamp_min(1e-12)).clamp_min(1e-12))
        all_error = error.square().reshape(n, -1).sum(dim=1)
        all_target = target_real.square().reshape(n, -1).sum(dim=1)
        nmse_all = 10.0 * torch.log10((all_error / all_target.clamp_min(1e-12)).clamp_min(1e-12))
        for i in range(n):
            rows.append({
                "sample": start + i,
                "nmse_all_db": float(nmse_all[i].cpu()),
                "nmse_omitted_db": float(nmse_om[i].cpu()),
                "aleatoric": float(ale[i]),
                "epistemic": float(epi[i]),
                "measured_snr_db": float(snr_stats["measured_snr_db_mean"]),
            })
    return rows


def summarize(rows: list[dict[str, float | int]], regime: str) -> dict[str, float | int | str]:
    result: dict[str, float | int | str] = {"regime": regime, "samples": len(rows)}
    for key in ("nmse_all_db", "nmse_omitted_db", "aleatoric", "epistemic"):
        values = np.asarray([float(r[key]) for r in rows])
        result[f"{key}_mean"] = float(values.mean())
        result[f"{key}_median"] = float(np.median(values))
        result[f"{key}_std"] = float(values.std(ddof=1))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/current_valid_baseline_condition_c.json")
    parser.add_argument("--channel-config", default="configs/dataset_prototype.json")
    parser.add_argument("--checkpoint", default="runs/current_valid_baseline/condition_c_5k_10ep_lambda1e3_corrected/checkpoint_with_provenance.pt")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples-per-regime", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    out = ROOT / args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing output: {out}")

    cfg = load_config(args.config)
    channel_cfg = load_sectioned_config(ROOT / args.channel_config)
    set_seeds(int(cfg["implementation_assumption"]["seed"]))
    device = torch.device(cfg["implementation_assumption"]["device"] if torch.cuda.is_available() else "cpu")
    model = _make_model(cfg, device)
    payload = torch.load(ROOT / args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"] if "model_state_dict" in payload else payload)
    model.eval()

    all_rows: dict[str, list[dict[str, float | int]]] = {}
    summaries = []
    for label, delay, offset in (("20 ns", 20.0, 20_000), ("80 ns", 80.0, 80_000)):
        rows = evaluate_regime(model, channel_cfg, delay, args.samples_per_regime, args.batch_size, device,
                               int(cfg["implementation_assumption"]["seed"]) + offset)
        all_rows[label] = rows
        summaries.append(summarize(rows, label))

    a20 = np.asarray([float(r["aleatoric"]) for r in all_rows["20 ns"]])
    a80 = np.asarray([float(r["aleatoric"]) for r in all_rows["80 ns"]])
    delta = bootstrap_delta(a20, a80, seed=20260914)
    delta["observed_sample_mean_difference"] = float(a80.mean() - a20.mean())
    if delta["ci95_high"] < 0:
        verdict = "REVERSED_ORDERING_CONFIRMED"
    elif delta["ci95_low"] <= 0 <= delta["ci95_high"]:
        verdict = "INSENSITIVE_OR_UNCERTAIN"
    else:
        verdict = "SMALL_SAMPLE_ORDERING_ARTIFACT_POSSIBLE"

    out.mkdir(parents=True, exist_ok=True)
    with (out / "sample_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["regime", "sample", "nmse_all_db", "nmse_omitted_db", "aleatoric", "epistemic", "measured_snr_db"])
        writer.writeheader()
        for regime, rows in all_rows.items():
            for row in rows:
                writer.writerow({"regime": regime, **row})
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader(); writer.writerows(summaries)
    result = {
        "experiment": "Current-valid large independent-test Aleatoric ordering audit",
        "config": str(args.config), "channel_config": str(args.channel_config), "checkpoint": str(args.checkpoint),
        "device": str(device), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "samples_per_regime": args.samples_per_regime, "batch_size": args.batch_size,
        "evaluation": {"mask": "uniform_grouping_mask", "Ng": 16, "observation": "direct CFR AWGN", "snr_db": 15.0},
        "summaries": summaries, "delta_ale_80_minus_20": delta, "verdict": verdict,
        "assumptions": ["IMPLEMENTATION-ASSUMPTION: TDL-A, zero mobility, normalize=false", "IMPLEMENTATION-ASSUMPTION: direct CFR AWGN stage"],
    }
    (out / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
