#!/usr/bin/env python3
"""RAM-safe Eq.(5) parameter decomposition for the frozen 100k x 1 model."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REGIMES = [("20 ns", "ID-Easy_20_ns.npz"), ("80 ns", "ID-Hard_80_ns.npz"),
           ("120 ns", "OOD-Near_120_ns.npz"), ("1 ms", "OOD-Far_1_ms.npz")]
K = 1024
PARAMETERS = ("psi", "kappa", "df", "kappa_factor", "inv_df", "scale_sq", "scale", "error", "error_scale_ratio")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class OnlineStats:
    """Exact count/sum and bounded reservoir for quantiles."""

    def __init__(self, reservoir_size: int = 8192, seed: int = 0) -> None:
        self.count = 0
        self.total = 0.0
        self.reservoir = np.empty(0, dtype=np.float64)
        self.reservoir_size = reservoir_size
        self.rng = np.random.default_rng(seed)

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if values.size == 0:
            return
        self.total += float(values.sum(dtype=np.float64))
        previous = self.count
        self.count += int(values.size)
        if self.reservoir.size < self.reservoir_size:
            take = min(self.reservoir_size - self.reservoir.size, values.size)
            self.reservoir = np.concatenate((self.reservoir, values[:take]))
            values = values[take:]
        if values.size:
            slots = self.rng.integers(0, self.count, size=values.size)
            keep = slots < self.reservoir_size
            self.reservoir[slots[keep]] = values[keep]

    def summary(self) -> dict[str, float | int]:
        q = self.reservoir
        return {"count": self.count, "mean": self.total / max(1, self.count),
                "median": float(np.quantile(q, .50)), "q90": float(np.quantile(q, .90)),
                "q95": float(np.quantile(q, .95)), "q99": float(np.quantile(q, .99))}


def rss_gib() -> float:
    # ru_maxrss is KiB on Linux.
    import resource
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024 ** 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--common-eval-dir", required=True)
    parser.add_argument("--samples-per-regime", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-seed", type=int, default=20262000)
    args = parser.parse_args()
    out = ROOT / args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(ROOT))
    from scripts.diagnose_predictor import _make_model
    from scripts.train_predictor import load_config, set_seeds
    from src.training.data import build_noisy_sparse_input, uniform_grouping_mask

    cfg = load_config("configs/current_valid_baseline_100k1_seed_20260819.json")
    checkpoint = ROOT / "runs/current_valid_baseline/paper_sample_alignment_100k_1ep_fig8_20260915/uacp_predictor_step4a.pt"
    common = ROOT / args.common_eval_dir
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cuda_info = {"is_available": bool(torch.cuda.is_available()), "device": str(device),
                 "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    print(json.dumps({"cuda_start": cuda_info, "rss_gib": rss_gib()}), flush=True)

    model = _make_model(cfg, device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload)
    model.eval()
    stats = {(regime, parameter): OnlineStats(seed=ri * 100 + pi)
             for ri, (regime, _) in enumerate(REGIMES)
             for pi, parameter in enumerate(PARAMETERS)}
    memory_rows = []

    for ri, (regime, filename) in enumerate(REGIMES):
        loaded = np.load(common / filename)
        cfr = loaded["cfr"]
        if len(cfr) < args.samples_per_regime:
            raise RuntimeError(f"{regime}: {len(cfr)} samples")
        cfr = cfr[:args.samples_per_regime]
        for start in range(0, len(cfr), args.batch_size):
            c = torch.from_numpy(cfr[start:start + args.batch_size]).to(device)
            mask = uniform_grouping_mask(c.shape[0], K, 16, device)
            set_seeds(args.eval_seed + ri * 100000 + start)
            x, target, _ = build_noisy_sparse_input(c, mask, 15.0)
            with torch.inference_mode():
                output = model(x)
                omitted = (mask < .5)[:, None, :]
                df = output.nu_expanded - 2 * K + 1.0
                kappa_factor = (output.kappa_expanded + 1.0) / output.kappa_expanded
                inv_df = 1.0 / df
                scale_sq = output.psi * kappa_factor * inv_df
                scale = torch.sqrt(scale_sq.clamp_min(1e-12))
                error = (target - output.gamma).abs()
                ratio = error / scale.clamp_min(1e-12)
                values = {"psi": output.psi, "kappa": output.kappa_expanded, "df": df,
                          "kappa_factor": kappa_factor, "inv_df": inv_df, "scale_sq": scale_sq,
                          "scale": scale, "error": error, "error_scale_ratio": ratio}
                keep = omitted.expand_as(output.psi)
                for parameter, tensor in values.items():
                    stats[(regime, parameter)].update(tensor[keep].detach().cpu().numpy())
            del c, mask, x, target, output, df, kappa_factor, inv_df, scale_sq, scale, error, ratio, values
        del cfr, loaded
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        memory_rows.append({"regime": regime, "max_rss_gib": rss_gib()})
        print(json.dumps(memory_rows[-1]), flush=True)

    summary_rows = []
    for regime, _ in REGIMES:
        row = {"regime": regime}
        for parameter in PARAMETERS:
            for key, value in stats[(regime, parameter)].summary().items():
                row[f"{parameter}_{key}"] = value
        summary_rows.append(row)
    write_csv(out / "parameter_summary.csv", summary_rows)

    baseline = summary_rows[0]
    relative_rows = []
    for row in summary_rows:
        relative = {"regime": row["regime"]}
        for parameter in PARAMETERS:
            relative[f"{parameter}_mean_ratio"] = row[f"{parameter}_mean"] / baseline[f"{parameter}_mean"]
            relative[f"{parameter}_median_ratio"] = row[f"{parameter}_median"] / baseline[f"{parameter}_median"]
        relative_rows.append(relative)
    write_csv(out / "relative_change.csv", relative_rows)
    write_csv(out / "memory_log.csv", memory_rows)

    labels = [row["regime"] for row in summary_rows]
    plt.figure(figsize=(7, 5))
    plt.plot(labels, [row["error_mean"] for row in summary_rows], "o-", label="|target-gamma|")
    plt.plot(labels, [row["scale_mean"] for row in summary_rows], "o-", label="predictive scale")
    plt.ylabel("Mean omitted-component value"); plt.grid(alpha=.25); plt.legend(); plt.tight_layout()
    plt.savefig(out / "error_vs_scale.png", dpi=180); plt.close()
    plt.figure(figsize=(7, 5))
    for parameter, label in (("psi", "Psi"), ("kappa_factor", "(kappa+1)/kappa"), ("inv_df", "1/df")):
        plt.plot(labels, [row[f"{parameter}_mean"] / baseline[f"{parameter}_mean"] for row in summary_rows], "o-", label=label)
    plt.axhline(1.0, color="k", linestyle="--", linewidth=.8); plt.ylabel("Ratio to 20 ns mean")
    plt.grid(alpha=.25); plt.legend(); plt.tight_layout(); plt.savefig(out / "parameter_factor_change.png", dpi=180); plt.close()

    result = {"checkpoint": str(checkpoint), "common_eval_dir": str(common),
              "samples_per_regime": args.samples_per_regime, "batch_size": args.batch_size,
              "eval_seed": args.eval_seed, "ng": 16, "snr_db": 15.0, "device": cuda_info,
              "max_rss_gib": max(row["max_rss_gib"] for row in memory_rows),
              "aggregation": "all valid omitted real/imag components; exact online sums/counts",
              "quantiles": "fixed-size reservoir (8192) per regime/parameter; means exact",
              "formula": "df=nu-2K+1; scale_sq=Psi*((kappa+1)/kappa)/df; scale=sqrt(scale_sq)",
              "no_training": True,
              "forbidden_runs": ["Full-Psi", "Fig.11/runtime", "Partial Fine-Tuning"],
              "assumptions": ["diagonal-Psi approximation", "corrected Real/Imag antenna-pair mapping",
                              "same CFR/mask/noise policy as prior Fig.9 evaluator"]}
    (out / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"cuda_end": cuda_info, "max_rss_gib": result["max_rss_gib"]}), flush=True)


if __name__ == "__main__":
    main()
