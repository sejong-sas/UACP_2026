#!/usr/bin/env python3
"""Fig.9-only audit/evaluation for the existing 100k checkpoint.

The only model forward in this script is the required common-10k evaluation.
It saves aggregate coverage counts, not a replacement for prior results.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
REGIMES = [("ID-Easy 20 ns", "ID"), ("ID-Hard 80 ns", "ID"),
           ("OOD-Near 120 ns", "OOD-Near"), ("OOD-Far 1 ms", "OOD-Far")]
NOMINALS = np.array([*np.arange(0.10, 1.00, 0.10), 0.95, 0.99])


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def synthetic_sanity(seed: int = 20260915, n: int = 300_000) -> dict:
    rng = np.random.default_rng(seed)
    df = 17.0
    scale = 1.7
    x = student_t.rvs(df, loc=0.0, scale=scale, size=n, random_state=rng)
    rows = []
    for nominal in [0.50, 0.80, 0.90, 0.95, 0.99]:
        q = student_t.ppf((1.0 + nominal) / 2.0, df)
        covered = np.abs(x) <= q * scale
        rows.append({"nominal": nominal, "empirical": float(covered.mean()),
                     "abs_error": float(abs(covered.mean() - nominal))})
    return {"distribution": "Student-t(df=17, loc=0, scale=1.7)",
            "samples": n, "seed": seed, "rows": rows,
            "pass": bool(max(r["abs_error"] for r in rows) < 0.005)}


@torch.no_grad()
def evaluate(args, out: Path) -> dict:
    import sys
    sys.path.insert(0, str(ROOT))
    from scripts.diagnose_predictor import _make_model
    from scripts.train_predictor import load_config, set_seeds
    from src.channel.sionna_channel import load_sectioned_config
    from src.training.data import build_noisy_sparse_input, uniform_grouping_mask

    cfg = load_config("configs/current_valid_baseline_100k1_seed_20260819.json")
    checkpoint = ROOT / "runs/current_valid_baseline/paper_sample_alignment_100k_1ep_fig8_20260915/uacp_predictor_step4a.pt"
    common = ROOT / args.common_eval_dir
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cuda_start = {"is_available": bool(torch.cuda.is_available()),
                  "device": str(device),
                  "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    print(json.dumps({"cuda_start": cuda_start}), flush=True)
    model = _make_model(cfg, device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload)
    model.eval()
    counts = {(label, float(nominal)): {"covered": 0, "count": 0}
              for label, _ in REGIMES for nominal in NOMINALS}
    for ri, (label, _) in enumerate(REGIMES):
        cfr = np.load(common / (label.replace(" ", "_").replace("/", "_") + ".npz"))["cfr"]
        if len(cfr) != args.samples_per_regime:
            raise RuntimeError(f"Expected {args.samples_per_regime} samples, got {len(cfr)} for {label}")
        for start in range(0, len(cfr), args.batch_size):
            c = torch.from_numpy(cfr[start:start + args.batch_size]).to(device)
            n = c.shape[0]
            mask = uniform_grouping_mask(n, 1024, 16, device)
            set_seeds(args.eval_seed + ri * 100000 + start)
            # Same noisy sparse observation is used for target alignment and model input.
            x, target, _ = build_noisy_sparse_input(c, mask, 15.0)
            # Same model output and Eq.(5) marginal used by the existing evaluator.
            output = model(x)
            omitted = (1.0 - mask).bool()
            d = 2 * 1024
            df = output.nu_expanded - d + 1.0
            scale_sq = ((output.kappa_expanded + 1.0) / (output.kappa_expanded * df)) * output.psi
            scale = torch.sqrt(scale_sq.clamp_min(1e-12))
            # nu/df is pair-level and broadcast over Real/Imag and frequency.
            # Compute quantiles at that native shape; this changes no formula.
            probs = ((1.0 + NOMINALS) / 2.0).reshape(-1, 1, 1, 1)
            df_pair = output.nu.detach().cpu().numpy() - d + 1.0
            q_all = student_t.ppf(probs, df_pair[None, ...])
            for qi, nominal in enumerate(NOMINALS):
                q_pair = torch.as_tensor(q_all[qi], device=device, dtype=scale.dtype)
                q = torch.cat((q_pair.expand(-1, -1, 1024), q_pair.expand(-1, -1, 1024)), dim=1)
                covered = (target - output.gamma).abs() <= q * scale
                keep = omitted[:, None, :].expand_as(covered)
                item = counts[(label, float(nominal))]
                item["covered"] += int(covered[keep].sum().item())
                item["count"] += int(keep.sum().item())
    cuda_end = {"is_available": bool(torch.cuda.is_available()),
                "device": str(device),
                "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    print(json.dumps({"cuda_end": cuda_end}), flush=True)

    rows = []
    for label, pool in REGIMES:
        for nominal in NOMINALS:
            item = counts[(label, float(nominal))]
            rows.append({"regime": label, "pool": pool, "nominal": float(nominal),
                         "covered": item["covered"], "count": item["count"],
                         "empirical": item["covered"] / item["count"]})
    write_csv(out / "coverage_by_regime.csv", rows)
    pool_rows = []
    for pool, labels in [("ID", [x[0] for x in REGIMES if x[1] == "ID"]),
                         ("OOD", [x[0] for x in REGIMES if x[1].startswith("OOD")]),
                         ("OOD-Near", [REGIMES[2][0]]), ("OOD-Far", [REGIMES[3][0]])]:
        for nominal in NOMINALS:
            z = [r for r in rows if r["regime"] in labels and r["nominal"] == float(nominal)]
            covered, count = sum(r["covered"] for r in z), sum(r["count"] for r in z)
            pool_rows.append({"pool": pool, "nominal": float(nominal), "covered": covered,
                              "count": count, "empirical": covered / count,
                              "abs_error": abs(covered / count - nominal)})
    write_csv(out / "calibration_curve.csv", pool_rows)
    ce = []
    for pool in ["ID", "OOD", "OOD-Near", "OOD-Far"]:
        z = [r for r in pool_rows if r["pool"] == pool]
        ce.append({"pool": pool, "ce_mae_0.1_0.9": float(np.mean([r["abs_error"] for r in z if r["nominal"] <= .9])),
                   "ce_mae_all_grid": float(np.mean([r["abs_error"] for r in z]))})
    write_csv(out / "calibration_error.csv", ce)
    for pool in ["ID", "OOD", "OOD-Near", "OOD-Far"]:
        z = [r for r in pool_rows if r["pool"] == pool]
        plt.plot([r["nominal"] for r in z], [r["empirical"] for r in z], "o-", label=pool)
    plt.plot([0, 1], [0, 1], "k--", label="ideal")
    plt.xlabel("Nominal Coverage"); plt.ylabel("Empirical Coverage")
    plt.grid(alpha=.25); plt.legend(); plt.tight_layout(); plt.savefig(out / "calibration_curve.png", dpi=180); plt.close()
    result = {"checkpoint": str(checkpoint), "common_eval_dir": str(common),
              "samples_per_regime": args.samples_per_regime, "batch_size": args.batch_size,
              "eval_seed": args.eval_seed, "snr_db": 15.0, "ng": 16,
              "cuda_start": cuda_start, "cuda_end": cuda_end, "ce": ce,
              "assumptions": [
                  "IMPLEMENTATION-ASSUMPTION: component-wise marginal Student-t intervals for diagonal-Psi output.",
                  "IMPLEMENTATION-ASSUMPTION: coverage uses omitted subcarrier Real/Imag components only.",
                  "IMPLEMENTATION-ASSUMPTION: ID=(20,80) and OOD=(120,1ms) equal-count pooling.",
                  "IMPLEMENTATION-ASSUMPTION: CE is mean absolute error over the stated nominal grid; paper exact reduction is not disclosed.",
                  "Diagonal-Psi approximation; this is Fig.9-style calibration, not exact full-covariance reproduction."],
              "no_training": True}
    (out / "results.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--common-eval-dir", required=True)
    p.add_argument("--samples-per-regime", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--eval-seed", type=int, default=20262000)
    args = p.parse_args()
    out = ROOT / args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True)
    sanity = synthetic_sanity()
    (out / "synthetic_student_t_sanity.json").write_text(json.dumps(sanity, indent=2))
    result = evaluate(args, out)
    result["synthetic_sanity"] = sanity
    (out / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
