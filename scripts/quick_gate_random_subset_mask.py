#!/usr/bin/env python3
"""Quick Gate for the random-subset mask scratch checkpoint."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
FACTORS = [4, 8, 16, 32, 64, 128]


def random_subset_mask(batch_size: int, ng: int, device: torch.device) -> torch.Tensor:
    k = 1024 // ng
    scores = torch.rand((batch_size, 1024), device=device)
    indices = scores.topk(k, dim=1).indices
    mask = torch.zeros((batch_size, 1024), dtype=torch.float32, device=device)
    mask.scatter_(1, indices, 1.0)
    return mask


def quantiles(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "p05": float(np.quantile(a, .05)), "p95": float(np.quantile(a, .95))}


def rank_auc(negative: list[float], positive: list[float]) -> float:
    """AUROC via pairwise ranking; avoids adding a package to the env."""
    a = np.asarray(negative, dtype=float)
    b = np.asarray(positive, dtype=float)
    comparisons = (b[:, None] > a[None, :]).sum() + 0.5 * (b[:, None] == a[None, :]).sum()
    return float(comparisons / (len(a) * len(b)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--samples-per-combination", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=25)
    args = ap.parse_args()
    out = ROOT / args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {out}")
    out.mkdir(parents=True)
    sys.path.insert(0, str(ROOT))
    from scripts.diagnose_predictor import _make_model
    from scripts.train_predictor import load_config, set_seeds
    from src.training.data import build_noisy_sparse_input

    cfg = load_config("runs/current_valid_baseline/paper_sample_alignment_100k_1ep_fig8_20260915/config.json")
    device = torch.device("cuda:0")
    model = _make_model(cfg, device)
    payload = torch.load(ROOT / args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload)
    model.eval()
    rows = []
    per_ng = {}
    data_dir = ROOT / "runs/baseline_reproduction/step2_delay_sweep_repro/generated_data"
    delays = [20, 80, 120]
    for di, delay in enumerate(delays):
        with np.load(data_dir / f"test_delay_{delay}_ns.npz") as z:
            cfr_np = z["cfr"][: args.samples_per_combination]
        for ng in [16, 32, 64, 128]:
            all_v, omitted_v, ale_v, epi_v = [], [], [], []
            nonfinite = 0
            for start in range(0, len(cfr_np), args.batch_size):
                set_seeds(20300000 + di * 100000 + ng * 1000 + start)
                cfr = torch.from_numpy(cfr_np[start:start + args.batch_size]).to(device)
                mask = random_subset_mask(cfr.shape[0], ng, device)
                x, target, _ = build_noisy_sparse_input(cfr, mask, 15.0)
                with torch.inference_mode():
                    output = model(x)
                    error = (output.predicted.float() - target.float()).square()
                    all_num = error.sum((1, 2))
                    all_den = target.float().square().sum((1, 2)).clamp_min(1e-12)
                    omitted = (1.0 - mask)[:, None, :].expand_as(target)
                    om_num = (error * omitted).sum((1, 2))
                    om_den = (target.float().square() * omitted).sum((1, 2)).clamp_min(1e-12)
                    all_v.extend((10 * torch.log10((all_num / all_den).clamp_min(1e-12))).cpu().numpy().tolist())
                    omitted_v.extend((10 * torch.log10((om_num / om_den).clamp_min(1e-12))).cpu().numpy().tolist())
                    ale = output.aleatoric.reshape(output.aleatoric.shape[0], 2, 4, 1024).sum(1).mean(1)
                    epi = output.epistemic.reshape(output.epistemic.shape[0], 2, 4, 1024).sum(1).mean(1)
                    count = (1.0 - mask).sum(1).clamp_min(1.0)
                    ale_v.extend(((ale * (1.0 - mask)).sum(1) / count).cpu().numpy().tolist())
                    epi_v.extend(((epi * (1.0 - mask)).sum(1) / count).cpu().numpy().tolist())
                    nonfinite += int((~torch.isfinite(output.nu_expanded)).sum().cpu())
            row = {"delay_ns": delay, "ng": ng, "samples": len(all_v),
                   "observed_subcarriers": 1024 // ng,
                   "all_nmse_mean_db": quantiles(all_v)["mean"],
                   "all_nmse_median_db": quantiles(all_v)["median"],
                   "omitted_nmse_mean_db": quantiles(omitted_v)["mean"],
                   "omitted_nmse_median_db": quantiles(omitted_v)["median"],
                   "aleatoric": quantiles(ale_v), "epistemic": quantiles(epi_v),
                   "nu_nonfinite_components": nonfinite}
            rows.append(row)
            per_ng.setdefault(ng, {})[delay] = {"aleatoric": ale_v, "epistemic": epi_v}
    auc_rows = []
    for ng, groups in per_ng.items():
        y = [0] * len(groups[20]["aleatoric"]) + [1] * len(groups[80]["aleatoric"])
        score = groups[20]["aleatoric"] + groups[80]["aleatoric"]
        auc_rows.append({"ng": ng, "aleatoric_20_vs_80_auroc": rank_auc(groups[20]["aleatoric"], groups[80]["aleatoric"]),
                         "aleatoric_20_mean": float(np.mean(groups[20]["aleatoric"])),
                         "aleatoric_80_mean": float(np.mean(groups[80]["aleatoric"])),
                         "aleatoric_delta_80_minus_20": float(np.mean(groups[80]["aleatoric"]) - np.mean(groups[20]["aleatoric"]))})
    flat_rows = []
    for row in rows:
        flat = {k: v for k, v in row.items() if k not in ("aleatoric", "epistemic")}
        flat.update({f"aleatoric_{k}": v for k, v in row["aleatoric"].items()})
        flat.update({f"epistemic_{k}": v for k, v in row["epistemic"].items()})
        flat_rows.append(flat)
    fields = list(flat_rows[0])
    with (out / "quick_gate.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(flat_rows)
    (out / "aleatoric_20_vs_80_auroc.csv").write_text(
        "ng,aleatoric_20_vs_80_auroc,aleatoric_20_mean,aleatoric_80_mean,aleatoric_delta_80_minus_20\n" +
        "\n".join(",".join(str(r[k]) for k in auc_rows[0]) for r in auc_rows) + "\n")
    result = {"checkpoint": args.checkpoint, "samples_per_combination": args.samples_per_combination,
              "combinations": len(rows), "delays_ns": delays, "ngs": [16, 32, 64, 128],
              "mask_placement": "random subset without replacement conditioned on Ng",
              "implementation_assumption": "uniform Ng sampling over {4,8,16,32,64,128}; Ng fixed per diagnostic combination",
              "nu_nonfinite_total": sum(r["nu_nonfinite_components"] for r in rows),
              "aleatoric_20_vs_80": auc_rows}
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
