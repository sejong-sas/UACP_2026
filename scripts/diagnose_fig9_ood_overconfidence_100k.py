#!/usr/bin/env python3
"""Three read-only Fig.9 OOD-overconfidence diagnostics for 100k×1.

One existing checkpoint is evaluated on the common 10k set. No training or
checkpoint mutation is performed. Metrics use all components for full-feedback
Ng=1 (omitted-only is undefined there), while both protocols are recorded.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import resource
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
REGIMES = [("20 ns", "ID-Easy_20_ns.npz"), ("80 ns", "ID-Hard_80_ns.npz"),
           ("120 ns", "OOD-Near_120_ns.npz"), ("1 ms", "OOD-Far_1_ms.npz")]
NGS = [16, 8, 4, 1]
NOMINALS = np.array([.50, .80, .90, .95, .99])


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


class OnlineStats:
    """Exact moments plus a bounded reservoir for approximate quantiles."""

    def __init__(self, reservoir_size: int = 8192, seed: int = 0) -> None:
        self.count = 0
        self.total = 0.0
        self.reservoir = np.empty(0, dtype=np.float64)
        self.reservoir_size = reservoir_size
        self.rng = np.random.default_rng(seed)

    def update(self, values: np.ndarray | list[float]) -> None:
        a = np.asarray(values, dtype=np.float64).reshape(-1)
        if not a.size:
            return
        self.total += float(a.sum(dtype=np.float64))
        self.count += int(a.size)
        if self.reservoir.size < self.reservoir_size:
            take = min(self.reservoir_size - self.reservoir.size, a.size)
            self.reservoir = np.concatenate((self.reservoir, a[:take]))
            a = a[take:]
        if a.size:
            slots = self.rng.integers(0, self.count, size=a.size)
            keep = slots < self.reservoir_size
            self.reservoir[slots[keep]] = a[keep]

    @property
    def mean(self) -> float:
        return self.total / max(1, self.count)

    def qstats(self) -> dict:
        a = self.reservoir
        return {"mean": self.mean, "median": float(np.quantile(a, .50)),
                "q90": float(np.quantile(a, .90)), "q95": float(np.quantile(a, .95)),
                "q99": float(np.quantile(a, .99))}


def rss_gib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024 ** 2)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--common-eval-dir", required=True)
    p.add_argument("--samples-per-regime", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--eval-seed", type=int, default=20262000)
    args = p.parse_args()
    out = ROOT / args.output_dir
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT))
    from scripts.diagnose_predictor import _make_model
    from scripts.train_predictor import load_config, set_seeds
    from src.training.data import build_noisy_sparse_input, uniform_grouping_mask

    cfg = load_config("configs/current_valid_baseline_100k1_seed_20260819.json")
    checkpoint = ROOT / "runs/current_valid_baseline/paper_sample_alignment_100k_1ep_fig8_20260915/uacp_predictor_step4a.pt"
    common = ROOT / args.common_eval_dir
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cuda_start = {"is_available": bool(torch.cuda.is_available()), "device": str(device),
                  "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    print(json.dumps({"cuda_start": cuda_start}), flush=True)
    model = _make_model(cfg, device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload)
    model.eval()

    # Only counts, moments, and bounded reservoirs are retained.
    cov = {(ng, protocol, regime, float(nom)): [0, 0]
           for ng in NGS for protocol in ("omitted-only", "all-subcarrier", "observed-only")
           for regime, _ in REGIMES for nom in NOMINALS}
    stats = {(regime, key): OnlineStats(seed=ri * 100 + ki) for ri, (regime, _) in enumerate(REGIMES)
             for ki, key in enumerate(("error", "scale", "ratio"))}
    decomp = {(ng, regime): {key: OnlineStats(seed=ng * 1000 + ri * 10 + ki)
                             for ki, key in enumerate(("psi", "kappa", "df", "kappa_factor", "inv_df", "scale_sq", "scale"))}
              for ng in NGS for ri, (regime, _) in enumerate(REGIMES)}

    for ri, (regime, filename) in enumerate(REGIMES):
        loaded = np.load(common / filename)
        cfr = loaded["cfr"]
        if len(cfr) < args.samples_per_regime: raise RuntimeError(f"{regime}: {len(cfr)} samples")
        cfr = cfr[:args.samples_per_regime]
        for ng in NGS:
            for start in range(0, len(cfr), args.batch_size):
                c = torch.from_numpy(cfr[start:start + args.batch_size]).to(device); n = c.shape[0]
                # Match the existing evaluator's deterministic noise policy.
                mask = uniform_grouping_mask(n, 1024, ng, device)
                set_seeds(args.eval_seed + ri * 100000 + start)
                x, target, _ = build_noisy_sparse_input(c, mask, 15.0)
                with torch.inference_mode():
                    output = model(x)
                    omitted = (mask < .5); observed = ~omitted; all_keep = torch.ones_like(omitted, dtype=torch.bool)
                    d = 2048
                    df = output.nu_expanded - d + 1.0
                    scale_sq = ((output.kappa_expanded + 1.0) / (output.kappa_expanded * df)) * output.psi
                    scale = torch.sqrt(scale_sq.clamp_min(1e-12))
                    error = (target - output.gamma).abs()
                    keep_c = omitted[:, None, :].expand_as(error)
                    flat_keep = keep_c.reshape(-1)
                    stats[(regime, "error")].update(error.reshape(-1)[flat_keep].detach().cpu().numpy())
                    stats[(regime, "scale")].update(scale.reshape(-1)[flat_keep].detach().cpu().numpy())
                    stats[(regime, "ratio")].update((error / scale.clamp_min(1e-12)).reshape(-1)[flat_keep].detach().cpu().numpy())
                    pair_df = (output.nu - d + 1.0).detach().cpu().numpy().reshape(-1)
                    pair_kappa = output.kappa.detach().cpu().numpy().reshape(-1)
                    pair_psi = output.psi.detach().cpu().numpy()
                    pair_scale_sq = scale_sq.detach().cpu().numpy()
                    pair_scale = scale.detach().cpu().numpy()
                    for key, vals in (("psi", pair_psi), ("kappa", pair_kappa), ("df", pair_df),
                                      ("kappa_factor", 1.0 + 1.0 / pair_kappa), ("inv_df", 1.0 / pair_df)):
                        decomp[(ng, regime)][key].update(vals)
                    decomp[(ng, regime)]["scale_sq"].update(pair_scale_sq)
                    decomp[(ng, regime)]["scale"].update(pair_scale)
                    for protocol, keep in (("omitted-only", omitted), ("all-subcarrier", all_keep), ("observed-only", observed)):
                        keep_comp = keep[:, None, :].expand_as(error)
                        keep_flat = keep_comp.reshape(-1)
                        for nominal in NOMINALS:
                            q_pair = student_t.ppf((1.0 + nominal) / 2.0, output.nu.detach().cpu().numpy() - d + 1.0)
                            q_pair = torch.as_tensor(q_pair, device=device, dtype=scale.dtype)
                            q = torch.cat((q_pair.expand(-1, -1, 1024), q_pair.expand(-1, -1, 1024)), dim=1)
                            covered = (error <= q * scale)
                            item = cov[(ng, protocol, regime, float(nominal))]
                            item[0] += int(covered[keep_comp].sum().item()); item[1] += int(keep_flat.sum().item())
                del c, n, x, target, mask, output, error, scale, scale_sq, df
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            print(json.dumps({"regime": regime, "ng": ng, "batch": args.batch_size,
                              "max_rss_gib": rss_gib()}), flush=True)
        del cfr, loaded
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    cuda_end = {"is_available": bool(torch.cuda.is_available()), "device": str(device),
                "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    print(json.dumps({"cuda_end": cuda_end}), flush=True)

    protocol_rows = []
    for ng in NGS:
        for protocol in ("omitted-only", "all-subcarrier", "observed-only"):
            for nominal in NOMINALS:
                z = []
                for regime, _ in REGIMES:
                    a, b = cov[(ng, protocol, regime, float(nominal))]; z.append((a, b))
                def pooled(ix):
                    denominator = sum(z[j][1] for j in ix)
                    return float("nan") if denominator == 0 else sum(z[j][0] for j in ix) / denominator
                protocol_rows.append({"ng": ng, "protocol": protocol, "nominal": float(nominal),
                    "id_empirical": pooled([0, 1]), "ood_empirical": pooled([2, 3]),
                    "near_empirical": pooled([2]), "far_empirical": pooled([3]),
                    "id_count": sum(z[j][1] for j in [0, 1]), "ood_count": sum(z[j][1] for j in [2, 3]),
                    "near_count": z[2][1], "far_count": z[3][1]})
    write_csv(out / "protocol_and_density_coverage.csv", protocol_rows)
    ce_rows = []
    for ng in NGS:
        for protocol in ("omitted-only", "all-subcarrier", "observed-only"):
            z = [r for r in protocol_rows if r["ng"] == ng and r["protocol"] == protocol]
            ce_rows.append({"ng": ng, "protocol": protocol,
                "id_ce": float(np.mean([abs(r["id_empirical"] - r["nominal"]) for r in z])),
                "ood_ce": float(np.mean([abs(r["ood_empirical"] - r["nominal"]) for r in z])),
                "near_ce": float(np.mean([abs(r["near_empirical"] - r["nominal"]) for r in z])),
                "far_ce": float(np.mean([abs(r["far_empirical"] - r["nominal"]) for r in z]))})
    write_csv(out / "protocol_and_density_ce.csv", ce_rows)

    err_rows = []
    for regime, _ in REGIMES:
        row = {"regime": regime}
        for prefix in ("error", "scale", "ratio"):
            row.update({f"{prefix}_{k}": v for k, v in stats[(regime, prefix)].qstats().items()})
        err_rows.append(row)
    write_csv(out / "error_vs_scale_summary.csv", err_rows)
    decomp_rows = []
    for ng in NGS:
        for regime, _ in REGIMES:
            row = {"ng": ng, "regime": regime}
            for key, vals in decomp[(ng, regime)].items():
                s = vals.qstats(); row[f"{key}_mean"] = s["mean"]; row[f"{key}_median"] = s["median"]
            decomp_rows.append(row)
    write_csv(out / "scale_decomposition.csv", decomp_rows)

    # Compact plots: A, B (all-subcarrier so Ng=1 is defined), C ratio samples.
    plt.figure(figsize=(7, 5))
    for protocol in ("omitted-only", "all-subcarrier", "observed-only"):
        z = [r for r in protocol_rows if r["ng"] == 16 and r["protocol"] == protocol]
        plt.plot([r["nominal"] for r in z], [r["ood_empirical"] for r in z], "o-", label=protocol)
    plt.plot([0, 1], [0, 1], "k--", label="ideal"); plt.xlabel("Nominal coverage"); plt.ylabel("OOD empirical coverage")
    plt.grid(alpha=.25); plt.legend(); plt.tight_layout(); plt.savefig(out / "protocol_calibration_ood.png", dpi=180); plt.close()
    plt.figure(figsize=(7, 5))
    for ng in NGS:
        z = [r for r in protocol_rows if r["ng"] == ng and r["protocol"] == "all-subcarrier"]
        plt.plot([r["nominal"] for r in z], [r["ood_empirical"] for r in z], "o-", label=f"Ng={ng}")
    plt.plot([0, 1], [0, 1], "k--", label="ideal"); plt.xlabel("Nominal coverage"); plt.ylabel("OOD empirical coverage")
    plt.grid(alpha=.25); plt.legend(); plt.tight_layout(); plt.savefig(out / "density_calibration_ood.png", dpi=180); plt.close()
    plt.figure(figsize=(8, 5))
    for regime, _ in REGIMES:
        plt.hist(stats[(regime, "ratio")].reservoir, bins=80, density=True, histtype="step", label=regime)
    plt.xlim(left=0); plt.xlabel("|target - gamma| / predictive scale"); plt.ylabel("Density")
    plt.grid(alpha=.25); plt.legend(); plt.tight_layout(); plt.savefig(out / "error_scale_ratio_distribution.png", dpi=180); plt.close()

    result = {"checkpoint": str(checkpoint), "common_eval_dir": str(common), "samples_per_regime": args.samples_per_regime,
              "batch_size": args.batch_size, "eval_seed": args.eval_seed, "snr_db": 15.0,
              "quantile_method": "fixed-size reservoir; means are exact online moments",
              "ngs": NGS, "cuda_start": cuda_start, "cuda_end": cuda_end,
              "protocols": ["omitted-only", "all-subcarrier", "observed-only"],
              "assumptions": [
                "IMPLEMENTATION-ASSUMPTION: paper coverage aggregation is undisclosed; protocol A is sensitivity analysis.",
                "IMPLEMENTATION-ASSUMPTION: ID=(20,80) and OOD=(120,1ms) equal-count pooling.",
                "IMPLEMENTATION-ASSUMPTION: coverage is component-wise marginal Student-t and uses all components for Ng=1 because omitted-only is empty.",
                "Full-feedback Ng=1 is a diagnostic outside the sparse training-mask distribution.",
                "Diagonal-Psi approximation; no training or covariance change."], "no_training": True}
    (out / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__": main()
