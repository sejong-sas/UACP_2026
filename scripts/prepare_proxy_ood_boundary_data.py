#!/usr/bin/env python3
"""Generate the fixed 160--500 ns proxy-OOD boundary-rehearsal split."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.generate_dataset import save_split_npz
from src.channel.sionna_channel import generate_cfr_for_delay_spreads, load_sectioned_config, paper_settings, assumption_settings, unknown_settings, environment_report


def sample_hashes(path):
    data = np.load(path, allow_pickle=True)["cfr"]
    return {hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest() for x in data}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/adaptation_120ns_protocol_20260921.json")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260922)
    args = p.parse_args()
    out = ROOT / args.output_dir
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    out.mkdir(parents=True)
    config = load_sectioned_config(ROOT / args.config)
    rng = np.random.default_rng(args.seed)
    delays = rng.uniform(160.0, 500.0, size=args.samples).astype(np.float32)
    cfr = generate_cfr_for_delay_spreads(config, delays, seed=args.seed + 1001)
    metadata = {
        "split": "proxy_ood_boundary_rehearsal",
        "role": "boundary-only uncertainty rehearsal; no reconstruction target loss",
        "delay_distribution_ns": "Uniform(160,500)",
        "samples": args.samples,
        "seed": args.seed,
        "config": args.config,
        "paper_specified": paper_settings(config),
        "implementation_assumption": assumption_settings(config),
        "unknown": unknown_settings(config),
        "environment": environment_report(),
        "mask_policy": "generated online; same sparse observation policy as adaptation",
        "snr_policy": "15 dB complex AWGN applied online to reported sparse CFR; clean CFR retained only as provenance, not a proxy reconstruction target",
        "blind_1ms_policy": "1 ms is not generated or read by this script",
    }
    path = out / "proxy_ood.npz"
    save_split_npz(path, cfr, delays, np.array(["proxy-ood"] * args.samples), metadata)
    existing = {
        "adapt_train": ROOT / "runs/current_valid_baseline/overnight_20260920_adaptation_data/train.npz",
        "adapt_validation": ROOT / "runs/current_valid_baseline/overnight_20260920_adaptation_data/validation.npz",
        "test_20ns": ROOT / "runs/current_valid_baseline/overnight_20260920_adaptation_data/test_id_easy.npz",
        "test_80ns": ROOT / "runs/current_valid_baseline/overnight_20260920_adaptation_data/test_id_hard.npz",
        "test_120ns": ROOT / "runs/current_valid_baseline/overnight_20260920_adaptation_data/test_ood_near.npz",
        "test_1ms": ROOT / "runs/current_valid_baseline/overnight_20260920_adaptation_data/test_ood_far.npz",
    }
    proxy_hashes = sample_hashes(path)
    overlaps = {name: len(proxy_hashes & sample_hashes(file)) for name, file in existing.items()}
    manifest = {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "sample_hash_count": len(proxy_hashes), "delay_min_ns": float(delays.min()), "delay_max_ns": float(delays.max()), "overlap_counts": overlaps, "one_ms_used": False, "gpu_environment": environment_report()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__": main()
