#!/usr/bin/env python3
"""Generate a small CFR prototype dataset for UACP reproduction."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.channel.sionna_channel import (
    assumption_settings,
    environment_report,
    generate_cfr_for_delay_spreads,
    load_sectioned_config,
    paper_settings,
    unknown_settings,
)


def load_dataset_config(path: str | Path) -> dict[str, Any]:
    return load_sectioned_config(path)


def save_split_npz(
    path: str | Path,
    cfr: np.ndarray,
    delay_spread_ns: np.ndarray,
    labels: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        cfr=cfr.astype(np.complex64, copy=False),
        delay_spread_ns=delay_spread_ns.astype(np.float32, copy=False),
        regime_label=labels.astype("U32", copy=False),
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )


def _uniform_delay_spreads(count: int, low_high: list[float], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(float(low_high[0]), float(low_high[1]), size=count).astype(np.float32)


def _metadata(config: dict[str, Any], split: str) -> dict[str, Any]:
    return {
        "split": split,
        "paper_specified": paper_settings(config),
        "implementation_assumption": assumption_settings(config),
        "unknown": unknown_settings(config),
        "environment": environment_report(),
        "layout": "[sample, subcarrier, rx_ant, tx_ant]",
        "mask_policy": "not stored; generate online in DataLoader/training",
        "snr_policy": "recorded from paper, not applied to clean CFR generation",
    }


def _generate_split(
    config: dict[str, Any],
    split: str,
    delay_spreads_ns: np.ndarray,
    labels: np.ndarray,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    start = time.perf_counter()
    cfr = generate_cfr_for_delay_spreads(config, delay_spreads_ns, seed=seed)
    elapsed = time.perf_counter() - start
    path = output_dir / f"{split}.npz"
    metadata = _metadata(config, split)
    metadata["generation_seconds"] = elapsed
    save_split_npz(path, cfr, delay_spreads_ns, labels, metadata)
    return {
        "split": split,
        "path": str(path),
        "samples": int(cfr.shape[0]),
        "shape": list(cfr.shape),
        "dtype": str(cfr.dtype),
        "generation_seconds": elapsed,
        "bytes": path.stat().st_size,
    }


def generate_dataset(config_path: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    config = load_dataset_config(config_path)
    dataset = config["_sections"]["dataset"]
    out_dir = Path(output_dir or dataset["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config["seed"])
    train_count = int(dataset["train_samples"])
    val_count = int(dataset["validation_samples"])
    test_count = int(dataset["test_samples_per_regime"])
    train_range = config["training_delay_spread_ns"]

    summary = {
        "config": str(config_path),
        "output_dir": str(out_dir),
        "splits": [],
    }

    train_delays = _uniform_delay_spreads(train_count, train_range, seed + 101)
    train_labels = np.array(["train"] * train_count)
    summary["splits"].append(_generate_split(config, "train", train_delays, train_labels, seed + 1001, out_dir))

    val_delays = _uniform_delay_spreads(val_count, train_range, seed + 202)
    val_labels = np.array(["validation"] * val_count)
    summary["splits"].append(_generate_split(config, "validation", val_delays, val_labels, seed + 2001, out_dir))

    for index, (label, delay_ns) in enumerate(config["test_regimes"].items()):
        delays = np.full(test_count, float(delay_ns), dtype=np.float32)
        labels = np.array([label] * test_count)
        split = f"test_{label.lower().replace('-', '_')}"
        summary["splits"].append(
            _generate_split(config, split, delays, labels, seed + 3001 + index * 1000, out_dir)
        )

    if bool(dataset.get("include_ood_far", False)):
        delay_ns = float(dataset["ood_far_delay_spread_ns"])
        delays = np.full(test_count, delay_ns, dtype=np.float32)
        labels = np.array(["OOD-Far"] * test_count)
        summary["splits"].append(_generate_split(config, "test_ood_far", delays, labels, seed + 9001, out_dir))

    summary_path = out_dir / "generation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dataset_prototype.json")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    print(json.dumps(generate_dataset(args.config, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
