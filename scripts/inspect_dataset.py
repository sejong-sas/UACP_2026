#!/usr/bin/env python3
"""Inspect UACP prototype CFR datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _adjacent_frequency_delta_db(cfr: np.ndarray) -> float:
    delta = cfr[:, 1:, :, :] - cfr[:, :-1, :, :]
    return float(20.0 * np.log10(np.mean(np.abs(delta)) + 1e-12))


def inspect_dataset(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        cfr = data["cfr"]
        delays = data["delay_spread_ns"]
        labels = data["regime_label"]
        metadata = json.loads(str(data["metadata_json"]))

        magnitude = np.abs(cfr)
        finite_real = np.isfinite(cfr.real)
        finite_imag = np.isfinite(cfr.imag)
        unique_labels, counts = np.unique(labels, return_counts=True)

        return {
            "path": str(path),
            "bytes": path.stat().st_size,
            "shape": list(cfr.shape),
            "dtype": str(cfr.dtype),
            "stored_on": "cpu_npz",
            "has_nan": bool(np.isnan(cfr.real).any() or np.isnan(cfr.imag).any()),
            "has_inf": bool((~finite_real).any() or (~finite_imag).any()),
            "magnitude": {
                "mean": float(np.mean(magnitude)),
                "std": float(np.std(magnitude)),
                "min": float(np.min(magnitude)),
                "max": float(np.max(magnitude)),
            },
            "delay_spread_ns": {
                "mean": float(np.mean(delays)),
                "std": float(np.std(delays)),
                "min": float(np.min(delays)),
                "max": float(np.max(delays)),
            },
            "labels": {str(label): int(count) for label, count in zip(unique_labels, counts)},
            "adjacent_frequency_delta_db": _adjacent_frequency_delta_db(cfr),
            "metadata": metadata,
        }


def inspect_path(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob("*.npz"))
    else:
        files = [path]
    return {"datasets": [inspect_dataset(file) for file in files]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="data/prototype")
    args = parser.parse_args()

    print(json.dumps(inspect_path(args.path), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
