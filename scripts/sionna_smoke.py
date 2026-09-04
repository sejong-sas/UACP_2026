#!/usr/bin/env python3
"""Minimal Sionna PHY smoke test for the UACP channel setup."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.channel.sionna_channel import (
    assumption_settings,
    environment_report,
    load_sectioned_config,
    paper_settings,
    smoke_result_for_delay,
    unknown_settings,
)


def load_config(path: str | Path) -> dict[str, Any]:
    return load_sectioned_config(path)


def run(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    results = {
        "environment": environment_report(),
        "paper_specified": paper_settings(config),
        "implementation_assumption": assumption_settings(config),
        "unknown": unknown_settings(config),
        "cfr_results": [],
    }

    for delay_spread_ns in config["delay_spreads_ns"]:
        results["cfr_results"].append(smoke_result_for_delay(config, delay_spread_ns))

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sionna_smoke.json")
    args = parser.parse_args()

    print(json.dumps(run(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
