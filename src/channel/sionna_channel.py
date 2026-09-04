"""Reusable Sionna PHY channel helpers for UACP reproduction work."""

from __future__ import annotations

import json
import platform
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def load_sectioned_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    config: dict[str, Any] = {}
    for section in ("paper_specified", "implementation_assumption", "unknown", "dataset"):
        config.update(raw.get(section, {}))
    config["_sections"] = raw
    return config


def paper_settings(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config["_sections"]["paper_specified"])


def assumption_settings(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config["_sections"]["implementation_assumption"])


def unknown_settings(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config["_sections"].get("unknown", {}))


def set_all_seeds(seed: int, device: str | None = None) -> None:
    import sionna.phy as phy
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    phy.config.seed = seed
    if device and device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)


def adjacent_frequency_delta_db(cfr):
    import torch

    adjacent_delta = cfr[:, 1:, :, :] - cfr[:, :-1, :, :]
    return 20.0 * torch.log10(torch.mean(torch.abs(adjacent_delta)) + 1e-12)


def real_imag_channel_view(cfr):
    import torch

    return torch.cat(
        [
            cfr.real.permute(0, 2, 3, 1).reshape(cfr.shape[0], -1, cfr.shape[1]),
            cfr.imag.permute(0, 2, 3, 1).reshape(cfr.shape[0], -1, cfr.shape[1]),
        ],
        dim=1,
    )


def generate_cfr_batch(config: dict[str, Any], delay_spread_ns: float, batch_size: int, seed: int):
    from sionna.phy.channel import GenerateOFDMChannel
    from sionna.phy.channel.tr38901 import TDL
    from sionna.phy.ofdm import ResourceGrid

    device = str(config["device"])
    set_all_seeds(seed, device)

    resource_grid = ResourceGrid(
        num_ofdm_symbols=int(config["num_ofdm_symbols"]),
        fft_size=int(config["fft_size"]),
        subcarrier_spacing=float(config["subcarrier_spacing_hz"]),
        num_tx=int(config["num_tx"]),
        num_streams_per_tx=int(config["num_streams_per_tx"]),
        cyclic_prefix_length=int(config["cyclic_prefix_length"]),
        precision=config["precision"],
        device=device,
    )

    channel_model = TDL(
        model=str(config["tdl_model"]),
        delay_spread=float(delay_spread_ns) * 1e-9,
        carrier_frequency=float(config["carrier_frequency_hz"]),
        min_speed=float(config["min_speed_mps"]),
        max_speed=float(config["max_speed_mps"]),
        num_rx_ant=int(config["num_rx_ant"]),
        num_tx_ant=int(config["num_tx_ant"]),
        precision=config["precision"],
        device=device,
    )

    generator = GenerateOFDMChannel(
        channel_model,
        resource_grid,
        normalize_channel=bool(config["normalize_channel"]),
        precision=config["precision"],
        device=device,
    )

    raw = generator(batch_size=int(batch_size))
    cfr = raw[:, 0, :, 0, :, 0, :].permute(0, 3, 1, 2).contiguous()
    return raw, cfr


def generate_cfr_for_delay_spreads(
    config: dict[str, Any],
    delay_spreads_ns: Iterable[float],
    seed: int,
) -> np.ndarray:
    import torch

    samples = []
    for index, delay_spread_ns in enumerate(delay_spreads_ns):
        _, cfr = generate_cfr_batch(config, float(delay_spread_ns), batch_size=1, seed=seed + index)
        samples.append(cfr.detach().cpu())

    return torch.cat(samples, dim=0).numpy().astype(np.complex64, copy=False)


def smoke_result_for_delay(config: dict[str, Any], delay_spread_ns: float) -> dict[str, Any]:
    import torch

    raw, cfr = generate_cfr_batch(
        config,
        delay_spread_ns=float(delay_spread_ns),
        batch_size=int(config["batch_size"]),
        seed=int(config["seed"]),
    )
    real_imag = real_imag_channel_view(cfr)

    return {
        "delay_spread_ns": float(delay_spread_ns),
        "raw_shape": list(raw.shape),
        "raw_dtype": str(raw.dtype),
        "raw_device": str(raw.device),
        "raw_is_complex": bool(torch.is_complex(raw)),
        "uacp_cfr_shape": list(cfr.shape),
        "uacp_cfr_dtype": str(cfr.dtype),
        "uacp_cfr_device": str(cfr.device),
        "uacp_cfr_is_complex": bool(torch.is_complex(cfr)),
        "real_imag_channels_shape": list(real_imag.shape),
        "mask_aware_predictor_input_channels": int(real_imag.shape[1] + 1),
        "mean_abs_cfr": float(torch.mean(torch.abs(cfr)).item()),
        "adjacent_frequency_delta_db": float(adjacent_frequency_delta_db(cfr).item()),
    }


def environment_report() -> dict[str, Any]:
    import sionna
    import torch

    report = {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_device_count": int(torch.cuda.device_count()),
        "sionna": getattr(sionna, "__version__", "UNKNOWN"),
    }
    if torch.cuda.is_available():
        report["torch_cuda_device_name"] = torch.cuda.get_device_name(0)
    return report
