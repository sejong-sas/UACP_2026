"""Dataset and mask utilities for UACP prototype training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def cfr_to_real_imag(cfr: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [
            cfr.real.permute(0, 2, 3, 1).reshape(cfr.shape[0], -1, cfr.shape[1]),
            cfr.imag.permute(0, 2, 3, 1).reshape(cfr.shape[0], -1, cfr.shape[1]),
        ],
        dim=1,
    )


def build_sparse_input(cfr: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # 관측된 위치만 남기고 나머지는 0으로 채운 뒤, 모델이 위치를 알도록 mask를 붙인다.
    target = cfr_to_real_imag(cfr)
    sparse = target * mask[:, None, :]
    x = torch.cat([sparse, mask[:, None, :].to(target.dtype)], dim=1)
    return x, target


def add_complex_awgn(
    cfr: torch.Tensor,
    mask: torch.Tensor,
    snr_db: float | None,
) -> tuple[torch.Tensor, dict[str, float | None]]:
    """Add sample-wise complex AWGN only at reported subcarriers."""
    # 논문에서 정확한 noise 위치가 공개되지 않았으므로, 현재 baseline의 가정인
    # reported CFR에만 sample별 복소 AWGN을 적용한다.
    if snr_db is None:
        return cfr, {"requested_snr_db": None, "measured_snr_db_mean": None, "measured_snr_db_std": None}
    if not cfr.is_complex():
        raise ValueError("Complex AWGN requires a complex CFR tensor.")
    mask_complex = mask[:, :, None, None].to(dtype=cfr.real.dtype)
    observation_count = mask_complex.expand_as(cfr.real).sum(dim=(1, 2, 3)).clamp_min(1.0)
    signal_power = (cfr.abs().square() * mask_complex).sum(dim=(1, 2, 3)) / observation_count
    noise_power = signal_power / (10.0 ** (float(snr_db) / 10.0))
    real_noise = torch.randn_like(cfr.real)
    imag_noise = torch.randn_like(cfr.real)
    noise = torch.complex(real_noise, imag_noise) * torch.sqrt(noise_power[:, None, None, None] / 2.0)
    observed_noise = noise * mask_complex
    noisy_cfr = cfr + observed_noise
    measured_noise_power = (observed_noise.abs().square() * mask_complex).sum(dim=(1, 2, 3)) / observation_count
    measured_snr = 10.0 * torch.log10(signal_power.clamp_min(1e-12) / measured_noise_power.clamp_min(1e-12))
    return noisy_cfr, {
        "requested_snr_db": float(snr_db),
        "measured_snr_db_mean": float(measured_snr.mean().detach().cpu()),
        "measured_snr_db_std": float(measured_snr.std(unbiased=False).detach().cpu()),
    }


def build_noisy_sparse_input(
    cfr: torch.Tensor,
    mask: torch.Tensor,
    snr_db: float | None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | None]]:
    # 입력에는 noisy sparse CFR을 사용하지만, 정답은 항상 clean full CFR로 유지한다.
    observed_cfr, snr_stats = add_complex_awgn(cfr, mask, snr_db)
    x, target = build_sparse_input(observed_cfr, mask)
    clean_target = cfr_to_real_imag(cfr)
    return x, clean_target, snr_stats


def build_pilot_ls_observation(
    cfr: torch.Tensor,
    mask: torch.Tensor,
    snr_db: float | None,
) -> tuple[torch.Tensor, dict[str, float | None]]:
    """Generate an explicit two-slot orthogonal-pilot LS CFR observation.

    The pilot matrix is X=I_2, so Y=H X+N and the LS estimate is H_hat=Y.
    Noise is applied to reported received-pilot entries only; unreported
    entries remain untouched and are later zeroed by sparse encoding.
    """
    if not cfr.is_complex() or cfr.ndim != 4 or tuple(cfr.shape[-2:]) != (2, 2):
        raise ValueError("Pilot LS observation requires CFR shape [B,K,2,2] with complex dtype.")
    if mask.ndim != 2 or mask.shape[:2] != cfr.shape[:2]:
        raise ValueError("Mask must have shape [B,K] matching CFR batch and subcarriers.")
    if snr_db is None:
        return cfr, {
            "requested_snr_db": None,
            "measured_snr_db_mean": None,
            "measured_snr_db_std": None,
        }

    mask_complex = mask[:, :, None, None].to(dtype=cfr.real.dtype)
    observation_count = mask_complex.expand_as(cfr.real).sum(dim=(1, 2, 3)).clamp_min(1.0)
    signal_power = (cfr.abs().square() * mask_complex).sum(dim=(1, 2, 3)) / observation_count
    noise_power = signal_power / (10.0 ** (float(snr_db) / 10.0))
    real_noise = torch.randn_like(cfr.real)
    imag_noise = torch.randn_like(cfr.real)
    received_noise = torch.complex(real_noise, imag_noise) * torch.sqrt(noise_power[:, None, None, None] / 2.0)
    received_noise = received_noise * mask_complex
    h_hat = cfr + received_noise
    measured_noise_power = (received_noise.abs().square() * mask_complex).sum(dim=(1, 2, 3)) / observation_count
    measured_snr = 10.0 * torch.log10(signal_power.clamp_min(1e-12) / measured_noise_power.clamp_min(1e-12))
    return h_hat, {
        "requested_snr_db": float(snr_db),
        "measured_snr_db_mean": float(measured_snr.mean().detach().cpu()),
        "measured_snr_db_std": float(measured_snr.std(unbiased=False).detach().cpu()),
        "pilot_sequence": "X=I2",
        "pilot_amplitude": 1.0,
        "estimator": "least_squares",
    }


def build_pilot_sparse_input(
    cfr: torch.Tensor,
    mask: torch.Tensor,
    snr_db: float | None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | None]]:
    """Build predictor input from a pilot-LS estimate and clean target."""
    estimated_cfr, stats = build_pilot_ls_observation(cfr, mask, snr_db)
    x, _ = build_sparse_input(estimated_cfr, mask)
    return x, cfr_to_real_imag(cfr), stats


def build_requested_hadamard_ls_observation(
    cfr: torch.Tensor,
    mask: torch.Tensor,
    snr_db: float | None,
) -> tuple[torch.Tensor, dict[str, float | None]]:
    """Estimate only requested subcarriers using a normalized Hadamard pilot.

    The request mask is applied before pilot generation.  For each requested
    k, Y[k] = H[k]X + N[k] with X=(1/sqrt(2))*[[1,1],[1,-1]], and the general
    LS estimate H_hat=Y X^H (X X^H)^-1 is evaluated (X X^H=I here).
    """
    if not cfr.is_complex() or cfr.ndim != 4 or tuple(cfr.shape[-2:]) != (2, 2):
        raise ValueError("Requested pilot LS requires complex CFR shape [B,K,2,2].")
    if mask.ndim != 2 or mask.shape[:2] != cfr.shape[:2]:
        raise ValueError("Mask must have shape [B,K] matching CFR.")
    pilot_real = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=cfr.real.dtype, device=cfr.device) / torch.sqrt(torch.tensor(2.0, dtype=cfr.real.dtype, device=cfr.device))
    pilot = torch.complex(pilot_real, torch.zeros_like(pilot_real))
    pilot_h = pilot.conj().transpose(0, 1)
    gram_inv = torch.linalg.inv(pilot @ pilot_h)
    request = mask[:, :, None, None].to(cfr.real.dtype)
    signal = torch.matmul(cfr, pilot)
    if snr_db is None:
        return cfr, {"requested_snr_db": None, "measured_snr_db_mean": None, "measured_snr_db_std": None, "pilot_sequence": "normalized_hadamard_2x2", "estimator": "least_squares"}
    count = request.expand_as(cfr.real).sum(dim=(1, 2, 3)).clamp_min(1.0)
    signal_power = (signal.abs().square() * request).sum(dim=(1, 2, 3)) / count
    noise_power = signal_power / (10.0 ** (float(snr_db) / 10.0))
    noise = torch.complex(torch.randn_like(signal.real), torch.randn_like(signal.real)) * torch.sqrt(noise_power[:, None, None, None] / 2.0)
    received = signal + noise * request
    estimate = torch.matmul(torch.matmul(received, pilot_h), gram_inv)
    h_hat = cfr.clone()
    h_hat = torch.where(request.bool(), estimate, h_hat)
    measured_noise = noise * request
    measured_power = (measured_noise.abs().square() * request).sum(dim=(1, 2, 3)) / count
    measured_snr = 10.0 * torch.log10(signal_power.clamp_min(1e-12) / measured_power.clamp_min(1e-12))
    return h_hat, {
        "requested_snr_db": float(snr_db),
        "measured_snr_db_mean": float(measured_snr.mean().detach().cpu()),
        "measured_snr_db_std": float(measured_snr.std(unbiased=False).detach().cpu()),
        "pilot_sequence": "normalized_hadamard_2x2",
        "pilot_amplitude": float(1.0 / 2.0**0.5),
        "estimator": "least_squares",
    }


def build_requested_pilot_sparse_input(
    cfr: torch.Tensor,
    mask: torch.Tensor,
    snr_db: float | None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | None]]:
    """Encode requested-subcarrier pilot-LS estimates as the 9-channel input."""
    estimated_cfr, stats = build_requested_hadamard_ls_observation(cfr, mask, snr_db)
    x, _ = build_sparse_input(estimated_cfr, mask)
    return x, cfr_to_real_imag(cfr), stats


def random_grouping_mask(batch_size: int, num_subcarriers: int, grouping_factors: list[int], device: torch.device) -> torch.Tensor:
    mask = torch.zeros((batch_size, num_subcarriers), dtype=torch.float32, device=device)
    factors = torch.tensor(grouping_factors, device=device)
    picks = factors[torch.randint(0, len(grouping_factors), (batch_size,), device=device)]
    for row, factor in enumerate(picks.tolist()):
        offset = int(torch.randint(0, factor, (1,), device=device).item())
        mask[row, offset::factor] = 1.0
    return mask


def uniform_grouping_mask(batch_size: int, num_subcarriers: int, grouping_factor: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros((batch_size, num_subcarriers), dtype=torch.float32, device=device)
    mask[:, :: int(grouping_factor)] = 1.0
    return mask


def linear_interpolate_real_imag(sparse: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    device = sparse.device
    dtype = sparse.dtype
    sparse_np = sparse.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()
    x = np.arange(sparse.shape[-1], dtype=np.float32)
    out = np.empty_like(sparse_np)

    for batch_idx in range(sparse_np.shape[0]):
        observed = mask_np[batch_idx] > 0.5
        observed_x = x[observed]
        if observed_x.size == 0:
            out[batch_idx] = 0.0
            continue
        for channel_idx in range(sparse_np.shape[1]):
            observed_y = sparse_np[batch_idx, channel_idx, observed]
            out[batch_idx, channel_idx] = np.interp(x, observed_x, observed_y)

    return torch.from_numpy(out).to(device=device, dtype=dtype)


class CFRNPZDataset(Dataset):
    def __init__(self, path: str | Path) -> None:
        with np.load(path, allow_pickle=False) as data:
            self.cfr = torch.from_numpy(data["cfr"])
            self.delay_spread_ns = torch.from_numpy(data["delay_spread_ns"])
            self.regime_label = data["regime_label"].astype(str)
            self.metadata: dict[str, Any] = json.loads(str(data["metadata_json"]))

    def __len__(self) -> int:
        return int(self.cfr.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "cfr": self.cfr[index],
            "delay_spread_ns": self.delay_spread_ns[index],
            "regime_label": self.regime_label[index],
        }
