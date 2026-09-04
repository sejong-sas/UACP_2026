"""Mask-aware 1D residual convolutional evidential CFR predictor."""

from __future__ import annotations

import torch
from torch import nn

from src.models.evidential import (
    EvidentialOutput,
    build_banded_cholesky,
    channels_to_pair_vectors,
    constrain_evidential,
    constrain_pair_scalar_evidential,
)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int = 192, kernel_size: int = 5, dropout: float = 0.05) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class UACPEvidentialPredictor(nn.Module):
    def __init__(
        self,
        input_channels: int = 9,
        output_channels: int = 8,
        hidden_channels: int = 192,
        residual_blocks: int = 32,
        kernel_size: int = 5,
        dropout: float = 0.05,
        num_subcarriers: int = 1024,
        evidential_mode: str = "elementwise",
        covariance_rank: int = 0,
        covariance_bandwidth: int = 32,
        covariance_factor_scale: float = 0.05,
    ) -> None:
        super().__init__()
        self.num_subcarriers = num_subcarriers
        self.output_channels = output_channels
        self.evidential_mode = evidential_mode
        self.covariance_rank = covariance_rank
        self.covariance_bandwidth = covariance_bandwidth
        self.covariance_factor_scale = covariance_factor_scale
        self.input_projection = nn.Conv1d(input_channels, hidden_channels, kernel_size=1)
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(hidden_channels, kernel_size, dropout) for _ in range(residual_blocks)]
        )
        if evidential_mode == "elementwise":
            self.evidential_head = nn.Conv1d(hidden_channels, output_channels * 4, kernel_size=1)
        elif evidential_mode == "pair_scalar":
            self.gamma_head = nn.Conv1d(hidden_channels, output_channels, kernel_size=1)
            self.psi_head = nn.Conv1d(hidden_channels, output_channels, kernel_size=1)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.kappa_head = nn.Conv1d(hidden_channels, output_channels // 2, kernel_size=1)
            self.nu_head = nn.Conv1d(hidden_channels, output_channels // 2, kernel_size=1)
        elif evidential_mode == "pair_scalar_lowrank":
            if covariance_rank <= 0:
                raise ValueError("pair_scalar_lowrank requires covariance_rank > 0")
            self.gamma_head = nn.Conv1d(hidden_channels, output_channels, kernel_size=1)
            self.psi_head = nn.Conv1d(hidden_channels, output_channels, kernel_size=1)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.kappa_head = nn.Conv1d(hidden_channels, output_channels // 2, kernel_size=1)
            self.nu_head = nn.Conv1d(hidden_channels, output_channels // 2, kernel_size=1)
            self.lowrank_head = nn.Conv1d(hidden_channels, (output_channels // 2) * 2 * covariance_rank, kernel_size=1)
        elif evidential_mode == "pair_scalar_banded":
            if covariance_bandwidth < 1:
                raise ValueError("pair_scalar_banded requires covariance_bandwidth > 0")
            self.gamma_head = nn.Conv1d(hidden_channels, output_channels, kernel_size=1)
            self.psi_head = nn.Conv1d(hidden_channels, output_channels, kernel_size=1)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.kappa_head = nn.Conv1d(hidden_channels, output_channels // 2, kernel_size=1)
            self.nu_head = nn.Conv1d(hidden_channels, output_channels // 2, kernel_size=1)
            self.banded_head = nn.Conv1d(hidden_channels, (output_channels // 2) * 4 * covariance_bandwidth, kernel_size=1)
        else:
            raise ValueError(f"Unknown evidential_mode: {evidential_mode}")

    def forward(self, x: torch.Tensor) -> EvidentialOutput:
        # 같은 backbone feature에서 CFR 평균과 uncertainty parameter를 분리해 출력한다.
        h = self.input_projection(x)
        for block in self.residual_blocks:
            h = block(h)
        if self.evidential_mode == "elementwise":
            raw = self.evidential_head(h)
            return constrain_evidential(raw, self.output_channels, self.num_subcarriers)

        pooled = self.pool(h)
        output = constrain_pair_scalar_evidential(
            gamma_raw=self.gamma_head(h),
            psi_raw=self.psi_head(h),
            kappa_raw=self.kappa_head(pooled),
            nu_raw=self.nu_head(pooled),
            num_subcarriers=self.num_subcarriers,
        )
        if self.evidential_mode == "pair_scalar_lowrank":
            raw_u = self.lowrank_head(h)
            batch_size, _, num_subcarriers = raw_u.shape
            raw_u = raw_u.reshape(batch_size, self.output_channels // 2, 2, self.covariance_rank, num_subcarriers)
            factor = raw_u.permute(0, 1, 2, 4, 3).reshape(batch_size, self.output_channels // 2, 2 * num_subcarriers, self.covariance_rank)
            factor = factor / (2 * num_subcarriers) ** 0.5
            output.covariance_factor = factor
        elif self.evidential_mode == "pair_scalar_banded":
            raw_band = self.banded_head(h)
            batch_size, _, num_subcarriers = raw_band.shape
            pairs = self.output_channels // 2
            raw_band = raw_band.reshape(batch_size, pairs, 4 * self.covariance_bandwidth, num_subcarriers).permute(0, 1, 3, 2)
            factor, padded_dim = build_banded_cholesky(
                raw_band,
                num_subcarriers=self.num_subcarriers,
                bandwidth=self.covariance_bandwidth,
                diagonal=channels_to_pair_vectors(output.psi),
                factor_scale=self.covariance_factor_scale,
            )
            output.banded_factor = factor
            output.banded_padded_dim = padded_dim
        return output

    def freeze_all_except_last_blocks_and_head(self, last_blocks: int) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        for block in self.residual_blocks[-last_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        head_names = ("evidential_head", "gamma_head", "psi_head", "kappa_head", "nu_head", "lowrank_head", "banded_head")
        for name in head_names:
            if hasattr(self, name):
                for parameter in getattr(self, name).parameters():
                    parameter.requires_grad = True
