"""Evaluation metrics for UACP prototype diagnostics."""

from __future__ import annotations

import torch


def nmse_all_db(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = torch.sum((pred - target).square(), dim=(1, 2))
    power = torch.sum(target.square(), dim=(1, 2)).clamp_min(1e-12)
    return 10.0 * torch.log10((mse / power).mean().clamp_min(1e-12))


def nmse_omitted_db(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    omitted = (1.0 - mask)[:, None, :]
    mse = torch.sum((pred - target).square() * omitted, dim=(1, 2))
    power = torch.sum(target.square() * omitted, dim=(1, 2)).clamp_min(1e-12)
    return 10.0 * torch.log10((mse / power).mean().clamp_min(1e-12))
