"""Uncertainty aggregation utilities for UACP diagnostics."""

from __future__ import annotations

import torch


def paper_subcarrier_uncertainty_map(uncertainty: torch.Tensor) -> torch.Tensor:
    # pair별 Real/Imag 두 성분의 trace를 만든 뒤 네 antenna pair를 평균한다.
    if uncertainty.ndim != 3 or uncertainty.shape[1] != 8:
        raise ValueError("Expected uncertainty shape [batch, 8, subcarrier] for 2x2 real/imag CFR.")
    batch_size, _, num_subcarriers = uncertainty.shape
    pair_real_imag = uncertainty.reshape(batch_size, 2, 4, num_subcarriers).permute(0, 2, 1, 3)
    return pair_real_imag.sum(dim=2).mean(dim=1)


def paper_omitted_uncertainty_score(uncertainty: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # 최종 score는 실제로 예측해야 했던 omitted subcarrier만 평균한다.
    score_map = paper_subcarrier_uncertainty_map(uncertainty)
    omitted = 1.0 - mask
    omitted_count = omitted.sum().clamp_min(1.0)
    return (score_map * omitted).sum() / omitted_count
