import numpy as np
import torch

from scripts.fig8_fig9_protocol_audit import (
    auc_from_scores,
    coverage_count,
    uncertainty_protocol_scores,
)


def test_uncertainty_reductions_follow_observed_mask():
    # Channels encode pair, Re/Imag and subcarrier in Eq.(12)'s [Re/Im,pair,K] layout.
    epi = torch.zeros((1, 8, 4))
    epi[:, 0, :] = 4.0  # Re pair 0
    epi[:, 4, :] = 8.0  # Im pair 0
    mask = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    got = uncertainty_protocol_scores(epi, mask)
    assert got["all_subcarrier"][0] == 3.0
    assert got["observed_only"][0] == 3.0
    assert got["omitted_only"][0] == 3.0


def test_coverage_uses_component_count_and_auc_rank_order():
    covered = torch.zeros((1, 8, 2), dtype=torch.bool)
    covered[0, 0, 0] = True
    covered[0, 1, 0] = True
    covered[0, 0, 1] = True
    submask = torch.tensor([[1.0, 0.0]])
    counts = coverage_count(covered, submask)
    assert counts["all_subcarrier"] == (3, 16)
    assert counts["observed_only"] == (2, 8)
    assert counts["omitted_only"] == (1, 8)
    assert auc_from_scores([0.1, 0.2], [0.8, 0.9]) == 1.0
    assert auc_from_scores([0.1, 0.2], [0.1, 0.2]) == 0.5
