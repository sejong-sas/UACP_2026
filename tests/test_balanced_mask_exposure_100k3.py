from pathlib import Path

import torch

from scripts.train_balanced_mask_exposure_100k3 import build_balanced_schedule, make_balanced_mask, schedule_counts


def test_schedule_has_exact_factor_counts_and_balanced_offsets():
    schedule = build_balanced_schedule(100000, [4, 8, 16, 32], 20260919)
    counts = schedule_counts(schedule, [4, 8, 16, 32])
    assert counts["ng_counts"] == {"4": 25000, "8": 25000, "16": 25000, "32": 25000}
    for offset_counts in counts["offset_counts"].values():
        values = list(offset_counts.values())
        assert max(values) - min(values) <= 1


def test_schedule_is_deterministic_and_mask_shape_is_preserved():
    first = build_balanced_schedule(32, [4, 8, 16, 32], 7)
    second = build_balanced_schedule(32, [4, 8, 16, 32], 7)
    assert first == second
    mask = make_balanced_mask(first[:8], 1024, torch.device("cpu"))
    assert mask.shape == (8, 1024)
    assert torch.all((mask.sum(dim=1) > 0))


def test_schedule_rejects_non_divisible_sample_count():
    try:
        build_balanced_schedule(10, [4, 8, 16, 32], 1)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
