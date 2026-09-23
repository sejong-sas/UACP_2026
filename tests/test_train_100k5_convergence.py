from copy import deepcopy

import pytest

from scripts.train_100k5_convergence import assert_epoch_only_change, replay_last_probe_rng


def sample_config():
    return {
        "data": {"train_path": "same.npz", "validation_path": "val.npz"},
        "paper_specified": {"learning_rate": 1e-4, "lambda_reg": 1e-3},
        "implementation_assumption": {"epochs": 1, "pilot_epochs": 1, "experiment_name": "baseline", "batch_size": 8, "seed": 7},
    }


def test_only_epoch_count_and_run_metadata_may_change():
    base = sample_config()
    candidate = deepcopy(base)
    candidate["implementation_assumption"].update(epochs=5, pilot_epochs=5, experiment_name="five-epoch")
    assert_epoch_only_change(base, candidate)

    candidate["paper_specified"]["lambda_reg"] = 0.01
    with pytest.raises(ValueError):
        assert_epoch_only_change(base, candidate)


def test_rng_replay_matches_last_probe_cfr_generation_call():
    calls = []

    def generate(config, delays, seed):
        calls.append((config, list(delays), seed))

    replay_last_probe_rng(
        samples=3,
        delay_spread_ns=1_000_000.0,
        baseline_seed=20260819,
        dataset_config={"sentinel": True},
        generate=generate,
    )

    assert calls == [({"sentinel": True}, [1_000_000.0], 20260819 + 94000 + 2)]
