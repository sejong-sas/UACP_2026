import numpy as np

from scripts.diagnose_fig9_ood_overconfidence_100k import OnlineStats


def test_online_stats_keeps_fixed_reservoir_and_matches_moments():
    stats = OnlineStats(reservoir_size=8, seed=7)
    values = np.arange(100, dtype=np.float64)
    stats.update(values)

    assert stats.count == 100
    assert stats.mean == 49.5
    assert stats.reservoir.size <= 8
    assert np.isfinite(stats.qstats()["median"])
