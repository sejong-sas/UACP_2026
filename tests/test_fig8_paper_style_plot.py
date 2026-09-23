import numpy as np

from scripts.fig8_paper_style_plot import histogram_density, regime_summary


def test_density_uses_all_samples_even_when_viewport_clips():
    samples = np.asarray([-3.0, -1.0, 0.0, 1.0, 3.0])
    edges = np.arange(-3.25, 3.51, 0.5)
    density, _ = histogram_density(samples, edges)
    assert np.isclose(np.sum(density * np.diff(edges)), 1.0)

    stats = regime_summary(samples, -2.0, 2.0)
    assert stats["below_range_count"] == 1
    assert stats["above_range_count"] == 1
    assert stats["inside_range_fraction"] == 3 / 5


def test_sample_level_db_statistics_and_minmax():
    values = np.asarray([0.001, 0.002, 0.004])
    stats = regime_summary(10 * np.log10(values), -37.5, -22.5, values)
    assert stats["count"] == 3
    assert np.isclose(stats["mean_db"], np.mean(10 * np.log10(values)))
    assert stats["min_db"] == np.min(10 * np.log10(values))
    assert stats["max_db"] == np.max(10 * np.log10(values))
