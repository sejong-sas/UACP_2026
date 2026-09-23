import math

import numpy as np

from scripts.exp1_ng_conditioned_epi_threshold import (
    auc_from_scores,
    summarize_scores,
    threshold_metrics,
)


def test_summary_and_threshold_metrics_are_scalar_and_correct():
    values = [1.0, 2.0, 3.0, 4.0]
    summary = summarize_scores(values)
    assert summary["count"] == 4
    assert summary["mean"] == 2.5
    assert summary["median"] == 2.5
    assert summary["q95"] == np.quantile(values, 0.95)

    metrics = threshold_metrics([0.1, 0.2, 0.3], 0.2)
    assert metrics == {"rate": 1.0 / 3.0, "count": 1, "total": 3}


def test_auc_and_empty_cases():
    assert auc_from_scores([0.1, 0.2], [0.8, 0.9]) == 1.0
    assert math.isnan(auc_from_scores([], [0.1]))
    assert math.isnan(auc_from_scores([0.1], []))
