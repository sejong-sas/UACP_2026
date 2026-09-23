import numpy as np

from scripts.fig9_paper_style_plot import calibration_mae, combine_pool_rows


def test_ood_pool_is_count_weighted_and_near_far_remain_separate():
    source = [
        {"pool": "ID", "nominal": 0.5, "empirical": 0.51, "count": 100},
        {"pool": "OOD", "nominal": 0.5, "empirical": 0.20, "count": 200},
        {"pool": "OOD-Near", "nominal": 0.5, "empirical": 0.30, "count": 100},
        {"pool": "OOD-Far", "nominal": 0.5, "empirical": 0.10, "count": 100},
    ]
    row = combine_pool_rows(source, 0.5)
    assert row["ID_empirical"] == 0.51
    assert row["OOD_empirical"] == 0.2
    assert row["OOD_Near_empirical"] == 0.3
    assert row["OOD_Far_empirical"] == 0.1


def test_mae_is_mean_absolute_nominal_error():
    assert np.isclose(calibration_mae([0.5, 0.8], [0.6, 0.7]), 0.1)
