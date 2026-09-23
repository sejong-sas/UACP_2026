import numpy as np

from scripts.exp2_feature_ood_mahalanobis import diagonal_mahalanobis, rank_corr


def test_diagonal_mahalanobis_and_rank_corr():
    mean = np.zeros(2)
    var = np.ones(2)
    values = np.asarray([[0.0, 0.0], [3.0, 4.0]])
    np.testing.assert_allclose(diagonal_mahalanobis(values, mean, var), [0.0, 5.0])
    assert rank_corr([1, 2, 3], [2, 4, 8]) == 1.0
