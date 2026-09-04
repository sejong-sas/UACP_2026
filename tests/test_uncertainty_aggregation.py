import unittest

import torch


class UncertaintyAggregationTests(unittest.TestCase):
    def test_paper_subcarrier_map_sums_real_imag_trace_per_pair_then_averages_pairs(self):
        from src.training.uncertainty import paper_subcarrier_uncertainty_map

        uncertainty = torch.zeros(1, 8, 3)
        uncertainty[:, 0, :] = 1.0
        uncertainty[:, 1, :] = 2.0
        uncertainty[:, 2, :] = 3.0
        uncertainty[:, 3, :] = 4.0
        uncertainty[:, 4, :] = 10.0
        uncertainty[:, 5, :] = 20.0
        uncertainty[:, 6, :] = 30.0
        uncertainty[:, 7, :] = 40.0

        score = paper_subcarrier_uncertainty_map(uncertainty)

        self.assertEqual(list(score.shape), [1, 3])
        self.assertTrue(torch.allclose(score, torch.full((1, 3), 27.5)))

    def test_paper_omitted_score_averages_only_unreported_subcarriers(self):
        from src.training.uncertainty import paper_omitted_uncertainty_score

        uncertainty = torch.arange(1, 33, dtype=torch.float32).reshape(1, 8, 4)
        mask = torch.tensor([[1.0, 0.0, 1.0, 0.0]])

        score = paper_omitted_uncertainty_score(uncertainty, mask)
        expected_map = uncertainty.reshape(1, 2, 4, 4).permute(0, 2, 1, 3).sum(dim=2).mean(dim=1)

        self.assertTrue(torch.allclose(score, expected_map[:, [1, 3]].mean()))


if __name__ == "__main__":
    unittest.main()
