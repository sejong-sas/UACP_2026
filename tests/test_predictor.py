import unittest

import torch


class PredictorTests(unittest.TestCase):
    def test_observation_noise_disabled_matches_clean_input(self):
        from src.training.data import build_sparse_input, build_noisy_sparse_input

        cfr = torch.ones((2, 16, 2, 2), dtype=torch.complex64)
        mask = torch.zeros((2, 16), dtype=torch.float32)
        mask[:, ::4] = 1.0
        clean_x, clean_target = build_sparse_input(cfr, mask)
        noisy_x, noisy_target, stats = build_noisy_sparse_input(cfr, mask, None)

        self.assertTrue(torch.equal(clean_x, noisy_x))
        self.assertTrue(torch.equal(clean_target, noisy_target))
        self.assertIsNone(stats["requested_snr_db"])

    def test_observation_noise_preserves_target_and_unreported_input(self):
        from src.training.data import build_noisy_sparse_input

        torch.manual_seed(7)
        cfr = torch.ones((2, 16, 2, 2), dtype=torch.complex64)
        mask = torch.zeros((2, 16), dtype=torch.float32)
        mask[:, ::4] = 1.0
        x, target, stats = build_noisy_sparse_input(cfr, mask, 15.0)

        self.assertTrue(torch.equal(target, torch.cat([torch.ones_like(target[:, :4]), torch.zeros_like(target[:, :4])], dim=1)))
        self.assertTrue(torch.all(x[:, :8, 1::4] == 0))
        self.assertAlmostEqual(stats["requested_snr_db"], 15.0)
        self.assertTrue(torch.isfinite(x).all())

    def test_observation_noise_same_seed_and_measured_snr(self):
        from src.training.data import build_noisy_sparse_input

        cfr = torch.ones((32, 64, 2, 2), dtype=torch.complex64)
        mask = torch.zeros((32, 64), dtype=torch.float32)
        mask[:, ::4] = 1.0
        torch.manual_seed(123)
        x1, _, stats1 = build_noisy_sparse_input(cfr, mask, 15.0)
        torch.manual_seed(123)
        x2, _, stats2 = build_noisy_sparse_input(cfr, mask, 15.0)

        self.assertTrue(torch.equal(x1, x2))
        self.assertAlmostEqual(stats1["measured_snr_db_mean"], 15.0, delta=0.5)
        self.assertAlmostEqual(stats2["measured_snr_db_mean"], 15.0, delta=0.5)
        torch.manual_seed(124)
        x3, _, _ = build_noisy_sparse_input(cfr, mask, 15.0)
        self.assertFalse(torch.equal(x1, x3))

    def test_pilot_ls_observation_is_clean_without_noise_and_preserves_unreported(self):
        from src.training.data import build_pilot_ls_observation

        cfr = torch.ones((2, 16, 2, 2), dtype=torch.complex64)
        mask = torch.zeros((2, 16), dtype=torch.float32)
        mask[:, ::4] = 1.0
        estimate, stats = build_pilot_ls_observation(cfr, mask, None)
        self.assertTrue(torch.equal(estimate, cfr))
        self.assertIsNone(stats["requested_snr_db"])

    def test_pilot_ls_observation_has_15db_snr_and_is_reproducible(self):
        from src.training.data import build_pilot_ls_observation

        cfr = torch.ones((32, 64, 2, 2), dtype=torch.complex64)
        mask = torch.zeros((32, 64), dtype=torch.float32)
        mask[:, ::4] = 1.0
        torch.manual_seed(321)
        estimate1, stats1 = build_pilot_ls_observation(cfr, mask, 15.0)
        torch.manual_seed(321)
        estimate2, stats2 = build_pilot_ls_observation(cfr, mask, 15.0)
        self.assertTrue(torch.equal(estimate1, estimate2))
        self.assertAlmostEqual(stats1["measured_snr_db_mean"], 15.0, delta=0.5)
        self.assertAlmostEqual(stats2["measured_snr_db_mean"], 15.0, delta=0.5)
        self.assertTrue(torch.all(estimate1[:, 1::4] == cfr[:, 1::4]))
        self.assertTrue(torch.isfinite(estimate1.real).all())

    def test_requested_hadamard_pilot_is_unitary_and_ls_is_exact_without_noise(self):
        from src.training.data import build_requested_hadamard_ls_observation

        cfr = torch.randn(2, 16, 2, 2, dtype=torch.complex64)
        mask = torch.zeros(2, 16)
        mask[:, ::4] = 1
        estimate, stats = build_requested_hadamard_ls_observation(cfr, mask, None)
        self.assertTrue(torch.allclose(estimate, cfr))
        self.assertIsNone(stats["requested_snr_db"])

    def test_requested_hadamard_pilot_reproducible_and_measured_snr(self):
        from src.training.data import build_requested_hadamard_ls_observation

        cfr = torch.ones(2, 32, 2, 2, dtype=torch.complex64)
        mask = torch.zeros(2, 32)
        mask[:, ::4] = 1
        torch.manual_seed(123)
        first, first_stats = build_requested_hadamard_ls_observation(cfr, mask, 15.0)
        torch.manual_seed(123)
        second, second_stats = build_requested_hadamard_ls_observation(cfr, mask, 15.0)
        self.assertTrue(torch.equal(first, second))
        self.assertAlmostEqual(first_stats["measured_snr_db_mean"], 15.0, delta=1.0)
        self.assertAlmostEqual(first_stats["measured_snr_db_mean"], second_stats["measured_snr_db_mean"], places=6)
        self.assertTrue(torch.isfinite(first.real).all())
        self.assertTrue(torch.isfinite(first.imag).all())
    def test_sparse_input_has_nine_channels_and_zeroes_unreported_subcarriers(self):
        from src.training.data import build_sparse_input

        cfr = torch.ones((2, 1024, 2, 2), dtype=torch.complex64)
        mask = torch.zeros((2, 1024), dtype=torch.float32)
        mask[:, ::4] = 1.0

        x, target = build_sparse_input(cfr, mask)

        self.assertEqual(list(x.shape), [2, 9, 1024])
        self.assertEqual(list(target.shape), [2, 8, 1024])
        self.assertTrue(torch.all(x[:, :8, 1::4] == 0))
        self.assertTrue(torch.all(x[:, 8, ::4] == 1))
        self.assertTrue(torch.all(x[:, 8, 1::4] == 0))

    def test_predictor_has_32_individually_addressable_blocks(self):
        from src.models.uacp_predictor import UACPEvidentialPredictor

        model = UACPEvidentialPredictor()
        self.assertEqual(len(model.residual_blocks), 32)

        x = torch.randn(2, 9, 1024)
        out = model(x)

        self.assertEqual(list(out.gamma.shape), [2, 8, 1024])
        self.assertEqual(list(out.kappa.shape), [2, 8, 1024])
        self.assertEqual(list(out.psi.shape), [2, 8, 1024])
        self.assertEqual(list(out.nu.shape), [2, 8, 1024])
        self.assertTrue(torch.all(out.kappa > 0))
        self.assertTrue(torch.all(out.psi > 0))
        self.assertTrue(torch.all(out.nu > 2049))

    def test_uncertainty_equations_follow_diagonal_niw_mapping(self):
        from src.models.evidential import EvidentialOutput

        gamma = torch.zeros(1, 8, 4)
        kappa = torch.full_like(gamma, 2.0)
        psi = torch.full_like(gamma, 6.0)
        nu = torch.full_like(gamma, 2052.0)
        out = EvidentialOutput(gamma=gamma, kappa=kappa, psi=psi, nu=nu)

        self.assertTrue(torch.allclose(out.predicted, gamma))
        self.assertTrue(torch.allclose(out.aleatoric, torch.full_like(gamma, 2.0)))
        self.assertTrue(torch.allclose(out.epistemic, torch.full_like(gamma, 1.0)))

    def test_loss_terms_are_finite(self):
        from src.models.evidential import EvidentialOutput, evidential_loss

        target = torch.randn(2, 8, 16)
        out = EvidentialOutput(
            gamma=torch.zeros_like(target),
            kappa=torch.ones_like(target),
            psi=torch.ones_like(target),
            nu=torch.full_like(target, 2052.0),
        )
        losses = evidential_loss(out, target, lambda_reg=1e-3)

        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertTrue(torch.isfinite(losses["nll"]))
        self.assertTrue(torch.isfinite(losses["reg"]))
        self.assertGreater(float(losses["total"]), 0.0)

    def test_pair_scalar_evidential_output_broadcasts_uncertainty(self):
        from src.models.evidential import EvidentialOutput

        gamma = torch.zeros(2, 8, 16)
        psi = torch.ones_like(gamma) * 6.0
        kappa = torch.ones(2, 4, 1) * 2.0
        nu = torch.ones(2, 4, 1) * 36.0
        out = EvidentialOutput(gamma=gamma, kappa=kappa, psi=psi, nu=nu, num_subcarriers=16)

        self.assertEqual(list(out.kappa_expanded.shape), [2, 8, 16])
        self.assertEqual(list(out.nu_expanded.shape), [2, 8, 16])
        self.assertTrue(torch.allclose(out.aleatoric, torch.full_like(gamma, 2.0)))
        self.assertTrue(torch.allclose(out.epistemic, torch.full_like(gamma, 1.0)))

    def test_pair_scalar_predictor_outputs_pair_level_kappa_nu(self):
        from src.models.uacp_predictor import UACPEvidentialPredictor

        model = UACPEvidentialPredictor(evidential_mode="pair_scalar")
        out = model(torch.randn(2, 9, 1024))

        self.assertEqual(list(out.gamma.shape), [2, 8, 1024])
        self.assertEqual(list(out.psi.shape), [2, 8, 1024])
        self.assertEqual(list(out.kappa.shape), [2, 4, 1])
        self.assertEqual(list(out.nu.shape), [2, 4, 1])

    def test_diagonal_multivariate_loss_terms_are_finite(self):
        from src.models.evidential import EvidentialOutput, evidential_loss

        target = torch.randn(2, 8, 16)
        out = EvidentialOutput(
            gamma=torch.zeros_like(target),
            kappa=torch.ones(2, 4, 1),
            psi=torch.ones_like(target),
            nu=torch.full((2, 4, 1), 36.0),
            num_subcarriers=16,
        )

        nll_losses = evidential_loss(out, target, lambda_reg=1e-3, nll_mode="diagonal_multivariate", reg_mode="elementwise")
        pair_losses = evidential_loss(out, target, lambda_reg=1e-3, nll_mode="diagonal_multivariate", reg_mode="pair")

        for losses in (nll_losses, pair_losses):
            self.assertTrue(torch.isfinite(losses["total"]))
            self.assertTrue(torch.isfinite(losses["nll"]))
            self.assertTrue(torch.isfinite(losses["reg"]))
            self.assertTrue(torch.isfinite(losses["lambda_reg_x_reg"]))

    def test_omitted_nmse_uses_only_dropped_subcarriers(self):
        from src.training.metrics import nmse_all_db, nmse_omitted_db

        target = torch.ones(1, 1, 4)
        pred = target.clone()
        pred[:, :, 1] = 0.0
        pred[:, :, 3] = 0.0
        mask = torch.tensor([[1.0, 0.0, 1.0, 0.0]])

        self.assertLess(float(nmse_all_db(pred, target)), float(nmse_omitted_db(pred, target, mask)))
        self.assertAlmostEqual(float(nmse_omitted_db(pred, target, mask)), 0.0, places=5)

    def test_linear_interpolation_reconstructs_between_observed_subcarriers(self):
        from src.training.data import linear_interpolate_real_imag

        sparse = torch.tensor([[[0.0, 0.0, 2.0, 0.0, 4.0]]])
        mask = torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0]])

        interpolated = linear_interpolate_real_imag(sparse, mask)

        self.assertTrue(torch.allclose(interpolated, torch.tensor([[[0.0, 1.0, 2.0, 3.0, 4.0]]])))

    def test_pair_scalar_broadcast_preserves_pair_channel_layout(self):
        from src.models.evidential import EvidentialOutput

        gamma = torch.zeros(1, 8, 2)
        kappa = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
        nu = torch.full((1, 4, 1), 10.0)
        out = EvidentialOutput(gamma=gamma, kappa=kappa, psi=torch.ones_like(gamma), nu=nu, num_subcarriers=2)
        expected = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]])
        self.assertTrue(torch.equal(out.kappa_expanded, expected))

    def test_channel_pair_mapping_and_roundtrip_are_exact(self):
        from src.models.evidential import channels_to_pair_vectors, pair_vectors_to_channels

        values = torch.zeros(1, 8, 3)
        for channel in range(8):
            values[:, channel, :] = channel + 1
        pairs = channels_to_pair_vectors(values)
        self.assertTrue(torch.equal(pairs[:, 0, :], torch.tensor([[1., 1., 1., 5., 5., 5.]])))
        self.assertTrue(torch.equal(pairs[:, 1, :], torch.tensor([[2., 2., 2., 6., 6., 6.]])))
        self.assertTrue(torch.equal(pairs[:, 2, :], torch.tensor([[3., 3., 3., 7., 7., 7.]])))
        self.assertTrue(torch.equal(pairs[:, 3, :], torch.tensor([[4., 4., 4., 8., 8., 8.]])))
        self.assertTrue(torch.equal(pair_vectors_to_channels(pairs, 3), values))

    def test_dense_reference_matches_diagonal_multivariate_nll(self):
        from src.models.evidential import EvidentialOutput, diagonal_multivariate_student_t_nll
        from src.models.evidential_reference import paper_predictive_student_t_log_prob

        torch.manual_seed(3)
        batch_size, pairs, dimension = 3, 4, 4
        target_pair = torch.randn(batch_size, pairs, dimension, dtype=torch.float64)
        gamma_pair = torch.randn_like(target_pair)
        psi_pair = torch.rand_like(target_pair) + 0.5
        kappa = torch.rand(batch_size, pairs, 1, dtype=torch.float64) + 0.5
        nu = torch.full((batch_size, pairs, 1), dimension + 3.0, dtype=torch.float64) + torch.rand(batch_size, pairs, 1, dtype=torch.float64)
        def to_channels(values):
            return values.reshape(batch_size, pairs, 2, dimension // 2).permute(0, 2, 1, 3).reshape(batch_size, 8, dimension // 2)
        output = EvidentialOutput(to_channels(gamma_pair), kappa, to_channels(psi_pair), nu, dimension // 2)
        target = to_channels(target_pair)
        optimized = diagonal_multivariate_student_t_nll(output, target)
        dense_log_prob = paper_predictive_student_t_log_prob(target_pair, gamma_pair, kappa.squeeze(-1), torch.diag_embed(psi_pair), nu.squeeze(-1))
        self.assertLess(abs(float(optimized) + float(dense_log_prob.mean())), 1e-8)

    def test_lowrank_nll_matches_dense_and_diagonal_when_zero(self):
        from src.models.evidential import EvidentialOutput, diagonal_multivariate_student_t_nll, lowrank_multivariate_student_t_nll
        from src.models.evidential_reference import paper_predictive_student_t_log_prob

        torch.manual_seed(4)
        batch_size, pairs, dimension, rank = 2, 4, 4, 2
        target_pair = torch.randn(batch_size, pairs, dimension, dtype=torch.float64)
        gamma_pair = torch.randn_like(target_pair)
        diagonal = torch.rand_like(target_pair) + 0.5
        factor = torch.randn(batch_size, pairs, dimension, rank, dtype=torch.float64) * 0.1
        kappa = torch.rand(batch_size, pairs, 1, dtype=torch.float64) + 0.5
        nu = torch.full((batch_size, pairs, 1), dimension + 3.0, dtype=torch.float64) + torch.rand(batch_size, pairs, 1, dtype=torch.float64)
        def to_channels(values):
            return values.reshape(batch_size, pairs, 2, dimension // 2).permute(0, 2, 1, 3).reshape(batch_size, 8, dimension // 2)
        target, gamma, psi = to_channels(target_pair), to_channels(gamma_pair), to_channels(diagonal)
        output = EvidentialOutput(gamma, kappa, psi, nu, dimension // 2, factor)
        structured = lowrank_multivariate_student_t_nll(output, target)
        dense_psi = torch.diag_embed(diagonal) + factor @ factor.transpose(-2, -1)
        dense = -paper_predictive_student_t_log_prob(target_pair, gamma_pair, kappa.squeeze(-1), dense_psi, nu.squeeze(-1)).mean()
        self.assertLess(abs(float(structured - dense)), 1e-8)
        zero = EvidentialOutput(gamma, kappa, psi, nu, dimension // 2, torch.zeros_like(factor))
        self.assertLess(abs(float(lowrank_multivariate_student_t_nll(zero, target) - diagonal_multivariate_student_t_nll(EvidentialOutput(gamma, kappa, psi, nu, dimension // 2), target))), 1e-8)

    def test_lowrank_predictor_shape_and_positive_diagonal(self):
        from src.models.uacp_predictor import UACPEvidentialPredictor

        model = UACPEvidentialPredictor(evidential_mode="pair_scalar_lowrank", covariance_rank=4, num_subcarriers=32)
        output = model(torch.randn(1, 9, 32))
        self.assertEqual(list(output.covariance_factor.shape), [1, 4, 64, 4])
        self.assertTrue(torch.all(output.covariance_diag > 0))

    def test_banded_factor_is_psd_and_has_frequency_local_support(self):
        from src.models.evidential import build_banded_cholesky, banded_covariance_dense

        raw = torch.zeros(1, 4, 8, 4 * 2, dtype=torch.float64)
        raw[..., 0] = 0.25
        raw[..., 1] = -0.1
        raw[..., 2] = 0.05
        raw[..., 3] = 0.2
        factors, padded_dim = build_banded_cholesky(raw, num_subcarriers=8, bandwidth=2, diagonal=torch.ones(1, 4, 16, dtype=torch.float64))
        dense = banded_covariance_dense(factors, num_subcarriers=8, padded_dim=padded_dim)
        self.assertEqual(list(dense.shape), [1, 4, 18, 18])
        eigenvalues = torch.linalg.eigvalsh(dense)
        self.assertGreater(float(eigenvalues.min()), 0.0)
        # Different frequency blocks are independent in the bounded pilot.
        self.assertEqual(float(dense[0, 0, 0, 2 * 3]), 0.0)

    def test_banded_nll_matches_dense_reference_on_small_dimension(self):
        from src.models.evidential import EvidentialOutput, banded_multivariate_student_t_nll
        from src.models.evidential_reference import paper_predictive_student_t_log_prob
        from src.models.evidential import build_banded_cholesky, banded_covariance_dense

        torch.manual_seed(11)
        batch_size, pairs, num_subcarriers, bandwidth = 1, 4, 4, 1
        target_pair = torch.randn(batch_size, pairs, 2 * num_subcarriers, dtype=torch.float64)
        gamma_pair = torch.randn_like(target_pair)
        diagonal = torch.full_like(target_pair, 1.5)
        raw = torch.randn(batch_size, pairs, num_subcarriers, 4 * bandwidth, dtype=torch.float64) * 0.05
        factors, padded_dim = build_banded_cholesky(raw, num_subcarriers, bandwidth, diagonal)
        target = target_pair.reshape(batch_size, pairs, 2, num_subcarriers).permute(0, 2, 1, 3).reshape(batch_size, 8, num_subcarriers)
        gamma = gamma_pair.reshape(batch_size, pairs, 2, num_subcarriers).permute(0, 2, 1, 3).reshape(batch_size, 8, num_subcarriers)
        psi = diagonal.reshape(batch_size, pairs, 2, num_subcarriers).permute(0, 2, 1, 3).reshape(batch_size, 8, num_subcarriers)
        kappa = torch.full((batch_size, pairs, 1), 2.0, dtype=torch.float64)
        nu = torch.full((batch_size, pairs, 1), 2 * num_subcarriers + 3.0, dtype=torch.float64)
        output = EvidentialOutput(gamma, kappa, psi, nu, num_subcarriers, banded_factor=factors, banded_padded_dim=padded_dim)
        actual = banded_multivariate_student_t_nll(output, target)
        dense_interleaved = banded_covariance_dense(factors, num_subcarriers, padded_dim)[:, :, : 2 * num_subcarriers, : 2 * num_subcarriers]
        permutation = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7])
        dense_psi = dense_interleaved.index_select(-2, permutation).index_select(-1, permutation)
        expected = -paper_predictive_student_t_log_prob(target_pair, gamma_pair, kappa.squeeze(-1), dense_psi, nu.squeeze(-1)).mean()
        self.assertLess(abs(float(actual - expected)), 1e-8)

    def test_banded_predictor_outputs_local_factor_and_positive_marginal(self):
        from src.models.uacp_predictor import UACPEvidentialPredictor

        model = UACPEvidentialPredictor(evidential_mode="pair_scalar_banded", covariance_bandwidth=2, num_subcarriers=8)
        output = model(torch.randn(1, 9, 8))
        self.assertEqual(list(output.banded_factor.shape), [1, 4, 3, 6, 6])
        self.assertEqual(list(output.covariance_diag.shape), [1, 8, 8])
        self.assertTrue(torch.all(output.covariance_diag > 0))


if __name__ == "__main__":
    unittest.main()
