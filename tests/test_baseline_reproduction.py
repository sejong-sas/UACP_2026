import tempfile
import unittest
from pathlib import Path


class BaselineReproductionTests(unittest.TestCase):
    def test_delay_label_formats_ns_and_ms(self):
        from scripts.baseline_reproduction import delay_label

        self.assertEqual(delay_label(20.0), "20 ns")
        self.assertEqual(delay_label(120.0), "120 ns")
        self.assertEqual(delay_label(1_000_000.0), "1 ms")

    def test_step_row_flattens_omitted_statistics(self):
        from scripts.baseline_reproduction import step_row

        metrics = {
            "samples": 2,
            "nmse_all_db": -1.0,
            "nmse_omitted_db": -2.0,
            "aleatoric_omitted": {"mean": 0.3, "std": 0.4},
            "epistemic_omitted": {"mean": 0.5, "std": 0.6},
            "psi_omitted": {"mean": 0.7, "std": 0.8},
            "kappa_omitted": {"mean": 0.9, "std": 1.0},
            "nu_omitted": {"mean": 2.0, "std": 3.0},
            "df_cov_omitted": {"mean": 4.0, "std": 5.0},
            "error_aleatoric_pearson": 0.1,
            "error_aleatoric_spearman": 0.2,
            "error_epistemic_pearson": 0.3,
            "error_epistemic_spearman": 0.4,
        }

        row = step_row("Experiment C", "20 ns", 20.0, metrics)

        self.assertEqual(row["experiment"], "Experiment C")
        self.assertEqual(row["regime"], "20 ns")
        self.assertEqual(row["delay_spread_ns"], 20.0)
        self.assertEqual(row["nmse_omitted_db"], -2.0)
        self.assertEqual(row["aleatoric_mean"], 0.3)
        self.assertEqual(row["epistemic_std"], 0.6)
        self.assertEqual(row["error_epistemic_spearman"], 0.4)

    def test_write_curve_pngs_creates_expected_files(self):
        from scripts.baseline_reproduction import write_curve_pngs

        rows = [
            {"delay_spread_ns": 20.0, "nmse_omitted_db": -10.0, "aleatoric_mean": 0.1, "epistemic_mean": 0.2, "psi_mean": 0.3, "kappa_mean": 1.0},
            {"delay_spread_ns": 80.0, "nmse_omitted_db": -8.0, "aleatoric_mean": 0.2, "epistemic_mean": 0.1, "psi_mean": 0.4, "kappa_mean": 1.1},
        ]

        with tempfile.TemporaryDirectory() as tmp:
            written = write_curve_pngs(rows, Path(tmp))

            self.assertEqual(len(written), 5)
            for path in written:
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 0)

    def test_serializable_args_drops_callable_dispatch_function(self):
        from argparse import Namespace

        from scripts.baseline_reproduction import serializable_args

        args = Namespace(step="step1", func=lambda value: value, output_dir="runs/x")

        self.assertEqual(serializable_args(args), {"output_dir": "runs/x", "step": "step1"})

    def test_epoch_probe_row_includes_epoch_and_regime_metrics(self):
        from scripts.baseline_reproduction import epoch_probe_row

        metrics = {
            "samples": 2,
            "nmse_all_db": -1.0,
            "nmse_omitted_db": -2.0,
            "aleatoric_omitted": {"mean": 0.3, "std": 0.4},
            "epistemic_omitted": {"mean": 0.5, "std": 0.6},
            "psi_omitted": {"mean": 0.7, "std": 0.8},
            "kappa_omitted": {"mean": 0.9, "std": 1.0},
            "nu_omitted": {"mean": 2.0, "std": 3.0},
            "df_cov_omitted": {"mean": 4.0, "std": 5.0},
            "error_aleatoric_pearson": 0.1,
            "error_aleatoric_spearman": 0.2,
            "error_epistemic_pearson": 0.3,
            "error_epistemic_spearman": 0.4,
            "nll": -6.0,
            "raw_l_reg": 7.0,
            "lambda_reg_x_l_reg": 0.007,
        }

        row = epoch_probe_row(3, "20 ns", 20.0, metrics)

        self.assertEqual(row["epoch"], 3)
        self.assertEqual(row["regime"], "20 ns")
        self.assertEqual(row["aleatoric_mean"], 0.3)
        self.assertEqual(row["nll"], -6.0)

    def test_step4_interpretation_identifies_case_b(self):
        from scripts.baseline_reproduction import interpret_step4

        rows = [
            {"epoch": 1, "regime": "ID-Easy 20 ns", "aleatoric_mean": 0.5, "epistemic_mean": 0.5, "nmse_omitted_db": -10.0, "error_aleatoric_pearson": 0.1, "error_epistemic_pearson": 0.1},
            {"epoch": 1, "regime": "ID-Hard 80 ns", "aleatoric_mean": 0.4, "epistemic_mean": 0.4, "nmse_omitted_db": -9.0, "error_aleatoric_pearson": 0.1, "error_epistemic_pearson": 0.1},
            {"epoch": 1, "regime": "OOD-Near 120 ns", "aleatoric_mean": 0.3, "epistemic_mean": 0.3, "nmse_omitted_db": -8.0, "error_aleatoric_pearson": 0.1, "error_epistemic_pearson": 0.1},
            {"epoch": 1, "regime": "OOD-Far 1 ms", "aleatoric_mean": 0.2, "epistemic_mean": 0.2, "nmse_omitted_db": 1.0, "error_aleatoric_pearson": 0.0, "error_epistemic_pearson": 0.0},
        ]

        interpretation = interpret_step4(rows)

        self.assertEqual(interpretation["case"], "Case B")
        self.assertFalse(interpretation["proceed_to_step4b"])

    def test_probe_metrics_as_evaluation_uses_omitted_uncertainty_means(self):
        from scripts.baseline_reproduction import probe_metrics_as_evaluation

        metrics = {
            "nmse_all_db": -1.0,
            "nmse_omitted_db": -2.0,
            "aleatoric_omitted": {"mean": 0.3},
            "epistemic_omitted": {"mean": 0.4},
            "nll": -5.0,
            "raw_l_reg": 6.0,
            "lambda_reg_x_l_reg": 0.006,
        }

        row = probe_metrics_as_evaluation(metrics)

        self.assertEqual(row["aleatoric"], 0.3)
        self.assertEqual(row["epistemic"], 0.4)
        self.assertEqual(row["reg"], 6.0)
        self.assertEqual(row["lambda_reg_x_reg"], 0.006)

    def test_diagnostic_ratio_and_error_bins_are_finite(self):
        from scripts.diagnose_predictor import _error_bin_calibration

        error = __import__("torch").tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        aleatoric = error * 2
        epistemic = error * 3
        bins = _error_bin_calibration(error, aleatoric, epistemic)

        self.assertEqual(len(bins), 3)
        self.assertTrue(all(item["count"] > 0 for item in bins))
        self.assertTrue(all(item["aleatoric_mean"] > 0 for item in bins))


if __name__ == "__main__":
    unittest.main()
