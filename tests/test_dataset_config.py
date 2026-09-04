import tempfile
import unittest
from pathlib import Path

import numpy as np


class DatasetConfigTests(unittest.TestCase):
    def test_dataset_config_keeps_paper_values_and_assumptions_separate(self):
        from scripts.generate_dataset import load_dataset_config
        from src.channel.sionna_channel import assumption_settings, paper_settings

        config = load_dataset_config("configs/dataset_prototype.json")

        paper = paper_settings(config)
        assumptions = assumption_settings(config)

        self.assertEqual(paper["fft_size"], 1024)
        self.assertEqual(paper["num_rx_ant"], 2)
        self.assertEqual(paper["num_tx_ant"], 2)
        self.assertEqual(paper["training_delay_spread_ns"], [10.0, 100.0])
        self.assertEqual(paper["snr_db"], 15.0)
        self.assertNotIn("tdl_model", paper)
        self.assertEqual(assumptions["tdl_model"], "A")

    def test_write_and_inspect_dataset_summary(self):
        from scripts.generate_dataset import save_split_npz
        from scripts.inspect_dataset import inspect_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "toy.npz"
            cfr = np.ones((3, 8, 2, 2), dtype=np.complex64)
            delay_spread_ns = np.array([20.0, 80.0, 120.0], dtype=np.float32)
            labels = np.array(["ID-Easy", "ID-Hard", "OOD-Near"])

            save_split_npz(
                path=path,
                cfr=cfr,
                delay_spread_ns=delay_spread_ns,
                labels=labels,
                metadata={"seed": 7, "channel_model": "TDL-A"},
            )
            summary = inspect_dataset(path)

        self.assertEqual(summary["shape"], [3, 8, 2, 2])
        self.assertEqual(summary["dtype"], "complex64")
        self.assertFalse(summary["has_nan"])
        self.assertFalse(summary["has_inf"])
        self.assertEqual(summary["delay_spread_ns"]["min"], 20.0)
        self.assertEqual(summary["delay_spread_ns"]["max"], 120.0)


if __name__ == "__main__":
    unittest.main()
