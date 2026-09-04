import unittest


class SionnaSmokeConfigTests(unittest.TestCase):
    def test_load_config_marks_paper_values_and_assumptions(self):
        from scripts.sionna_smoke import load_config, paper_settings, assumption_settings

        config = load_config("configs/sionna_smoke.json")

        self.assertEqual(config["num_rx_ant"], 2)
        self.assertEqual(config["num_tx_ant"], 2)
        self.assertEqual(config["fft_size"], 1024)
        self.assertEqual(config["carrier_frequency_hz"], 3.5e9)
        self.assertEqual(config["subcarrier_spacing_hz"], 30e3)
        self.assertEqual(config["snr_db"], 15.0)
        self.assertEqual(config["delay_spreads_ns"], [20.0, 80.0, 120.0])
        self.assertIn("tdl_model", assumption_settings(config))
        self.assertNotIn("tdl_model", paper_settings(config))


if __name__ == "__main__":
    unittest.main()
