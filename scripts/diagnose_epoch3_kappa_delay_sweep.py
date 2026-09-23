#!/usr/bin/env python3
"""Run the existing κ loss-gradient diagnosis over the full delay sweep."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.diagnose_epoch3_kappa_loss_gradients as diagnosis

diagnosis.REGIMES = [10, 20, 40, 60, 80, 100, 120]
diagnosis.OUT = diagnosis.ROOT / "runs/current_valid_baseline/epoch3_kappa_delay_sweep_diagnosis_20260920"

if __name__ == "__main__":
    diagnosis.main()
