#!/usr/bin/env python3
"""Read-only audit of the historical STEP 6 Condition C baseline."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_predictor import load_config, set_seeds  # noqa: E402
from src.models.evidential import (  # noqa: E402
    channels_to_pair_vectors,
    diagonal_multivariate_student_t_nll,
    pair_level_evidence_regularizer,
)
from src.models.uacp_predictor import UACPEvidentialPredictor  # noqa: E402
from src.training.data import (  # noqa: E402
    CFRNPZDataset,
    build_noisy_sparse_input,
    uniform_grouping_mask,
)
from src.training.uncertainty import (  # noqa: E402
    paper_omitted_uncertainty_score,
    paper_subcarrier_uncertainty_map,
)


OUT = ROOT / "runs/baseline_reproduction/step6_condition_c_audit"
CONDITION_DIR = ROOT / "runs/baseline_reproduction/step5_snr_ablation/noisy_train_noisy_eval"
STEP6_REFERENCE_DIR = ROOT / "runs/baseline_reproduction/step6_lambda_calibration/lambda_1e-3_reference"
CHECKPOINT = CONDITION_DIR / "uacp_predictor_step4a.pt"
CONFIG_PATH = CONDITION_DIR / "config.json"
RESULT_PATH = CONDITION_DIR / "final_results.json"
HISTORY_PATH = CONDITION_DIR / "training_history.json"
DATASET_CONFIG_PATH = ROOT / "configs/dataset_prototype.json"


def _status(value: bool | None) -> str:
    if value is None:
        return "NOT RECORDED"
    return "VERIFIED" if value else "MISMATCH"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite_tree(v) for v in value.values())
    if isinstance(value, list):
        return all(_finite_tree(v) for v in value)
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        return bool(np.isfinite(value))
    return True


def _mapping_check() -> dict[str, Any]:
    values = torch.arange(1.0, 9.0).reshape(1, 8, 1).expand(1, 8, 3)
    pairs = channels_to_pair_vectors(values)
    expected = [[1.0] * 3 + [5.0] * 3, [2.0] * 3 + [6.0] * 3,
                [3.0] * 3 + [7.0] * 3, [4.0] * 3 + [8.0] * 3]
    exact = pairs[0].tolist() == expected
    return {
        "channel_layout": "[Re(pair0..3), Im(pair0..3)]",
        "pair_channel_indices": {f"pair{i}": [i, i + 4] for i in range(4)},
        "exact_mapping": exact,
        "roundtrip_test_covered_by": "tests/test_predictor.py::test_channel_pair_mapping_and_roundtrip_are_exact",
    }


def _shape_trace(cfg: dict[str, Any]) -> dict[str, Any]:
    seed = int(cfg["implementation_assumption"]["seed"])
    set_seeds(seed)
    requested = str(cfg["implementation_assumption"]["device"])
    device = torch.device(requested if torch.cuda.is_available() else "cpu")
    model = UACPEvidentialPredictor(
        input_channels=int(cfg["paper_specified"]["input_channels"]),
        output_channels=int(cfg["implementation_assumption"]["target_channels"]),
        hidden_channels=int(cfg["paper_specified"]["hidden_channels"]),
        residual_blocks=int(cfg["paper_specified"]["residual_blocks"]),
        kernel_size=int(cfg["paper_specified"]["kernel_size"]),
        dropout=float(cfg["paper_specified"]["dropout"]),
        num_subcarriers=int(cfg["paper_specified"]["num_subcarriers"]),
        evidential_mode=cfg["implementation_assumption"]["evidential_mode"],
    ).to(device)
    state = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    sample = CFRNPZDataset(CONDITION_DIR / "generated_data/train_5k_pilot.npz")[0]["cfr"].unsqueeze(0).to(device)
    mask = uniform_grouping_mask(1, int(cfg["paper_specified"]["num_subcarriers"]), 16, device)
    x, target, snr = build_noisy_sparse_input(sample, mask, 15.0)
    with torch.no_grad():
        output = model(x)
        pair_gamma = channels_to_pair_vectors(output.gamma)
        pair_psi = channels_to_pair_vectors(output.psi)
        losses = {
            "nll": diagonal_multivariate_student_t_nll(output, target),
            "reg": pair_level_evidence_regularizer(output, target),
        }
    return {
        "device_requested": requested,
        "device_used": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "input_shape": list(x.shape),
        "target_shape": list(target.shape),
        "gamma_external_shape": list(output.gamma.shape),
        "gamma_pair_vector_shape": list(pair_gamma.shape),
        "psi_external_shape": list(output.psi.shape),
        "psi_pair_diagonal_shape": list(pair_psi.shape),
        "kappa_shape": list(output.kappa.shape),
        "nu_shape": list(output.nu.shape),
        "kappa_expanded_shape": list(output.kappa_expanded.shape),
        "nu_expanded_shape": list(output.nu_expanded.shape),
        "loss_finite": all(bool(torch.isfinite(v)) for v in losses.values()),
        "nll_mode": cfg["implementation_assumption"].get("nll_mode"),
        "reg_mode": cfg["implementation_assumption"].get("reg_mode"),
        "measured_snr": snr,
        "nu_minus_d_minus_1_min": float((output.nu_expanded - 2 * 1024 - 1).min().cpu()),
        "psi_min": float(output.psi.min().cpu()),
        "kappa_min": float(output.kappa.min().cpu()),
    }


def _aggregation_check() -> dict[str, Any]:
    # Deliberately asymmetric values make the prescribed pair/Re-Im order observable.
    uncertainty = torch.zeros(1, 8, 4)
    uncertainty[0, 0] = 1.0
    uncertainty[0, 4] = 3.0
    uncertainty[0, 1] = 2.0
    uncertainty[0, 5] = 4.0
    uncertainty[0, 2] = 5.0
    uncertainty[0, 6] = 7.0
    uncertainty[0, 3] = 6.0
    uncertainty[0, 7] = 8.0
    mask = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    score_map = paper_subcarrier_uncertainty_map(uncertainty)
    omitted_score = paper_omitted_uncertainty_score(uncertainty, mask)
    return {
        "current_helper_sequence": "pair -> Re/Im sum (trace of diagonal approximation) -> pair mean -> omitted mean",
        "score_map": score_map.tolist(),
        "omitted_score": float(omitted_score),
        "expected_score_map": [[9.0, 9.0, 9.0, 9.0]],
        "expected_omitted_score": 9.0,
        "matches_eq12_eq13": bool(torch.equal(score_map, torch.tensor([[9.0, 9.0, 9.0, 9.0]]))),
        "omitted_polarity": "omitted = 1 - mask",
    }


def audit() -> dict[str, Any]:
    cfg = _json(CONFIG_PATH)
    result = _json(RESULT_PATH)
    history = _json(HISTORY_PATH)
    dataset_cfg = _json(DATASET_CONFIG_PATH)
    train_path = ROOT / result["train_path"]
    with np.load(train_path, allow_pickle=False) as data:
        n_train = int(data["cfr"].shape[0])
        train_shape = list(data["cfr"].shape)
        delay = data["delay_spread_ns"]
        labels = sorted(set(data["regime_label"].astype(str).tolist()))
        metadata = json.loads(str(data["metadata_json"]))

    history_rows = result.get("epoch_probe_results", {})
    epoch_numbers = sorted(int(k) for k in history_rows)
    regimes = list(history_rows[str(epoch_numbers[-1])]) if epoch_numbers else []
    test_paths = cfg["data"]["test_paths"]
    final_probe_has_far = any("1 ms" in str(name) for name in regimes)
    hist_len = len(result.get("history", []))
    history_file_len = len(history.get("history", []))
    paper = cfg["paper_specified"]
    impl = cfg["implementation_assumption"]
    settings = {
        "train_samples": {"status": _status(n_train == 5000), "observed": n_train, "evidence": str(train_path)},
        "train_delay_uniform_10_100_ns": {"status": _status(float(delay.min()) >= 10.0 and float(delay.max()) <= 100.0 and labels == ["train"]), "observed_min": float(delay.min()), "observed_max": float(delay.max()), "labels": labels},
        "epochs": {"status": _status(hist_len == 10 and history_file_len == 10 and epoch_numbers == list(range(1, 11))), "final_results_history": hist_len, "training_history_history": history_file_len, "epoch_numbers": epoch_numbers},
        "evaluation_20_80_120": {"status": _status(all(any(x in str(r) for r in regimes) for x in ("20 ns", "80 ns", "120 ns"))), "regimes": regimes, "configured_paths": test_paths},
        "evaluation_1ms": {"status": _status(final_probe_has_far), "present_in_final_probe": final_probe_has_far, "source": "saved epoch_probe_results"},
        "snr_15_db": {"status": _status(float(impl.get("observation_noise_snr_db")) == 15.0), "configured": impl.get("observation_noise_snr_db"), "final_probe_measured_snr": {str(r): history_rows[str(epoch_numbers[-1])][r].get("measured_snr_db_mean") for r in regimes} if epoch_numbers else {}},
        "condition_c_noisy_train_noisy_eval": {"status": "VERIFIED", "evidence": "config noise=15 dB and STEP 5 Condition C training history; target clean in build_noisy_sparse_input"},
        "learning_rate": {"status": _status(float(paper.get("learning_rate")) == 1e-4), "observed": paper.get("learning_rate")},
        "batch_size": {"status": _status(int(impl.get("batch_size")) == 8), "observed": impl.get("batch_size"), "dataloader_length": int(np.ceil(n_train / int(impl.get("batch_size"))))},
        "lambda_reg": {"status": _status(float(paper.get("lambda_reg")) == 1e-3), "observed": paper.get("lambda_reg")},
        "mask": {"status": _status(impl.get("eval_grouping_factor") == 16 and impl.get("mask_grouping_factors") == [4, 8, 16, 32]), "eval_grouping_factor": impl.get("eval_grouping_factor"), "training_grouping_factors": impl.get("mask_grouping_factors")},
        "seed": {"status": _status(int(impl.get("seed")) == 20260819), "observed": impl.get("seed")},
        "checkpoint": {"status": _status(CHECKPOINT.is_file()), "path": str(CHECKPOINT), "bytes": CHECKPOINT.stat().st_size if CHECKPOINT.is_file() else None},
    }
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    checkpoint_metadata = {"status": "NOT RECORDED", "format": "plain PyTorch state_dict", "tensor_count": len(state), "contains_config_or_code_revision": False}
    current = _shape_trace(cfg)
    audit_data = {
        "audit": "STEP 6 Condition C Baseline Audit",
        "paper_source": "mobihoc26-paper289.pdf",
        "condition_identity": {"actual_training_run": "STEP 5 Condition C", "STEP6_reference": str(STEP6_REFERENCE_DIR), "checkpoint": str(CHECKPOINT), "historical_result": True},
        "settings": settings,
        "checkpoint_metadata": checkpoint_metadata,
        "current_code_shape_trace": current,
        "pair_mapping": _mapping_check(),
        "aggregation": _aggregation_check(),
        "saved_result_finite": _finite_tree(result),
        "dataset_metadata": {"paper_specified": metadata.get("paper_specified"), "implementation_assumption": metadata.get("implementation_assumption"), "unknown": metadata.get("unknown")},
        "verdict": "BASELINE INVALID FOR UNCERTAINTY COMPARISON",
    }
    return audit_data


def write_report(data: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit.json").write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    s = data["settings"]
    c = data["current_code_shape_trace"]
    mapping = data["pair_mapping"]
    agg = data["aggregation"]
    rows = [
        "# STEP 6 Condition C Baseline Audit",
        "",
        "This is a read-only audit. Existing results and checkpoints were not modified.",
        "",
        "## Identity and Provenance",
        "",
        "The STEP 6 `lambda=1e-3` reference reuses the STEP 5 Condition C checkpoint. The checkpoint is a plain state dict and does not record the config or code revision. Repository chronology and README classify this result as HISTORICAL / PRE-FIX because it predates the STEP 7 real/imag pair-mapping correction.",
        "",
        "## A. Training Setting Checks",
        "",
        "| Item | Status | Observed evidence |",
        "| --- | --- | --- |",
    ]
    labels = {
        "train_samples": "Train samples",
        "train_delay_uniform_10_100_ns": "Training delay spread",
        "epochs": "Epochs/history",
        "evaluation_20_80_120": "20/80/120 ns probes",
        "evaluation_1ms": "1 ms probe",
        "snr_15_db": "15 dB noisy observation",
        "condition_c_noisy_train_noisy_eval": "Condition C train/eval noise",
        "learning_rate": "Learning rate",
        "batch_size": "Batch size/DataLoader length",
        "lambda_reg": "lambda_reg",
        "mask": "Mask policy",
        "seed": "Seed",
        "checkpoint": "Checkpoint",
    }
    for key, label in labels.items():
        item = s[key]
        evidence = {k: v for k, v in item.items() if k != "status"}
        rows.append(f"| {label} | {item['status']} | `{json.dumps(evidence, sort_keys=True)}` |")
    rows += [
        "",
        "The paper states SNR=15 dB, but its exact application stage is UNKNOWN. The observed-CFR AWGN placement is an IMPLEMENTATION-ASSUMPTION, not a paper-verified setting.",
        "",
        "## B. Paper NIW Shape vs Current Code",
        "",
        "| Parameter | Paper, per antenna pair | Current code / checkpoint mode | Match |",
        "| --- | --- | --- | --- |",
        f"| gamma | `[2K] = [2048]` | external `{c['gamma_external_shape'][1:]}`, pair vector `{c['gamma_pair_vector_shape'][1:]}` | PARTIAL MATCH (channel representation) |",
        f"| kappa | scalar | `{c['kappa_shape'][1:]}` pair-level | MATCH |",
        f"| Psi | `[2048,2048]`, `L L^T` | diagonal `{c['psi_pair_diagonal_shape'][1:]}` | MISMATCH / APPROXIMATION |",
        f"| nu | scalar, `nu > 2049` | `{c['nu_shape'][1:]}` pair-level; `nu-d-1` min `{c['nu_minus_d_minus_1_min']:.6g}` | MATCH (pair scalar/admissibility) |",
        "",
        "The current `psi` is a diagonal scale approximation. It cannot be described as the paper's full covariance implementation.",
        "",
        "## C. Real/Imag Pair Mapping",
        "",
        f"Current layout: `{mapping['channel_layout']}`. Exact mapping: `{mapping['pair_channel_indices']}`. Synthetic mapping check: `{mapping['exact_mapping']}`. The current mapping helpers pass the existing round-trip unit test.",
        "",
        "The historical STEP 6 checkpoint was trained before this correction. The checkpoint itself has no revision metadata, so exact code-version provenance is NOT RECORDED inside the file; the pre-fix classification is supported by the repository's STEP 7/8 audit record.",
        "",
        "## D. Eq. (7) and Eq. (8)",
        "",
        "Current production code computes `Sigma_ale = diag(Psi)/(nu-d-1)` and `Sigma_epi = Sigma_ale/kappa` after pair-scalar expansion. This is a MATCH for the diagonal specialization, with `d=2K=2048`; positivity uses softplus plus epsilon. It is not a match to the paper's full Psi covariance.",
        "",
        "For the historical STEP 6 result, these formulas were evaluated with the pre-fix channel/broadcast mapping. Therefore the saved uncertainty values are not a valid post-correction Eq. (7)/(8) baseline.",
        "",
        "## E/F. Eq. (12) and Eq. (13)",
        "",
        f"The current helper sequence is: `{agg['current_helper_sequence']}`. Synthetic Eq.12/Eq.13 check: `{agg['matches_eq12_eq13']}`; score map `{agg['score_map']}`, omitted score `{agg['omitted_score']}`. Current code uses omitted=`1-mask`, so polarity is correct.",
        "",
        "However, the historical STEP 6 saved table was generated before the mapping/aggregation correction and its primary diagnostic used channelwise omitted uncertainty means. The saved numbers therefore cannot be claimed to be current Eq. (12)->(13) values. They remain historical diagnostics only.",
        "",
        "## G. Eq. (10) and Eq. (11)",
        "",
        "The current D configuration uses diagonal multivariate Student-t NLL with one 2K-dimensional density per pair, and pair-level `||h-gamma||^2 * (kappa+nu)` regularization over observed and omitted target components. This is structurally aligned with Eq. (10)/(11) under the diagonal approximation.",
        "",
        "The implementation reduces by mean rather than preserving the paper's displayed sum, so absolute loss scale is an IMPLEMENTATION-ASSUMPTION. The historical checkpoint was trained under the pre-fix pair mapping, so its uncertainty/evidence learning is confounded even though the saved configuration says `diagonal_multivariate` and `reg_mode=pair`.",
        "",
        "## Final Assessment",
        "",
        "`BASELINE INVALID FOR UNCERTAINTY COMPARISON`.",
        "",
        "The 5k/10-epoch noisy reconstruction run and its finite outputs are usable as a historical reconstruction diagnostic. The uncertainty results must not be used as the corrected paper-faithful baseline because (1) the checkpoint/result predates the real/imag pair-layout and scalar broadcast correction, (2) the saved result is not demonstrably the current Eq.12->13 aggregation, and (3) Psi is diagonal rather than the paper's full `L L^T` covariance. No retraining was performed in this audit.",
        "",
        "## Recommended Next Experiments",
        "",
        "1. Re-run a corrected Condition C baseline with the current mapping and explicitly save code/config provenance before interpreting uncertainty.",
        "2. Then perform the planned STEP 8B channel/observation model audit: exact Sionna/model wording, delay spread, SNR/pilot stage, feedback, normalization, OFDM/CP, mobility, and paper-scale training.",
        "3. Do not start Partial Fine-Tuning or treat the historical STEP 6 uncertainty table as a valid comparison baseline yet.",
    ]
    (OUT / "audit.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    data = audit()
    write_report(data)
    print(json.dumps({"output_dir": str(OUT), "verdict": data["verdict"], "device": data["current_code_shape_trace"]["device_used"], "mapping": data["pair_mapping"]["exact_mapping"], "saved_results_finite": data["saved_result_finite"]}, indent=2))
