# STEP 6 Condition C Baseline Audit

This is a read-only audit. Existing results and checkpoints were not modified.

## Identity and Provenance

The STEP 6 `lambda=1e-3` reference reuses the STEP 5 Condition C checkpoint. The checkpoint is a plain state dict and does not record the config or code revision. Repository chronology and README classify this result as HISTORICAL / PRE-FIX because it predates the STEP 7 real/imag pair-mapping correction.

## A. Training Setting Checks

| Item | Status | Observed evidence |
| --- | --- | --- |
| Train samples | VERIFIED | `{"evidence": "/home/saslab01/Desktop/UACP_2026/runs/baseline_reproduction/step5_snr_ablation/noisy_train_noisy_eval/generated_data/train_5k_pilot.npz", "observed": 5000}` |
| Training delay spread | VERIFIED | `{"labels": ["train"], "observed_max": 99.99331665039062, "observed_min": 10.002578735351562}` |
| Epochs/history | VERIFIED | `{"epoch_numbers": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "final_results_history": 10, "training_history_history": 10}` |
| 20/80/120 ns probes | VERIFIED | `{"configured_paths": {"ID-Easy 20 ns": "data/prototype/test_id_easy.npz", "ID-Hard 80 ns": "data/prototype/test_id_hard.npz", "OOD-Near 120 ns": "data/prototype/test_ood_near.npz"}, "regimes": ["ID-Easy 20 ns", "ID-Hard 80 ns", "OOD-Far 1 ms", "OOD-Near 120 ns"]}` |
| 1 ms probe | VERIFIED | `{"present_in_final_probe": true, "source": "saved epoch_probe_results"}` |
| 15 dB noisy observation | VERIFIED | `{"configured": 15.0, "final_probe_measured_snr": {"ID-Easy 20 ns": 14.990790939331054, "ID-Hard 80 ns": 14.986900787353516, "OOD-Far 1 ms": 14.993080215454102, "OOD-Near 120 ns": 15.00957145690918}}` |
| Condition C train/eval noise | VERIFIED | `{"evidence": "config noise=15 dB and STEP 5 Condition C training history; target clean in build_noisy_sparse_input"}` |
| Learning rate | VERIFIED | `{"observed": 0.0001}` |
| Batch size/DataLoader length | VERIFIED | `{"dataloader_length": 625, "observed": 8}` |
| lambda_reg | VERIFIED | `{"observed": 0.001}` |
| Mask policy | VERIFIED | `{"eval_grouping_factor": 16, "training_grouping_factors": [4, 8, 16, 32]}` |
| Seed | VERIFIED | `{"observed": 20260819}` |
| Checkpoint | VERIFIED | `{"bytes": 47317003, "path": "/home/saslab01/Desktop/UACP_2026/runs/baseline_reproduction/step5_snr_ablation/noisy_train_noisy_eval/uacp_predictor_step4a.pt"}` |

The paper states SNR=15 dB, but its exact application stage is UNKNOWN. The observed-CFR AWGN placement is an IMPLEMENTATION-ASSUMPTION, not a paper-verified setting.

## B. Paper NIW Shape vs Current Code

| Parameter | Paper, per antenna pair | Current code / checkpoint mode | Match |
| --- | --- | --- | --- |
| gamma | `[2K] = [2048]` | external `[8, 1024]`, pair vector `[4, 2048]` | PARTIAL MATCH (channel representation) |
| kappa | scalar | `[4, 1]` pair-level | MATCH |
| Psi | `[2048,2048]`, `L L^T` | diagonal `[4, 2048]` | MISMATCH / APPROXIMATION |
| nu | scalar, `nu > 2049` | `[4, 1]` pair-level; `nu-d-1` min `17.0134` | MATCH (pair scalar/admissibility) |

The current `psi` is a diagonal scale approximation. It cannot be described as the paper's full covariance implementation.

## C. Real/Imag Pair Mapping

Current layout: `[Re(pair0..3), Im(pair0..3)]`. Exact mapping: `{'pair0': [0, 4], 'pair1': [1, 5], 'pair2': [2, 6], 'pair3': [3, 7]}`. Synthetic mapping check: `True`. The current mapping helpers pass the existing round-trip unit test.

The historical STEP 6 checkpoint was trained before this correction. The checkpoint itself has no revision metadata, so exact code-version provenance is NOT RECORDED inside the file; the pre-fix classification is supported by the repository's STEP 7/8 audit record.

## D. Eq. (7) and Eq. (8)

Current production code computes `Sigma_ale = diag(Psi)/(nu-d-1)` and `Sigma_epi = Sigma_ale/kappa` after pair-scalar expansion. This is a MATCH for the diagonal specialization, with `d=2K=2048`; positivity uses softplus plus epsilon. It is not a match to the paper's full Psi covariance.

For the historical STEP 6 result, these formulas were evaluated with the pre-fix channel/broadcast mapping. Therefore the saved uncertainty values are not a valid post-correction Eq. (7)/(8) baseline.

## E/F. Eq. (12) and Eq. (13)

The current helper sequence is: `pair -> Re/Im sum (trace of diagonal approximation) -> pair mean -> omitted mean`. Synthetic Eq.12/Eq.13 check: `True`; score map `[[9.0, 9.0, 9.0, 9.0]]`, omitted score `9.0`. Current code uses omitted=`1-mask`, so polarity is correct.

However, the historical STEP 6 saved table was generated before the mapping/aggregation correction and its primary diagnostic used channelwise omitted uncertainty means. The saved numbers therefore cannot be claimed to be current Eq. (12)->(13) values. They remain historical diagnostics only.

## G. Eq. (10) and Eq. (11)

The current D configuration uses diagonal multivariate Student-t NLL with one 2K-dimensional density per pair, and pair-level `||h-gamma||^2 * (kappa+nu)` regularization over observed and omitted target components. This is structurally aligned with Eq. (10)/(11) under the diagonal approximation.

The implementation reduces by mean rather than preserving the paper's displayed sum, so absolute loss scale is an IMPLEMENTATION-ASSUMPTION. The historical checkpoint was trained under the pre-fix pair mapping, so its uncertainty/evidence learning is confounded even though the saved configuration says `diagonal_multivariate` and `reg_mode=pair`.

## Final Assessment

`BASELINE INVALID FOR UNCERTAINTY COMPARISON`.

The 5k/10-epoch noisy reconstruction run and its finite outputs are usable as a historical reconstruction diagnostic. The uncertainty results must not be used as the corrected paper-faithful baseline because (1) the checkpoint/result predates the real/imag pair-layout and scalar broadcast correction, (2) the saved result is not demonstrably the current Eq.12->13 aggregation, and (3) Psi is diagonal rather than the paper's full `L L^T` covariance. No retraining was performed in this audit.

## Recommended Next Experiments

1. Re-run a corrected Condition C baseline with the current mapping and explicitly save code/config provenance before interpreting uncertainty.
2. Then perform the planned STEP 8B channel/observation model audit: exact Sionna/model wording, delay spread, SNR/pilot stage, feedback, normalization, OFDM/CP, mobility, and paper-scale training.
3. Do not start Partial Fine-Tuning or treat the historical STEP 6 uncertainty table as a valid comparison baseline yet.
