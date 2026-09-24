# Partial Fine-Tuning Readiness — 2026-09-21

The readiness package below describes the pre-training state. The controlled experiment was subsequently run and is reported in `docs/PARTIAL_FT_RESULTS_20260921.md`.

## Status

This overnight package completed baseline diagnosis and Partial Fine-Tuning readiness only. No Partial-vs-Full adaptation training was started. New artifacts are under `runs/current_valid_baseline/overnight_20260920_*`; existing checkpoints and results were not overwritten.

The control is `runs/current_valid_baseline/training_convergence_100k_5ep_20260916_final_controlled/uacp_predictor_100k_5ep_epoch_3.pt`. Inference and smoke checks used NVIDIA GB10 on `cuda:0`.

## 1. Experiments performed

- Phase 1: Ng=16/32 conditional residual inference. Common evaluation data supplied 2,000 samples each at 20 and 80 ns; the existing fixed-delay archive supplied 200 each at 40, 60, and 100 ns. The latter limitation is recorded, not hidden.
- Phase 2: inference-only absolute amplitude scaling at α ∈ {0.75, 1.0, 1.25} for 20/60/100 ns and Ng=16/32, preserving shape, target relationship, mask, and SNR relationship.
- Actual parameter inventory and one optimizer-step smoke checks for `head_only`, `last_block_plus_head`, `last_8_blocks_plus_head`, and `full`.

## 2. Ng32 tail judgment

The conditional diagnostic was `log(κ) ~ scaled RMS + scaled adjacent/frequency variation + scaled NMSE + scaled delay + Ng32 indicator`, with even samples for fitting and odd samples held out. Held-out R² was 0.884 and the Ng32 coefficient remained negative (-0.558). This is a conditional association diagnostic, not causal proof.

The pooled Ng32 ID q99 Epistemic threshold was 0.0192. Its held-out q99 tail had median actual/conditional κ ratio 0.463 (q05 0.085, q95 0.531), so residual κ collapse remains in the extreme tail. Overall held-out residual medians were 1.007 for Ng16 and 1.028 for Ng32. The strongest conclusion is **C**: conditioning removes much of the mean Ng effect, but a sample-dependent evidential-head/distribution-shape tail remains. The 40/60/100 ns cells are underpowered at 200 samples each and do not support a full 2,000-sample claim.

Phase 2 showed strong absolute-scale sensitivity. For Ng32, median κ/Epistemic ratios at α=0.75 were 1.227/0.723 (20 ns), 1.464/0.568 (60 ns), and 1.927/0.422 (100 ns). At α=1.25 they were 0.733/1.593, 0.521/2.410, and 0.253/5.048. Thus `normalize_channel=false` is an important calibration candidate, but remains `IMPLEMENTATION-ASSUMPTION / DIAGNOSTIC ONLY`; no normalization training was started.

## 3. Baseline readiness

### Baseline behavior reproduced

- CFR reconstruction is finite and operational on the epoch-3 control path.
- ID-Hard 80 ns remains harder than ID-Easy 20 ns in reconstruction error in the saved delay diagnostics.
- Aleatoric uncertainty generally rises with delay/difficulty in the saved epoch-3 sweep.
- Far-OOD 1 ms has strongly degraded reconstruction and increased Epistemic relative to ordinary ID in the saved baseline.
- No new NaN/inf was observed in the completed inference or smoke runs.

### Remaining implementation/reproduction gap

- Fig.11 dynamic threshold/controller remains limited: the saved batch-400 lite path collapses to Ng=1 and omitted-NMSE is undefined under full feedback.
- Diagonal Ψ is used instead of the paper's full covariance: `APPROXIMATION`.
- Reported-CFR sample-wise complex AWGN with clean target is used: `IMPLEMENTATION-ASSUMPTION`; paper noise stage is unknown.
- `normalize_channel=false` is not paper-confirmed: `IMPLEMENTATION-ASSUMPTION`.
- Partial layer selection, adaptation counts, epochs, LR, and trigger details below are `RESEARCH-EXTENSION / IMPLEMENTATION-ASSUMPTION`, not paper settings.

## 4. Actual parameter groups and candidate scopes

The actual model contains `input_projection`, 32 `ResidualBlock`s, and pair-scalar heads `gamma_head`, `psi_head`, `kappa_head`, `nu_head`. Full tensor inventory is in `runs/current_valid_baseline/overnight_20260920_partial_readiness_inventory/parameter_inventory.csv`.

| Scope | Trainable params | Frozen params | Trainable % | Adam state FP32 |
|---|---:|---:|---:|---:|
| `head_only` | 4,632 | 11,810,688 | 0.0392% | 0.035 MiB |
| `last_block_plus_head` (block 31) | 373,656 | 11,441,664 | 3.1625% | 2.851 MiB |
| `last_8_blocks_plus_head` (blocks 24–31) | 2,956,824 | 8,858,496 | 25.0253% | 22.559 MiB |
| `full` | 11,815,320 | 0 | 100% | 90.144 MiB |

These are real module-name scopes, not claimed paper policy.

## 5. Adaptation split plan

`configs/adaptation_120ns_protocol_20260921.json` defines a new provenance root:

- `adapt-train`: new 120 ns samples, full feedback S=K, 1,000 samples (`IMPLEMENTATION-ASSUMPTION`).
- `adapt-validation`: separate new 120 ns samples, 200 samples (`IMPLEMENTATION-ASSUMPTION`).
- untouched ID-Easy 20 ns, ID-Hard 80 ns, Near-OOD 120 ns, and Far-OOD 1 ms tests, 1,000 each (`IMPLEMENTATION-ASSUMPTION`).
- Existing 120 ns evaluation samples are not reused for adaptation training.

The paper allows full feedback during OOD adaptation but does not publish these counts or a Partial layer policy. This is therefore `RESEARCH-EXTENSION / IMPLEMENTATION-ASSUMPTION`, not paper reproduction.

## 6. Comparison protocol

Use the same epoch-3 starting checkpoint, adaptation samples, seed, order, optimizer, LR, epochs, batch size, and untouched evaluation sets. Change only `--trainable-scope`. Compare pre/post NMSE, Aleatoric, Epistemic, ID forgetting, Near-OOD improvement, NaN/inf, trainable count/percentage, wall-clock time, peak VRAM, and runtime if needed. `scripts/evaluate_partial_ft_comparison.py` is the common no-training evaluator.

## 7. Commands

Generate the separate adaptation/evaluation archive:

    python scripts/generate_dataset.py --config configs/adaptation_120ns_protocol_20260921.json --output-dir runs/current_valid_baseline/overnight_20260920_adaptation_data

Recommended first Partial scope is `last_block_plus_head`: it is the smallest feature-adaptive scope after head-only and passed the smoke invariant. The first command tomorrow is:

    python scripts/partial_ft_adapt.py --checkpoint runs/current_valid_baseline/training_convergence_100k_5ep_20260916_final_controlled/uacp_predictor_100k_5ep_epoch_3.pt --trainable-scope last_block_plus_head --adapt-train-data runs/current_valid_baseline/overnight_20260920_adaptation_data/train.npz --adapt-val-data runs/current_valid_baseline/overnight_20260920_adaptation_data/validation.npz --epochs 1 --batch-size 8 --lr 1e-4 --seed 20260921 --output-dir runs/current_valid_baseline/partial_ft_20260921_last_block_plus_head --dry-run

This is a smoke gate, not a performance result. The script supports the real multi-epoch path, but `--dry-run` was intentionally used overnight; remove it only after recording the research-extension budget tomorrow.

The actual first experiment command, after the split-generation command completes and the research-extension budget is recorded, is the same command without `--dry-run`:

    python scripts/partial_ft_adapt.py --checkpoint runs/current_valid_baseline/training_convergence_100k_5ep_20260916_final_controlled/uacp_predictor_100k_5ep_epoch_3.pt --trainable-scope last_block_plus_head --adapt-train-data runs/current_valid_baseline/overnight_20260920_adaptation_data/train.npz --adapt-val-data runs/current_valid_baseline/overnight_20260920_adaptation_data/validation.npz --epochs 1 --batch-size 8 --lr 1e-4 --seed 20260921 --output-dir runs/current_valid_baseline/partial_ft_20260921_last_block_plus_head

## 8. Dry-run results

All four scopes used the same starting checkpoint hash `d3c864788ee7e0e58bbe9f683aa47615039a470258ff054ec2828179b1f23986`, full-feedback mask, and one optimizer step. Each produced finite loss, intended gradients, and unchanged frozen parameters on GB10/cuda:0. Results:

- `runs/current_valid_baseline/overnight_20260920_partial_dryrun_head_only/`
- `runs/current_valid_baseline/overnight_20260920_partial_dryrun_last_block_plus_head/`
- `runs/current_valid_baseline/overnight_20260920_partial_dryrun_last_8_blocks_plus_head/`
- `runs/current_valid_baseline/overnight_20260920_partial_dryrun_full/`

These are smoke results only and must not be interpreted as research performance.

## 9. First experiment tomorrow

After generating the new split and recording the final budget choice, run the one-variable `last_block_plus_head` vs `full` comparison under identical conditions. Do not use existing 120 ns evaluation samples as adaptation training data.
