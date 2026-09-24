# UACP Partial FT Final Confirmatory Validation (2026-09-21)

## Final conclusion

The confirmatory hypothesis is **not supported**. `last_4_blocks_plus_head` (12.5323%) retained the expected 120 ns reconstruction benefit in the fixed 10-epoch runs, but its Far-OOD uncertainty preservation was not robust: 1 ms `kappa` inflation and `nu-margin` growth accumulated with duration, and the 1 ms Epistemic median fell to 1.7% (Ng16) / 0.9% (Ng32) of Pre by epoch10. Therefore it is not adopted as a final operating point under the present dataset, model, and adaptation schedule.

This is a **research-extension validation**, not a reproduction of a paper-specified Partial FT policy.

## Question and fixed protocol

The single question was whether ResidualBlock 28–31 plus the four evidential heads can preserve Full-like 120 ns adaptation while reducing ID forgetting and compute cost. The only compared scopes were:

| Scope | Trainable parameters | Percent |
|---|---:|---:|
| `last_block_plus_head` | 373,656 | 3.1625% |
| `last_4_blocks_plus_head` | 1,480,728 | 12.5323% |
| `last_8_blocks_plus_head` | 2,956,824 | 25.0253% |
| `full` | 11,815,320 | 100% |

Three fresh training seeds (20260921, 20260922, 20260923) were run for 10 epochs from the same epoch3 checkpoint. Training used the same 120 ns adapt-train/validation split, sparse CFR input, 15 dB complex AWGN, mask channel, clean full-CFR target, Adam, LR 1e-4, batch 8, FP32, and λreg=1e-3. Epochs 1/3/5/10 were evaluated; epoch10 was fixed as the primary endpoint.

All scope policies, 120 ns adaptation, sample counts, 10 epochs, LR, and seed plan are `RESEARCH-EXTENSION / IMPLEMENTATION-ASSUMPTION`; they are not paper settings. Current implementation assumptions retained in the interpretation include diagonal Ψ, the repository AWGN/observation pipeline, and `normalize_channel=false`.

## Data and integrity

The fixed evaluation sets were 20 ns, 80 ns, 120 ns, and 1 ms, each evaluated with Ng 4/8/16/32. Existing provenance recorded zero train/validation/test overlap and no reuse of 120 ns test data for adaptation. The evaluator reused the same ordered evaluation data and deterministic mask/noise construction for every scope and seed. Sparse observation → clean full-CFR target was preserved; no full-CFR → full-CFR identity training was used.

Pre checkpoint SHA-256 remained `d3c864788ee7e0e58bbe9f683aa47615039a470258ff054ec2828179b1f23986`.

## Reconstruction and forgetting

Epoch10 seed means are in `final_seed_summary.csv`. For Ng16/Ng32, last4 120 ns omitted-NMSE improvements were 1.513±0.003 dB and 1.145±0.007 dB. Full was 1.505±0.048 dB and 1.132±0.039 dB, respectively. Thus the reconstruction retention was approximately 100.6% (Ng16) and 101.2% (Ng32), satisfying the pre-specified ≥90% criterion.

The fixed evaluation also shows that the apparent adaptation gain does not imply uncertainty preservation. At epoch10, last4 20 ns ΔNMSE was −2.390±0.034 dB (Ng16) and −2.152±0.032 dB (Ng32); 80 ns ΔNMSE was +0.017±0.017 dB and +0.000±0.022 dB. These values are not worse than Full for 80 ns, while the 20 ns comparison is affected by the common omitted-NMSE convention and should be read together with the raw per-seed tables.

## Uncertainty and AUROC

At epoch10, 1 ms Epistemic median ratios relative to Pre were:

| Scope | Ng16 | Ng32 |
|---|---:|---:|
| 3.16% | 0.345 | 0.352 |
| 12.53% last4 | 0.017 | 0.009 |
| 25.03% | 0.001 | 0.001 |
| Full | 0.004 | 0.003 |

Thus last4 did not preserve Far-OOD uncertainty under the 10-epoch protocol; it was intermediate between 3.16% and the severe 25%/Full collapse, but still a large collapse. Epoch trajectories show that last4 1 ms Epistemic ratios already fell to 0.106/0.051 at epoch1, 0.038/0.015 at epoch3, 0.028/0.011 at epoch5, and 0.017/0.009 at epoch10 for Ng16/Ng32.

The 1 ms AUROC remained numerically near one because the ID and 1 ms distributions remain highly rank-separated despite the large absolute uncertainty reduction: last4 epoch10 AUROC was 0.999994 (Ng16) and 0.9999005 (Ng32), averaged across the fixed seed-specific evaluations in `auroc_summary_deduplicated.csv`. This demonstrates why AUROC alone is insufficient here; absolute Epistemic calibration and ID-to-1 ms gaps must also be reported.

## Evidential mechanism

The direct mechanism is kappa inflation, with additional nu-margin growth. At epoch10, median ratios to Pre for last4 were:

| Ng | Ψ | ν-margin | κ | Epistemic |
|---:|---:|---:|---:|---:|
| 16, 1 ms | 0.973 | 1.323 | 43.327 | 0.017 |
| 32, 1 ms | 0.957 | 1.668 | 67.492 | 0.009 |

Ψ was a secondary factor. The same ratios grew with epoch, confirming a duration-dependent confidence-boundary drift rather than a single transient epoch3 artifact. The complete raw statistics, including mean/std/median/p10/p90 for Ψ, κ, ν-margin, Aleatoric, Epistemic, and total uncertainty, are in `epoch_trajectory.csv` and `evidential_parameter_trajectory.csv`.

## Representation drift

At epoch10 on 1 ms, normalized activation drift averaged over Ng16/Ng32 was:

| Scope | block23 | block27 | block31 | head input |
|---|---:|---:|---:|---:|
| 3.16% | 0.000 | 0.000 | 0.000 | 0.094 |
| 12.53% | 0.000 | 0.000 | 0.286 | 0.290 |
| 25.03% | 0.000 | 0.323 | 0.411 | 0.413 |
| Full | 0.343 | 0.439 | 0.453 | 0.455 |

This confirms the previously identified late-feature mechanism: expanding the trainable boundary increases 1 ms feature drift, and that drift accompanies κ inflation and Epistemic collapse.

## Efficiency

Mean training time, seconds/step, and peak allocated VRAM over the three seeds were:

| Scope | Time (s) | sec/step | Peak VRAM (MiB) |
|---|---:|---:|---:|
| 3.16% | 403.6±31.7 | 0.270±0.021 | 117.8 |
| 12.53% | 459.9±19.6 | 0.313±0.013 | 184.8 |
| 25.03% | 488.0±21.2 | 0.338±0.015 | 275.6 |
| Full | 757.0±69.9 | 0.556±0.051 | 816.2 |

Last4 reduced mean wall-clock time to about 60.8% of Full and peak VRAM to about 22.6% of Full, with 12.53% of trainable parameters.

## Dynamic controller

`SKIPPED_WITH_REASON`: the repository Fig.11 controller is hard-wired to an older checkpoint and delay-sweep sequence, with a different candidate-Ng/threshold provenance and no direct 20→80→120→1 ms sequence matching this fixed validation. Replacing those inputs would create a new controller experiment rather than reuse the existing pipeline. Static uncertainty evaluation is therefore the primary evidence for this confirmatory validation.

## Final judgement and limitation

**CASE 3 — Not supported.** Last4 met the reconstruction retention target, but Far-OOD preservation failed across the duration trajectory and was not rescued by the three-seed repetition. The earlier 3-epoch last4 result was therefore a transient/short-duration operating point, not a robust final scope under this schedule. This conclusion is limited to the current UACP implementation, dataset, sparse-observation construction, optimizer, LR, and 10-epoch adaptation extension.

Exactly one next research step is recommended: run a single controlled uncertainty-preserving adaptation objective/constraint on the fixed last4 scope, keeping the same data and optimizer budget, rather than beginning another scope sweep.

## Artifacts

- Training checkpoints: `runs/current_valid_baseline/partial_ft_final_validation_20260921_retry1/`
- Evaluation outputs: `runs/current_valid_baseline/partial_ft_final_validation_20260921_analysis_retry5/`
- Failed/restarted evaluator provenance is preserved in `...analysis_retry`, `...analysis_retry2`, `...analysis_retry3`, and `...analysis_retry4`.
- Modified scripts: `scripts/partial_ft_adapt.py`, `scripts/final_partial_ft_validation.py`, `scripts/summarize_final_partial_ft_validation.py`.
