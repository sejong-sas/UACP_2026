# Controlled Partial Fine-Tuning: `last_8_blocks_plus_head` (2026-09-21)

## Scope and status

This is a single-variable controlled research extension. The question is whether expanding the trainable scope from the existing 3.16% scope to approximately 25% approaches Full FT adaptation while retaining lower cost and lower ID forgetting. Partial-layer selection, the 120 ns adaptation split, 3 epochs, batch size 8, Adam learning rate `1e-4`, seed `20260921`, and `lambda_reg=1e-3` are **RESEARCH-EXTENSION / IMPLEMENTATION-ASSUMPTION**, not paper settings. The paper only motivates OOD detection followed by full feedback during adaptation; it does not specify this layer policy or budget.

The 25% run was the only new training run. The prior 3.16% and Full results were reused after audit.

## Audit and protocol

- Starting checkpoint: `runs/current_valid_baseline/training_convergence_100k_5ep_20260916_final_controlled/uacp_predictor_100k_5ep_epoch_3.pt`
- Starting checkpoint SHA-256: `d3c864788ee7e0e58bbe9f683aa47615039a470258ff054ec2828179b1f23986`
- Adaptation data: existing leakage-checked 120 ns train/validation split, 1,000/200 samples.
- Untouched evaluation data: existing 20 ns, 80 ns, 120 ns, and 1 ms splits, 1,000 samples each.
- Cross-split duplicate count: 0; existing 120 ns test was not reused for adaptation.
- Same schedule: seed `20260921`, Ng candidates `{4,8,16,32}`, batch 8, 3 epochs/375 steps, deterministic mask/noise schedule digest `43f88e0e797f62b220a31a5fe0dd8ea99a42d956e981ee4cf242a570f60bdcb5`.
- Same evaluation: sparse input, fixed canonical Ng16/Ng32 masks, deterministic 15 dB complex AWGN, clean full CFR target, FP32, `cuda:0` on NVIDIA GB10.
- Adaptation input was verified as sparse observed CFR; full feedback was used only to provide the clean target. No full-CFR-to-full-CFR identity training was used.

## Actual trainable scope

`last_8_blocks_plus_head` means `ResidualBlock 24–31` plus `gamma_head`, `psi_head`, `kappa_head`, and `nu_head`.

| Scope | Trainable parameters | Trainable % | Optimizer state (FP32) |
|---|---:|---:|---:|
| Existing Partial | 373,656 | 3.1625% | 2.85 MiB |
| New Partial | 2,956,824 | 25.0253% | 22.56 MiB |
| Full | 11,815,320 | 100% | 90.14 MiB |

Smoke passed before training: 40 intended trainable tensors had gradients, frozen tensors were unchanged, frozen parameters were excluded from the optimizer, input was sparse, target was clean full CFR, loss was finite, and GB10/cuda:0 was used. In the completed run, comparing epoch-3 state to the starting checkpoint gave frozen-tensor maximum absolute difference `0.0`.

## Training result

The new 25% run completed 3 epochs and 375 optimizer steps with finite losses:

| Epoch | Train loss | Validation loss |
|---:|---:|---:|
| 1 | -2437.4136 | -2520.8029 |
| 2 | -2513.7102 | -2461.9406 |
| 3 | -2499.9045 | -2491.9308 |

Runtime was 149.34 s total, 0.3431 s/step mean, 21.84 samples/s, and 275.55 MiB peak allocated VRAM.

## Epoch-3 omitted-NMSE (dB)

| Ng | Model | 20 ns | 80 ns | 120 ns | 1 ms |
|---:|---|---:|---:|---:|---:|
| 16 | Pre | -20.9567 | -18.7954 | -16.8910 | 1.6995 |
| 16 | 3.16% | -19.5916 | -18.8713 | -18.1338 | 1.9763 |
| 16 | 25% | -18.4849 | -18.7680 | -18.4120 | 1.9742 |
| 16 | Full | -18.1859 | -18.6005 | -18.3077 | 2.0330 |
| 32 | Pre | -18.1301 | -15.9997 | -14.2149 | 1.8416 |
| 32 | 3.16% | -16.8207 | -16.1124 | -15.1330 | 2.0880 |
| 32 | 25% | -15.8773 | -15.9308 | -15.3437 | 2.1423 |
| 32 | Full | -15.6162 | -15.7736 | -15.2478 | 2.1653 |

## Epoch-3 epistemic uncertainty

| Ng | Model | 20 ns | 80 ns | 120 ns | 1 ms |
|---:|---|---:|---:|---:|---:|
| 16 | Pre | 0.003150 | 0.005356 | 0.008393 | 32985.8 |
| 16 | 3.16% | 0.003221 | 0.005424 | 0.008263 | 27697.6 |
| 16 | 25% | 0.004985 | 0.006298 | 0.007924 | 22.10 |
| 16 | Full | 0.005245 | 0.006154 | 0.007759 | 353.28 |
| 32 | Pre | 0.010469 | 0.022117 | 0.053175 | 3.0569e6 |
| 32 | 3.16% | 0.010014 | 0.020909 | 0.046497 | 2.2521e6 |
| 32 | 25% | 0.014438 | 0.019514 | 0.026943 | 82.30 |
| 32 | Full | 0.016572 | 0.019284 | 0.026344 | 1195.07 |

## Adaptation and forgetting

Definitions: `120 NMSE improvement = NMSE_pre - NMSE_post`; ID forgetting is `NMSE_post - NMSE_pre`; epistemic change is `Epi_post - Epi_pre`. Values below are per Ng; no aggregate score is used.

| Ng | Metric | 3.16% | 25% | Full |
|---:|---|---:|---:|---:|
| 16 | 120 ns NMSE improvement | 1.2428 | 1.5210 | 1.4167 |
| 16 | 120 ns Epi change | -0.000130 | -0.000469 | -0.000633 |
| 16 | 20 ns NMSE change | 1.3651 | 2.4718 | 2.7708 |
| 16 | 80 ns NMSE change | -0.0758 | 0.0275 | 0.1950 |
| 16 | 1 ms Epi change | -5288.2 | -32963.7 | -32632.5 |
| 32 | 120 ns NMSE improvement | 0.9180 | 1.1287 | 1.0329 |
| 32 | 120 ns Epi change | -0.006678 | -0.026232 | -0.026831 |
| 32 | 20 ns NMSE change | 1.3094 | 2.2528 | 2.5139 |
| 32 | 80 ns NMSE change | -0.1127 | 0.0689 | 0.2261 |
| 32 | 1 ms Epi change | -804748.7 | -3056779.5 | -3055666.8 |

The 25% NMSE retention relative to Full is `1.074` (Ng16) and `1.093` (Ng32), compared with the existing 3.16% values `0.877` and `0.889`. Thus reconstruction adaptation moved at least as close to Full as measured by this fixed definition. However, 25% uncertainty adaptation also nearly matched Full at 120 ns while collapsing the 1 ms epistemic signal: post-adaptation `Epi(1ms)/Epi(120ns)` was about `2.79e3` (Ng16) and `3.05e3` (Ng32), versus Full `4.55e4` and `4.54e4`, and 3.16% `3.35e6` and `4.84e7`.

## Efficiency

| Metric | 3.16% | 25% | Full |
|---|---:|---:|---:|
| Trainable parameters | 373,656 | 2,956,824 | 11,815,320 |
| Trainable % | 3.1625% | 25.0253% | 100% |
| Training time | 137.32 s | 149.34 s | 285.94 s |
| Seconds/step | 0.3035 | 0.3431 | 0.6982 |
| Peak VRAM | 117.84 MiB | 275.55 MiB | 816.16 MiB |

Relative to Full, the 25% run used 25.0253% of parameters, 52.23% of wall time, 49.14% of seconds/step, and 33.76% of peak allocated VRAM. The prior 3.16% run used 48.02% of Full wall time and 14.44% of Full peak VRAM.

## Judgment

**Partial support, with an important Far-OOD limitation.** Expanding to 25% clearly improved reconstruction adaptation relative to 3.16% and reached Full-like 120 ns epistemic reduction. It remained substantially cheaper than Full and had smaller 20/80 ns NMSE forgetting than Full. It did not preserve the Far-OOD epistemic separation: 1 ms uncertainty collapsed by roughly three orders of magnitude relative to the adapted 120 ns uncertainty. Therefore 25% is a strong reconstruction/near-OOD adaptation candidate, but it is not a uniformly better trade-off for uncertainty-preserving adaptation. No additional scope, LR, epoch, controller, threshold, or normalization experiment was run.

## Artifacts

- Training: `runs/current_valid_baseline/partial_ft_20260921_sparse_25pct_3ep/`
- Evaluation: `runs/current_valid_baseline/partial_ft_20260921_sparse_25pct_evaluation/`
- Prior comparison reused: `runs/current_valid_baseline/partial_ft_20260921_sparse_evaluation/`
- New evaluator wrapper: `scripts/evaluate_partial_ft_25pct.py`
