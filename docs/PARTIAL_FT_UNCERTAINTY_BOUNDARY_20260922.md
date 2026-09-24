# Boundary-Distilled Partial Fine-Tuning — 2026-09-22

## Final judgment

**CASE 3 — Not supported.** The boundary-distillation objective preserved 120 ns reconstruction, but it did not produce a stable uncertainty boundary. It drove κ downward and ν-margin toward zero for unseen conditions; Ng32 produced non-finite Aleatoric/Epistemic values at 1 ms. The method therefore cannot be accepted as an uncertainty-preserving Partial FT solution in the current implementation.

This is a `RESEARCH-EXTENSION / IMPLEMENTATION-ASSUMPTION`, not a paper reproduction. The paper-aligned element is the OOD full-feedback adaptation flow. The fixed last4 scope, auxiliary losses, proxy-OOD range, data counts, 10 epochs, seed set, and normalization policy are implementation assumptions.

## Fixed hypothesis and data roles

The student was fixed to ResidualBlock 28–31 plus the four evidential heads: 1,480,728 trainable parameters (12.5323%). The Pre epoch-3 checkpoint was a frozen teacher. Every optimizer step used three independent batches:

1. 120 ns sparse CFR + mask + 15 dB complex AWGN with clean full-CFR reconstruction target;
2. an independent 10–100 ns ID subset for uncertainty distillation;
3. a proxy-OOD split with delay spread Uniform(160,500) ns for uncertainty-only boundary rehearsal.

The proxy split has 1,000 samples, delay range 160.09–499.19 ns, and zero sample-hash overlap with adaptation train/validation, 20 ns, 80 ns, 120 ns, and 1 ms files. Proxy samples never contributed reconstruction target loss. The 1 ms file was not loaded by training, coefficient selection, or model selection; it was used only in final evaluation.

The fixed auxiliary losses were:

`L_ID = MSE(log(Epi_s+eps), log(Epi_t+eps)) + MSE(log(Ale_s+eps), log(Ale_t+eps))`

`L_boundary = mean(ReLU(log(Epi_t+eps) - log(Epi_s+eps))²)`

Each auxiliary term was normalized by its detached first-batch magnitude and used with fixed `lambda_ID=lambda_boundary=1.0`. Because teacher and student are initially identical, the one-sided boundary term can be exactly zero; its scale then falls back to the ID auxiliary scale. This fallback is explicitly an implementation assumption and was not tuned after evaluation.

## Reconstruction and ID behavior

At epoch 10, 120 ns omitted-NMSE improvement and Full retention were:

| Ng | improvement (mean ± std, dB) | Full retention |
|---:|---:|---:|
| 16 | 1.491 ± 0.014 | 99.1% |
| 32 | 1.131 ± 0.013 | 99.9% |

The ≥90% reconstruction requirement passed. ID forgetting, expressed as post-minus-pre NMSE, was approximately:

| Ng | 20 ns ΔNMSE | 80 ns ΔNMSE |
|---:|---:|---:|
| 16 | +1.969 ± 0.138 dB | −0.056 ± 0.023 dB |
| 32 | +1.914 ± 0.089 dB | −0.032 ± 0.021 dB |

ID uncertainty remained near the Pre reference for Ng16/32, with epoch-10 Epistemic median ratios around 0.986/0.966 at 20 ns and 0.997/0.983 at 80 ns. Thus the ID distillation branch itself did not cause a large ID uncertainty shift.

## Blind 1 ms result and failure mechanism

At 1 ms, Ng16 epoch-10 Epistemic median/Pre median was **8.85**, not a calibrated preservation near the Pre boundary. For Ng32, Aleatoric and Epistemic were non-finite at epoch 10 for all three seeds. Raw inspection showed finite Ψ, ν, and κ, but `nu_margin = nu - 2K - 1` reached exactly zero for some samples, causing division by zero in Eq.(7)/(8). The NaN counts were recorded rather than clipped or hidden.

The mechanism trajectory at 1 ms was:

| Ng | epoch | Ψ ratio | ν-margin ratio | κ ratio | Epistemic ratio |
|---:|---:|---:|---:|---:|---:|
| 16 | 1 | 1.014 | 0.864 | 0.252 | 5.28 |
| 16 | 3 | 1.012 | 0.797 | 0.182 | 9.86 |
| 16 | 5 | 1.028 | 0.717 | 0.128 | non-finite in aggregate |
| 16 | 10 | 1.001 | 0.737 | 0.168 | 8.85 |
| 32 | 1 | 1.017 | 0.778 | 0.287 | non-finite in aggregate |
| 32 | 3 | 1.003 | 0.658 | 0.173 | non-finite in aggregate |
| 32 | 5 | 1.039 | 0.457 | 0.102 | non-finite in aggregate |
| 32 | 10 | 0.996 | 0.573 | 0.175 | non-finite in aggregate |

This is the opposite failure mode from unregularized last4: κ inflation was suppressed too aggressively, while ν-margin was pushed toward its numerical boundary. The boundary loss prevented confidence collapse by producing excessive/unstable uncertainty rather than a well-calibrated Far-OOD boundary.

AUROC was not sufficient as a success criterion. Boundary-distilled ID-vs-1 ms AUROC was numerically 1.0 for Ng16 and approximately 1.0 for Ng32 despite the non-finite/over-dispersed uncertainty, demonstrating that ranking does not certify finite calibration.

## Confidence-boundary curve

The interpretation-only curve used delay spreads 20/40/60/80/100/120/160/250/500 ns and the untouched 1 ms test at 1000 ns. New non-1 ms curve points used 200 samples per delay; 1 ms used the fixed 1,000-sample blind test. At epoch 10, median Epistemic increased monotonically in the proxy range:

| delay | Ng16 median | Ng32 median |
|---:|---:|---:|
| 20 ns | 0.00034 | 0.00092 |
| 80 ns | 0.00061 | 0.00213 |
| 100 ns | 0.00076 | 0.00308 |
| 120 ns | 0.00095 | 0.00467 |
| 160 ns | 0.00155 | 0.01055 |
| 250 ns | 0.00416 | 0.04985 |
| 500 ns | 0.275 | 2.108 |
| 1000 ns | 149.3 | 4,492.9 |

The desired qualitative direction—uncertainty increasing outside 120 ns—was obtained, but the curve is excessively steep and not numerically stable across the full Ng grid. It is therefore evidence of boundary separation, not evidence of successful calibrated uncertainty preservation.

## Representation drift and efficiency

The method did not directly anchor features. Representation drift was therefore retained as a diagnostic; the primary failure was already visible in evidential parameters. At epoch 10, normalized head-input drift remained nonzero and increased with adaptation. The boundary objective changed the uncertainty head outputs more strongly than it constrained the late feature representation.

All three seeds used CUDA `cuda:0`, NVIDIA GB10, FP32. Training used 1,480,728 parameters, about `482.7`, `483.6`, and `536.0` seconds for the three seeds, `0.369–0.410 sec/step`, and `397.9 MiB` peak allocated VRAM. Existing Pre/Partial/Full artifacts were reused read-only.

## Integrity and limitations

- Pre SHA-256 unchanged: `d3c864788ee7e0e58bbe9f683aa47615039a470258ff054ec2828179b1f23986`.
- Teacher parameters had `requires_grad=False`; smoke tests showed intended gradients and frozen tensors unchanged.
- Proxy/data split hash overlap was zero.
- Sparse input → clean full-CFR target was retained for 120 ns.
- Proxy-OOD had no reconstruction target loss.
- 1 ms was blind until final evaluation.
- All non-finite 1 ms values were retained and reported.

The main limitation is that the fixed one-sided log-Epistemic penalty does not control the lower bound of ν-margin or the minimum κ. Its apparent success in raising Far-OOD uncertainty can therefore become an unstable over-uncertainty solution.

## One next research step

Run exactly one controlled follow-up that freezes `psi_head`, `kappa_head`, and `nu_head` at Pre while adapting the same last4 reconstruction path and `gamma_head`, with the same data, seeds, schedule, and evaluation grid. Do not add another proxy range or tune auxiliary weights in that run.

## Artifacts

- Training/checkpoints: `runs/current_valid_baseline/partial_ft_uncertainty_boundary_distillation_20260922/train_seed_20260921/`, `train_seed_20260922/`, `train_seed_20260923/`
- Fixed-grid evaluation: `runs/current_valid_baseline/partial_ft_uncertainty_boundary_distillation_20260922/evaluation_retry1/`
- Boundary curve: `runs/current_valid_baseline/partial_ft_uncertainty_boundary_distillation_20260922/boundary_curve_retry2/`
- Tables/plots: `runs/current_valid_baseline/partial_ft_uncertainty_boundary_distillation_20260922/`
- Scripts: `scripts/prepare_proxy_ood_boundary_data.py`, `scripts/partial_ft_boundary_distill.py`, `scripts/evaluate_boundary_distillation.py`, `scripts/evaluate_boundary_curve.py`, `scripts/summarize_boundary_distillation.py`
