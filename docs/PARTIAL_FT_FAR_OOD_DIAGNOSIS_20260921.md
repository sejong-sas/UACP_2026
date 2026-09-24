# Far-OOD Epistemic Collapse Diagnosis after 25% Partial FT

## One-line conclusion

The 25% Far-OOD collapse is primarily caused by **late residual feature drift from ResidualBlock 24–31**, not by an evidential-head-only parameter change. The drift changes the pooled evidence, especially increasing `kappa`, while also increasing `nu - 2K - 1` and slightly lowering `Psi`.

This work is a **RESEARCH-EXTENSION / IMPLEMENTATION-ASSUMPTION**. The paper does not specify these Partial layer scopes, adaptation split size, optimizer budget, or component-swap diagnostics.

## Audit and fixed protocol

The repository, README, active pipeline, Partial FT scripts, model/evidential implementation, and prior result documents were audited before analysis. The common starting checkpoint remained:

`runs/current_valid_baseline/training_convergence_100k_5ep_20260916_final_controlled/uacp_predictor_100k_5ep_epoch_3.pt`

Its SHA-256 remained `d3c864788ee7e0e58bbe9f683aa47615039a470258ff054ec2828179b1f23986`.

All diagnosis and confirmation used the existing leakage-checked splits and the established sparse evaluator: Ng16/32, fixed canonical mask offset 0, deterministic 15 dB complex AWGN keyed by `seed + Ng*1000`, sparse CFR plus mask input, and clean full-CFR target. No existing checkpoint/result was overwritten. The diagnosis ran on NVIDIA GB10 `cuda:0`.

The implementation audit found:

- `Psi` is the diagonal covariance approximation (`diagonal Psi`).
- `kappa` and `nu` are produced by pooled heads.
- The current output applies `aleatoric = Psi / (nu - 2K - 1)` and `epistemic = aleatoric / kappa`, matching the current Eq.(7)/(8) implementation.
- Final uncertainty is aggregated over omitted subcarriers using the current Eq.(12)/(13)-style score path.

## Distribution-level reproduction

The 25% 1 ms distribution shifted toward ID, not merely its mean. Median and selected quantiles of omitted Epistemic were:

| Ng | Model | p01 | p10 | median | p90 | p99 | fraction < 1 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 16 | Pre | 0.141 | 2.747 | 147.18 | 22,615 | 602,540 | 5.6% |
| 16 | 3.16% | 0.122 | 2.093 | 106.50 | 15,929 | 486,652 | 6.6% |
| 16 | 25% | 0.0276 | 0.0849 | 0.841 | 22.46 | 413.7 | 52.4% |
| 16 | Full | 0.0560 | 0.238 | 4.069 | 195.7 | 6,436 | 26.3% |
| 32 | Pre | 4.252 | 57.97 | 2,361.8 | 196,970 | 51.55e6 | 0.3% |
| 32 | 3.16% | 3.162 | 43.44 | 1,695.8 | 124,054 | 44.50e6 | 0.3% |
| 32 | 25% | 0.152 | 0.605 | 5.572 | 90.02 | 1,756 | 16.2% |
| 32 | Full | 0.310 | 1.376 | 21.28 | 531.3 | 19,120 | 7.3% |

The ID-vs-1 ms Epistemic AUROC remained numerically high because the 1 ms distribution still generally exceeds ID, but it degraded slightly for 25%: Ng16 `0.9999895`, Ng32 `0.9999615`, versus Pre `1.0/0.9999995`. Thus AUROC alone hides the severe loss of uncertainty magnitude and distribution separation.

## Eq.(7)/(8) decomposition

At 1 ms, mean omitted statistics were:

| Ng | Model | Psi | nu margin | kappa | Aleatoric | Epistemic |
|---:|---|---:|---:|---:|---:|---:|
| 16 | Pre | 1.766 | 104.37 | 0.2079 | 0.1759 | 32,985.8 |
| 16 | 25% | 1.715 | 150.03 | 2.4236 | 0.0945 | 22.10 |
| 16 | Full | 1.869 | 128.58 | 1.0784 | 0.1242 | 353.28 |
| 32 | Pre | 2.501 | 61.20 | 0.01840 | 3.6474 | 3.0569e6 |
| 32 | 25% | 2.307 | 114.46 | 0.6590 | 0.1679 | 82.30 |
| 32 | Full | 2.456 | 99.14 | 0.3205 | 0.2157 | 1,195.07 |

Using per-sample median ratios of 25% post/pre at 1 ms:

| Ng | Psi ratio | nu-margin ratio | kappa ratio | Aleatoric ratio | Epistemic ratio |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.977 | 1.416 | 120.47 | 0.689 | 0.00565 |
| 32 | 0.933 | 1.846 | 213.50 | 0.506 | 0.00226 |

Therefore the direct multiplicative driver is the large `kappa` increase. The larger `nu` margin further lowers Aleatoric uncertainty, while `Psi` is not the primary driver: it is nearly unchanged at Ng16 and only modestly lower at Ng32. This is a feature-conditioned confidence change, not an evidential-head-only shift.

## Component-swap diagnosis

All swaps were in-memory and evaluated on the same samples, masks, and noise schedule.

| Hybrid | 1 ms Epi mean Ng16 | 1 ms Epi median Ng16 | 1 ms Epi mean Ng32 | 1 ms Epi median Ng32 |
|---|---:|---:|---:|---:|
| Pre | 32,985.8 | 147.18 | 3.0569e6 | 2,361.8 |
| H-BLOCK: 25% blocks 24–31 only | 23.67 | 0.889 | 84.70 | 5.607 |
| H-HEAD: 25% four heads only | 29,965.8 | 138.31 | 2.8997e6 | 2,320.3 |
| H-GAMMA: gamma head only | 32,985.8 | 147.18 | 3.0569e6 | 2,361.8 |
| H-UNC: psi/kappa/nu heads only | 29,965.8 | 138.31 | 2.8997e6 | 2,320.3 |
| H25: blocks 24–31 + heads | 22.10 | 0.841 | 82.30 | 5.572 |

H-BLOCK reproduces essentially the full 25% collapse, while H-HEAD/H-UNC preserve the Pre-like distribution. H-GAMMA has no uncertainty effect, as expected. This supports **Case A: late residual feature drift is the primary cause**. The heads map the adapted feature into larger pooled evidence; replacing heads alone is insufficient to cause the collapse.

## Representation and parameter drift

Block 23 is frozen and had exactly zero drift. Median activation drift from Pre to 25% at block 31/head input was:

| Ng | 20 ns relative L2 / cosine | 120 ns relative L2 / cosine | 1 ms relative L2 / cosine |
|---:|---:|---:|---:|
| 16 | 0.256 / 0.967 | 0.210 / 0.978 | 0.302 / 0.954 |
| 32 | 0.299 / 0.955 | 0.274 / 0.962 | 0.342 / 0.940 |

The 1 ms representation moved farther than 20/120 ns, especially at Ng32. Relative parameter drift for the 25% trainable components was largest in blocks 24–27 (`6.92%`, `5.91%`, `5.37%`, `5.12%`), followed by blocks 28–31 (`4.85%`, `4.23%`, `3.56%`, `4.14%`). Head parameter drift was smaller: gamma `1.85%`, Psi `2.57%`, kappa `0.95%`, nu `0.56%`.

## Confirmatory training

Based on H-BLOCK, exactly one confirmatory training was selected: `last_4_blocks_plus_head` = ResidualBlock 28–31 plus all four heads. This is a **RESEARCH-EXTENSION / IMPLEMENTATION-ASSUMPTION**. It used the same epoch3 checkpoint, 120 ns split, seed, schedule digest, sparse input, clean target, Adam/LR, batch size, 3 epochs, and evaluator. No other scope or hyperparameter was trained.

Actual trainable parameters were `1,480,728` (`12.5323%`), with 375 optimizer steps, 139.69 s total, 0.3164 s/step, and 184.79 MiB peak allocated VRAM. Smoke passed with 24 intended gradient tensors, finite loss, sparse input, clean target, frozen unchanged, and GB10/cuda:0.

| Ng | Model | 120 ns NMSE improvement | Retention vs Full | 120 ns Epi change | 20 ns ΔNMSE | 80 ns ΔNMSE | 1 ms/120 ns Epi |
|---:|---|---:|---:|---:|---:|---:|---:|
| 16 | 3.16% | 1.243 | 87.7% | -0.000130 | +1.365 | -0.076 | 3.35e6 |
| 16 | 25% | 1.521 | 107.4% | -0.000469 | +2.472 | +0.027 | 2.79e3 |
| 16 | Full | 1.417 | 100.0% | -0.000633 | +2.771 | +0.195 | 4.55e4 |
| 16 | Confirmatory | 1.426 | 100.7% | -0.000305 | +1.925 | -0.053 | 6.12e4 |
| 32 | 3.16% | 0.918 | 88.9% | -0.006678 | +1.309 | -0.113 | 4.84e7 |
| 32 | 25% | 1.129 | 109.3% | -0.026232 | +2.253 | +0.069 | 3.05e3 |
| 32 | Full | 1.033 | 100.0% | -0.026831 | +2.514 | +0.226 | 4.54e4 |
| 32 | Confirmatory | 1.105 | 107.0% | -0.015299 | +1.730 | -0.065 | 3.57e4 |

Confirmatory ID-vs-1 ms Epistemic AUROC was `0.999993` (Ng16) and `0.9999715` (Ng32). The 1 ms Epistemic/120 ns Epistemic ratio recovered from 25% values `2.79e3/3.05e3` to `6.12e4/3.57e4`, close to or within the Full scale, while preserving smaller ID forgetting than Full. Reconstruction remained Full-like.

## Final answers to the five questions

1. **Why did 1 ms Epistemic decrease?** Adapted late residual blocks moved 1 ms representations toward a region that the pooled evidential heads interpreted as high-confidence evidence.
2. **Which Eq.(7)/(8) quantity directly caused it?** `kappa` was dominant, increasing by median factors about `120x/214x`; `nu` margin amplified the reduction; `Psi` was secondary.
3. **Which component caused it?** H-BLOCK reproduced collapse and H-HEAD did not: late residual blocks, with feature-to-head interaction, not heads alone.
4. **Why can 120 ns improve while 1 ms worsens?** The same late feature adaptation improves reconstruction and uncertainty for the trained 120 ns regime but broadens the high-confidence feature region into a much farther OOD regime, producing uncertainty-boundary forgetting.
5. **Next scope principle:** keep the adaptive block range below the point where 1 ms feature drift becomes large; the confirmatory result supports `last_4_blocks_plus_head` as the next candidate, subject to independent future validation.

## Artifacts

- Diagnosis: `runs/current_valid_baseline/partial_ft_far_ood_diagnosis_20260921/`
- Confirmatory checkpoint: `runs/current_valid_baseline/partial_ft_far_ood_confirmatory_last4_20260921_3ep/`
- Confirmatory evaluation: `runs/current_valid_baseline/partial_ft_far_ood_confirmatory_last4_20260921_evaluation/`
- Diagnosis script: `scripts/diagnose_partial_ft_far_ood.py`
- Summary script: `scripts/summarize_partial_ft_far_ood.py`
- Evaluator update: `scripts/evaluate_partial_ft_25pct.py`
- Scope support/test update: `scripts/partial_ft_adapt.py`, `tests/test_partial_ft_readiness.py`
