# Feature-Anchored Partial Fine-Tuning — 2026-09-22

## Final verdict

**CASE 3 — Not supported.** With the pre-specified normalized ID feature anchor (`lambda_anchor=1.0`), last4 adaptation retained 120 ns reconstruction performance but did not reduce the long-duration Far-OOD uncertainty collapse. Therefore feature anchoring alone is not sufficient under this controlled setting.

This is a `RESEARCH-EXTENSION / IMPLEMENTATION-ASSUMPTION`, not a reproduction of an unpublished Partial FT policy. The paper-aligned part is the full-feedback OOD adaptation flow; the sparse-input/clean-target adaptation construction, last4 scope, 10 epochs, three seeds, ID subset, and anchor coefficient are implementation assumptions.

## Question and protocol

The fixed scope was ResidualBlock 28–31 plus `gamma_head`, `psi_head`, `kappa_head`, and `nu_head`: 1,480,728 trainable parameters (12.5323%). The student started independently from the unchanged Pre epoch-3 checkpoint for seeds 20260921, 20260922, and 20260923. All runs used the same 120 ns adaptation train/validation data, sparse CFR + mask + 15 dB complex AWGN input, clean full-CFR target, Adam, batch 8, LR 1e-4, FP32, `lambda_reg=1e-3`, and 10 epochs. Checkpoints at epochs 1/3/5/10 were preserved.

The teacher was a frozen copy of Pre. At each step, a separate ID-reference batch from the original 10–100 ns training distribution was used. The anchor was applied once at the final ResidualBlock31 output, which is the evidential-head input in this architecture:

`mean(||z_student-z_teacher||² / (||z_teacher||² + 1e-8))`, with `lambda_anchor=1.0`.

The ID subset contains 1,000 samples from the original training-distribution source and has zero CFR-hash overlap with adaptation train, validation, 120 ns test, and 1 ms test. The 1 ms set was not used for training, anchor construction, lambda selection, or model selection.

## Reconstruction and ID retention

At epoch 10, anchored-last4 120 ns omitted-NMSE improvement was:

| Ng | improvement, mean ± std (dB) | Full retention |
|---:|---:|---:|
| 16 | 1.5124 ± 0.0074 | 100.6% ± 2.8% |
| 32 | 1.1455 ± 0.0090 | 101.2% ± 2.7% |

Thus the pre-specified ≥90% reconstruction gate passed. However, ID forgetting was not improved: 20 ns ΔNMSE was +2.390 ± 0.042 dB for Ng16 and +2.150 ± 0.030 dB for Ng32; 80 ns ΔNMSE was −0.0147 ± 0.0236 dB and −0.0012 ± 0.0220 dB, respectively. The 20 ns degradation is consistent with the unregularized last4 long-run behavior and is not evidence that the anchor preserved all ID behavior.

## Far-OOD uncertainty

The epoch-10 anchored-last4 1 ms Epistemic median / Pre median ratios were:

| Ng | anchored ratio | unregularized last4 ratio | required practical gate |
|---:|---:|---:|---:|
| 16 | 0.0171 | 0.0171 | ≥0.17 |
| 32 | 0.0087 | 0.0086 | ≥0.18 |

Both preservation gates failed. The absolute 1 ms Epistemic mean remained very large relative to ID because the distribution is heavy-tailed, but the median and parameter trajectory show that most samples moved toward the low-confidence ID-like branch. AUROC therefore remained misleadingly high: ID-vs-1 ms AUROC was 0.999983 ± 0.0000003 (Ng16) and 0.999902 ± 0.000015 (Ng32). ID-vs-120 ns AUROC was 0.8568 ± 0.0068 and 0.8638 ± 0.0026. Absolute calibration and ID-to-Far-OOD separation, not AUROC alone, determine the preservation result.

## Mechanism and representation drift

At 1 ms, epoch-10 median ratios relative to Pre were:

| Ng | Ψ | ν-margin | κ | Epistemic |
|---:|---:|---:|---:|---:|
| 16 | 0.974 | 1.322 | 44.0 | 0.0171 |
| 32 | 0.956 | 1.668 | 67.2 | 0.0087 |

The direct mechanism remains κ inflation, with ν-margin growth contributing and Ψ only a secondary factor. The ratios were already moving in the wrong direction at epoch 1 and worsened through epochs 3, 5, and 10, so the failure is not an epoch-3-only transient.

Anchoring also did not materially reduce final head-input drift. The normalized drift trajectory (Ng16 / Ng32) was approximately 0.176 / 0.206 at epoch 1, 0.216 / 0.256 at epoch 3, 0.237 / 0.279 at epoch 5, and 0.267 / 0.313 at epoch 10. These values match the prior unregularized last4 trajectory within the observed seed variation. The final ResidualBlock31 output and evidential-head input are the same tensor here, so they are reported once.

## Efficiency and integrity

Anchored-last4 used 1,480,728 parameters (12.5323%), 0.2364 ± 0.0079 seconds/step, 311.6 MiB peak allocated VRAM, and 316.5 ± 10.5 seconds total training time across seeds. The comparison remains substantially smaller than Full (11,815,320 parameters, 100%; prior final-validation mean about 757 s and 816 MiB), but the anchor did not buy Far-OOD robustness.

All training/evaluation runs used NVIDIA GB10 on `cuda:0` and FP32. Smoke tests verified finite loss, intended gradients, frozen parameters, sparse input, clean target, and the frozen teacher. The Pre checkpoint SHA-256 remained:

`d3c864788ee7e0e58bbe9f683aa47615039a470258ff054ec2828179b1f23986`.

Existing checkpoints/results were reused read-only; new outputs were written under `runs/current_valid_baseline/partial_ft_uncertainty_preserving_feature_anchor_20260922/`.

## Limitations and one next step

The anchor subset, coefficient, and adaptation duration are not paper-specified. The result tests one fixed normalized feature anchor only; it does not rule out all ID-preserving objectives. The next single experiment should therefore be a **frozen-uncertainty-head controlled run**: keep ResidualBlock 28–31 and the same sparse adaptation protocol, but freeze `psi_head`, `kappa_head`, and `nu_head` at Pre while adapting only the reconstruction path and `gamma_head`; use the same seeds and evaluation grid. No such follow-up was run here.

## Artifacts

- Training: `runs/current_valid_baseline/partial_ft_uncertainty_preserving_feature_anchor_20260922/train_seed_20260921/`, `train_seed_20260922/`, `train_seed_20260923/`
- Evaluation: `runs/current_valid_baseline/partial_ft_uncertainty_preserving_feature_anchor_20260922/evaluation_retry1/`
- AUROC supplement: `runs/current_valid_baseline/partial_ft_uncertainty_preserving_feature_anchor_20260922/auroc_retry1/`
- Tables/plots: `runs/current_valid_baseline/partial_ft_uncertainty_preserving_feature_anchor_20260922/`
- Implementation: `scripts/partial_ft_feature_anchor.py`, `scripts/prepare_id_anchor_data.py`, `scripts/evaluate_feature_anchor.py`, `scripts/evaluate_feature_anchor_auroc.py`, `scripts/summarize_feature_anchor.py`
