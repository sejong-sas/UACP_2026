# Controlled Partial vs Full Fine-Tuning Result — 2026-09-21

## Experiment label

This is a `RESEARCH-EXTENSION / IMPLEMENTATION-ASSUMPTION`, not a reproduction of a paper-specified Partial Fine-Tuning policy. The paper-aligned part is OOD detection followed by full feedback; the sparse adaptation construction, 1,000/200 split sizes, layer policy, 3 epochs, LR, and seed are implementation choices.

## Protocol audit

- Starting checkpoint for both runs: `runs/current_valid_baseline/training_convergence_100k_5ep_20260916_final_controlled/uacp_predictor_100k_5ep_epoch_3.pt`.
- Starting checkpoint SHA-256: `d3c864788ee7e0e58bbe9f683aa47615039a470258ff054ec2828179b1f23986`.
- GPU: NVIDIA GB10, `cuda:0`; precision: FP32.
- Adaptation input: complete 120 ns CFR is used to obtain the clean target; predictor input is rebuilt as sparse CFR with canonical Ng ∈ {4, 8, 16, 32}, periodic mask, masked 15 dB complex AWGN, zero omitted positions, and mask channel.
- Partial and Full used the same sample order, deterministic mask/noise schedule, seed `20260921`, Adam, LR `1e-4`, λreg `1e-3`, batch size 8, and 3 epochs / 375 steps.
- Schedule SHA-256 for both runs: `43f88e0e797f62b220a31a5fe0dd8ea99a42d956e981ee4cf242a570f60bdcb5`.
- Adaptation split leakage check: 0 cross-split CFR byte-hash duplicates; existing 120 ns evaluation data was not reused for train or validation.
- All 56,000 evaluation sample records were finite.

## Data provenance

Generated data and split provenance are in `runs/current_valid_baseline/overnight_20260920_adaptation_data/`. The split manifest is `provenance_manifest.json`; all split roles are explicitly marked `RESEARCH-EXTENSION / IMPLEMENTATION-ASSUMPTION`.

## Primary performance — epoch 3

Values are omitted-subcarrier NMSE in dB. Improvement is `NMSE_pre - NMSE_post`, so positive is improvement.

| Model | Ng | 20 ns | 80 ns | 120 ns | 1 ms |
|---|---:|---:|---:|---:|---:|
| Pre | 16 | -20.957 | -18.795 | -16.891 | 1.700 |
| Partial | 16 | -19.592 | -18.871 | -18.134 | 1.976 |
| Full | 16 | -18.186 | -18.600 | -18.308 | 2.033 |
| Pre | 32 | -18.130 | -16.000 | -14.215 | 1.842 |
| Partial | 32 | -16.821 | -16.112 | -15.133 | 2.088 |
| Full | 32 | -15.616 | -15.774 | -15.248 | 2.165 |

| Metric | Ng16 Partial | Ng16 Full | Ng32 Partial | Ng32 Full |
|---|---:|---:|---:|---:|
| 120 ns NMSE improvement | 1.243 dB | 1.417 dB | 0.918 dB | 1.033 dB |
| Partial retention vs Full | 87.7% | — | 88.9% | — |
| 120 ns Epistemic change | -0.000130 | -0.000633 | -0.006678 | -0.026831 |

## Uncertainty and forgetting

At 120 ns, Partial reduced Epistemic less than Full. Therefore Partial achieved similar reconstruction adaptation but not the same evidential adaptation magnitude.

| Metric | Partial Ng16 | Full Ng16 | Partial Ng32 | Full Ng32 |
|---|---:|---:|---:|---:|
| 20 ns ΔNMSE (post − pre) | +1.365 dB | +2.771 dB | +1.309 dB | +2.514 dB |
| 80 ns ΔNMSE (post − pre) | -0.076 dB | +0.195 dB | -0.113 dB | +0.226 dB |
| 1 ms / 120 ns Epistemic ratio, pre | 3,930,343 | 3,930,343 | 57,486,604 | 57,486,604 |
| 1 ms / 120 ns Epistemic ratio, post | 3,352,129 | 45,529 | 48,435,804 | 45,364 |

The Far-OOD signal remained much larger after Partial than after Full: epoch3 Epistemic was 27,697.6 vs 353.3 for Ng16 and 2,252,113 vs 1,195.1 for Ng32. This indicates stronger Far-OOD uncertainty preservation under Partial in this run, while also showing that the absolute scale is highly regime-dependent.

## Efficiency

| Metric | Partial 3.1625% | Full 100% |
|---|---:|---:|
| Trainable parameters | 373,656 | 11,815,320 |
| Training time | 137.3 s | 285.9 s |
| Seconds/step | 0.304 | 0.698 |
| Samples/second | 21.85 | 10.49 |
| Peak allocated VRAM | 117.8 MiB | 816.2 MiB |
| Estimated Adam state FP32 | 2.851 MiB | 90.144 MiB |

Partial used 3.16% of parameters, 48.0% of wall-clock time, 43.5% of seconds/step, and 14.4% of peak allocated VRAM relative to Full.

## Scope and checkpoint integrity

- Partial epoch3 frozen tensors: 126; maximum absolute difference from starting checkpoint: exactly 0.0.
- Full changed parameter tensors: 138/138.
- Original starting checkpoint hash after experiment: unchanged; see `partial_ft_20260921_sparse_evaluation/scope_verification.json`.
- Epoch checkpoints are in `runs/current_valid_baseline/partial_ft_20260921_sparse_3ep/` and `runs/current_valid_baseline/full_ft_20260921_sparse_3ep/`.

## Hypothesis judgment

**Partial support.** The 3.16% scope obtained approximately 88–89% of Full's 120 ns NMSE improvement, had less ID forgetting, preserved more Far-OOD Epistemic signal, and was substantially cheaper. However, Full produced a larger 120 ns Epistemic reduction, so the claim cannot be extended to equivalent uncertainty-head adaptation.

No LR, epoch, mask, data, or scope tuning was performed after observing the result. `diagonal Ψ`, current AWGN/observation pipeline, `normalize_channel=false`, adaptation sample count, epoch/LR, and Partial layer policy remain the documented implementation assumptions.

## Result artifacts

- Evaluation summary: `runs/current_valid_baseline/partial_ft_20260921_sparse_evaluation/summary.csv`
- Primary table: `primary_performance_epoch3.csv`
- Uncertainty table: `uncertainty_epoch3.csv`
- Adaptation/forgetting table: `adaptation_forgetting.csv`
- Efficiency table: `efficiency.csv`
- Scope verification: `scope_verification.json`
- Evaluation manifest: `evaluation_manifest.json`
