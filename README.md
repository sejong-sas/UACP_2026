
- 현재 Valid Baseline: `runs/current_valid_baseline/condition_c_5k_10ep_lambda1e3_corrected/`
- 현재 연구 흐름: `docs/CURRENT_RESEARCH_STORY.md`
- 전체 실험 지도: `docs/EXPERIMENT_MAP.md`
- 코드 구조 가이드: `docs/CODE_GUIDE.md`
- 주요 기능 코드 조회표: `docs/CODE_LOOKUP.md`
- 스크립트 목록: `scripts/README.md`
- 결과 폴더 목록: `runs/README.md`
- 정리 후보: `docs/CLEANUP_CANDIDATES.md`

기존 상세 기록은 아래에 그대로 보존되어 있습니다. 위 문서부터 읽으면 현재
valid baseline과 핵심 진단 흐름을 먼저 파악할 수 있습니다.

# UACP_2026

This project uses `mobihoc26-paper289.pdf` as the primary source for reproducing
the experiment setup of "UACP: Uncertainty-Aware Channel Prediction in MIMO
Networks".

# 2026-09-17 Batch-400 Baseline Revalidation: Fig.8 / Fig.9 / Fig.11-lite

## Purpose

This run checks whether earlier small-batch dynamics were materially different
from a scaled condition: 10,000 unique CFRs, 5 epochs, batch size 400. It is
baseline-reproduction reinforcement only. Full-Psi, Partial Fine-Tuning, and
new research losses were not run.

## Conditions and provenance

- Training archive: `runs/current_valid_baseline/diversity_ablation/tenk_5ep_seed_20260819/generated_data/train_5k_pilot.npz` (10,000 CFRs)
- Epochs: 5; seed `20260819`; Adam; LR `1e-4`; `lambda_reg=1e-3`
- Diagonal-Psi, pair-scalar kappa/nu, diagonal multivariate Student-t NLL, pair regularizer
- Reported-CFR sample-wise complex AWGN, SNR 15 dB; clean target; uniform `Ng=16`
- Common evaluation set reused unchanged from `diversity_ablation/reproducibility_20260914/common_eval`
- GPU start/end: NVIDIA GB10, `cuda:0`; see `gpu_start_training.csv` and `gpu_end_all.csv`
- Direct batch 400 passed the 20-step feasibility benchmark; no gradient accumulation was used.
- Benchmark peak allocated `32372.5 MiB`, mean step `16.748 s`, `23.884 samples/s`, estimated 5-epoch time `2093.5 s`.
- Actual training time `2165.5 s` (36.1 min), peak allocated `32372.5 MiB`.

`IMPLEMENTATION-ASSUMPTION`: 10k sample-count scaling is not an exact paper
training reproduction. `IMPLEMENTATION-ASSUMPTION`: existing delay-sweep
samples represent the Fig.11 sequence because no temporal-correlated generator
was found. Diagonal-Psi remains an approximation to the paper's full covariance.

## Artifacts and commands

All new artifacts are under
`runs/current_valid_baseline/batch400_10k5ep_20260917_retry/`.

Scripts used: `scripts/train_batch400_10k5.py`,
`scripts/fig8_fig9_canonical_fast.py`, `scripts/fig11_lite_batch400.py`, and
`scripts/plot_batch400_results.py`. Existing artifacts were not overwritten;
failed evaluator attempts are retained only in the new run directory.

## Results

Final validation omitted-NMSE was `-14.0807 dB`.

| Fig.8 regime | Mean epistemic | dB diagnostic |
| --- | ---: | ---: |
| ID-Easy 20 ns | 3.554813 | 5.508 dB |
| ID-Hard 80 ns | 3.495747 | 5.436 dB |
| OOD-Near 120 ns | 3.496323 | 5.437 dB |
| OOD-Far 1 ms | 3.496331 | 5.437 dB |

Fig.8 AUROC: pooled OOD `0.46273`, Near `0.46372`, Far `0.46174`.
The 80 ns versus 120 ns separation did not improve; Near-OOD is below the
maximum ID mean by about `-0.05849` raw-score units.

Fig.9 used the validated Student-t interval formula exactly, but the full
10,000-sample/regime pass was stopped after 45 minutes because the existing
per-element `scipy.stats.t.ppf` implementation had not reached its save stage.
The bounded exact result uses the first 20 common-eval samples per regime
(307,200 omitted components per pool/nominal): ID calibration MAE `0.45377`,
pooled OOD calibration MAE `0.23792`. Coverage is above nominal for both pools;
this is exploratory bounded evidence, not a full 10k estimate.

Fig.11-lite used 240 steps with `20 → 80 → 10 → 40 → 60 → 120 ns`, 40 samples
per regime, and only Ng/NMSE. The controller quickly falls to `Ng=1`; omitted
NMSE is undefined at full feedback and is retained as NaN. All-NMSE means in
sequence order were `3.016, 3.297, 3.236, 3.338, 3.223, 3.320 dB`. No
BER/EVM/precoding/adaptation update was run.

The compact comparison is in `comparison_summary.csv` and
`comparison_summary.json`; existing 50k×1, 100k×1, and prior batch-8 values are
preserved there.

## Judgment and next experiment

- Near-OOD epistemic separation: **No improvement** at batch 400; worse than the saved batch-8 gap.
- OOD calibration: **No evidence of improvement**; bounded exact calibration remains over-covered. Full 10k Fig.9 is pending a faster evaluator.
- Nu boundary/inf: no new saved `nu` boundary/inf was observed; full-feedback omitted-NMSE NaNs are a mask-definition effect.
- Fig.11 control linkage: not improved; the threshold/controller collapses to `Ng=1`.

Recommended next experiment: increase training scale to **100k×5** (or a
controlled 100k×150 approximation), after making Fig.9 Student-t coverage
streaming/bounded. Avoid another batch-size-only run until that evaluator is
computationally bounded.

# Current Valid UACP Baseline

## Purpose

This is the new provenance-recorded control run for all future diagnostics. It
was trained from scratch with the corrected Real/Imag antenna-pair mapping.
Historical STEP 6 Condition C was pre-fix and is invalid for uncertainty
comparison; it remains preserved as a historical result.

## Channel and Observation

- Sionna `2.0.1`, 2x2 MIMO, `K=1024`, 3.5 GHz, 30 kHz spacing
- TDL-A, zero mobility, and `normalize=false`: `IMPLEMENTATION-ASSUMPTION`
- Clean full CFR is the target.
- Reported subcarriers receive sample-wise 15 dB complex AWGN before sparse
  encoding: `IMPLEMENTATION-ASSUMPTION`; the paper's exact SNR stage is UNKNOWN.
- Evaluation uses uniform `Ng=16` and omitted mask `1-mask`.

## Predictor Formulation

Experiment D is used: 32 residual blocks, 192 hidden channels, pair-level scalar
`kappa` and `nu`, diagonal `Psi`, diagonal multivariate Student-t NLL, and
pair-level evidence regularizer. The real layout is
`[Re(pair0..3), Im(pair0..3)]`, so pairs are `(0,4)`, `(1,5)`, `(2,6)`, and
`(3,7)`. `gamma` and diagonal `Psi` map to four `[2048]` pair vectors;
`kappa` and `nu` have shape `[B,4,1]`. Full paper `Psi=L L^T` is not implemented;
diagonal `Psi` is an `APPROXIMATION`.

## Current-Valid Training Setting

- 5,000 samples, Uniform `[10,100] ns`, 10 epochs
- Adam, learning rate `1e-4`, batch size `8`, `lambda_reg=1e-3`
- Seed `20260819`, GPU `cuda:0` NVIDIA GB10
- 5k/10 epochs is `IMPLEMENTATION-ASSUMPTION / PILOT`; `lambda_reg=1e-3` is
  `PAPER-SPECIFIED` numerically.

## Baseline Result

The primary Aleatoric/Epistemic scores use the current Eq.12 -> Eq.13 helper:
pair covariance -> Re/Im trace -> four-pair mean -> omitted-subcarrier mean.

| Regime | NMSE all (dB) | NMSE omitted (dB) | Aleatoric | Epistemic | Err-Ale P | Err-Epi P |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ID-Easy 20 ns | -17.6702 | -17.6709 | 0.020904 | 0.002677 | 0.2358 | 0.2415 |
| ID-Hard 80 ns | -17.2629 | -17.2605 | 0.020815 | 0.002642 | 0.2646 | 0.2685 |
| OOD-Near 120 ns | -15.2965 | -15.2930 | 0.021007 | 0.002673 | 0.2534 | 0.2558 |
| OOD-Far 1 ms | 1.3591 | 1.5622 | 0.028924 | 0.004138 | 0.0613 | 0.0698 |

## Provenance and Artifacts

The current-valid run is stored in
`runs/current_valid_baseline/condition_c_5k_10ep_lambda1e3_corrected/`.
It includes `checkpoint_with_provenance.pt`, `model_state_dict.pt`,
`config.json`, `provenance.json`, `evaluation_setup.json`, epoch probe results,
training history, summary CSV, and curves. The wrapped checkpoint records the
model state, config, git commit, code hashes, seed, epoch, and experiment name.

## Next Diagnostics

This checkpoint is the control for: (1) noisy sparse CFR plus frequency-axis
linear interpolation reconstruction-only evaluation, and (2) antenna-pair
`gamma`, diagonal `Psi`, `kappa`, `nu`, `Sigma_ale`, and `Sigma_epi` diagnostics.
Both experiments must reuse the saved evaluation setup. Partial Fine-Tuning is
not started by this baseline step.

## Diagnostic Experiment 1 - No-Predictor Linear Reconstruction

### Purpose and Fixed Setup

This read-only experiment tested whether the same 15 dB noisy sparse observation
used by the current-valid predictor produces the expected reconstruction
difficulty without a predictor or uncertainty model. The in-memory probes were
shared with Experiment 2: the same clean targets, `Ng=16` mask, and deterministic
sample-wise reported-CFR AWGN policy were used. No training was performed.

### Method

Each of the four antenna-pair CFRs was reconstructed independently along the
frequency axis by real/imaginary linear interpolation. `np.interp` endpoint
values were used at the two edges. No spline, nearest, or cubic interpolation
was used. This is a reconstruction-only control conceptually aligned with
uniform sparse feedback plus interpolation; it is not a learned UACP model.

### Results

| Regime | NMSE all (dB) | NMSE omitted (dB) | Pair0 | Pair1 | Pair2 | Pair3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ID-Easy 20 ns | -16.7449 | -16.8870 | -14.5907 | -15.4204 | -14.1789 | -14.2122 |
| ID-Hard 80 ns | -16.4345 | -16.5537 | -15.5201 | -15.4274 | -15.2857 | -15.1853 |
| OOD-Near 120 ns | -15.9135 | -15.9823 | -14.8488 | -14.9392 | -15.0460 | -15.1411 |

Overall H1 `20 < 80 < 120` in reconstruction difficulty is `PASS`. Pair-level
values are heterogeneous, so the aggregate ordering should not be interpreted
as every antenna pair following the same monotonic curve. The 1 ms secondary
control was also strongly worse at `2.4655 dB` omitted NMSE.

### Artifacts

Results are in `runs/current_valid_baseline/diagnostics/exp1_linear_interpolation_no_predictor/`.

## Diagnostic Experiment 2 - Antenna-Pair Evidential Parameter Trace

### Purpose and Fixed Setup

This read-only experiment traced the current-valid checkpoint from `gamma` to
diagonal `Psi`, pair-scalar `nu` and `kappa`, then to Eq. (7)/(8) and the
validated Eq. (12)/(13) omitted-subcarrier scores. The model, loss, checkpoint,
noise policy, mask, and channel data were unchanged. The same in-memory noisy
probes as Experiment 1 were used.

### Parameter Roles and Shapes

- `gamma`: external `[B,8,1024]`, pair-vector view `[B,4,2048]`
- `Psi`: diagonal approximation, pair-vector view `[B,4,2048]`; not the paper's full covariance
- `kappa`: pair scalar `[B,4,1]`
- `nu`: pair scalar `[B,4,1]`
- `denom = nu - 2K - 1`, with `K=1024`
- `Sigma_ale = Psi_diag / denom`, `Sigma_epi = Sigma_ale / kappa`

Final Eq.12/13 scores were calculated as pair covariance -> Real/Imag trace ->
four-pair mean -> omitted-subcarrier mean.

### Final Parameter Trace

| Regime | Mean gamma NMSE | Mean Psi | Mean kappa | Mean nu | Mean denom | Mean pair Ale | Mean pair Epi |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 ns | -16.9020 | 0.190119 | 7.8855 | 2067.3243 | 18.3243 | 0.010424 | 0.001334 |
| 80 ns | -16.7745 | 0.190251 | 7.8990 | 2067.3137 | 18.3137 | 0.010320 | 0.001309 |
| 120 ns | -14.9314 | 0.191583 | 7.8741 | 2067.2734 | 18.2734 | 0.010406 | 0.001324 |
| 1 ms | -4.4284 | 0.231994 | 7.0540 | 2065.1498 | 16.1498 | 0.014367 | 0.002054 |

### Final Eq.12/13 Scores

| Regime | NMSE all (dB) | NMSE omitted (dB) | U_ale omitted | U_epi omitted |
| --- | ---: | ---: | ---: | ---: |
| ID-Easy 20 ns | -17.7000 | -17.7009 | 0.0209063 | 0.0026773 |
| ID-Hard 80 ns | -17.1943 | -17.1921 | 0.0208260 | 0.0026432 |
| OOD-Near 120 ns | -15.2339 | -15.2307 | 0.0210129 | 0.0026743 |
| OOD-Far 1 ms | 1.3607 | 1.5639 | 0.0289447 | 0.0041414 |

The gaps are `Ale80-20=-8.0274e-5`, `Epi120-ID=-2.9623e-6`, and
`Epi1ms-ID=+1.4641e-3`. Thus the current valid run preserves a clear Far-OOD
epistemic signal, while the Near-OOD epistemic gap is slightly negative and
the ID-Hard aleatoric gap remains negative.

## Diagnostic Experiment A - Sample/Pair Evidence Calibration

### Purpose and fixed baseline

This read-only experiment tested whether the current-valid checkpoint lowers
evidence and raises uncertainty for individual sample/antenna-pair predictions
with larger reconstruction error. The checkpoint, noisy `Ng=16` probes, target,
mask policy, and model were unchanged; no training was performed. The primary
error was the omitted-subcarrier squared L2 error of one `[2048]` pair vector.

For each sample and pair, the diagnostic recorded `kappa`, `nu`,
`Phi=kappa+nu`, `denom=nu-2K-1`, diagonal `Psi`, and pair-level Eq. (7)/(8)
Aleatoric/Epistemic scores. Pair scores were computed before aggregation; the
overall scores use Eq. (12) followed by omitted-subcarrier Eq. (13).

### Results

| Regime | NMSE omitted (dB) | U_ale omitted | U_epi omitted |
| --- | ---: | ---: | ---: |
| ID-Easy 20 ns | -17.7174 | 0.020891650 | 0.002674789 |
| ID-Hard 80 ns | -17.2750 | 0.020813461 | 0.002641180 |
| OOD-Near 120 ns | -15.2648 | 0.020994240 | 0.002670816 |
| OOD-Far 1 ms | 1.5592 | 0.028925674 | 0.004138705 |

The overall gaps are `Ale80-20=-7.8188e-5`,
`Epi120-ID=-3.9737e-6`, and `Epi1ms-ID=+1.4639e-3`.

Across all sample/pair observations, error had negative Pearson correlation
with evidence and positive correlation with uncertainty:

| Regime | error/evidence | error/Aleatoric | error/Epistemic |
| --- | ---: | ---: | ---: |
| 20 ns | -0.4480 | 0.5942 | 0.5739 |
| 80 ns | -0.6805 | 0.6891 | 0.6871 |
| 120 ns | -0.4847 | 0.5323 | 0.5335 |
| 1 ms | -0.3571 | 0.3859 | 0.3793 |

The same direction held in error tertiles: for 20 ns, Aleatoric increased
from `0.009542` to `0.011416` and Epistemic from `0.001185` to `0.001501`
from low to high error; for 80 ns the corresponding values were `0.009745`
to `0.010968` and `0.001216` to `0.001416`. Pair-level correlations were
heterogeneous but remained negative for evidence and positive for uncertainty
for all four pairs in the primary regimes. This is `Case A1` at sample/pair
level: local calibration exists, but it does not produce the desired regime
mean ordering.

### Artifacts and interpretation

Results and plots are in
`runs/current_valid_baseline/diagnostics/expA_sample_pair_evidence_calibration/`.
The result supports a regime-level representation/aggregation limitation more
than a complete failure of sample-level evidence calibration. It does not by
itself justify changing the model or loss.

## Diagnostic Experiment B - Diagonal Psi Approximation Diagnostic

### Purpose and method

This read-only experiment tested whether the clean channel distribution has
frequency-correlation structure that is absent from the learned diagonal
`Psi`. Empirical covariance was computed across samples for each regime and
antenna pair using the vector `[Re(H[0..K-1]), Im(H[0..K-1])]`. The reported
complex lag correlation is

`mean_k | E[(H_k-mean) conjugate(H_{k+lag}-mean)] /
sqrt(var_k var_{k+lag}) |`.

The empirical covariance is a diagnostic of the clean ground-truth channel;
the learned `Psi` values were traced from the unchanged noisy-observation
checkpoint. The current learned `Psi` remains a diagonal
`IMPLEMENTATION-ASSUMPTION / APPROXIMATION`, not the paper's full covariance.

### Covariance energy and learned diagonal Psi

| Regime | Off/total covariance energy | Mean diagonal variance sum | Learned Psi mean |
| --- | ---: | ---: | ---: |
| 20 ns | 0.998574 | 1088.3186 | 0.190062 |
| 80 ns | 0.997906 | 1043.9868 | 0.190175 |
| 120 ns | 0.997465 | 1022.1192 | 0.191485 |
| 1 ms | 0.993790 | 1030.2365 | 0.231841 |

The off/total energy ratio is high in every regime because the CFR is strongly
correlated across frequency and this energy metric includes all covariance
entries. Its regime separation is modest, while the lag decay is much clearer:

| Regime | R(1) | R(2) | R(4) | R(8) | R(16) | R(32) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 ns | 0.999993 | 0.999973 | 0.999893 | 0.999573 | 0.998294 | 0.993236 |
| 80 ns | 0.999887 | 0.999550 | 0.998205 | 0.992886 | 0.972563 | 0.904606 |
| 120 ns | 0.999750 | 0.998999 | 0.996019 | 0.984399 | 0.942434 | 0.830189 |
| 1 ms | 0.087439 | 0.330331 | 0.158990 | 0.589011 | 0.312575 | 0.343175 |

At lag 32, the pair-averaged correlation is `0.993236` for 20 ns,
`0.904606` for 80 ns, and `0.830189` for 120 ns, while the learned diagonal
Psi means are nearly unchanged for the three regimes. This is `Case B1`
evidence: the channel contains frequency-correlation differences that a
diagonal covariance cannot represent directly. The 1 ms result is a separate,
strongly shifted regime and is not used to claim a smooth physical trend.

### Artifacts and next-step boundary

Results and plots are in
`runs/current_valid_baseline/diagnostics/expB_diagonal_psi_covariance_diagnostic/`.
No low-rank or full-covariance model was trained. The evidence makes a bounded
structured-covariance experiment the highest-priority next diagnostic, but it
must be designed and reviewed before execution. The existing baseline,
checkpoint, loss, NIW parameterization, and all historical outputs were left
unchanged.

The denominator at 80 ns is slightly lower than at 20 ns, which would increase
aleatoric uncertainty. The observed lower 80 ns aleatoric score is therefore
not explained by a larger `nu-2K-1` denominator. Mean `Psi` is nearly unchanged
between 20 and 80 ns, with a slight increase at 80 ns, so the remaining gap is
small and results from the combined per-sample/per-pair ratio rather than a
simple regime-level `Psi` increase. Pair-level distributions are preserved in
`sample_pair_distributions.csv` for further inspection.

### Artifacts

Results are in `runs/current_valid_baseline/diagnostics/exp2_evidential_parameter_trace/`:
`final_uncertainty_summary.csv`, `pair_parameter_summary.csv`,
`sample_pair_distributions.csv`, and `results.json`. The shared observation
manifest is `diagnostics/shared_observation_manifest.json`.

### Diagnostic Conclusion

Experiment 1 passes, so the noisy sparse observation itself produces the
expected aggregate difficulty ordering. Experiment 2 shows that the predictor
mean error also increases toward 120 ns and collapses at 1 ms, but the
evidential parameters do not convert the ID-Hard difficulty into a larger
aleatoric score. The strongest remaining candidates are pair/sample-level
evidence calibration and the diagonal-`Psi` approximation; this does not by
itself establish either as the cause. No baseline model or training condition
was changed, and Partial Fine-Tuning was not started.

## Phase B Sionna PHY Smoke Test - 2026-08-15

### Goal

Verify the minimum channel simulation foundation for later UACP reproduction and
OOD adaptation experiments. This phase does not implement the UACP evidential
predictor, does not create the 100,000-sample dataset, and does not run
fine-tuning.

Future model code must keep the 32 residual blocks individually addressable so
the following adaptation policies can be compared fairly:

- No Adaptation
- Head Only
- Last 1 Block + Head
- Last 4 Blocks + Head
- Last 8 Blocks + Head
- Full Fine-Tuning

Primary future metrics: OOD NMSE recovery, recovery step/epoch, adaptation
time, trainable parameter count, and trainable parameter ratio.

### Installed Environment

The existing conda `base` environment was not modified. A project-local virtual
environment was created at `.venv-uacp`.

Installed package versions observed in `.venv-uacp`:

- Python: `3.13.9`
- PyTorch: `2.13.0+cu130`
- PyTorch CUDA: `13.0`
- Sionna: `2.0.1`
- Sionna package installed: `sionna-no-rt==2.0.1`
- NumPy: `2.5.2`
- SciPy: `1.18.0`

System/GPU observed from the smoke test:

- Architecture: `aarch64`
- Platform: `Linux-6.11.0-1014-nvidia-aarch64-with-glibc2.39`
- GPU: `NVIDIA GB10`
- CUDA available in PyTorch: `true`
- CUDA device count: `1`

### Commands Executed

```bash
cd /home/saslab01/Desktop/UACP_2026
python -m venv .venv-uacp
.venv-uacp/bin/python -m pip install --upgrade pip
.venv-uacp/bin/python -m pip install sionna-no-rt==2.0.1
.venv-uacp/bin/python -m unittest discover -s tests -t . -v
.venv-uacp/bin/python scripts/sionna_smoke.py --config configs/sionna_smoke.json
```

### Paper-Specified Settings Used

These values are directly from the paper's Evaluation Setup or evaluation
regime text:

- MIMO antenna dimensions: `2 x 2`
- Number of subcarriers: `K = 1024`
- Carrier frequency: `3.5 GHz`
- Subcarrier spacing: `30 kHz`
- SNR: `15 dB`
- Delay spread smoke-test regimes: `20 ns`, `80 ns`, `120 ns`

Note: `SNR = 15 dB` is recorded in the config because it is paper-specified, but
this smoke test only generates CFR tensors and does not yet apply AWGN or a full
link simulation.

### Implementation Assumptions

The paper does not specify the exact Sionna version, channel model, TDL/CDL
profile, antenna geometry, cyclic prefix, or number of OFDM symbols. The smoke
test therefore uses the following implementation assumptions only to verify that
Sionna PHY can generate CFR tensors under paper-level dimensions:

- Sionna package: `sionna-no-rt==2.0.1`
- Channel model: `sionna.phy.channel.tr38901.TDL`
- TDL profile: `A`
- Batch size: `2`
- Number of OFDM symbols: `1`
- Cyclic prefix length: `0`
- Mobility: `min_speed_mps = 0.0`, `max_speed_mps = 0.0`
- Channel normalization: `false`
- Precision: `single`
- Device: `cuda:0`
- Seed: `20260815`

These are not claimed as original UACP paper settings.

### Smoke Test Results

The command completed successfully on `cuda:0`.

Raw Sionna CFR tensor shape:

```text
[batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]
= [2, 1, 2, 1, 2, 1, 1024]
```

UACP-ready complex CFR view:

```text
[batch, subcarrier, rx_ant, tx_ant] = [2, 1024, 2, 2]
dtype = torch.complex64
device = cuda:0
complex = true
```

Real/imag concatenated predictor channels before adding the mask:

```text
[batch, 8, 1024]
```

After adding one binary mask channel, this matches the paper's 2x2 MIMO
mask-aware predictor input channel count:

```text
8 real/imag CFR channels + 1 mask channel = 9 channels
```

Delay spread applicability:

| RMS delay spread | CFR generated | UACP-ready shape | Device | Adjacent-frequency delta proxy |
| --- | --- | --- | --- | --- |
| `20 ns` | yes | `[2, 1024, 2, 2]` | `cuda:0` | `-48.2130 dB` |
| `80 ns` | yes | `[2, 1024, 2, 2]` | `cuda:0` | `-34.9444 dB` |
| `120 ns` | yes | `[2, 1024, 2, 2]` | `cuda:0` | `-31.6420 dB` |

The adjacent-frequency delta proxy is an implementation-side diagnostic only.
It suggests stronger frequency variation as RMS delay spread increases in this
TDL-A smoke setup, but it is not a paper metric and must not be used as a final
UACP reproduction result.

### Unknown / Not Yet Reproduced

- Exact Sionna version used by the paper: `UNKNOWN`
- Exact channel model used by the paper: `UNKNOWN`
- TDL/CDL profile or scenario model: `UNKNOWN`
- Antenna geometry and correlation assumptions: `UNKNOWN`
- Mobility/Doppler settings: `UNKNOWN`
- Cyclic prefix and number of OFDM symbols: `UNKNOWN`
- Full link behavior using `SNR = 15 dB`: not implemented in this smoke phase
- UACP evidential predictor, loss, training, and adaptation: not implemented in
  this smoke phase

### Next Issues Before Dataset Reproduction

- Determine whether the paper authors released code or additional artifact
  details for the exact Sionna channel model and version.
- Decide how to map the paper's unspecified channel model to Sionna 2.0.1
  without presenting the choice as paper-specified.
- Add a small, bounded dataset-generation script only after the channel model
  assumption is approved.
- Keep every dataset parameter in config, including seed, sample count,
  delay-spread distribution, channel model, and output tensor layout.

## Prototype CFR Dataset - 2026-08-19

### Paper / Artifact Check

The local paper `mobihoc26-paper289.pdf` was searched again for channel
simulation details. Confirmed paper-specified items:

- Simulator family: Sionna PHY
- MIMO dimensions: `Nr = 2`, `Nt = 2`
- Number of subcarriers: `K = 1024`
- Carrier frequency: `3.5 GHz`
- Subcarrier spacing: `30 kHz`
- SNR: `15 dB`
- Training CFR realizations in the paper: `100,000`
- Training delay spread distribution: Uniform `[10, 100] ns`
- Evaluation regimes: `20 ns`, `80 ns`, OOD `> 100 ns`
- OOD-near / OOD-far diagnostic regimes: `120 ns` / `1 ms`

Still `UNKNOWN` in the paper:

- Exact Sionna version
- Sionna 1.x vs 2.x
- Exact channel model: TDL/CDL/UMi/UMa/RMa/Rayleigh
- TDL/CDL profile
- Antenna geometry and spatial correlation
- Mobility, Doppler, or velocity
- OFDM symbol count
- Cyclic prefix length
- Channel normalization policy
- Exact CFR generation implementation
- Exact stage where `15 dB` SNR is applied

Official code/artifact status: `OFFICIAL CODE NOT FOUND`. Searches by paper
title, `mobihoc26-paper289`, `UACP`, `Sionna`, `CFR`, and `evidential` did not
identify an official author repository. The PDF appears submission/anonymized
and does not expose author names or an artifact URL.

### Channel Model Decision

Selected channel model for the prototype dataset:

```text
IMPLEMENTATION-ASSUMPTION: Sionna 2.0.1 PHY, 3GPP TR 38.901 TDL, profile A
```

Candidate comparison:

| Candidate | Delay spread control | 2x2 MIMO CFR | 1024 OFDM | ID/OOD by RMS delay | 100k feasibility | Fit to paper |
| --- | --- | --- | --- | --- | --- | --- |
| TDL | Direct scalar `delay_spread` | Yes | Yes via `GenerateOFDMChannel` | Direct | Practical | Strongest available fit |
| CDL | Direct delay spread, richer geometry | Yes | Yes | Direct | Heavier | Plausible but adds unspecified geometry |
| UMi/UMa/RMa | Scenario-level delay/LSP behavior | Yes | Yes | Less direct | Heavier | Adds many unspecified assumptions |
| Rayleigh block fading | Yes for MIMO but no multipath RMS delay profile | Yes | Weak frequency selectivity | Poor | Practical | Too simple for delay-spread study |

TDL-A is therefore the most conservative prototype choice because it directly
controls RMS delay spread, generates 2x2 MIMO CFR, works naturally with
Sionna OFDM channel generation, and avoids the extra geometry assumptions
required by CDL or scenario models. It remains an implementation assumption,
not a paper-specified setting.

### SNR Interpretation

The paper states that the system operates at `15 dB SNR`, and separately
describes sounding/pilot CFR estimation, sparse feedback, prediction,
precoding, BER/EVM, and throughput evaluation. It does not explicitly state
whether `15 dB` is applied to channel observation noise, pilot estimation,
feedback, data transmission, BER/EVM evaluation, or all link-level operations.

Prototype dataset policy:

```text
Clean full CFR is generated and stored.
SNR is recorded in metadata but not applied to this clean CFR dataset.
Observation noise / sparse feedback must be a later explicit stage:
Ground-truth full CFR -> Observation/noise process -> Sparse CFR + mask -> UACP Predictor.
```

### Dataset Generated

Config:

```text
configs/dataset_prototype.json
```

Primary output:

```text
data/prototype/
```

Reproducibility check output:

```text
data/prototype_repro/
```

Generated splits:

| Split | Samples | Shape | dtype | Size |
| --- | ---: | --- | --- | ---: |
| train | 1000 | `[1000, 1024, 2, 2]` | `complex64` | 32M |
| validation | 200 | `[200, 1024, 2, 2]` | `complex64` | 6.3M |
| test_id_easy | 200 | `[200, 1024, 2, 2]` | `complex64` | 6.3M |
| test_id_hard | 200 | `[200, 1024, 2, 2]` | `complex64` | 6.3M |
| test_ood_near | 200 | `[200, 1024, 2, 2]` | `complex64` | 6.3M |

Generation command:

```bash
/usr/bin/time -f 'elapsed=%E max_rss_kb=%M' \
  .venv-uacp/bin/python scripts/generate_dataset.py \
  --config configs/dataset_prototype.json
```

Observed generation time:

```text
elapsed=0:06.86 max_rss_kb=1290472
```

### Sanity Check

Inspection command:

```bash
.venv-uacp/bin/python scripts/inspect_dataset.py data/prototype
```

All splits:

- Stored as CPU `.npz`
- dtype `complex64`
- CFR layout `[sample, subcarrier, rx_ant, tx_ant]`
- No NaN
- No Inf
- No sparse masks stored

Delay spread sanity:

| Split | Delay spread |
| --- | --- |
| train | min `10.1038 ns`, mean `55.4137 ns`, max `99.9843 ns` |
| validation | min `10.3797 ns`, mean `51.2907 ns`, max `99.5274 ns` |
| test_id_easy | fixed `20 ns` |
| test_id_hard | fixed `80 ns` |
| test_ood_near | fixed `120 ns` |

Frequency-selectivity diagnostic only, not a UACP paper metric:

| Regime | Adjacent-frequency delta proxy |
| --- | ---: |
| ID-Easy `20 ns` | `-47.0485 dB` |
| ID-Hard `80 ns` | `-34.9083 dB` |
| OOD-Near `120 ns` | `-31.5484 dB` |

Same-seed reproducibility was checked by regenerating into
`data/prototype_repro` and comparing arrays. The `.npz` file hashes differ
because metadata records generation time, but `cfr`, `delay_spread_ns`, and
`regime_label` arrays are exactly identical for every split.

### Commands

```bash
.venv-uacp/bin/python -m unittest discover -s tests -t . -v
.venv-uacp/bin/python scripts/sionna_smoke.py --config configs/sionna_smoke.json
.venv-uacp/bin/python scripts/generate_dataset.py --config configs/dataset_prototype.json
.venv-uacp/bin/python scripts/inspect_dataset.py data/prototype
.venv-uacp/bin/python scripts/generate_dataset.py --config configs/dataset_prototype.json --output-dir data/prototype_repro
```

### Remaining Gap Before UACP Predictor

- The exact paper channel model remains unknown.
- `TDL-A` must continue to be labeled `IMPLEMENTATION-ASSUMPTION`.
- `15 dB SNR` needs an explicit design decision before noisy observation or
  BER/EVM experiments.
- The future DataLoader should generate masks online to match the paper's
  training procedure.
- The future predictor implementation must expose 32 residual blocks as
  individually freezeable modules for fair partial fine-tuning comparisons.

## Prototype UACP Predictor Baseline - 2026-08-19

### Predictor Architecture

Paper-specified structure rechecked from Sections 3.2, 3.2.1, 3.2.2 and
Evaluation Setup:

- Mask-aware 1D residual convolutional inpainting network
- 2x2 complex CFR converted to real/imag representation
- Input channels: `8` real/imag CFR channels + `1` binary mask = `9`
- Initial `1x1` convolution
- Hidden channels: `192`
- Residual blocks: `32`
- Each residual block: Conv1D kernel size `5`, GELU, dropout `0.05`, Conv1D
  kernel size `5`, identity skip connection
- Final evidential head
- Learning rate `1e-4`, `lambda_reg = 1e-3`

Implemented:

- `UACPEvidentialPredictor`
- `input_projection: Conv1d(9, 192, kernel_size=1)`
- `residual_blocks: ModuleList` with exactly 32 individually addressable blocks
- `evidential_head: Conv1d(192, 32, kernel_size=1)`
- Output channels: `gamma/kappa/psi/nu` for 8 real-valued CFR channels
- Total/trainable parameters in baseline: `11,816,864`

`IMPLEMENTATION-ASSUMPTION`:

- The paper describes per-antenna-pair multivariate NIW with covariance
  `Psi in R^{2K x 2K}` but does not specify a feasible head tensor layout.
- Prototype uses a diagonal per-real/imag-channel evidential approximation:
  `gamma, kappa, psi, nu` each have shape `[batch, 8, 1024]`.
- This is not claimed to be the exact paper implementation.

### Input / Mask

Dataset CFR layout:

```text
[N, 1024, 2, 2] complex64
```

Training/evaluation target:

```text
[N, 8, 1024]
```

Sparse input:

```text
[N, 9, 1024]
```

Mask policy:

- Masks are not stored in the dataset.
- During training, masks are generated online.
- Training mask grouping factors: `[4, 8, 16, 32]`
- Evaluation grouping factor: `16`

Mask grouping choices are `IMPLEMENTATION-ASSUMPTION`; the paper states online
randomized masks but does not list exact sparsity candidates.

### Evidential Head

Equation mapping:

- Eq. (6): predicted CFR mean is `gamma`
- Eq. (7): aleatoric uncertainty is `psi / (nu - 2K - 1)`
- Eq. (8): epistemic uncertainty is `aleatoric / kappa`
- Eq. (9): `kappa = softplus(raw_kappa) + eps`,
  `psi = softplus(raw_psi) + eps`,
  `nu = 2K + 1 + softplus(raw_nu) + eps`

Unit tests check positivity/admissibility and the Eq. (6)-(8) diagonal mapping.

### Loss

Implemented:

```text
L = L_NLL + lambda_reg * L_reg
```

- `L_NLL`: Student-t negative log likelihood from the diagonal form of Eq. (5)
- `L_reg`: `(target - gamma)^2 * (kappa + nu)`, following Eq. (11)'s evidence
  regularization idea
- `lambda_reg = 1e-3`

Loss terms are logged separately as `total`, `nll`, and `reg`.

### Prototype Training

Config:

```text
configs/train_prototype.json
```

Command:

```bash
/usr/bin/time -f 'elapsed=%E max_rss_kb=%M' \
  .venv-uacp/bin/python scripts/train_predictor.py \
  --config configs/train_prototype.json \
  --output-dir runs/prototype_predictor
```

Training settings:

- Epochs: `5`
- Batch size: `8`
- Eval batch size: `16`
- Learning rate: `1e-4`
- Optimizer: Adam
- GPU: `cuda:0` / NVIDIA GB10
- Deterministic mode enabled for reproducibility

Observed runtime:

```text
elapsed_seconds = 117.4392
deterministic shell time = 1:59.39
max_rss_kb = 1775276
```

Training curve saved:

```text
runs/prototype_predictor/training_curve.csv
runs/prototype_predictor/training_results.json
```

Loss/NMSE summary:

| Epoch | Train total | Val total | Train NMSE dB | Val NMSE dB |
| ---: | ---: | ---: | ---: | ---: |
| 1 | `1.1397` | `0.9588` | `-3.3238` | `-3.8607` |
| 2 | `0.2275` | `-0.5932` | `-6.5825` | `-14.2789` |
| 3 | `-0.1039` | `-0.2682` | `-8.4358` | `-11.4973` |
| 4 | `-0.3576` | `-0.8311` | `-9.8061` | `-15.8388` |
| 5 | `-0.4925` | `-0.7799` | `-10.7412` | `-15.4303` |

### Evaluation

Evaluation table saved:

```text
runs/prototype_predictor/evaluation_table.csv
```

Uncertainty values below are averaged over dropped/unreported subcarriers.

| Regime | NMSE dB | Aleatoric | Epistemic |
| --- | ---: | ---: | ---: |
| ID-Easy 20 ns | `-15.4818` | `0.013111` | `0.006620` |
| ID-Hard 80 ns | `-15.4925` | `0.012807` | `0.006449` |
| OOD-Near 120 ns | `-15.0930` | `0.012676` | `0.006371` |

### Expected Behavior Check

- NMSE: OOD-Near is worse than ID-Easy/ID-Hard, but ID-Hard is not worse than
  ID-Easy in this short prototype run.
- Aleatoric: does not increase from 20 ns to 80 ns.
- Epistemic: does not increase for OOD-Near 120 ns; it slightly decreases.

This means the prototype model learns CFR reconstruction, but it does not yet
reproduce the paper's desired uncertainty separation behavior.

### Errors / Fixes

- Reproducibility issue: two same-seed GPU training runs initially produced
  slightly different histories and final metrics.
  - Cause: GPU operation nondeterminism.
  - Fix: enabled `torch.use_deterministic_algorithms(True)`, disabled cuDNN
    benchmarking, forced cuDNN deterministic mode, and disabled TF32.
  - Effect on paper structure: none. This only affects reproducibility.
- No NaN/Inf occurred in output uncertainties or loss.
- No GPU memory issue occurred.

### Reproducibility

Reproducibility command:

```bash
.venv-uacp/bin/python scripts/train_predictor.py \
  --config configs/train_prototype.json \
  --output-dir runs/prototype_predictor_repro
```

Comparison result after deterministic settings:

```text
history_equal=True
test_results_equal=True
params_equal=True
```

### Decision

The current prototype is sufficient to prove:

- 9-channel sparse CFR + mask input works
- 32-block predictor forward pass works
- Evidential outputs are finite
- Loss decreases on the prototype dataset
- NMSE can be evaluated on ID/OOD regimes
- Same-seed training results are reproducible

It is not sufficient to proceed to Full Fine-Tuning vs Partial Fine-Tuning yet,
because OOD-Near does not show increased epistemic uncertainty.

## Predictor Diagnostic - 2026-08-27

Goal: keep the current checkpoint and model formulation unchanged, then separate
dataset difficulty from evidential uncertainty behavior.

Command:

```bash
.venv-uacp/bin/python scripts/diagnose_predictor.py \
  --config configs/train_prototype.json \
  --checkpoint runs/prototype_predictor/uacp_predictor_prototype.pt \
  --training-results runs/prototype_predictor/training_results.json \
  --output-dir runs/prototype_diagnostics
```

Outputs:

```text
runs/prototype_diagnostics/diagnostics.json
runs/prototype_diagnostics/parameter_diagnostics.csv
runs/prototype_diagnostics/linear_interpolation_baseline.csv
runs/prototype_diagnostics/training_loss_scale.csv
```

### Predictor Parameter Diagnostic

All values below are computed with the same `Ng=16` evaluation mask. Parameter
statistics are over omitted subcarriers.

| Regime | NMSE all dB | NMSE omitted dB | psi mean/std | kappa mean/std | nu mean/std | nu-2K-1 mean/std | aleatoric mean/std | epistemic mean/std |
| --- | ---: | ---: | --- | --- | --- | --- | --- | --- |
| ID-Easy 20 ns | `-15.4818` | `-15.4141` | `0.037939 / 0.017579` | `2.094594 / 0.350328` | `2052.0029 / 0.391686` | `3.002902 / 0.391686` | `0.013111 / 0.010726` | `0.006620 / 0.009887` |
| ID-Hard 80 ns | `-15.4925` | `-15.4255` | `0.037259 / 0.017248` | `2.101161 / 0.351226` | `2052.0156 / 0.389066` | `3.015725 / 0.389066` | `0.012807 / 0.010613` | `0.006449 / 0.009847` |
| OOD-Near 120 ns | `-15.0930` | `-15.0074` | `0.036981 / 0.017159` | `2.104024 / 0.349693` | `2052.0232 / 0.386174` | `3.023168 / 0.386174` | `0.012676 / 0.010593` | `0.006371 / 0.009842` |

Observed mechanism:

- `psi` decreases as delay spread increases.
- `nu - 2K - 1` increases as delay spread increases.
- `kappa` also increases as delay spread increases.
- Therefore `aleatoric = psi / (nu - 2K - 1)` decreases.
- `epistemic = aleatoric / kappa` decreases further.

This shows the current evidential model is learning higher confidence, not lower
confidence, on the 120 ns OOD-Near split.

### Training Loss Scale Diagnostic

| Epoch | Train NLL | Train raw L_reg | Train lambda_reg x L_reg | Val NLL | Val raw L_reg | Val lambda_reg x L_reg |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `0.6461` | `493.6600` | `0.4937` | `0.5457` | `413.1374` | `0.4131` |
| 2 | `-0.0088` | `236.3519` | `0.2364` | `-0.6311` | `37.8892` | `0.0379` |
| 3 | `-0.2620` | `158.0932` | `0.1581` | `-0.3339` | `65.7211` | `0.0657` |
| 4 | `-0.4761` | `118.5148` | `0.1185` | `-0.8564` | `25.3210` | `0.0253` |
| 5 | `-0.5876` | `95.0352` | `0.0950` | `-0.8066` | `26.7865` | `0.0268` |

The regularization term is not negligible. Because the current reduction and
element-wise evidence shape differ from the paper's pair-level formulation, the
numeric value `lambda_reg=1e-3` should not be treated as equivalent to the
paper's effective regularization strength.

### Linear Interpolation Baseline

Same dataset, same `Ng=16` mask, no predictor.

| Regime | Linear interpolation NMSE omitted dB |
| --- | ---: |
| ID-Easy 20 ns | `-44.2249` |
| ID-Hard 80 ns | `-28.9145` |
| OOD-Near 120 ns | `-23.4312` |

Interpretation:

- Dataset/channel/mask do create a clear sparse reconstruction difficulty
  ordering: `20 ns` is easiest, `80 ns` is harder, `120 ns` is hardest.
- The current failure is therefore concentrated in the predictor/evidential
  formulation, loss calibration, or prototype training setup rather than the
  generated CFR dataset itself.

## Evidential Formulation Experiments - 2026-08-27

Goal: keep the Sionna-generated dataset, deterministic seed, `Ng=16` mask
policy, optimizer, learning rate, and prototype epoch count fixed, then modify
only the evidential formulation in stages.

These experiments are implementation-side approximations to the paper's
antenna-pair-level NIW meaning. They do not claim to reproduce an unspecified
full `2048 x 2048` covariance implementation.

### Experiment Definitions

| Name | Change | Config | Output |
| --- | --- | --- | --- |
| Baseline A | Element-wise `gamma/kappa/psi/nu`, element-wise NLL, element-wise regularizer | `configs/train_prototype.json` | `runs/prototype_predictor`, `runs/prototype_diagnostics` |
| Experiment B | Pair-level scalar `kappa/nu`; element-wise NLL and regularizer retained | `configs/train_formulation_b_pair_scalar.json` | `runs/formulation_b_pair_scalar` |
| Experiment C | B + diagonal multivariate Student-t NLL per antenna pair, `d=2048` | `configs/train_formulation_c_diag_mvnll.json` | `runs/formulation_c_diag_mvnll` |
| Experiment D | C + pair-level evidence regularizer | `configs/train_formulation_d_pair_reg.json` | `runs/formulation_d_pair_reg` |

Commands:

```bash
.venv-uacp/bin/python -m unittest tests.test_predictor -v

.venv-uacp/bin/python scripts/train_predictor.py \
  --config configs/train_formulation_b_pair_scalar.json \
  --output-dir runs/formulation_b_pair_scalar
.venv-uacp/bin/python scripts/diagnose_predictor.py \
  --config configs/train_formulation_b_pair_scalar.json \
  --checkpoint runs/formulation_b_pair_scalar/uacp_predictor_prototype.pt \
  --training-results runs/formulation_b_pair_scalar/training_results.json \
  --output-dir runs/formulation_b_pair_scalar_diagnostics

.venv-uacp/bin/python scripts/train_predictor.py \
  --config configs/train_formulation_c_diag_mvnll.json \
  --output-dir runs/formulation_c_diag_mvnll
.venv-uacp/bin/python scripts/diagnose_predictor.py \
  --config configs/train_formulation_c_diag_mvnll.json \
  --checkpoint runs/formulation_c_diag_mvnll/uacp_predictor_prototype.pt \
  --training-results runs/formulation_c_diag_mvnll/training_results.json \
  --output-dir runs/formulation_c_diag_mvnll_diagnostics

.venv-uacp/bin/python scripts/train_predictor.py \
  --config configs/train_formulation_d_pair_reg.json \
  --output-dir runs/formulation_d_pair_reg
.venv-uacp/bin/python scripts/diagnose_predictor.py \
  --config configs/train_formulation_d_pair_reg.json \
  --checkpoint runs/formulation_d_pair_reg/uacp_predictor_prototype.pt \
  --training-results runs/formulation_d_pair_reg/training_results.json \
  --output-dir runs/formulation_d_pair_reg_diagnostics
```

### Key Results

Omitted-subcarrier metrics, same `Ng=16` evaluation mask:

| Experiment | 20 NMSE dB | 80 NMSE dB | 120 NMSE dB | 20 Ale | 80 Ale | 120 Ale | 20 Epi | 80 Epi | 120 Epi |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Linear interpolation | `-44.2249` | `-28.9145` | `-23.4312` |  |  |  |  |  |  |
| Baseline A | `-15.4141` | `-15.4255` | `-15.0074` | `0.013111` | `0.012807` | `0.012676` | `0.006620` | `0.006449` | `0.006371` |
| Experiment B | `-16.1657` | `-16.1730` | `-15.7228` | `0.014447` | `0.013962` | `0.013771` | `0.007609` | `0.007196` | `0.007082` |
| Experiment C | `-18.5640` | `-18.5147` | `-17.7412` | `0.411422` | `0.404211` | `0.401234` | `0.416726` | `0.400847` | `0.397144` |
| Experiment D | `-18.0396` | `-18.0662` | `-17.5160` | `0.141441` | `0.136947` | `0.135489` | `0.132010` | `0.125037` | `0.123482` |

Parameter/correlation details are stored in:

```text
runs/formulation_comparison/summary.csv
runs/formulation_comparison/summary.md
runs/formulation_b_pair_scalar_diagnostics/parameter_diagnostics.csv
runs/formulation_c_diag_mvnll_diagnostics/parameter_diagnostics.csv
runs/formulation_d_pair_reg_diagnostics/parameter_diagnostics.csv
```

### Loss Scale

With `lambda_reg=1e-3`, Experiment C's validation regularizer contribution at
epoch 5 is small relative to the NLL:

```text
NLL = -2519.0610
raw L_reg = 13.5837
lambda_reg x L_reg = 0.0136
```

Experiment D's pair-level regularizer changes the effective scale substantially:

```text
NLL = -2312.7717
raw L_reg = 32146.8723
lambda_reg x L_reg = 32.1469
```

Therefore the paper-specified numeric `lambda_reg=1e-3` should be retained for
the controlled comparison, but its effective strength is not equivalent across
the element-wise and pair-level reductions.

### Interpretation

- Linear interpolation still confirms the dataset difficulty ordering
  `20 ns < 80 ns < 120 ns`.
- B/C/D all improve reconstruction and make `120 ns` harder than ID in
  predictor NMSE, especially C and D.
- None of B/C/D restores the expected uncertainty behavior. Aleatoric and
  epistemic uncertainty remain largest at `20 ns` and decrease toward `120 ns`.
- The first formulation change that clearly improves reconstruction under OOD
  is Experiment C, the diagonal multivariate NLL. The expected epistemic OOD
  increase does not appear in any tested stage.
- The next baseline-reproduction problem is not the CFR dataset or mask policy;
  it is the evidence learning objective/calibration and possibly the missing
  full covariance or an unspecified paper detail such as observation noise or
  channel-estimation SNR handling.

Decision: do not proceed to Full Fine-Tuning vs Partial Fine-Tuning yet.

## Baseline Reproduction STEP 1 - 2026-08-27

### 목적

OOD-Near `120 ns`가 training range `[10, 100] ns`에 가까워서 epistemic
uncertainty가 민감하게 반응하지 않는 것인지, 아니면 OOD-Far에서도
evidence mechanism 자체가 동작하지 않는 것인지 확인한다.

### 이전 단계에서 확인된 문제

Dataset/channel/mask는 Linear Interpolation 기준으로 `20 ns < 80 ns <
120 ns` 난이도를 만든다. 그러나 Experiment C까지도 aleatoric과
epistemic uncertainty가 OOD로 갈수록 감소했다.

### 변경 파일 / 함수

- Added [scripts/baseline_reproduction.py](/home/saslab01/Desktop/UACP_2026/scripts/baseline_reproduction.py): STEP별 diagnostic runner.
- Added [tests/test_baseline_reproduction.py](/home/saslab01/Desktop/UACP_2026/tests/test_baseline_reproduction.py): output row/curve/config serialization tests.

### 실험 조건

- Dataset: existing `data/prototype` for `20/80/120 ns`; newly generated clean
  CFR only for `1 ms`
- Checkpoint: `runs/formulation_c_diag_mvnll/uacp_predictor_prototype.pt`
- Model mode: Experiment C, pair-level `kappa/nu` + diagonal multivariate NLL
- Epoch: no retraining
- Batch size: eval batch `16`
- Mask: uniform `Ng=16`
- Seed: `20260819`
- SNR: `15 dB` is PAPER-SPECIFIED but not applied to clean CFR generation
- lambda: `1e-3`

### 실행 명령

```bash
/usr/bin/time -f 'STEP1 elapsed=%E max_rss_kb=%M' \
  .venv-uacp/bin/python scripts/baseline_reproduction.py step1-ood-far \
  --model-config configs/train_formulation_c_diag_mvnll.json \
  --checkpoint runs/formulation_c_diag_mvnll/uacp_predictor_prototype.pt \
  --dataset-config configs/dataset_prototype.json \
  --samples 200 \
  --experiment-name 'Experiment C + diag MV NLL' \
  --output-dir runs/baseline_reproduction/step1_ood_far
```

### 결과

| Regime | NMSE_all | NMSE_omitted | Aleatoric | Epistemic | psi | kappa | df_cov |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ID-Easy 20 ns | `-18.6034` | `-18.5640` | `0.411422` | `0.416726` | `0.451061` | `0.998764` | `1.188094` |
| ID-Hard 80 ns | `-18.5585` | `-18.5147` | `0.404211` | `0.400847` | `0.448519` | `1.002625` | `1.190580` |
| OOD-Near 120 ns | `-17.8214` | `-17.7412` | `0.401234` | `0.397144` | `0.446655` | `1.005048` | `1.194963` |
| OOD-Far 1 ms | `2.5615` | `2.8407` | `0.390166` | `0.389986` | `0.433749` | `0.996152` | `1.199741` |

Output:

```text
runs/baseline_reproduction/step1_ood_far/config.json
runs/baseline_reproduction/step1_ood_far/results.json
runs/baseline_reproduction/step1_ood_far/summary.csv
runs/baseline_reproduction/step1_ood_far/summary.md
```

### 논문에서 기대한 behavior

- `Aleatoric(80 ns) > Aleatoric(20 ns)`
- `Epistemic(120 ns or 1 ms) > Epistemic(ID)`

### 실제 behavior

- `Aleatoric(80) > Aleatoric(20)`: `False`
- `Epistemic(120) > Epistemic(ID)`: `False`
- `Epistemic(1 ms) > Epistemic(ID)`: `False`
- `1 ms` NMSE is much worse, but epistemic uncertainty is lower than ID.

### 판단

`FAIL`. STEP 1 Case B: OOD-Far에서도 epistemic이 증가하지 않는다. 이는
Near-OOD sensitivity만의 문제가 아니라 evidence learning/objective 자체가
OOD evidence를 제대로 낮추지 못하는 문제로 보는 것이 타당하다.

### 다음 단계

Delay spread sweep으로 uncertainty trend가 어느 구간에서 무너지는지 확인한다.

## Baseline Reproduction STEP 2 - 2026-08-27

### 목적

`20/80/120 ns` 세 점만으로는 trend 판단이 불안정하므로, delay spread sweep을
통해 reconstruction difficulty와 uncertainty parameter가 어떻게 움직이는지
curve로 확인한다.

### 이전 단계에서 확인된 문제

STEP 1에서 `1 ms` OOD-Far의 NMSE는 크게 악화되었지만 epistemic uncertainty는
증가하지 않았다.

### 변경 파일 / 함수

- [scripts/baseline_reproduction.py](/home/saslab01/Desktop/UACP_2026/scripts/baseline_reproduction.py): `step2-delay-sweep` subcommand, CSV/PNG curve writer.

### 실험 조건

- Delay spread: `10, 20, 40, 60, 80, 100, 120, 200, 500 ns, 1 ms`
- Samples per point: `200`
- Checkpoint/model/mask/seed/lambda: same as STEP 1
- SNR/noise: clean CFR only; `15 dB` not applied
- Channel model: `IMPLEMENTATION-ASSUMPTION`, TDL-A

### 실행 명령

```bash
/usr/bin/time -f 'STEP2 elapsed=%E max_rss_kb=%M' \
  .venv-uacp/bin/python scripts/baseline_reproduction.py step2-delay-sweep \
  --model-config configs/train_formulation_c_diag_mvnll.json \
  --checkpoint runs/formulation_c_diag_mvnll/uacp_predictor_prototype.pt \
  --dataset-config configs/dataset_prototype.json \
  --samples 200 \
  --experiment-name 'Experiment C + diag MV NLL' \
  --output-dir runs/baseline_reproduction/step2_delay_sweep
```

### 결과

| Delay | NMSE_omitted | Aleatoric | Epistemic | psi | kappa | df_cov |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 ns | `-18.5180` | `0.404249` | `0.405860` | `0.445690` | `1.008568` | `1.198792` |
| 20 ns | `-18.6843` | `0.408938` | `0.416737` | `0.444614` | `0.992642` | `1.183206` |
| 40 ns | `-18.6654` | `0.407385` | `0.405552` | `0.449207` | `1.007006` | `1.192078` |
| 60 ns | `-18.5999` | `0.406157` | `0.402288` | `0.450132` | `1.005582` | `1.188278` |
| 80 ns | `-18.4304` | `0.404099` | `0.401731` | `0.447075` | `1.000392` | `1.188685` |
| 100 ns | `-18.1335` | `0.403191` | `0.399932` | `0.446801` | `1.002582` | `1.189869` |
| 120 ns | `-17.7550` | `0.401886` | `0.398657` | `0.445722` | `1.002767` | `1.191450` |
| 200 ns | `-15.1675` | `0.401083` | `0.398140` | `0.444434` | `1.001720` | `1.191295` |
| 500 ns | `-6.9807` | `0.396750` | `0.393519` | `0.440156` | `1.002127` | `1.194090` |
| 1 ms | `2.8507` | `0.389992` | `0.389016` | `0.433172` | `0.998561` | `1.200148` |

Curve files:

```text
runs/baseline_reproduction/step2_delay_sweep/delay_vs_nmse_omitted.png
runs/baseline_reproduction/step2_delay_sweep/delay_vs_aleatoric.png
runs/baseline_reproduction/step2_delay_sweep/delay_vs_epistemic.png
runs/baseline_reproduction/step2_delay_sweep/delay_vs_psi.png
runs/baseline_reproduction/step2_delay_sweep/delay_vs_kappa.png
```

### 논문에서 기대한 behavior

- ID `[10,100] ns`: delay spread increases, NMSE and aleatoric increase,
  epistemic remains relatively stable.
- OOD `>100 ns`: epistemic increases.

### 실제 behavior

- NMSE worsens strongly after `100 ns`.
- Aleatoric does not increase over the ID range.
- Epistemic does not exceed ID in OOD.
- At `1 ms`, error-uncertainty correlation is near zero.

### 판단

`FAIL`. Reconstruction difficulty is visible in NMSE, but the evidential
parameters do not track the difficulty. This points away from the dataset and
toward evidence learning/calibration.

### 다음 단계

Before retraining, verify whether the current uncertainty score aggregation
differs from Eq. (12)-(13).

## Baseline Reproduction STEP 4A

### 목적

STEP 1-3에서 확인한 reconstruction difficulty와 반대 방향의 uncertainty가
`1k samples / 5 epochs`라는 작은 training scale 때문인지 확인하기 위해
Experiment D를 5k samples, 10 epochs로 제한하여 pilot했다.

### 설정

- `5,000 samples / 10 epochs`: `IMPLEMENTATION-ASSUMPTION / PILOT`, PAPER-SPECIFIED 아님
- Training CFR: clean TDL-A channel, RMS delay spread Uniform `[10, 100] ns`
- Validation/test: 기존 prototype split 유지; probe는 `20, 80, 120 ns, 1 ms`, 각 200 samples
- Mask: 기존 deterministic `Ng=16` fixed evaluation mask와 동일
- Seed: `20260819`; optimizer, learning rate, batch size는 Experiment D와 동일
- SNR: clean CFR 실험이며 `15 dB` noise는 적용하지 않음 (`UNKNOWN / NEEDS DESIGN DECISION`)

Experiment D를 primary로 선택했다. Experiment C가 reconstruction은 더 좋았지만
regularizer가 없고, D는 pair-level evidence regularizer를 포함하여 현재 구현 중
논문 Eq. (11)의 의미에 더 가깝다. 단, regularizer scale이 크므로 영향은 별도
logging했다.

### 저장 단계 오류와 원인

최초 학습은 10 epochs까지 완료했으나 `diagnose_regime()`의
`aleatoric_omitted` schema와 `write_training_csvs()`의 `aleatoric` lookup이
달라 후처리에서 실패했다. 학습 checkpoint와 epoch history는 보존되었다.

### 후처리 복구 방법

`probe_metrics_as_evaluation()` helper와 `step4a-finalize` subcommand를
추가하여 재학습 없이 `evaluation_table.csv`, curves, `summary.csv`,
`summary.md`를 재생성했다.

### 변경 파일 / 함수

- `scripts/baseline_reproduction.py`: `run_step4a`, `epoch_probe_row`,
  `probe_metrics_as_evaluation`, `interpret_step4`, curve/summary writers,
  `step4a-finalize`
- `tests/test_baseline_reproduction.py`: epoch probe, finalize schema,
  STEP 4 interpretation tests

### 실행 명령

```bash
.venv-uacp/bin/python scripts/baseline_reproduction.py step4a-scaled-training-pilot \
  --model-config configs/train_formulation_d_pair_reg.json \
  --dataset-config configs/dataset_prototype.json \
  --train-samples 5000 --probe-samples 200 --epochs 10 \
  --output-dir runs/baseline_reproduction/step4_scaled_training/pilot_5k_10ep_repro
```

최초 원본 run의 후처리는 다음으로 복구했다.

```bash
.venv-uacp/bin/python scripts/baseline_reproduction.py step4a-finalize \
  --output-dir runs/baseline_reproduction/step4_scaled_training/pilot_5k_10ep
```

### 예상 Runtime / 실제 Runtime

원본 run의 runtime estimate는 `1330.0 s`, dataset generation은 `14.3 s`,
training/probe는 `1257.3 s`였다. shell elapsed는 `21:14.20`, maximum RSS는
`2,317,000 KB`였다. 동일 seed repro도 별도 directory에서 완료되었다.

### Epoch별 uncertainty 변화

| Epoch | Ale 20 | Ale 80 | Epi 20 | Epi 120 | Epi 1 ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.158216 | 0.154795 | 0.145417 | 0.141778 | 0.136347 |
| 3 | 0.024471 | 0.023791 | 0.007085 | 0.006893 | 0.007609 |
| 5 | 0.009421 | 0.009128 | 0.001650 | 0.001587 | 0.002200 |
| 10 | 0.003307 | 0.003178 | 0.000351 | 0.000334 | 0.000464 |

Far-OOD epistemic은 epoch 3부터 두 ID regime보다 높아졌고 epoch 10까지
유지되었다. 그러나 80 ns aleatoric은 모든 epoch에서 20 ns보다 낮았으며,
120 ns epistemic도 모든 epoch에서 ID보다 낮았다.

### 최종 5k / 10 epoch 결과

| Regime | NMSE all | NMSE omitted | Aleatoric | Epistemic |
| --- | ---: | ---: | ---: | ---: |
| ID-Easy 20 ns | -21.3436 | -21.2917 | 0.003307 | 0.000351 |
| ID-Hard 80 ns | -21.2558 | -21.2026 | 0.003178 | 0.000333 |
| OOD-Near 120 ns | -20.4320 | -20.3570 | 0.003184 | 0.000334 |
| OOD-Far 1 ms | 2.3117 | 2.5856 | 0.004149 | 0.000464 |

### 논문에서 기대한 Behavior

- `Aleatoric(80 ns) > Aleatoric(20 ns)`
- `Epistemic(120 ns) > Epistemic(ID)`
- OOD reconstruction error 증가와 positive error-uncertainty relation

### 실제 Behavior

- `Aleatoric(80) > Aleatoric(20)`: `False`
- `Epistemic(120) > Epistemic(ID)`: `False`
- `Epistemic(1 ms) > Epistemic(ID)`: `True`
- OOD-Far NMSE는 크게 악화되었고, 최종 error-uncertainty Pearson correlation은 positive였으나 1 ms에서는 약했다.

### Same-seed reproducibility 결과

원본과 repro의 checkpoint SHA-256는 동일했다.
`final_results.json`의 history와 epoch probe 결과도 exact equality였다.
CSV의 수치도 동일하고 차이는 regime 행 순서와 output path뿐이었다.

### STEP 4A 최종 PASS/FAIL 또는 Partial PASS 판단

전체 UACP baseline behavior 기준으로는 `FAIL`이다. 다만 training scale을
늘리면 Far-OOD epistemic signal이 epoch 3부터 reproducibly 나타난다는 점에서
STEP 4A의 Far-OOD 가설은 `PARTIAL PASS`다. 이는 1k/5가 Far-OOD evidence
learning에 부족했을 가능성을 지지하지만, ID-Hard aleatoric과 Near-OOD
epistemic 실패까지 정리하지는 못한다. 초기부터 정상 ordering이었다가 후기에
무너지는 Case C 패턴은 관찰되지 않았다.

### STEP 4B 진행 여부

사용자 기준의 Case A에 해당하므로 10k/30 pilot을 수행할 정보 가치는 있다.
다만 5k/10도 약 21분이 걸렸고 핵심 두 behavior가 전혀 접근하지 않았으므로,
이번 단계에서는 자동 실행하지 않는다. 다음 실험은 bounded `10k/30`에서
epoch 10과 20에 early checkpoint를 저장하여 Far-OOD 신호의 지속 여부만 먼저
확인하거나, clean/noisy observation 차이를 직접 검증하는 STEP 5를 비교해
선택해야 한다. 현재 증거만으로는 120 ns sensitivity와 aleatoric ordering을
위해 STEP 5 noisy-observation ablation을 우선 추천한다.

### 다음 단계

STEP 4B를 승인할 경우 10k/30을 early-stop probe와 함께 실행한다. 그렇지 않으면
15 dB observation noise를 `IMPLEMENTATION-ASSUMPTION`으로 명시한 STEP 5를
진행한다. Partial Fine-Tuning은 baseline uncertainty behavior가 복구될 때까지
시작하지 않는다.

## Baseline Reproduction STEP 3 - 2026-08-27

### 목적

논문 Eq. (12)-(13)의 subcarrier-level uncertainty map과 omitted-subcarrier
score aggregation이 현재 8-channel 평균과 다른 ordering을 만드는지 확인한다.

### 이전 단계에서 확인된 문제

STEP 2에서 NMSE는 delay spread 증가를 반영했지만 aleatoric/epistemic score는
둘 다 감소했다.

### 변경 파일 / 함수

- Added [src/training/uncertainty.py](/home/saslab01/Desktop/UACP_2026/src/training/uncertainty.py): paper-style uncertainty aggregation utilities.
- Added [tests/test_uncertainty_aggregation.py](/home/saslab01/Desktop/UACP_2026/tests/test_uncertainty_aggregation.py): trace aggregation tests.
- [scripts/baseline_reproduction.py](/home/saslab01/Desktop/UACP_2026/scripts/baseline_reproduction.py): `step3-uncertainty-aggregation` subcommand.

### 실험 조건

- Same checkpoint/model/mask/seed/lambda as STEP 1
- Regimes: `20 ns`, `80 ns`, `120 ns`, `1 ms`
- No retraining
- Head/loss unchanged

### 실행 명령

```bash
/usr/bin/time -f 'STEP3 elapsed=%E max_rss_kb=%M' \
  .venv-uacp/bin/python scripts/baseline_reproduction.py step3-uncertainty-aggregation \
  --model-config configs/train_formulation_c_diag_mvnll.json \
  --checkpoint runs/formulation_c_diag_mvnll/uacp_predictor_prototype.pt \
  --dataset-config configs/dataset_prototype.json \
  --samples 200 \
  --experiment-name 'Experiment C + diag MV NLL' \
  --output-dir runs/baseline_reproduction/step3_uncertainty_aggregation
```

### 결과

| Regime | Ale current | Ale paper | Epi current | Epi paper | Total current | Total paper |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ID-Easy 20 ns | `0.411422` | `0.822844` | `0.416726` | `0.833453` | `0.828148` | `1.656297` |
| ID-Hard 80 ns | `0.404211` | `0.808421` | `0.400847` | `0.801694` | `0.805058` | `1.610115` |
| OOD-Near 120 ns | `0.401234` | `0.802469` | `0.397144` | `0.794289` | `0.798379` | `1.596757` |
| OOD-Far 1 ms | `0.388641` | `0.777281` | `0.387491` | `0.774982` | `0.776132` | `1.552263` |

### 논문에서 기대한 behavior

Eq. (12)는 subcarrier별 real/imag covariance trace를 antenna pair 평균으로
집계하고, Eq. (13)는 unreported subcarrier 평균을 취한다.

### 실제 behavior

Diagonal covariance approximation에서는 paper-style aggregation이 current
8-channel mean의 약 `2x` scale이지만 ordering은 바뀌지 않는다.

### 판단

`FAIL`. Aggregation mismatch alone is not the reason for uncertainty trend
failure.

### 다음 단계

STEP 4 scaled training is the next controlled experiment. Based on the observed
prototype runtime, `10,000 samples` and `30-50 epochs` at the current
implementation batch size may take hours. Before launching that long run, create
a bounded config and estimate runtime/storage, then run only after confirming
the intended scale.

## Baseline Reproduction STEP 5 - Clean vs 15 dB Observation Noise

### STEP 4A에서 확인된 사실

Clean training에서는 Far-OOD `1 ms` epistemic ordering만 회복되었고,
`Aleatoric(80) > Aleatoric(20)` 및 `Epistemic(120) > Epistemic(ID)`는 실패했다.

### STEP 5 목적

SNR 15 dB를 observed CFR noise로 해석했을 때 aleatoric calibration과
Near-OOD epistemic separation이 회복되는지 controlled ablation했다.

### 논문의 SNR 정보

- `PAPER-SPECIFIED`: SNR `15 dB`
- `UNKNOWN`: 정확한 noise 적용 stage는 PDF에서 확인되지 않음
- `IMPLEMENTATION-ASSUMPTION`: reported complex CFR에만 sample-wise AWGN 적용,
  target은 clean full CFR

### Noise 정의

각 sample의 reported CFR에 대해 `signal_power = mean(|H|^2)`,
`noise_power = signal_power / 10^(15/10)`을 사용했다. Complex noise는
`sqrt(noise_power/2) * (N_real + j N_imag)`이며 unreported 위치에는 noise를
추가하지 않았다. 실측 SNR은 `10 log10(signal_power / noise_power)`로 계산했다.

### 변경 파일 / 함수

- `src/training/data.py`: `add_complex_awgn`, `build_noisy_sparse_input`
- `scripts/train_predictor.py`: optional observation noise in train/eval
- `scripts/diagnose_predictor.py`: noisy diagnostic and measured SNR statistics
- `scripts/baseline_reproduction.py`: STEP 5 evaluation/training/comparison
- `tests/test_predictor.py`: noise boundary, target, SNR, seed tests

### Condition A

`Clean Train / Clean Eval`: 기존 STEP 4A 결과를 재사용했다.

### Condition B

`Clean Train / Noisy Eval`: STEP 4A checkpoint에만 15 dB observation noise를
적용했다. measured SNR은 네 regime에서 `14.986-15.016 dB`였다.

### Condition C

`Noisy Train / Noisy Eval`: Experiment D, 5k samples, 10 epochs를 유지하고
observed CFR에 15 dB noise를 적용하여 학습했다. runtime은 `21:50.54`,
max RSS는 `2,356,176 KB`였다.

### Optional Condition D

`Noisy Train / Clean Eval`: Condition C checkpoint를 재학습 없이 clean input으로
평가했다.

### 실험 조건

Dataset/channel/TDL-A/mask/seed/optimizer/LR/batch size/lambda는 STEP 4A와
동일하다. TDL-A는 계속 `IMPLEMENTATION-ASSUMPTION`이며 SNR stage는 `UNKNOWN`이다.
Probe는 `20, 80, 120 ns, 1 ms`, 각 200 samples, `Ng=16`이다.

### 최종 비교표

| Condition | Regime | NMSE omitted | Aleatoric | Epistemic | Err-Ale P | Err-Epi P |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A Clean->Clean | 20 ns | -21.2917 | 0.003307 | 0.000351 | 0.3505 | 0.3538 |
| A Clean->Clean | 80 ns | -21.2026 | 0.003178 | 0.000333 | 0.3344 | 0.3341 |
| A Clean->Clean | 120 ns | -20.3570 | 0.003184 | 0.000334 | 0.3743 | 0.3732 |
| A Clean->Clean | 1 ms | 2.5856 | 0.004149 | 0.000464 | 0.0192 | 0.0218 |
| B Clean->Noisy | 20 ns | -15.6318 | 0.003481 | 0.000373 | 0.1674 | 0.1731 |
| B Clean->Noisy | 80 ns | -15.6058 | 0.003327 | 0.000352 | 0.1414 | 0.1434 |
| B Clean->Noisy | 120 ns | -15.3486 | 0.003334 | 0.000352 | 0.1806 | 0.1814 |
| B Clean->Noisy | 1 ms | 2.6368 | 0.004363 | 0.000492 | 0.0216 | 0.0230 |
| C Noisy->Noisy | 20 ns | -18.2480 | 0.011062 | 0.001384 | 0.1730 | 0.1760 |
| C Noisy->Noisy | 80 ns | -17.6463 | 0.011011 | 0.001378 | 0.2300 | 0.2320 |
| C Noisy->Noisy | 120 ns | -15.6202 | 0.011164 | 0.001406 | 0.3230 | 0.3240 |
| C Noisy->Noisy | 1 ms | 1.5948 | 0.015788 | 0.002271 | 0.0685 | 0.0767 |
| D Noisy->Clean | 20 ns | -23.0056 | 0.010931 | 0.001363 | 0.4056 | 0.4087 |
| D Noisy->Clean | 80 ns | -21.4330 | 0.010903 | 0.001361 | 0.4244 | 0.4278 |
| D Noisy->Clean | 120 ns | -17.5771 | 0.011041 | 0.001386 | 0.3360 | 0.3382 |
| D Noisy->Clean | 1 ms | 1.5556 | 0.015614 | 0.002238 | 0.0650 | 0.0731 |

### Epoch별 결과

Condition C에서 `Epistemic(120)`은 epoch 6부터 20 ns보다 높아졌고,
epoch 7부터 80 ns보다도 높아졌다. Far-OOD epistemic은 epoch 3부터 ID보다
높았다. Aleatoric은 epoch 1-10 전체에서 `80 < 20`이었다.

### Aleatoric / Near-OOD / Far-OOD behavior

`Aleatoric(80) > Aleatoric(20)`은 Condition B/C 모두 실패했다. 반면 Condition C는
처음으로 `Epistemic(120) > Epistemic(20/80)`를 만족했고, `Epistemic(1 ms) >
Epistemic(ID)`도 유지했다. Error-uncertainty Pearson은 C에서 20/80/120 ns 기준
aleatoric `0.173/0.230/0.323`, epistemic `0.176/0.232/0.324`로 양수였으나
1 ms에서는 각각 `0.0685/0.0767`로 약했다.

### STEP 5 PASS / PARTIAL PASS / FAIL

`PARTIAL PASS`. Observation noise training은 Near-OOD와 Far-OOD epistemic을
개선했지만 ID-Hard aleatoric ordering은 복구하지 못했다. 따라서 전체 UACP
baseline behavior가 복구된 것은 아니다.

### 원인 해석

Observation noise는 input corruption과 Near-OOD pattern의 separation에는
도움이 되었지만 현재 evidence objective는 20/80 ns difficulty를 aleatoric
ordering으로 표현하지 못했다. 이는 논문의 실제 SNR 적용 위치가 확인됐다는
뜻이 아니다.

### STEP 6 진행 여부

다음은 `STEP 6 lambda_reg / evidence calibration`을 권장한다. 현재 noisy
formulation에서 regularizer scale과 uncertainty calibration을 먼저 분리하고,
그 뒤 10k/30 또는 covariance 확장을 판단한다. Partial Fine-Tuning은 보류한다.

## Baseline Reproduction STEP 6 - Lambda / Evidence Calibration

### STEP 5에서 확인된 사실

15 dB noisy training에서 `Epistemic(120) > Epistemic(ID)`와
`Epistemic(1 ms) > Epistemic(ID)`가 복구되었지만 `Aleatoric(80) >
Aleatoric(20)`은 실패했다.

### STEP 6 목적

pair-level evidence regularizer의 effective strength가 남은 aleatoric
failure의 원인인지 확인하기 위해 `lambda_reg`만 변경했다.

### Loss 구조

`L = L_NLL + lambda_reg * L_reg`

NLL은 diagonal multivariate Student-t likelihood이며 regularizer는
pair-level prediction error와 evidence를 결합한다.

### NLL의 역할

Clean target에 대한 CFR reconstruction 및 uncertainty distribution fitting을
담당한다.

### Evidence Regularizer의 역할

높은 prediction error에서 높은 evidence가 출력되는 것을 penalize한다.

### 왜 lambda calibration이 필요한가

논문 numeric `1e-3`를 사용하더라도 현재 pair-level reduction과 diagonal
`Psi` approximation의 effective scale이 논문과 동일하다고 보장할 수 없다.

### Paper-Specified

- 논문 명시값: `lambda_reg = 1e-3`
- `0`, `1e-4`, `1e-2`: `IMPLEMENTATION-ASSUMPTION / calibration sweep`

### Controlled Variables

TDL-A (`IMPLEMENTATION-ASSUMPTION`), dataset, 5k train samples, 10 epochs,
Experiment D, 15 dB observed complex AWGN, clean target, mask, seed, optimizer,
learning rate, batch size와 fixed probes를 고정했다.

### Lambda Sweep

- `0`: regularizer 제거, NLL-only control
- `1e-4`: 약한 regularization
- `1e-3`: STEP 5 Condition C reference 재사용
- `1e-2`: 현재보다 10배 강한 regularization

### Loss Scale 결과

| Lambda | Train NLL | Train lambda*L_reg | Train ratio | Val NLL | Val lambda*L_reg | Val ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | -2244.46 | 0.00 | 0.0000 | -2133.63 | 0.00 | 0.0000 |
| 1e-4 | -2243.54 | 3.55 | 0.0016 | -2172.25 | 3.34 | 0.0015 |
| 1e-3 | -2248.98 | 35.26 | 0.0157 | -2196.60 | 32.78 | 0.0149 |
| 1e-2 | -2270.75 | 339.20 | 0.1494 | -2249.91 | 312.91 | 0.1391 |

`lambda=1e-2`는 divergence는 없었지만 regularizer가 NLL의 약 14-15%를
차지했다.

### Final Comparison Table

| Lambda | Regime | NMSE omitted | Aleatoric | Epistemic | Err-Ale P | Err-Epi P | ratio |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 20 ns | -17.9311 | 0.010976 | 0.001428 | 0.257 | 0.257 | 0.0000 |
| 0 | 80 ns | -17.4089 | 0.010968 | 0.001429 | 0.272 | 0.271 | 0.0000 |
| 0 | 120 ns | -15.5175 | 0.011140 | 0.001462 | 0.264 | 0.264 | 0.0000 |
| 0 | 1 ms | 1.6109 | 0.015592 | 0.002330 | 0.064 | 0.073 | 0.0000 |
| 1e-4 | 20 ns | -18.1487 | 0.010515 | 0.001331 | 0.255 | 0.256 | 0.0016 |
| 1e-4 | 80 ns | -17.6035 | 0.010480 | 0.001330 | 0.214 | 0.287 | 0.0018 |
| 1e-4 | 120 ns | -15.6451 | 0.010595 | 0.001352 | 0.273 | 0.275 | 0.0035 |
| 1e-4 | 1 ms | 1.5695 | 0.014254 | 0.002038 | 0.062 | 0.070 | 0.1156 |
| 1e-3 | 20 ns | -18.2480 | 0.011062 | 0.001384 | 0.266 | 0.271 | 0.0153 |
| 1e-3 | 80 ns | -17.6463 | 0.011011 | 0.001378 | 0.292 | 0.296 | 0.0177 |
| 1e-3 | 120 ns | -15.6202 | 0.011164 | 0.001406 | 0.273 | 0.276 | 0.0350 |
| 1e-3 | 1 ms | 1.5948 | 0.015788 | 0.002271 | 0.069 | 0.077 | 1.1596 |
| 1e-2 | 20 ns | -18.6130 | 0.013579 | 0.002292 | 0.173 | 0.176 | 0.1375 |
| 1e-2 | 80 ns | -17.8195 | 0.013549 | 0.002277 | 0.221 | 0.224 | 0.1664 |
| 1e-2 | 120 ns | -15.6224 | 0.013813 | 0.002338 | 0.322 | 0.324 | 0.3490 |
| 1e-2 | 1 ms | 1.5199 | 0.020495 | 0.003979 | 0.076 | 0.083 | 11.4831 |

### Aleatoric 80 vs 20

| Lambda | Ale(80)-Ale(20) |
| ---: | ---: |
| 0 | -0.00000740 |
| 1e-4 | -0.00003520 |
| 1e-3 | -0.00005084 |
| 1e-2 | -0.00002992 |

모든 lambda에서 gap이 음수다. Regularizer strength만으로 ID-Hard
aleatoric failure를 정리할 수 없는 `Case C`다.

### Near-OOD Epistemic

모든 lambda에서 `epi_gap_120_id > 0`이었다. 값은 `3.30e-5`, `2.08e-5`,
`2.13e-5`, `4.65e-5`이며 STEP 5 Near-OOD signal은 유지됐다.

### Far-OOD Epistemic

모든 lambda에서 `epi_gap_1ms_id > 0`이었다. `lambda=1e-2`의 gap이 가장
컸지만 regularizer ratio도 가장 컸다.

### Error-bin Calibration Diagnostic

Omitted error를 tertile로 나눈 low/medium/high bin 결과를
`error_bin_calibration.csv`에 저장했다. 신규 lambda 모두 high-error bin의
aleatoric/epistemic 평균이 low-error bin보다 높아 global correlation과 같은
양의 방향이었다. `lambda=1e-3`은 기존 checkpoint에 대한 evaluation-only
refresh에서 error-bin을 추가 기록했다.

### Reconstruction Trade-off

Mean ID NMSE는 `lambda=0: -17.6700`, `1e-4: -17.8761`, `1e-3: -17.9471`,
`1e-2: -18.2162 dB`였다. 큰 lambda가 pilot reconstruction을 붕괴시키지는
않았지만 evidence scale과 regularizer dominance를 키웠다.

### Best Lambda

전체 목표를 만족하는 lambda는 없다. `1e-2`가 OOD gap은 가장 크지만 aleatoric
gap은 음수이고 ratio가 과도하다. 따라서 calibration 관점의 best lambda는
선택하지 않으며 `1e-3` reference를 유지한다.

### STEP 6 PASS / PARTIAL PASS / FAIL

전체 behavior recovery 기준 `FAIL`, `Case C`다. 모든 lambda에서 Aleatoric(80)이
20보다 낮았다. 반면 OOD epistemic behavior는 모든 조건에서 유지됐다.

### 원인 해석

Regularizer strength는 uncertainty scale과 OOD gap을 조절했지만 ID-Hard와
ID-Easy의 aleatoric ordering을 뒤집지 못했다. 남은 후보는 diagonal `Psi`
approximation, exact likelihood/reduction, 또는 aleatoric 정의와 clean/noisy
observation 관계다.

Gradient-only diagnostic은 실행하지 않았다. 각 lambda 학습 시간이 약 21분인
점을 고려해 다음 구조 진단 단계에서 대표 batch gradient를 추가하는 편이 낫다.

### 다음 단계

STEP 7 covariance 확장은 즉시 진행하지 않고, 먼저 aleatoric parameterization과
likelihood reduction을 재검토한다. 이후 low-rank covariance를 설계할 때 본 coarse
sweep과 `lambda=1e-3` reference를 control로 사용한다. Partial Fine-Tuning과
10k/30 학습은 계속 보류한다.

## Baseline Reproduction STEP 7 - Evidential Formula and Covariance Structure

### STEP 6까지 확인된 사실

STEP 6 lambda sweep에서 `lambda_reg`만으로는 `Aleatoric(80) > Aleatoric(20)`을 복구하지 못했다. 15 dB noisy training의 Near/Far-OOD epistemic ordering은 유지됐지만 ID-Hard aleatoric gap은 음수였다.

### STEP 7 목적

먼저 원 논문의 Eq. (5), (7)--(11)과 현재 diagonal multivariate 구현을 작은 차원 dense reference로 검증했다. 통과 후 full covariance를 피하는 bounded rank-4 structured covariance pilot을 수행했다.

### Paper Formulation

논문은 antenna pair별 `h`를 `d=2K` 차원 Gaussian으로 두고 NIW `(gamma, kappa, Psi, nu)`를 사용한다. Eq. (5)의 predictive Student-t는 `df=nu-d+1`, scale matrix `((kappa+1)/(kappa*df))*Psi`를 사용한다. Eq. (7)은 `Sigma_ale=Psi/(nu-d-1)`, Eq. (8)은 `Sigma_epi=Sigma_ale/kappa`이다. Eq. (9)는 softplus 제약과 `Psi=L L^T`를 요구한다. Eq. (10)은 pair별 predictive log density의 합이고, Eq. (11)은 전체 observed+omitted CFR의 pair squared error와 `Phi=kappa+nu`를 결합한다.

`PAPER-SPECIFIED`: full per-pair `Psi=L L^T`, `d=2K`, 위 식의 정의. `IMPLEMENTATION-ASSUMPTION`: batch/pair mean reduction과 finite prototype 계산을 위한 diagonal 또는 low-rank approximation.

### Current Implementation Mapping

| Paper item | Current implementation | Status |
| --- | --- | --- |
| Eq. (5) df | `nu-d+1`, `d=2K` | Exact |
| Eq. (5) scale | `((kappa+1)/(kappa*df))*psi_diag` | Exact diagonal specialization |
| Eq. (5) density | multivariate gamma, log determinant, diagonal Mahalanobis | Exact diagonal specialization |
| Eq. (7) | `psi_diag/(nu-d-1)` | Exact diagonal specialization |
| Eq. (8) | aleatoric divided by pair `kappa` | Exact |
| Eq. (9) | softplus positivity and `nu>d+1`; full `L` absent | Approximation |
| Eq. (10) | mean of pair log densities | Reduction assumption; paper writes a sum |
| Eq. (11) | pair error norm times `kappa+nu`, all target frequencies | Formula aligned; reduction is assumption |

The real layout is `[Re(pair0..3,k), Im(pair0..3,k)]`. The audit found direct `reshape(B,4,2K)` and interleaved scalar expansion were wrong for this layout. `channels_to_pair_vectors()` and corrected pair expansion now make the mapping explicit. Existing result files/checkpoints were preserved.

### STEP 7A Equation Audit

The dense diagonal reference for `d=4` and `d=8` matched the optimized diagonal NLL with maximum absolute log-probability error `0.0` at float64 precision (tolerance `1e-8`). Eq. (7), Eq. (8), total covariance consistency, and Eq. (11) full-target scope were also checked. STEP 7A is `PASS` after the layout correction. Audit artifacts are in `runs/baseline_reproduction/step7_covariance_structure/equation_audit/`.

### Why Diagonal Psi Is a Candidate

The paper factorizes across antenna pairs but retains a `2K`-dimensional frequency-domain covariance within each pair. The diagonal implementation loses off-diagonal frequency correlation, so a bounded structured pilot tested this approximation without allocating a dense `2048x2048` matrix.

### STEP 7B Low-Rank + Diagonal Psi

`Psi=D+UU^T` with rank `r=4` was used. `D` is the positive softplus diagonal head and `U` has shape `[B,4,2K,4]`, mapped with the target's real/imag ordering. The factor uses `1/sqrt(d)` scaling for numerical control. This is an `IMPLEMENTATION-ASSUMPTION / APPROXIMATION`, not a paper claim. NLL uses the matrix determinant lemma and Woodbury identity.

### Training Conditions

The pilot fixed TDL-A (`IMPLEMENTATION-ASSUMPTION`), existing CFR data, 5,000 training samples, 10 epochs, Experiment D, 15 dB observed-subcarrier complex AWGN (`IMPLEMENTATION-ASSUMPTION`), clean target, same mask, seed, optimizer, learning rate, batch size, and `lambda_reg=1e-3` as STEP 5 Condition C. The 5k/10 epoch scale is a prototype pilot, not paper-specified.

### Final Results

| Model | Regime | NMSE omitted | Aleatoric | Epistemic | Err-Ale P | Err-Epi P |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Diagonal corrected control | 20 ns | -18.4342 | 0.010542 | 0.001290 | 0.2392 | 0.2412 |
| Diagonal corrected control | 80 ns | -17.7426 | 0.010528 | 0.001287 | 0.2887 | 0.2917 |
| Diagonal corrected control | 120 ns | -15.6018 | 0.010654 | 0.001310 | 0.2802 | 0.2831 |
| Diagonal corrected control | 1 ms | 1.5445 | 0.014483 | 0.002003 | 0.0620 | 0.0697 |
| Low-rank rank4 | 20 ns | -15.6119 | 0.002177 | 0.018429 | 0.1965 | 0.1990 |
| Low-rank rank4 | 80 ns | -15.4903 | 0.002171 | 0.018268 | 0.2185 | 0.2458 |
| Low-rank rank4 | 120 ns | -13.7204 | 0.002164 | 0.018170 | 0.2133 | 0.2379 |
| Low-rank rank4 | 1 ms | 1.1698 | 0.002121 | 0.017694 | 0.0460 | 0.0430 |

The corrected diagonal gaps were `Ale80-20=-1.4048e-5`, `Epi120-ID=+2.0253e-5`, and `Epi1ms-ID=+7.1337e-4`. Rank-4 gave `Ale80-20=-6.6513e-6`, `Epi120-ID=-2.5827e-4`, and `Epi1ms-ID=-7.3487e-4`. Rank-4 therefore slightly reduced the aleatoric gap magnitude but broke the corrected diagonal Near/Far-OOD epistemic ordering. The original STEP5 table remains preserved as a historical result.

### Frequency Correlation Diagnostic

Rank-4 off-diagonal energy ratio was approximately `0.9974--0.9977` across regimes. Mean lag-1/2/4/8 covariance magnitudes were about `0.005--0.007` and did not show useful regime separation.

### Error-Uncertainty Calibration

Low-rank error correlations remained positive for ID regimes (`0.196--0.246`) but were weak at 1 ms (`0.043`, `0.043`). Error-tertile diagnostics are stored in `final_results.json`; rank-4 did not provide stronger OOD calibration than the diagonal control.

### Computational Cost

| Model | Parameters | Covariance-head parameters | Inference ms/sample | Peak GPU MB |
| --- | ---: | ---: | ---: | ---: |
| Diagonal corrected control | 11,815,320 | 4,632 | 2.672 | 94.69 |
| Low-rank rank4 | 11,821,496 | 10,808 | 3.036 | 94.69 |

The pilot added 6,176 parameters (`+0.052%`) and approximately `1.4%` measured single-sample inference latency. Training plus fixed probes took `1292.0 s` (`21.5 min`); data generation took `19.4 s`.

### STEP 7 PASS / PARTIAL PASS / FAIL

`FAIL / Case C`. The equation audit passed, but rank-4 covariance did not restore `Aleatoric(80)>Aleatoric(20)` and made both Near/Far-OOD epistemic gaps negative. No rank-8 run was started because rank-4 showed neither behavioral improvement nor useful regime-dependent covariance separation.

### 원인 해석

The diagonal likelihood is mathematically consistent after layout correction, so a basic Student-t formula mismatch is not the remaining explanation. Rank-4 can represent off-diagonal terms, but this pilot did not learn the desired regime separation and degraded established epistemic behavior. Remaining high-priority gaps are exact channel/Sionna setup, exact observation/pilot noise process, paper-scale training, and the paper's full `L` parameterization and reduction details. Partial Fine-Tuning remains blocked.

### 다음 단계

Do not run rank-8 automatically. First resolve the exact channel and observation pipeline or design a separately audited full-`L` feasibility experiment at much smaller dimension. Future predictor baselines must retain corrected pair layout and equation reference tests.

## STEP 6 Condition C Baseline Audit

### Audit Purpose

This was a read-only audit of the historical STEP 6 `lambda=1e-3` reference.
That reference reuses the STEP 5 Condition C noisy-train/noisy-eval checkpoint;
no training, checkpoint rewrite, or result rewrite was performed.

### Verified Condition C Settings

The actual training NPZ contains 5,000 samples with delay spread min/max
`10.002579/99.99332 ns`, label `train`, and metadata `Uniform[10,100] ns`.
The saved result contains 10 epochs and fixed probes for 20, 80, 120 ns and
1 ms. The model config records `learning_rate=1e-4`, `batch_size=8`,
`lambda_reg=1e-3`, seed `20260819`, evaluation `Ng=16`, and training grouping
factors `[4,8,16,32]`. The checkpoint was
`runs/baseline_reproduction/step5_snr_ablation/noisy_train_noisy_eval/uacp_predictor_step4a.pt`.
The recorded measured SNR is approximately 15 dB.

| Item | Audit status | Evidence / qualification |
| --- | --- | --- |
| 5,000 training samples | VERIFIED | Actual NPZ shape and DataLoader length |
| 10 epochs | VERIFIED | `final_results.json` and `training_history.json` |
| Uniform `[10,100] ns` | VERIFIED | Actual delay array and NPZ metadata |
| 20/80/120 ns probes | VERIFIED | Saved epoch probe results |
| 1 ms probe | VERIFIED | Saved epoch probe results |
| 15 dB observation | VERIFIED as implementation | Config and measured SNR; exact paper stage is UNKNOWN |
| LR, batch, lambda, mask, seed | VERIFIED | Config and training code |
| Checkpoint metadata/code revision | NOT RECORDED | Checkpoint is a plain `state_dict` |

`SNR=15 dB` is PAPER-SPECIFIED. Applying sample-wise complex AWGN to reported
CFR is an IMPLEMENTATION-ASSUMPTION because the paper does not specify the
noise/pilot/feedback stage.

### Paper NIW Shapes vs Implementation

For `K=1024`, the paper models one antenna pair as a real vector of dimension
`d=2K=2048` with `gamma_p:[2048]`, scalar `kappa_p`, `Psi_p:[2048,2048]`, and
scalar `nu_p` with `nu_p>2049`.

| Parameter | Paper per pair | Current `pair_scalar` implementation | Assessment |
| --- | --- | --- | --- |
| gamma | `[2048]` | external `[8,1024]`, converted to `[4,2048]` | PARTIAL MATCH |
| kappa | scalar | `[B,4,1]` | MATCH |
| Psi | `[2048,2048]`, `L L^T` | diagonal `[B,4,2048]` | MISMATCH / APPROXIMATION |
| nu | scalar, `>2049` | `[B,4,1]`, denominator positive | MATCH (pair scalar/admissibility) |

The current diagonal representation must not be described as the paper's full
covariance model.

### Real/Imag Pair Layout

The current real encoding is `[Re(pair0..3), Im(pair0..3)]`. Therefore pair
indices are `(0,4)`, `(1,5)`, `(2,6)`, and `(3,7)` in the eight real channels.
`channels_to_pair_vectors()` and `pair_vectors_to_channels()` implement and
round-trip this mapping; the synthetic mapping and existing unit tests pass.
The STEP 6 Condition C checkpoint/result predates this BUG-FIX. Its state dict
does not contain a code revision, so the exact revision is NOT RECORDED inside
the checkpoint; repository chronology and the STEP 7/8 audit identify it as
HISTORICAL / PRE-FIX.

### Eq. (7), Eq. (8), Eq. (12), and Eq. (13)

Current code computes the diagonal specialization
`Sigma_ale=diag(Psi)/(nu-d-1)` and `Sigma_epi=Sigma_ale/kappa` after pair-scalar
expansion, with `d=2048`. This is a MATCH for the diagonal approximation and
not for the paper's full `Psi`.

The current Eq.12/13 utility follows `pair -> Re/Im trace -> four-pair mean ->
omitted-subcarrier mean`, with `omitted=1-mask`; the synthetic audit passed.
However, the historical STEP 6 saved uncertainty table was produced before
the mapping/aggregation correction and its primary diagnostic used channelwise
omitted means. It therefore remains a historical diagnostic, not a verified
current Eq.12-to-Eq.13 result.

### Eq. (10), Eq. (11), and Loss Qualification

The saved configuration records diagonal multivariate Student-t NLL and
pair-level evidence regularization over all target components. Current code
implements one diagonal `2K`-dimensional density per pair and
`||h-gamma||^2*(kappa+nu)`, structurally aligned under the diagonal
approximation. The code uses mean reduction whereas the paper displays sums;
absolute loss scale is therefore an IMPLEMENTATION-ASSUMPTION. The historical
checkpoint learned with the pre-fix pair mapping, so its evidence/uncertainty
behavior is confounded.

### Final Assessment

`BASELINE INVALID FOR UNCERTAINTY COMPARISON`.

The historical run is usable as a finite, noisy reconstruction diagnostic, but
its Aleatoric/Epistemic values must not be used as the corrected paper-faithful
baseline. The reasons are the pre-fix pair mapping/broadcast semantics, the
lack of checkpoint code provenance, and the diagonal `Psi` approximation.
The read-only audit artifacts are in
`runs/baseline_reproduction/step6_condition_c_audit/`.

### Recommended Next Experiments

First rerun Condition C with the current corrected mapping while saving config
and code provenance before interpreting uncertainty. Then perform STEP 8B's
channel/observation model audit against the paper wording for Sionna/model,
delay spread, SNR/pilot stage, feedback, normalization, OFDM/CP, mobility, and
paper-scale training. Do not begin Partial Fine-Tuning or treat the historical
STEP 6 uncertainty table as a valid comparison baseline yet.

## Baseline Reproduction STEP 8A - Corrected Baseline Revalidation

### STEP 7 Pair Layout Bug and Purpose

The current encoding is `[Re(pair0..3), Im(pair0..3)]`; therefore pair 0 is
channels 0 and 4, pair 1 is 1 and 5, pair 2 is 2 and 6, and pair 3 is 3 and 7.
The former direct reshape grouped adjacent channels as real/imag of one pair,
and the former scalar broadcast used interleaved pairs. `channels_to_pair_vectors()`
and `pair_vectors_to_channels()` now enforce the corrected mapping. STEP 6 was
`HISTORICAL / PRE-FIX`; STEP 8A checks whether its lambda conclusion survives in
the `CURRENT VALID BASELINE`.

### Controlled Variables and Lambda Conditions

TDL-A (`IMPLEMENTATION-ASSUMPTION`), existing CFR data, 15 dB observed-CFR AWGN
(`IMPLEMENTATION-ASSUMPTION`), clean target, Experiment D diagonal formulation,
5k samples, 10 epochs, mask, optimizer, learning rate, batch size, seed, and
fixed probes were held constant. Lambda values were `0`, `1e-3`, and `1e-2`.
The 5k/10 epoch scale is `IMPLEMENTATION-ASSUMPTION / PILOT`; `lambda=1e-3` is
the `PAPER-SPECIFIED` numeric value. The corrected STEP 7 `lambda=1e-3` result
was reused without retraining.

### Final Corrected Results

| Lambda | Regime | NMSE omitted | Aleatoric | Epistemic | psi | nu-d-1 | kappa | Err-Ale P | Err-Epi P |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 20 ns | -18.0936 | 0.010998 | 0.001466 | 0.193581 | 17.71461 | 7.56846 | 0.254 | 0.251 |
| 0 | 80 ns | -17.5201 | 0.010984 | 0.001463 | 0.193382 | 17.64064 | 7.51859 | 0.288 | 0.290 |
| 0 | 120 ns | -15.5629 | 0.011177 | 0.001501 | 0.195249 | 17.50324 | 7.45801 | 0.273 | 0.275 |
| 0 | 1 ms | 1.5861 | 0.015269 | 0.002311 | 0.234604 | 15.51060 | 6.68229 | 0.062 | 0.070 |
| 1e-3 | 20 ns | -18.4342 | 0.010542 | 0.001290 | 0.193981 | 18.50625 | 8.23536 | 0.239 | 0.241 |
| 1e-3 | 80 ns | -17.7426 | 0.010528 | 0.001287 | 0.193554 | 18.41296 | 8.18860 | 0.289 | 0.292 |
| 1e-3 | 120 ns | -15.6018 | 0.010654 | 0.001310 | 0.194851 | 18.31402 | 8.14331 | 0.280 | 0.283 |
| 1e-3 | 1 ms | 1.5445 | 0.014483 | 0.002003 | 0.233414 | 16.24528 | 7.30009 | 0.062 | 0.070 |
| 1e-2 | 20 ns | -18.6654 | 0.013234 | 0.002242 | 0.193701 | 14.71178 | 5.94332 | 0.252 | 0.254 |
| 1e-2 | 80 ns | -17.8788 | 0.013166 | 0.002224 | 0.192869 | 14.68324 | 5.93198 | 0.319 | 0.322 |
| 1e-2 | 120 ns | -15.6704 | 0.013444 | 0.002287 | 0.195377 | 14.56355 | 5.88903 | 0.303 | 0.305 |
| 1e-2 | 1 ms | 1.5340 | 0.020038 | 0.003913 | 0.245949 | 12.43187 | 5.18849 | 0.078 | 0.085 |

### Gap and Parameter Analysis

| Lambda | Ale80-20 | psi80-20 | denom80-20 | Epi120-ID | Epi1ms-ID |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | -1.4354e-05 | -1.9871e-04 | -7.3973e-02 | +3.5227e-05 | +8.4446e-04 |
| 1e-3 | -1.4048e-05 | -4.2689e-04 | -9.3288e-02 | +2.0253e-05 | +7.1337e-04 |
| 1e-2 | -6.8218e-05 | -8.3260e-04 | -2.8533e-02 | +4.5849e-05 | +1.6709e-03 |

All corrected aleatoric gaps are negative. The denominator is also lower at
80 ns, which would increase, not decrease, aleatoric uncertainty. The dominant
cause is therefore lower learned `psi` at 80 ns. Within-regime error-bin
calibration remains positive: for lambda 0 ID-Easy aleatoric rises from
`0.010413` to `0.011994` and epistemic from `0.001385` to `0.001604` across
low-to-high error tertiles.

### Epoch-wise Gap Analysis

Each training directory contains all 10 epochs in `epoch_probe_results.csv`.
Aleatoric 80-vs-20 remained negative throughout, approaching zero late in
training. Near-OOD epistemic became positive around epochs 6--8, while Far-OOD
became positive after the early epochs. No corrected run showed a late positive
aleatoric ordering.

### Pre-fix STEP 6 vs Corrected STEP 8A

Pre-fix values are historical diagnostics and corrected values are the valid
baseline:

| Lambda | Pre Ale gap | Corrected Ale gap | Pre Near Epi | Corrected Near Epi | Pre Far Epi | Corrected Far Epi |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | -7.4003e-06 | -1.4354e-05 | +3.3001e-05 | +3.5227e-05 | +9.0108e-04 | +8.4446e-04 |
| 1e-3 | -5.0836e-05 | -1.4048e-05 | +2.1330e-05 | +2.0253e-05 | +8.8630e-04 | +7.1337e-04 |
| 1e-2 | -2.9922e-05 | -6.8218e-05 | +4.6513e-05 | +4.5849e-05 | +1.6875e-03 | +1.6709e-03 |

### STEP 8A PASS / PARTIAL PASS / FAIL

`FAIL / Case C`. Corrected mapping does not change the conclusion: lambda
strength alone is not the primary cause of the aleatoric regime-ordering failure.
Near/Far-OOD epistemic remains positive in the corrected diagonal control, but
there is no lambda that satisfies all target behaviors. Next recommendation is
`STEP 8B Channel / Observation Model Audit`, comparing paper wording for exact
Sionna/channel generation, delay spread, SNR, pilot/channel estimation,
feedback, normalization, sample count, and epochs against current TDL-A,
one OFDM symbol, CP=0, mobility=0, normalize=false, and sample-wise AWGN.
Do not start rank-8, full covariance, Partial Fine-Tuning, or 100k/150-epoch
training yet.

## Covariance Experiment - Frequency-Local Banded Psi

### Background and hypothesis

Experiment A showed that sample/pair-level evidence calibration exists, while
Experiment B showed a strong difference in clean-CFR frequency correlation:
the pair-averaged `R(32)` was `0.993236` (20 ns), `0.904606` (80 ns), and
`0.830189` (120 ns). The single hypothesis tested here was that diagonal
`Psi` misses this frequency-local off-diagonal structure and therefore misses
the ID-Hard Aleatoric separation.

The historical STEP 7 rank-4 pilot used a generic global `Psi=D+UU^T` factor
with `U:[B,4,2048,4]`. It was not frequency-local and its learned lag
diagnostics did not separate the regimes. The present treatment is therefore
not a rerun of rank-4.

### Banded representation

The treatment uses corrected pair vectors in interleaved frequency blocks
`[Re(k), Im(k)]`. A lower block-banded factor `L` is constructed with factor
bandwidth 32 and `Psi=L L^T`; each block contains 33 frequencies and the
final block is identity-padded. This is a bounded block-local implementation,
not the paper's unspecified feasible head and not a full 2048x2048 matrix.
The off-diagonal 2x2 Re/Im blocks are bounded with `tanh` and factor scale
`0.05`; the positive diagonal comes from the existing positive `Psi` head.
These choices are `IMPLEMENTATION-ASSUMPTION / DIAGNOSTIC PILOT`.

Bandwidth 32 was selected because lag 1 was nearly identical across regimes,
lag 8 began to separate them, and lag 32 showed clear 20/80/120 separation.
The Student-t df, scale factor, pair-scalar `kappa/nu`, Eq. (7)/(8), Eq. (11),
and lambda `1e-3` were unchanged. TDL-A and observed-CFR 15 dB AWGN remain
`IMPLEMENTATION-ASSUMPTION`; all other training conditions match the current
valid baseline.

### Feasibility and training

The K=1024 forward/backward feasibility test passed on `cuda:0 / NVIDIA GB10`
with factor shape `[1,4,32,66,66]`, finite loss, approximately 1.16 seconds,
and 200.3 MiB peak allocation. The 5k/10 epoch treatment used the command:

`.venv-uacp/bin/python scripts/train_banded_covariance.py --config configs/covariance_banded_lag32_5k_10ep.json --output-dir runs/current_valid_baseline/covariance_experiments/banded_lag32_5k_10ep`

Training took `3338.439 s` (55.6 minutes). The treatment has `11,914,136`
parameters versus `11,815,320` for the diagonal control, an increase of
`98,816` parameters. Paired inference was `164.39 ms` versus `70.46 ms` per
sample in the measurement script; peak allocation was `161.88 MiB` for both
measurements.

### Diagonal versus banded result

The following uses identical read-only probe seeds for both models:

| Model | Regime | NMSE omitted (dB) | Aleatoric | Epistemic |
| --- | --- | ---: | ---: | ---: |
| Diagonal control | 20 ns | -17.6840 | 0.020907141 | 0.002677559 |
| Diagonal control | 80 ns | -17.2224 | 0.020818593 | 0.002642030 |
| Diagonal control | 120 ns | -15.2716 | 0.021020940 | 0.002675008 |
| Diagonal control | 1 ms | 1.5576 | 0.028841565 | 0.004121700 |
| Banded lag32 | 20 ns | -17.3219 | 0.009978971 | 0.001630803 |
| Banded lag32 | 80 ns | -16.9415 | 0.009986444 | 0.001629854 |
| Banded lag32 | 120 ns | -14.9227 | 0.009986794 | 0.001629742 |
| Banded lag32 | 1 ms | 1.3940 | 0.010126974 | 0.001659934 |

| Model | Ale 80-20 | Epi 120-ID | Epi 1ms-ID |
| --- | ---: | ---: | ---: |
| Diagonal control | -8.8548e-5 | -2.5515e-6 | +1.4441e-3 |
| Banded lag32 | +7.4732e-6 | -1.0609e-6 | +2.9131e-5 |

The positive Aleatoric gap is only `7.47e-6`, while the overall uncertainty
scale is about half of the control. Reconstruction is also worse on the ID
regimes by about `0.32 dB`. Near-OOD Epistemic remains slightly negative and
Far-OOD Epistemic is much weaker than the diagonal control.

### Learned covariance diagnostic

The banded factor learned nonzero local covariance: mean off-diagonal/total
energy was approximately `0.5812` for 20/80/120 ns. However, its lag summaries
were nearly identical across those regimes:

| Regime | Learned lag 1 | Learned lag 8 | Learned lag 16 | Learned lag 32 |
| --- | ---: | ---: | ---: | ---: |
| 20 ns | 0.022759 | 0.021301 | 0.016066 | 0.001737 |
| 80 ns | 0.022761 | 0.021303 | 0.016068 | 0.001736 |
| 120 ns | 0.022760 | 0.021301 | 0.016067 | 0.001738 |
| 1 ms | 0.022806 | 0.021332 | 0.016086 | 0.001707 |

Thus the treatment represented nonzero frequency-local structure but did not
learn the regime-dependent decay observed in the ground-truth channel.
Sample/pair error correlations also weakened from the control's approximately
`0.23--0.26` ID Pearson values to `0.05--0.06`; Far-OOD correlations became
slightly negative.

### Historical STEP 7 rank-4 comparison

STEP 7 rank-4 is historical and not a valid control-equivalent run. Its
reported gaps were Ale `-6.65e-6`, Near Epi `-2.58e-4`, and Far Epi
`-7.35e-4`; its lag diagnostics were approximately `0.005--0.007` and did
not separate regimes. The new banded treatment improves the Ale gap relative
to that historical run, but does not establish regime-dependent learned
covariance or preserve the control's OOD signal.

### Judgment and next step

`FAIL`. The small positive Aleatoric gap is not sufficient evidence for H1:
the learned lag covariance was effectively regime-invariant, reconstruction
degraded, and sample/pair calibration plus Far-OOD Epistemic weakened. The
primary hypothesis is therefore not confirmed by this bounded pilot. Do not
automatically run bandwidth/rank sweeps, full covariance, paper-scale
training, channel estimation, or Partial Fine-Tuning. The next single
recommended experiment is a channel/observation-model audit against the
paper wording, especially exact channel model, SNR/feedback stage, and
normalization assumptions.

Artifacts are in
`runs/current_valid_baseline/covariance_experiments/banded_lag32_5k_10ep/`.

## Observation Experiment - Pilot-based LS CFR Estimation

### Purpose

This diagnostic tests whether the uncertainty-ordering failure is caused by
the direct-CFR noise approximation rather than by the predictor. The current
valid diagonal baseline was not changed or retrained; a separate treatment was
trained with the pilot observation mode.

### Paper status and assumptions

- `PAPER-SPECIFIED`: SNR = 15 dB.
- `UNKNOWN`: the paper does not disclose the exact pilot sequence, allocation,
  estimator, pilot power normalization, or noise injection stage.
- `IMPLEMENTATION-ASSUMPTION`: two orthogonal pilot slots with `X=I2`, unit
  amplitude, received-pilot complex AWGN at 15 dB, and least-squares CFR
  estimation. The target remains the clean full CFR.
- `IMPLEMENTATION-ASSUMPTION`: TDL-A, 5k/10-epoch pilot, zero mobility, and
  disabled channel normalization remain unchanged from the current baseline.

### Pipelines

Control: `H -> sparse reported selection -> direct complex AWGN -> sparse
input + mask -> predictor`.

Treatment: `H -> Y=HX+N -> LS H_hat=Y X^H(XX^H)^-1 -> sparse reported selection
-> sparse input + mask -> predictor`.

For `X=I2`, the LS expression reduces exactly to `H_hat=H+N`. The implementation
uses this algebraically equivalent form, so this particular pilot design has
the same observation distribution as direct CFR AWGN; it is not evidence that
all pilot estimators are equivalent.

### Fixed conditions

Both runs use the current corrected diagonal Experiment D model, 5,000 train
samples, 10 epochs, batch size 8, Adam, learning rate `1e-4`, lambda
`1e-3`, seed `20260819`, training masks `[4,8,16,32]`, evaluation `Ng=16`,
clean targets, and 15 dB noise. The treatment ran on CUDA `NVIDIA GB10` for
`1229.642 s`.

### Observation sanity

| Regime | Direct observed NMSE (dB) | Pilot-LS observed NMSE (dB) | Direct measured SNR (dB) | Pilot-LS measured SNR (dB) |
| --- | ---: | ---: | ---: | ---: |
| 20 ns | -14.9794 | -14.9794 | 14.9983 | 14.9983 |
| 80 ns | -15.0043 | -15.0043 | 15.0165 | 15.0165 |
| 120 ns | -15.0311 | -15.0311 | 15.0420 | 15.0420 |
| 1 ms | -15.0239 | -15.0239 | 15.0271 | 15.0271 |

### Reconstruction and uncertainty

The paired linear interpolation and predictor results are in
`runs/current_valid_baseline/observation_experiments/pilot_ls_15db_5k_10ep/comparison_v2/`.
The treatment and control rows are numerically identical because of `X=I2`.

| Model | 20 NMSE omitted | 80 NMSE omitted | 120 NMSE omitted | 1 ms NMSE omitted |
| --- | ---: | ---: | ---: | ---: |
| Direct-CFR-AWGN | -17.7460 | -17.2458 | -15.2809 | 1.5601 |
| Pilot-LS | -17.7460 | -17.2458 | -15.2809 | 1.5601 |

| Model | Ale(80)-Ale(20) | Epi(120)-Epi(ID) | Epi(1ms)-Epi(ID) |
| --- | ---: | ---: | ---: |
| Direct-CFR-AWGN | -7.9554e-5 | -2.1623e-6 | +1.4624e-3 |
| Pilot-LS | -7.9554e-5 | -2.1623e-6 | +1.4624e-3 |

The predictor preserves the expected NMSE ordering `20 < 80 < 120`; however,
`Ale(80)>Ale(20)` remains false and Near-OOD Epistemic remains slightly below
the ID maximum. Far-OOD Epistemic remains positive. Psi, kappa, nu, and the
pair denominator are identical between conditions to the reported precision,
and sample-level error/uncertainty Pearson correlations are also identical.

### Result

`FAIL` for the stated treatment hypothesis. The experiment was numerically
valid and stable, but the identity orthogonal pilot makes the treatment the
same effective observation model as the control. Therefore it cannot explain
or repair the remaining uncertainty ordering failure. Existing baseline,
historical results, and checkpoints were preserved.

### Next recommendation

The next single experiment should use a genuinely non-identity pilot-based
observation audit, with a pilot allocation and estimator documented as a new
`IMPLEMENTATION-ASSUMPTION`, only after choosing a pilot design that does not
algebraically collapse to direct CFR AWGN. Do not change the model or start
fine-tuning before that controlled observation comparison is reviewed.

## Observation Experiment - Requested-Subcarrier Orthogonal Pilot LS

### Purpose

This is a Gate-1/Gate-2 observation-only diagnostic of the paper's stated
flow: the requested subset `S_u` is selected first, sounding pilots are
received, the receiver estimates CFR only on `S_u`, and the sparse estimate
plus mask is fed back. No predictor training was performed in this experiment.

### Paper-specified and unknown details

`PAPER-SPECIFIED`: sounding pilots are known at the receiver; the receiver
estimates requested CFR entries and feeds back a sparse CFR/mask for full CFR
prediction. `UNKNOWN IN PAPER`: pilot sequence, pilot-symbol count, pilot power,
estimator, noise stage, and exact SNR normalization.

`IMPLEMENTATION-ASSUMPTION`: `T_p=2` normalized equal-power Hadamard pilot,
`X=(1/sqrt(2))*[[1,1],[1,-1]]`, complex AWGN added to received pilot `Y`, and
ordinary LS `H_hat=Y X^H (X X^H)^-1`. The requested mask is applied before
generating/estimating pilot observations. The nominal received-pilot SNR is
15 dB, defined from reported `||HX||_F^2 / ||N||_F^2`.

### Control and treatment

Control: `H -> requested entries -> direct CFR AWGN -> sparse feedback`.

Treatment: `S_u -> requested pilot sounding -> Y[k]=H[k]X+N[k] -> LS H_hat[k]
-> sparse feedback`.

The treatment does not estimate a full CFR before selecting `S_u`.

### Gate 1 - observation equivalence

| Regime | Direct NMSE (dB) | Pilot-LS NMSE (dB) | Direct SNR (dB) | Pilot SNR (dB) | KS distance |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20 ns | -14.9793 | -14.9793 | 14.9901 | 14.9901 | 0.00377 |
| 80 ns | -14.9866 | -14.9866 | 14.9985 | 14.9985 | 0.00240 |
| 120 ns | -15.0121 | -15.0121 | 15.0253 | 15.0253 | 0.00306 |
| 1 ms | -14.9852 | -14.9852 | 14.9994 | 14.9994 | 0.00329 |

The same-seed element-wise errors are not identical because the Hadamard pilot
rotates white received noise (`max abs difference` approximately `1.04--1.48`).
However, their variances differ by at most `6.7e-7`, error second moments differ
by about `2e-9`, and KS distances are below `0.004`. Thus the two observation
distributions are equivalent for this diagnostic.

### Gate 2 - requested-subcarrier estimation dependence

The pilot-LS observation NMSE is approximately constant across delay regimes:
`-14.9793`, `-14.9866`, `-15.0121`, and `-14.9852` dB for 20, 80, 120 ns,
and 1 ms. Independent per-subcarrier LS does not use frequency correlation,
so it does not create a meaningful receiver-side delay-spread difficulty curve.

### Decision

`OBS-EQUIVALENT`. Under a full-rank unitary pilot and white AWGN, requested
subcarrier LS reduces in distribution to additive CFR noise. The UACP
treatment was therefore not trained, avoiding a duplicate 5k/10-epoch run.
The result does not rule out a frequency-aware pilot-grid estimator; it shows
that this simple per-requested-subcarrier LS model cannot test that hypothesis.

Artifacts: `runs/current_valid_baseline/observation_experiments/requested_subcarrier_hadamard_ls_15db/gate1_complete/`.

## Paper Figure 5-8 Style Uncertainty Diagnostics

### Purpose and fixed baseline

This is a read-only evaluation of the current-valid checkpoint; no training,
model, loss, lambda, or observation change was made. The checkpoint is
`runs/current_valid_baseline/condition_c_5k_10ep_lambda1e3_corrected/checkpoint_with_provenance.pt`
and its SHA-256 is
`c9df50913b5b48688c8576bbb174b5645a842b269667cacf2822dc049cd626fc`.
Evaluation uses Condition C, reported-subcarrier 15 dB AWGN
(`IMPLEMENTATION-ASSUMPTION`), `Ng=16`, corrected pair mapping, diagonal Psi,
and Eq.12 -> Eq.13 sample-level aggregation. The paper's exact Figure 5-8
dataset/plot implementation is not available, so these are qualitative
diagnostics rather than numeric reproduction.

Artifacts:
`runs/paper_style_uncertainty_diagnostics_20260904/`.

### P1 - Total uncertainty as error proxy

For each sample, omitted-subcarrier NMSE in dB was paired with
`U_total = U_ale + U_epi`, where each score was computed sample-wise using
Eq.12 and Eq.13. Pearson and Spearman are within-regime ranking diagnostics;
Spearman is not a paper metric.

| Regime | Mean NMSE (dB) | Mean U_total | Pearson | Spearman |
| --- | ---: | ---: | ---: | ---: |
| 20 ns | -17.7118 | 0.023562 | -0.2846 | -0.3265 |
| 80 ns | -17.2502 | 0.023466 | -0.6201 | -0.6257 |
| 120 ns | -15.3734 | 0.023690 | -0.6356 | -0.6698 |
| 1 ms | 1.5548 | 0.033010 | -0.1197 | -0.1184 |

Using NMSE dB as requested, all correlations are negative. Higher uncertainty
does not identify higher-error samples in this checkpoint. P1 verdict: `FAIL`.

### P2 - Risk coverage

Samples were sorted by low total uncertainty and retained at 10% increments.
Because less-negative NMSE is worse, a successful curve would start below the
100% value and generally rise toward it.

| Regime | 10% NMSE | 30% NMSE | 100% NMSE |
| --- | ---: | ---: | ---: |
| 20 ns | -17.2104 | -17.5207 | -17.7118 |
| 80 ns | -16.4594 | -16.7441 | -17.2502 |
| 120 ns | -14.2708 | -14.6155 | -15.3734 |
| 1 ms | 1.6850 | 1.6138 | 1.5548 |

The low-uncertainty subsets are worse, not safer, in every regime. P2 verdict:
`FAIL`.

### P3 - ID delay-spread sweep

The 10, 20, 40, 60, 80, and 100 ns probes are all inside the training range
`[10,100] ns`. Each has 200 newly generated CFR samples, `Ng=16`, and 15 dB
observation noise. Results include sample standard deviations and 95% CI
half-widths in `p3_delay_sweep.csv`.

| Delay | NMSE (dB) | Aleatoric | Epistemic |
| ---: | ---: | ---: | ---: |
| 10 ns | -17.6697 | 0.020513 | 0.002608 |
| 20 ns | -17.7283 | 0.020427 | 0.002591 |
| 40 ns | -17.7392 | 0.020275 | 0.002562 |
| 60 ns | -17.6313 | 0.020542 | 0.002602 |
| 80 ns | -17.2591 | 0.020791 | 0.002637 |
| 100 ns | -16.4009 | 0.020765 | 0.002634 |

P3-1 NMSE trend: `PASS` only from the broad 40-100 ns portion; overall
`PARTIAL` because 10-40 ns is effectively flat within variability. P3-2
Aleatoric hardness trend: `PARTIAL`; it rises from 40 to 80 ns but is nearly
flat overall and does not provide a clear monotonic ID calibration. P3-3 ID
Epistemic stability: `PASS` qualitatively; it remains near `0.00256--0.00264`.

### P4 - OOD Epistemic detection

Epistemic distributions were evaluated over 200 samples per regime.

| Regime | Mean | Median | Std | P05 | P25 | P75 | P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 ns | 0.002677 | 0.002565 | 0.000529 | 0.002035 | 0.002265 | 0.002961 | 0.003790 |
| 80 ns | 0.002644 | 0.002622 | 0.000274 | 0.002281 | 0.002438 | 0.002808 | 0.003177 |
| 120 ns | 0.002677 | 0.002655 | 0.000258 | 0.002323 | 0.002487 | 0.002828 | 0.003141 |
| 1 ms | 0.004132 | 0.003975 | 0.000813 | 0.003146 | 0.003561 | 0.004572 | 0.005454 |

| OOD task | AUROC |
| --- | ---: |
| Overall: 120 ns + 1 ms vs ID | 0.7598 |
| Near: 120 ns vs ID | 0.5492 |
| Far: 1 ms vs ID | 0.9705 |

P4-1 Near-OOD verdict: `PARTIAL`; AUROC is only slightly above random.
P4-2 Far-OOD verdict: `GOOD`; 1 ms is clearly right-shifted and separable.

### Combined status

| Diagnostic | Function | Result | Verdict |
| --- | --- | --- | --- |
| P1 | Total uncertainty as error proxy | Negative within-regime correlations | FAIL |
| P2 | Low-uncertainty risk ranking | Low-uncertainty subsets are worse | FAIL |
| P3-1 | Delay spread -> NMSE | Clear degradation mainly 40-100 ns | PARTIAL |
| P3-2 | ID hardness -> Aleatoric | Weak/non-monotonic | PARTIAL |
| P3-3 | ID Epistemic stability | Approximately stable | PASS |
| P4-1 | Near-OOD Epistemic | AUROC 0.5492 | PARTIAL |
| P4-2 | Far-OOD Epistemic | AUROC 0.9705 | GOOD |

### Final baseline status

`YELLOW` overall, with uncertainty-based feedback control still blocked. The
checkpoint preserves useful coarse Far-OOD information, stable ID Epistemic
scale, and a partial ID NMSE trend, so it is not a completely uninformative
uncertainty output. However, P1/P2 show that uncertainty currently cannot be
used as a safe sample-level risk-ranking signal, Aleatoric ID hardness is weak,
and Near-OOD separation is close to random. The next priority is to audit the
evaluation/uncertainty semantics and sample-level target/error alignment before
any Partial Fine-Tuning experiment; no downstream adaptation was started here.

## P1 Diagnostic - Sample CFR Power Normalization

### Purpose and hypothesis

This read-only experiment tests whether the negative P1 correlation between
total uncertainty and sample NMSE was caused by variation in per-sample CFR
power. The current-valid checkpoint was reused without retraining, and the
same 800 evaluation samples and deterministic observation seeds were used.

### Normalization definition

For each sample, `a = sqrt(mean(|H|^2))` was computed from the clean full CFR.
Both target and prediction were transformed as `H_norm=H/a` and
`gamma_norm=gamma/a`. Because uncertainty is a variance/covariance quantity,
the covariance-derived scores were transformed as
`Sigma_norm=Sigma/a^2`, equivalently `U_ale_norm=U_ale/a^2` and
`U_epi_norm=U_epi/a^2`. Uncertainty was not left at its original amplitude
scale. NMSE is theoretically scale-invariant and was recomputed after
normalization as a check.

### Power and unnormalized relationships

| Regime | U vs power Pearson | U vs abs MSE Pearson | U vs existing NMSE Pearson |
| --- | ---: | ---: | ---: |
| 20 ns | 0.7815 | 0.7371 | -0.2846 |
| 80 ns | 0.9598 | 0.8918 | -0.6201 |
| 120 ns | 0.9357 | 0.7910 | -0.6356 |
| 1 ms | 0.8843 | 0.8561 | -0.1197 |

The uncertainty is strongly related to absolute CFR error, but the existing
NMSE in dB is negatively related. This is consistent with sample amplitude
being a major confounder for the original P1 score.

### Before versus after normalization

| Regime | Before Pearson | Before Spearman | After Pearson | After Spearman |
| --- | ---: | ---: | ---: | ---: |
| 20 ns | -0.2846 | -0.3265 | 0.4409 | 0.4185 |
| 80 ns | -0.6201 | -0.6257 | 0.6991 | 0.6891 |
| 120 ns | -0.6356 | -0.6698 | 0.7984 | 0.7746 |
| 1 ms | -0.1197 | -0.1184 | 0.1342 | 0.0983 |

The normalized NMSE values were unchanged within numerical precision, while
the normalized total uncertainty changed as expected under the `1/a^2`
variance scaling. Mean normalized absolute MSE was `0.0685`, `0.0760`,
`0.1183`, and `5.7293` for 20 ns, 80 ns, 120 ns, and 1 ms respectively.

### Verdict and interpretation

`PASS` for the stated power-confounding hypothesis. Sample power variation is
a major cause of the original P1 inversion: after jointly normalizing CFR and
variance scale, the ID and Near-OOD error-ranking correlations become
positive. Far-OOD remains weak but positive, so normalization alone does not
fully solve OOD ranking. The next diagnostic priority is therefore a precise
Eq.12->Eq.13 sample-level aggregation and mask/target alignment audit using
the normalized representation. No model or checkpoint was changed.

Artifacts:
`runs/paper_style_uncertainty_diagnostics_20260904/p1_power_normalization/`.

## Normalized Figure 7 ID Delay-Spread Sweep

### Purpose

This read-only experiment tests whether sample-wise CFR power normalization
restores the Figure 7 semantic behavior inside the training range. The fixed
current-valid checkpoint was evaluated on 200 newly generated samples at each
of 10, 20, 40, 60, 80, and 100 ns. All six points are ID because they lie in
the training interval `[10,100] ns`.

### Method

The baseline observation and model were unchanged: `Ng=16`, reported-CFR
15 dB AWGN (`IMPLEMENTATION-ASSUMPTION`), clean full-CFR target, corrected pair
mapping, diagonal Psi, and Eq.12 -> Eq.13 aggregation. For each sample,
`a=sqrt(mean(|H_true|^2))`; `H_true` and `gamma` were divided by `a`, while
Aleatoric and Epistemic covariance-derived scores were divided by `a^2`.
Normalized NMSE was recomputed as a consistency check and is unchanged.

### Raw versus normalized result

| Delay | Raw NMSE (dB) | Raw Ale | Raw Epi | Normalized NMSE (dB) | Normalized Ale | Normalized Epi |
|---:|---:|---:|---:|---:|---:|---:|
| 10 ns | -17.6697 | 0.020513 | 0.002608 | -17.6697 | 0.024138 | 0.003032 |
| 20 ns | -17.7283 | 0.020427 | 0.002591 | -17.7283 | 0.023310 | 0.002924 |
| 40 ns | -17.7392 | 0.020275 | 0.002562 | -17.7392 | 0.022690 | 0.002845 |
| 60 ns | -17.6313 | 0.020542 | 0.002602 | -17.6313 | 0.022936 | 0.002887 |
| 80 ns | -17.2591 | 0.020791 | 0.002637 | -17.2591 | 0.022293 | 0.002809 |
| 100 ns | -16.4009 | 0.020764 | 0.002634 | -16.4009 | 0.022465 | 0.002839 |

Sample standard deviations and 95% confidence intervals are stored in
`normalized_delay_sweep.csv`.

### Interpretation and verdict

NMSE shows a partial delay-spread trend: it is broadly flat through 40 ns and
then worsens from 40 to 100 ns. The normalized Aleatoric mean decreases from
0.024138 at 10 ns to 0.022465 at 100 ns; its Pearson correlation with delay is
`-0.8507`, so the intended increasing Aleatoric behavior is not recovered.
Normalized Epistemic also decreases mildly (`0.003032` to `0.002839`, delay
correlation `-0.8006`) and is therefore only approximately stable, not a
positive ID-hardness signal.

Verdicts: NMSE trend `PARTIAL`, normalized Aleatoric trend `FAIL`, ID
Epistemic stability `PARTIAL`. The power normalization fixed the sample-level
P1 inversion, but it does not restore the Figure 7 Aleatoric semantics across
the ID delay sweep. No checkpoint or model was modified.

Artifacts:
`runs/paper_style_uncertainty_diagnostics_20260904/p3_normalized_delay_sweep/`.

## Normalized Aleatoric Eq.12 -> Eq.13 Aggregation Audit

### Purpose and fixed conditions

This read-only audit tests whether the remaining delay-spread Aleatoric trend
failure is caused by final aggregation or by mask/target alignment. The
current-valid checkpoint was not retrained or modified. The same 200-sample
probes were used for 20, 80, and 120 ns; the 1 ms result is a secondary
reference. The model input remains the direct reported-CFR 15 dB AWGN
observation (`IMPLEMENTATION-ASSUMPTION`) with `Ng=16`.

### Exact aggregation path

The channel layout is `[Re(pair0..3), Im(pair0..3)]`, so pair `p` uses
channels `(p,p+4)`. For each sample and pair, the audit first computed
`Sigma_ale = Psi/(nu-2K-1)` and, for normalized scores, divided this variance
by `a^2`, where `a=sqrt(mean(|H_true|^2))`. Eq.12 was then recomputed as the
Re and Im diagonal variances summed for each subcarrier, followed by the mean
over four antenna pairs. Eq.13 averaged only entries where `mask=0`.

The mask audit is exact for every regime: 64 observed indices
`0,16,...,1008`, 960 omitted indices, with `1=observed` and `0=omitted`.
The clean target and prediction use the same corrected pair mapping. Thus no
reported subcarrier enters the final omitted score.

### 20 ns versus 80 ns intermediate values

| Regime | Omitted pair error | Normalized omitted pair error | Raw Eq.12 observed | Raw Eq.13 Ale | Normalized Eq.12 observed | Normalized Eq.13 Ale |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 ns | 0.0163714 | 0.0168964 | 0.0204030 | 0.0204326 | 0.0234499 | 0.0233126 |
| 80 ns | 0.0191251 | 0.0192335 | 0.0205995 | 0.0207934 | 0.0222053 | 0.0223008 |

The omitted prediction error is higher at 80 ns, and the raw Aleatoric score
also rises slightly (`+0.0003608`). After power normalization, however, the
final Eq.13 score reverses (`0.0223008 - 0.0233126 = -0.0010118`). This is not
caused by including observed samples: the observed-only Eq.12 means show the
same 80 ns decrease after normalization.

### Intermediate correlation with final normalized Aleatoric

| Regime | Eq.12 observed score | Normalized Psi mean | Denom mean | Normalized omitted error | Sample scale `a` |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20 ns | 0.9961 | 0.9919 | 0.3154 | 0.4696 | -0.8951 |
| 80 ns | 0.9988 | 0.9990 | 0.7508 | 0.7027 | -0.9278 |

These are within-regime sample correlations, not causal coefficients. The
final normalized Aleatoric score is almost entirely aligned with normalized
Psi (`0.9919` and `0.9990` Pearson), while denominator variation is weaker.
Across regimes, normalized Psi and Eq.13 score are lower at 80 ns despite the
higher reconstruction error. Therefore the failure occurs before the final
omitted averaging, in the learned covariance/evidence values, rather than in
the Eq.12-to-Eq.13 mask reduction.

### Verdict

`PASS` for aggregation and mask/target alignment: pair mapping, Re/Im trace,
observed/omitted separation, and Eq.13 omitted-only averaging are consistent.
The single hypothesis that the final aggregation itself causes the
normalized Aleatoric delay-spread failure is not supported. The remaining
priority is a learned `Psi`/evidence behavior audit or the underlying
observation/channel semantics; no training or model change was made here.

Artifacts:
`runs/current_valid_baseline/diagnostics/normalized_aleatoric_aggregation_audit_20260904_v2/`.

## Psi-Error Alignment Diagnostic

### Purpose and fixed conditions

This read-only diagnostic tests whether the 80 ns Aleatoric failure is mainly
caused by learned `Psi` following absolute CFR power/error scale instead of
normalized reconstruction difficulty. Only the current-valid checkpoint was
used; no training, model, loss, mask, noise, or checkpoint change was made.
The 20 ns and 80 ns probes each contain 200 samples, with the same `Ng=16`
mask and direct reported-CFR 15 dB AWGN observation
(`IMPLEMENTATION-ASSUMPTION`).

### Sample/pair quantities

For each of 800 sample/pair observations per regime, the audit saved CFR
power, absolute and power-normalized omitted reconstruction MSE, raw and
normalized diagonal `Psi` scale, raw and normalized Aleatoric, `kappa`, and
`nu`. Normalized `Psi` and Aleatoric use the variance scaling `1/a^2`, where
`a=sqrt(mean(|H_true|^2))`.

### Overall correlations

| Regime | Raw Psi vs power | Raw Psi vs abs. error | Norm. Psi vs norm. error | Norm. Psi vs power |
| --- | ---: | ---: | ---: | ---: |
| 20 ns | 0.8615 | 0.7006 | 0.2389 | -0.8538 |
| 80 ns | 0.9304 | 0.7758 | 0.3766 | -0.8804 |

Spearman values are in `correlations.csv`. Raw `Psi` is strongly power
aligned in both regimes. After normalization, its relation to normalized
error remains positive but modest, while the strong negative relation to
power remains because the predictor's learned raw scale is being divided by
sample power.

### Regime averages

| Regime | Power | Abs. error | Norm. error | Raw Psi | Norm. Psi | Raw Ale | Norm. Ale |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 ns | 0.990879 | 0.016371 | 0.016896 | 0.187615 | 0.216686 | 0.020433 | 0.023313 |
| 80 ns | 1.024089 | 0.019125 | 0.019233 | 0.190210 | 0.205569 | 0.020793 | 0.022301 |

Thus the unconditioned normalized `Psi` and Aleatoric means are lower at
80 ns despite its higher normalized error.

### Common power-bin comparison

The two regimes were compared in five common CFR-power quantile bins. This is
a power-matched diagnostic, not a new training or evaluation protocol.
Equal-bin averages were:

| Quantity | 20 ns | 80 ns | 80 - 20 |
| --- | ---: | ---: | ---: |
| Normalized error MSE | 0.016835 | 0.019370 | +0.002535 |
| Normalized Psi | 0.210882 | 0.211078 | +0.000197 |
| Normalized Aleatoric | 0.022735 | 0.022856 | +0.000120 |

The per-bin values are retained in `power_matched_bins.csv`; the sign is not
uniform in every bin, so this is evidence of a small partial recovery rather
than a decisive effect.

### Verdict

`PARTIAL`: power confounding is a substantial factor because raw `Psi` tracks
CFR power strongly, and power matching changes the 80-vs-20 normalized `Psi`
and Aleatoric gaps from negative to slightly positive while normalized error
remains higher at 80 ns. However, the effect is small and mixed across bins;
the learned evidential behavior is not fully explained by power alone. The
next diagnostic should use tighter paired-power matching and audit the
normalized `Psi`/target scale alignment before changing the model or loss.

Artifacts:
`runs/current_valid_baseline/diagnostics/psi_error_alignment_20260904_v2/`.

## 1:1 Paired-Power Matching Diagnostic

### Purpose and method

This read-only follow-up tests the Psi/error alignment after controlling CFR
power more strictly. The same 200 samples per regime and current-valid
checkpoint were used. Each 20 ns sample was assigned to one distinct 80 ns
sample by minimum-total absolute CFR-power distance (Hungarian 1:1 assignment,
no replacement). Sample values were the mean of the four antenna pairs.

The matching tolerance was mean absolute power difference `0.060636`, median
`0.012266`, 95th percentile `0.295633`, and maximum `0.518449`; mean relative
tolerance was `0.088701` and the 95th percentile was `0.539643`. The broad
upper-tail tolerance is retained as a limitation of this diagnostic rather
than hidden.

### Paired differences (80 ns - 20 ns)

| Quantity | Mean | Median | Positive-pair ratio | Wilcoxon p | Sign-test p |
| --- | ---: | ---: | ---: | ---: | ---: |
| Normalized error | +0.002337 | +0.002429 | 0.800 | 1.05e-18 | 3.38e-18 |
| Normalized Psi | -0.011117 | -0.001957 | 0.430 | 0.0178 | 0.0560 |
| Normalized Aleatoric | -0.001012 | -0.000215 | 0.475 | 0.0819 | 0.5246 |

The higher 80 ns reconstruction error remains after power matching, while
normalized Psi and normalized Aleatoric do not increase consistently. Thus
the main diagnostic criterion `DeltaError>0` with consistent `DeltaPsi>0` and
`DeltaAle>0` is not met. Power confounding contributes to the unconditioned
result, but it is not sufficient to explain the Aleatoric failure; the
learned evidential alignment remains the stronger candidate.

Artifacts:
`runs/current_valid_baseline/diagnostics/psi_error_alignment_20260904_v2/paired_power_matching/`.

## Loss-Psi Gradient Alignment Audit

### Purpose and fixed baseline

This read-only autograd audit tested whether the current evidential objective
provides enough Psi-increase pressure for harder 80 ns samples. The current
valid checkpoint, `Ng=16` Condition C observation, corrected pair mapping,
diagonal Psi, pair-scalar `kappa/nu`, and `lambda_reg=1e-3` were unchanged.
No optimizer step was called. The existing 20/80 probe data used by the
power-matching diagnostic were reused.

### Gradient definition

For each sample and antenna pair, the audit computed the exact current
diagonal multivariate Student-t pair NLL and pair-level Eq.11 regularizer.
It recorded gradients with respect to the raw pre-softplus `Psi`, `kappa`,
and `nu` heads. `Psi increase pressure` is defined as
`-dL/d(raw_Psi)`, the direction of a gradient-descent update.

### Main results

| Regime | Norm. error | Norm. Psi | Norm. Ale | Psi pressure mean | Psi pressure L2 | Positive ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 ns | 0.008448 | 0.216686 | 0.011661 | -2.57e-5 | 0.4445 | 0.327 |
| 80 ns | 0.009617 | 0.205569 | 0.011147 | -7.77e-6 | 0.4561 | 0.323 |

Both regimes have negative mean Psi-increase pressure, meaning the average
gradient-descent direction decreases raw Psi. The 80 ns pressure is less
negative but does not become positive. Normalized error remains higher at
80 ns while normalized Psi and Aleatoric remain lower.

### Loss-component gradient finding

The pair-level regularizer is implemented as
`||h-gamma||^2 * (kappa+nu)`. It contains no Psi term, so autograd measured
exactly zero for `d(lambda_reg L_reg)/d(raw_Psi)` in both regimes. Therefore
all direct Psi gradient in this implementation comes from the Student-t NLL;
the regularizer directly updates the evidence scalars instead. Mean loss
scales were:

| Regime | NLL | Raw L_reg | lambda x L_reg | Approx. abs(weighted/NLL) |
| --- | ---: | ---: | ---: | ---: |
| 20 ns | -2144.99 | 34797.65 | 34.80 | 1.62% |
| 80 ns | -1948.95 | 40623.38 | 40.62 | 2.08% |

The negative NLL is a log-density value, not a numerical failure. The
regularizer has nonzero `kappa/nu` gradients, but their total gradient
magnitudes are lower at 80 ns: `kappa 7.15e-4` versus `1.21e-3`, and `nu
2.18e-3` versus `3.37e-3`.

### Error-bin behavior

Across global normalized-error quintiles, both regimes show a local increase
of Psi and Aleatoric for high-error samples, but the 80 ns pressure remains
mostly negative until the highest bin:

| Regime | Error bin | Error | Norm. Psi | Norm. Ale | Psi pressure |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20 ns | Low | 0.005851 | 0.204052 | 0.010980 | -5.84e-5 |
| 20 ns | High | 0.013604 | 0.251158 | 0.013329 | +1.16e-5 |
| 80 ns | Low | 0.006166 | 0.166401 | 0.009153 | -2.61e-5 |
| 80 ns | High | 0.012803 | 0.234332 | 0.012630 | +3.43e-6 |

Error-to-Psi-pressure Pearson correlations were `0.4341` for 20 ns and
`0.2225` for 80 ns; normalized error-to-Psi correlations were `0.2389` and
`0.3766`. Error-to-total-kappa-gradient correlations were `-0.2974` and
`-0.1179`, and error-to-total-nu-gradient correlations were `-0.3262` and
`-0.2496`.

### Verdict

`Case A/C` is the best description. Harder 80 ns samples do not receive a
positive average Psi-increase pressure, and the regularizer gives Psi no
direct gradient at all. Local error calibration exists, but regime-level
calibration is weak; the objective can leave uncertainty adjustment to
`kappa/nu` while the learned Psi baseline remains low. This supports an
evidential objective/parameter-coupling limitation rather than a final
aggregation bug. No model or loss was modified; this audit only traced the
existing gradients.

Artifacts:
`runs/current_valid_baseline/diagnostics/loss_psi_gradient_alignment_20260904_v4/`.

## Evidential Parameter Contribution Audit

### Purpose

This read-only audit tested whether the harder 80 ns samples receive an
Aleatoric-increasing update through `Psi`, `nu`, or `kappa`. The current-valid
checkpoint and the existing 20/80 paired-power probe setup were reused; no
model weight was changed and `optimizer.step()` was never called.

### Signed local contributions

For normalized uncertainty score `A`, each contribution is
`S=-dA/d(raw_parameter) * dL/d(raw_parameter)`. Positive values mean that a
gradient-descent update would increase the uncertainty. `S_Ale_Kappa` was
exactly zero for every sample/pair because Aleatoric does not depend directly
on `kappa`.

| Regime | Norm. error | S_Ale_Psi | S_Ale_Nu | S_Ale_Total | Positive total ratio | Psi dominance | Nu dominance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 ns | 0.008448 | -1.055e-6 | -2.041e-6 | -3.096e-6 | 0.174 | 0.459 | 0.541 |
| 80 ns | 0.009617 | -0.609e-6 | -0.971e-6 | -1.580e-6 | 0.273 | 0.536 | 0.464 |

Both regime-level total contributions are negative, so the current local
update direction would reduce Aleatoric on average. The 80 ns contribution is
less negative, but it is not positive. The Nu path is slightly larger than
the Psi path at 20 ns; at 80 ns the Psi path is slightly dominant by absolute
magnitude. This does not support a simple claim that 80 ns difficulty is
absorbed by a positive Nu adjustment.

### Loss-path decomposition

The regularizer gives no direct Psi contribution: `d(lambda L_reg)/d(raw_Psi)
= 0` exactly. Its Nu contribution to Aleatoric was positive in both regimes
(`+1.56e-7` at 20 ns and `+1.80e-7` at 80 ns), while the NLL Nu contribution
was negative (`-2.20e-6` and `-1.15e-6`). Thus the NLL determines the net Nu
direction. The regularizer directly affects `kappa/nu`, not Psi.

### Error alignment and bins

Error versus total Aleatoric contribution had Pearson/Spearman correlations
`0.290/0.356` at 20 ns and `0.291/0.342` at 80 ns. Error versus Nu
contribution was `0.394/0.452` and `0.334/0.412`; Error versus Psi
contribution was only `0.021/0.072` and `0.109/0.087`.

In the highest normalized-error bin, `S_Ale_Total` was `-1.207e-6` at 20 ns
and `-0.412e-6` at 80 ns. Therefore high-error samples receive a less
negative update at 80 ns, but not a positive uncertainty-increase update.

### Power-controlled paired result

Using the existing 1:1 power-matched sample pairs, the 80 ns minus 20 ns
differences were:

| Quantity | Mean difference | Median | Positive ratio | Wilcoxon p |
| --- | ---: | ---: | ---: | ---: |
| Normalized error | +0.001169 | +0.001215 | 0.800 | 1.05e-18 |
| S_Ale_Psi | +0.446e-6 | +0.317e-6 | 0.600 | 2.89e-4 |
| S_Ale_Nu | +1.070e-6 | +0.667e-6 | 0.845 | 4.19e-23 |
| S_Ale_Total | +1.516e-6 | +1.083e-6 | 0.725 | 8.95e-17 |

After power matching, harder 80 ns samples have higher error and a less
negative, significantly larger contribution through both Psi and especially
Nu. This is a relative improvement, not absolute positive pressure.

### Verdict

`Case B with partial relative recovery`: the 80 ns total contribution remains
negative, so the current loss does not provide a strong absolute Aleatoric
increase signal. The regularizer cannot correct Psi directly, and the NLL
dominates the Nu direction. Power-controlled differences show a meaningful
relative 80-vs-20 shift, but the learned model still settles at lower 80 ns
Aleatoric. The primary remaining issue is loss/evidential parameter coupling,
with optimization sufficiency as a secondary possibility.

Artifacts:
`runs/current_valid_baseline/diagnostics/evidential_parameter_contribution_20260904_v6/`.

## Evidential Formula / Semantic Consistency Audit

### Purpose and scope

This is a read-only audit of the current-valid corrected diagonal baseline.
No model, loss, lambda, checkpoint, or optimizer state was changed, and no
training or optimizer step was run. The audit separates algebraic equation
correctness from reduction scale, parameter-role coupling, and the semantic
cost of the diagonal covariance approximation.

### Paper-specified formulation vs current implementation

| Item | Paper | Current implementation | Status |
| --- | --- | --- | --- |
| Pair vector | `h_p in R^(2K)`, `d=2K` | `K=1024`, pair vector `[2048]` | PAPER-SPECIFIED semantics |
| `gamma_p` | pair CFR mean vector | `[B,8,K]`, mapped to `[B,4,2048]` | PAPER-SPECIFIED semantics; layout is implementation |
| `kappa_p` | pair scalar, positive | `[B,4,1]`, softplus | PAPER-SPECIFIED semantics |
| `Psi_p` | full positive-definite `L L^T` matrix | diagonal `[B,4,2048]` | PAPER-SPECIFIED full form; current diagonal is APPROXIMATION |
| `nu_p` | pair scalar, `nu>2K+1` | `[B,4,1]`, `2K+1+softplus+eps` | PAPER-SPECIFIED semantics |
| Eq.5 | multivariate Student-t, `df=nu-d+1`, scale `((kappa+1)/(kappa*df))*Psi` | exact diagonal specialization in active path | MATCH under diagonal approximation |
| Eq.7/8 | `Psi/(nu-d-1)` and `/kappa` | pair-first diagonal covariance and pair scalar broadcast | MATCH under diagonal approximation |
| Eq.9 | `Psi=L L^T` with valid positive diagonal | softplus diagonal only | APPROXIMATION |
| Eq.10 | sum of pair/sample log densities | mean over batch and pair terms | IMPLEMENTATION-ASSUMPTION |
| Eq.11 | full target, pair error norm times `Phi=kappa+nu` | same scope and pair norm, mean reduction | MATCH in scope; reduction assumption |

Exact pilot size, TDL-A, direct reported-CFR AWGN placement, and reduction
convention are not specified in the paper and remain
`IMPLEMENTATION-ASSUMPTION` or `UNKNOWN IN PAPER`, not paper claims.

### Eq.5 mapping and scale/covariance distinction

For one pair, `r=h-gamma`, `df=nu-d+1`, and
`S=((kappa+1)/(kappa*df))*Psi`. The paper log density is:

`lgamma((df+d)/2)-lgamma(df/2)-0.5*(d*log(df*pi)+log|S|)-((df+d)/2)*log1p(r^T S^-1 r/df)`.

The active `diagonal_multivariate_student_t_nll` maps these terms to
`degrees`, `scale_diag`, `sum(log(scale_diag))`, and
`sum(residual^2/scale_diag)`. It returns the negative mean pair log density.
The predictive Student-t scale `S` is not Eq.7's Aleatoric covariance;
the latter uses `Psi/(nu-d-1)`. A repository-wide audit found no active
helper that substitutes one for the other. The old elementwise
`student_t_nll` remains as a historical compatibility path but is not active
for the current-valid config.

### Eq.10 reduction audit

Using the same representative batch (`B=8`, four pairs), paper-pair-sum and
current-pair-mean values differ by the expected factor 32. The corrected
same-reduction regularizer-to-NLL ratios are approximately 2.164% (20 ns) and
2.015% (80 ns) for `lambda=1e-3`; the sum and mean variants have the same
ratio when both terms use the same reduction. The elementwise compatibility
path has a materially different objective and gradient scale, so it must not
be treated as the active multivariate baseline.

### Eq.11 regularizer audit

The active regularizer is `||h_p-gamma_p||^2*(kappa_p+nu_p)` over the full
clean target, hence observed and omitted frequencies are included. Autograd
measured exactly zero direct gradient from the regularizer to raw `Psi`; this
is intrinsic to Eq.11's implemented evidence factor, not a numerical failure
of diagonal covariance. The regularizer directly pressures `gamma`, `kappa`,
and `nu`, while `Psi` is updated directly through NLL.

### Psi/Nu sensitivity

For `A=Psi/(nu-d-1)`, the analytic raw derivatives are
`dA/drawPsi=sigmoid(rawPsi)/(nu-d-1)` and
`dA/drawNu=-Psi*sigmoid(rawNu)/(nu-d-1)^2`; the displayed pair-mean score
also includes the `1/2048` average. Autograd and analytic values agreed
within `5.5e-12` for Psi and `1.2e-10` for Nu.

| Regime | `|dA/drawPsi|` | `|dA/drawNu|` | Psi/Nu sensitivity ratio |
| --- | ---: | ---: | ---: |
| 20 ns | 4.849e-6 | 6.410e-4 | 0.00756 |
| 80 ns | 4.558e-6 | 5.740e-4 | 0.00794 |

Under this pair-mean score, Nu is the more sensitive local control path. This
does not by itself prove that Nu is the learned cause; it establishes strong
parameter coupling in the current parameterization.

### Operating range and diagonal semantic risk

Across 200 samples and four pairs, mean learned Psi was `0.18996` at 20 ns
and `0.19020` at 80 ns; mean Nu-denominator was `18.3291` and `18.3156`.
Mean Aleatoric was `0.0104385` and `0.0104025`, respectively. The active
parameters were finite and admissible, with `nu` well above `d+1` and positive
Psi/kappa. Full Psi can place frequency correlation into the determinant and
Mahalanobis terms, while diagonal Psi retains only marginal variances.
Existing empirical correlation at lag 32 (`0.993236`, `0.904606`, `0.830189`
for 20/80/120 ns) therefore makes diagonal semantic loss a credible risk,
but this audit does not establish it as the sole cause of the Aleatoric
ordering failure.

### Tiny full-vs-diagonal reference

With equal marginal variances but fixed off-diagonal correlation, the toy
reference changed both NLL and Nu gradient allocation: for `d=4`, NLL was
`4.9405` diagonal vs `5.7340` correlated and `|grad raw Nu|` was `0.1628`
vs `0.2538`; for `d=8`, NLL was `8.2672` vs `9.0250` and Nu gradient was
`0.1373` vs `0.2326`. This demonstrates a possible semantic mechanism, not
a production-model result.

### Final audit verdict

- **CONSISTENT (algebraic active path):** active diagonal Student-t, Eq.7,
  and Eq.8 agree with the diagonal specialization of the paper equations.
- **PARAMETER-COUPLING:** Eq.11 gives no direct Psi gradient and the local
  Aleatoric score is much more sensitive to Nu than Psi under the reported
  reduction.
- **DIAGONAL-APPROXIMATION-RISK:** full frequency covariance is paper-
  specified, while the current diagonal form removes off-diagonal information
  that is empirically regime-dependent.
- **Not classified as IMPLEMENTATION-BUG:** no active Eq.5/7/8 algebraic bug
  was found in this audit.

These findings explain why the implementation can be algebraically correct
yet fail to learn the paper's Aleatoric semantics: the active objective sends
no direct Eq.11 signal to Psi, Nu is a highly sensitive denominator control,
and diagonal Psi cannot encode the observed frequency-correlation structure.

Artifacts:
`runs/current_valid_baseline/diagnostics/evidential_formula_semantic_audit_20260904_v3/`

### Recommended next experiment

Run one controlled, short **Psi-aware evidence-objective ablation** with the
current valid data/model setup fixed, adding an explicitly documented Psi
alignment term and comparing its Psi/Nu gradient allocation against this
audit. This was not executed here; no loss or model was modified.

## Psi/Nu counterfactual diagnostic (2026-09-11)

### 목적과 직전 실험과의 연결

Current Valid Baseline과 power-normalized diagnostic에서 80 ns의
reconstruction error는 20 ns보다 컸지만 normalized Aleatoric ordering은
증가하지 않았다. Eq.12→Eq.13 aggregation/indexing audit에서는 ordering을
뒤집는 명확한 bug를 찾지 못했고, 이전 `S_Psi`/`S_Nu`는 현재 위치에서의
다음 gradient step local direction만 보여주었다. 이번 diagnostic은 재학습
없이 이미 학습된 20 ns/80 ns parameter state를 counterfactual하게 조합해
최종 normalized Aleatoric 차이에 normalized Psi와 Nu가 각각 얼마나
기여하는지 분리했다.

가설은 ``Nu가 Psi 증가를 상쇄한다``로 미리 정하지 않았다. 실제 parameter
state 차이에서 Psi contribution과 Nu contribution의 부호와 크기를
측정하는 것이 목적이었다.

### 실행 파일, config, checkpoint, 실행 조건

- 실행 script: `scripts/psi_nu_counterfactual_diagnostic.py`
- config: `runs/current_valid_baseline/condition_c_5k_10ep_lambda1e3_corrected/config.json`
- checkpoint: `runs/current_valid_baseline/condition_c_5k_10ep_lambda1e3_corrected/checkpoint_with_provenance.pt`
- 결과: `runs/current_valid_baseline/diagnostics/psi_nu_counterfactual_20260911_final/`
- 평가: ID-Easy 20 ns, ID-Hard 80 ns, direct CFR AWGN 15 dB, grouping factor 16
- normalization: `a=sqrt(mean(|H_true|^2))`, `Psi_norm=Psi/a^2`
- aggregation: 현재 active pair mapping과 기존 `paper_subcarrier_uncertainty_map`
  및 omitted-subcarrier Eq.13 reduction
- training/optimizer: 수행하지 않음. checkpoint parameter는 변경하지 않음.
- device: `cuda:0`, `NVIDIA GB10`

현재 모델은 full covariance가 아니라 diagonal `Psi`를 사용하므로 이
counterfactual도 그 diagonal baseline 의미를 그대로 따른다. 이는 논문의
full `Psi=L L^T`를 단순화한 `APPROXIMATION`이다.

### Pairing 판단과 counterfactual 방법

`scripts/generate_dataset.py`는 regime별 seed를
`seed + 3001 + index*1000` 형태로 다르게 만들고, NPZ metadata에도 공유
base realization ID가 없다. 따라서 20 ns sample 0과 80 ns sample 0은
index-wise paired라고 볼 근거가 없다. 단일 index pairing을 사용하지 않고,
100개 permutation을 `seed=20260911`로 deterministic하게 생성해
without-replacement pairing을 반복했다. 이 반복수와 permutation 해석은
논문에서 주어진 방법이 아닌 `IMPLEMENTATION-ASSUMPTION`이다.

각 조합은 다음처럼 기존 forward state와 동일한 F를 사용했다.

| 조합 | 의미 |
| --- | --- |
| `A20_20` | `F(Psi20_norm, nu20)` |
| `A80_20` | `F(Psi80_norm, nu20)`; Psi만 80 ns로 교체 |
| `A20_80` | `F(Psi20_norm, nu80)`; Nu만 80 ns로 교체 |
| `A80_80` | `F(Psi80_norm, nu80)` |

대응하는 symmetric diagnostic contribution은
`C_Psi=0.5*((A80_20-A20_20)+(A80_80-A20_80))`,
`C_Nu=0.5*((A20_80-A20_20)+(A80_80-A80_20))`로 계산했다. 이 분해는
원 논문의 식이 아니라 `IMPLEMENTATION-ASSUMPTION`이며,
`A80_80-A20_20 = C_Psi+C_Nu` identity를 확인했다.

### 실제 결과

| quantity | mean | permutation mean std | 해석 |
| --- | ---: | ---: | --- |
| `A20_20` | 0.0222106361 | 0 | actual 20 ns |
| `A80_20` | 0.0222198296 | 2.29e-5 | Psi-only swap |
| `A20_80` | 0.0223529394 | 1.24e-5 | Nu-only swap |
| `A80_80` | 0.0220699114 | 0 | actual 80 ns |
| `A80_80-A20_20` | -1.407247e-4 | 0 | actual normalized ordering |
| `C_Psi` | -1.369173e-4 | 1.15e-5 | 전체 gap의 약 97.3% |
| `C_Nu` | -3.807415e-6 | 1.15e-5 | 평균은 작고 permutation에 민감 |

Baseline-dependent 단순 delta도 함께 기록했다. `Delta_Psi|20 =
A80_20-A20_20 = +9.194e-6`이고 `Delta_Nu|20 = A20_80-A20_20 =
 +1.423e-4`이다. 즉 20 ns 상태에 고정한 한쪽 경로에서는 Nu 교체가
Aleatoric을 올리는 방향이었다. 그러나 두 경로를 대칭 평균한
`C_Nu=-3.807e-6`은 상호작용과 독립 dataset permutation의 영향을 포함한
값이며, 단순 delta와 같은 의미가 아니다. 따라서 Nu의 방향을 하나의
단정적인 보상 메커니즘으로 해석하지 않았다.

따라서 이번 실행에서 80 ns normalized Aleatoric은 20 ns보다
`0.0220699114 < 0.0222106361`로 낮았다. 이 차이는 평균적으로 Psi state
변화가 주로 만들었다. Nu state 변화의 평균 contribution은 음수였지만
크기가 작고 permutation별 부호가 안정적이지 않았다. 그러므로
``Nu가 Psi 증가를 상쇄한다``는 설명은 확인되지 않았다. 더 정확히는
Psi state difference가 최종 ordering을 주도했고, Nu는 평균적으로 작은
감소 방향이었으나 독립 dataset pairing의 uncertainty 안에서 robust한
주원인으로 특정되지 않았다. 이 결과는 ``Case B``를 단순 적용하기보다,
Psi는 robust negative contribution, Nu는 작고 unstable한 contribution으로
판정하는 `부분 성공`이다.

parameter 통계는 `parameter_stats.csv`에 저장했다. 전체 평균은 다음과 같다.

| regime | normalized Psi mean | Nu mean | denominator mean | actual normalized Aleatoric |
| --- | ---: | ---: | ---: | ---: |
| 20 ns | 0.2046162 | 2067.3281 | 18.32808 | 0.02221064 |
| 80 ns | 0.2029358 | 2067.3127 | 18.31264 | 0.02206991 |

따라서 20 ns→80 ns에서 normalized Psi와 denominator 모두 평균적으로
작아졌다. 직접적인 state decomposition에서는 Psi 변화가 훨씬 큰
negative contribution을 만들었고, denominator/Nu 변화는 평균적으로
작은 negative contribution을 만들었다. 이는 이전의 local
`S_Psi`/`S_Nu`와 같은 값이 아니다. 이전 값은 현재 state에서 gradient
descent 한 step의 local direction이고, 이번 결과는 학습이 끝난 두 regime
state의 차이를 counterfactual하게 교체한 finite state effect이다.

### 결과 artifact와 재현 정보

- `summary.csv`: 네 counterfactual mean/std/median 및 delta
- `parameter_stats.csv`: overall 및 4 antenna-pair parameter 통계
- `counterfactual_results.csv`: 100 permutation × 200 sample 결과
- `state_sample_summary.csv`: sample별 scale, actual normalized Aleatoric,
  custom Eq.12→Eq.13 check
- `provenance.json`: checkpoint SHA256, config, CUDA/GPU, pairing,
  normalization, aggregation, no-training 기록
- `analysis.json`: contribution 정의, identity residual, state check
- `counterfactual_aleatoric_bar.png`, `symmetric_contribution_bar.png`

`analysis.json`의 `identity_mean_residual`은 0이고, 기존 actual Aleatoric
경로와 counterfactual 계산 경로의 최대 차이는 `5.59e-9`이다. 따라서 이번
결과는 별도 수식으로 만든 숫자가 아니라 현재 active pair mapping과
Eq.12→Eq.13 utility를 재사용한 결과로 해석된다.

### 판단과 다음 실험

판정은 **부분 성공**이다. Psi와 Nu를 분리하는 목적은 달성했고 Psi state
차이가 실제 normalized Aleatoric gap의 대부분을 설명했다. 다만 Nu 평균
기여가 작고 permutation에 민감하므로 Nu가 robust하게 ordering을 만든다고
말할 수 없다. 따라서 이전 Aleatoric failure는 단순 aggregation/indexing
bug나 ``Nu compensation`` 하나로 설명되지 않으며, 현재 diagonal Psi와
evidence objective가 Psi state를 학습시키는 방식이 우선적인 후속 검증
대상이다.

다음 실험은 하나만 권장한다: 현재 valid baseline data/model을 고정한
**Psi-aware evidence-objective ablation**을 수행해 Psi parameter에 직접
정렬 신호를 추가했을 때 Aleatoric ordering과 Psi/Nu gradient allocation이
변하는지 확인한다. 이 실험은 아직 실행하지 않았다.

## Psi-Head-Only Calibration Diagnostic (2026-09-11)

### 실험 목적과 직전 counterfactual과의 연결

직전 Psi–Nu counterfactual은 20 ns와 80 ns normalized Aleatoric 차이의
약 97.3%가 Psi state difference로 설명되며, Nu가 robust하게 Psi를
상쇄한다는 증거는 없다고 보였다. 이번 실험은 그 다음의 단일 가설,
즉 shared backbone에 difficulty 정보는 있지만 joint training에서 Psi head가
충분히 calibration되지 않았을 가능성을 검증했다.

Current Valid Baseline checkpoint를 시작점으로 두고, backbone과
gamma/kappa/nu head를 고정한 뒤 `Psi head` parameter만 기존 Student-t
NLL과 기존 regularizer를 사용해 10 epoch 추가 최적화했다. Psi ordering이나
uncertainty ranking을 직접 가르치는 새 loss는 추가하지 않았다.

### 실제 구조와 설정

실제 `pair_scalar` predictor module은 다음과 같다.

| 역할 | 실제 module |
| --- | --- |
| input projection | `model.input_projection` |
| shared backbone | `model.residual_blocks` |
| clean CFR mean | `model.gamma_head` |
| diagonal Psi | `model.psi_head` |
| pair scalar kappa | `model.kappa_head` |
| pair scalar nu | `model.nu_head` |
| pooling | `model.pool` |

trainable parameter는 `psi_head.weight`, `psi_head.bias`뿐이었다. 총
parameter는 `11,815,320`, trainable parameter는 `1,544`, frozen parameter는
`11,813,776`이었다. trainable parameter name 외의 항목이 없는 assert도
통과했다.

- source checkpoint:
  `runs/current_valid_baseline/condition_c_5k_10ep_lambda1e3_corrected/checkpoint_with_provenance.pt`
- train data: 실제 baseline의 `train_5k_pilot.npz`, 5,000 samples
- delay spread: 실제 NPZ metadata의 Uniform `[10,100] ns`
- optimizer: Adam
- learning rate: `1e-4`
- batch size: `8`
- additional epochs: `10`
- evaluation: ID-Easy 20 ns, ID-Hard 80 ns, `Ng=16`, direct CFR AWGN 15 dB
- device: `cuda:0`, `NVIDIA GB10`
- calibrated output:
  `runs/current_valid_baseline/diagnostics/psi_head_only_calibration_20260911/`

20 ns와 80 ns test dataset은 calibration training에 사용하지 않았다. 두
regime의 평가 입력은 calibration 전후 동일한 mask와 AWGN realization을
미리 고정했다. 평가 seed는 각각 `97000`, `98000`이며 이는 논문 설정이
아닌 `IMPLEMENTATION-ASSUMPTION`이다.

frozen backbone dropout이 평가마다 달라지지 않도록 전체 모델을
`eval()` 상태로 유지했다. weight freeze 및 dropout mode 선택은 논문에
명시된 설정이 아니라 diagnostic `IMPLEMENTATION-ASSUMPTION`이다.

현재 `Psi`는 `softplus(raw_Psi)+eps`로 양수화된 diagonal covariance이며,
full `Psi=L L^T`가 아니다. 따라서 이 실험도 full covariance 논문식을
diagonal로 단순화한 `APPROXIMATION` 위에서 수행했다. loss는
`src.models.evidential.evidential_loss`를 그대로 호출했으며, config의
`diagonal_multivariate` Student-t NLL과 `pair` evidence regularizer를
수정하지 않았다.

### 실행 파일과 결과 artifact

- 실행 script: `scripts/psi_head_only_calibration_diagnostic.py`
- calibration checkpoint: `calibrated_checkpoint.pt`
- `training_history.csv`
- `evaluation_before.csv`, `evaluation_after.csv`
- `parameter_stats_before.csv`, `parameter_stats_after.csv`
- `provenance.json`, `analysis.json`
- `counterfactual_after/`: 기존 100-permutation protocol 재사용 결과

### Freeze sanity check

고정된 동일 evaluation input에서 다음 최대 차이를 측정했다.

| Regime | gamma max abs diff | kappa max abs diff | nu max abs diff | Psi max abs diff |
| --- | ---: | ---: | ---: | ---: |
| 20 ns | 0 | 0 | 0 | 1.3508856 |
| 80 ns | 0 | 0 | 0 | 1.3575209 |

또한 clean CFR prediction이 변하지 않았기 때문에 omitted NMSE도
calibration 전후 동일했다. 따라서 실제로 Psi head만 바뀌었다고 판단할
수 있다.

### Calibration 전후 결과

| Metric | Before | After | 변화 |
| --- | ---: | ---: | ---: |
| `Psi_norm_20` | 0.20461618 | 0.18842776 | -0.01618842 |
| `Psi_norm_80` | 0.20293583 | 0.18569328 | -0.01724254 |
| `Delta_Psi` | -0.00168035 | -0.00273448 | 더 음수 |
| `A_norm_20` | 0.02221064 | 0.02044477 | -0.00176587 |
| `A_norm_80` | 0.02206991 | 0.02021724 | -0.00185268 |
| `Delta_A` | -0.00014073 | -0.00022753 | 더 음수 |
| `NMSE_20` | -17.66961 dB | -17.66961 dB | 0 |
| `NMSE_80` | -17.21492 dB | -17.21492 dB | 0 |
| `C_Psi` | -1.369173e-4 | -2.416105e-4 | 더 음수 |
| `C_Nu` | -3.807415e-6 | +1.407701e-5 | 작은 양수로 전환 |

Calibration 후에도 `Psi_norm_80 <= Psi_norm_20`이고
`A_norm_80 <= A_norm_20`이었다. `C_Nu`는 양수로 바뀌었지만 크기가 작고,
최종 ordering을 뒤집을 정도가 아니었다. 반대로 `C_Psi`는 더 음수가
되었다.

validation NLL은 고정 validation input에서 `-2070.4585`에서
`-2094.0426`으로 낮아졌다. 그러나 validation NLL 개선이 ID-Hard
uncertainty ordering 개선을 의미하지는 않았다.

### Epoch trajectory

| Epoch | Psi20 | Psi80 | Delta Psi | A20 | A80 | Delta A | Psi grad norm |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.200030 | 0.197265 | -0.002765 | 0.0216852 | 0.0214530 | -0.0002322 | 199.15 |
| 2 | 0.196281 | 0.193620 | -0.002661 | 0.0212848 | 0.0210645 | -0.0002203 | 190.54 |
| 3 | 0.194444 | 0.191726 | -0.002718 | 0.0210893 | 0.0208626 | -0.0002267 | 189.12 |
| 4 | 0.193420 | 0.190762 | -0.002658 | 0.0209844 | 0.0207631 | -0.0002213 | 189.11 |
| 5 | 0.191871 | 0.189198 | -0.002673 | 0.0208189 | 0.0205969 | -0.0002220 | 190.46 |
| 6 | 0.190840 | 0.188135 | -0.002705 | 0.0207069 | 0.0204830 | -0.0002239 | 189.02 |
| 7 | 0.190002 | 0.187312 | -0.002690 | 0.0206123 | 0.0203905 | -0.0002219 | 186.92 |
| 8 | 0.189670 | 0.187003 | -0.002668 | 0.0205810 | 0.0203585 | -0.0002225 | 188.39 |
| 9 | 0.189207 | 0.186538 | -0.002668 | 0.0205296 | 0.0203066 | -0.0002230 | 188.34 |
| 10 | 0.188428 | 0.185693 | -0.002734 | 0.0204448 | 0.0202172 | -0.0002275 | 185.54 |

trajectory 전반에서 Psi20과 Psi80 모두 감소했고, 두 regime 차이의 부호는
한 번도 양수가 되지 않았다. Aleatoric도 동일하게 ordering을 복구하지
못했다.

### 판정과 연구적 해석

판정은 **FAILURE**이다. 이번 실험의 정의에서 Psi-head-only 추가
calibration은 ID-Hard Psi ordering과 Aleatoric ordering을 복구하지
못했다. 오히려 마지막 `Delta_Psi`와 `Delta_A`가 calibration 전보다 더
음수가 되었다.

따라서 현재 Aleatoric failure를 단순히 ``Psi head가 충분히 calibration되지
않았기 때문``이라고 볼 수 없다. 더 정확한 결론은 다음과 같다.

> 고정된 shared backbone, gamma, kappa, nu에서 현재 Student-t NLL만으로
> Psi head를 추가 학습해도 ID-Hard uncertainty ordering은 복구되지 않았다.

이 결과만으로 backbone에 difficulty 정보가 없다고 단정할 수는 없다.
또한 Student-t NLL이 틀렸거나 Nu가 문제가 아니라고 단정할 수도 없다.
이번 실험이 직접 배제한 것은 오직 ``현재 representation을 그대로 둔
Psi-head-only calibration만으로 충분하다``는 설명이다.

### 다음 실험

다음 실험은 하나만 권장한다: **Psi-aware evidence-objective ablation**을
수행해 현재 Student-t NLL의 Psi gradient allocation을 명시적으로 바꿨을
때 ordering이 달라지는지 검증한다. 이 실험은 아직 실행하지 않았다.

## Structured Covariance Approximation Ablation: Diagonal Psi vs Banded Psi (2026-09-12)

### 목적과 prior 실험 복원

이번 실험의 단일 가설은 diagonal `Psi`가 제거한 frequency-domain local correlation을 제한적인 banded representation으로 복원하면 ID-Hard 80 ns의 Aleatoric ordering이 개선되는지 확인하는 것이다. 이는 논문의 exact full covariance reproduction이 아니라 `APPROXIMATION / IMPLEMENTATION-ASSUMPTION`이다.

기존 `runs/current_valid_baseline/covariance_experiments/banded_lag32_5k_10ep/` artifact를 먼저 확인했다. prior run은 train path, 5k/10 epoch, batch 8, lr `1e-4`, lambda `1e-3`, mask `[4,8,16,32]`, eval `Ng=16`, direct CFR AWGN 15 dB, seed `20260819`, pair mapping, backbone, regularizer 및 `bandwidth=32`를 사용했다. 따라서 실험 의도와 대부분의 training 조건은 current valid baseline의 covariance-only ablation과 같다. 그러나 prior artifact는 raw Aleatoric만 저장했고 power-normalized Aleatoric을 primary endpoint로 평가하지 않았으며, provenance의 source hashes가 현재 valid snapshot과 달랐다. 따라서 prior 결과를 현재 clean ablation의 최종 근거로 사용하지 않고 current source로 별도 실행했다.

Prior raw result는 `Ale80-Ale20=+7.4732e-6`였지만 Near-OOD Epistemic gap은 `-1.0609e-6`이고 Far-OOD gap은 `+2.9131e-5`로 diagonal control보다 약해졌으며, 결과는 historical FAIL로 정리되어 있었다.

### 구현 감사와 sanity check

실행 branch는 기존 `pair_scalar_banded`이다. `src/models/uacp_predictor.py`의 forward는 `banded_head`를 `[B,4,K,4*bw]`로 정리한 뒤 `src/models/evidential.py:37-97`의 `build_banded_cholesky()`로 interleaved `[Re(k),Im(k)]` frequency block factor를 만든다. 각 antenna pair마다 block-local lower factor를 사용하고 `Psi=L L^T`를 정의한다. 이는 논문의 full `2048x2048` covariance를 모든 frequency pair에 대해 표현하는 구조가 아니므로 `APPROXIMATION`이다. `bandwidth=32`와 `factor_scale=0.05`도 논문 값이 아닌 `IMPLEMENTATION-ASSUMPTION`이다.

`src/models/evidential.py:291-323`의 banded NLL은 diagonal만 추출하지 않고 `torch.linalg.solve_triangular()`로 Mahalanobis term을 계산하며, `2*sum(log(diag(L)))`로 log determinant를 계산한다. Aleatoric/Epistemic 평가는 `src/models/evidential.py:136-156`에서 `diag(L L^T)`를 기존 Eq.12→Eq.13 map에 전달한다. 따라서 off-diagonal은 NLL의 solve 경로에는 실제로 들어가지만, 현재 uncertainty aggregation은 `L L^T`의 diagonal을 사용한다.

작은 toy sanity check는 `sanity_checks.json`에 저장했다. lower-triangular, symmetric, positive-definite(min eigenvalue `0.898456`), nonzero off-diagonal, band 밖 zero, dense inverse와 triangular solve 일치, dense logdet 일치, finite backward/loss가 모두 통과했다. 생산 경로는 dense covariance를 materialize하지 않았다.

### 실행 조건과 artifact

실행 command는 다음과 같다.

```bash
.venv-uacp/bin/python scripts/train_banded_covariance.py \
  --config configs/covariance_banded_lag32_5k_10ep.json \
  --output-dir runs/current_valid_baseline/structured_covariance/banded_20260912_current_valid_clean
```

current-valid 조건은 train `train_5k_pilot.npz`, Uniform `[10,100] ns`, 5,000 samples, 10 epochs, batch 8, Adam `1e-4`, lambda `1e-3`, grouping `[4,8,16,32]`, eval grouping 16, observation 15 dB, clean target, seed `20260819`이다. TDL-A/zero mobility/normalize=false 및 direct CFR AWGN은 `IMPLEMENTATION-ASSUMPTION`이다. 실행은 `cuda:0 / NVIDIA GB10`에서 `3356.227 s`가 걸렸다. 기존 checkpoint/result는 수정하지 않았다.

Power-normalized metric은 기존 `scripts/p1_power_normalization_diagnostic.py`를 사용한 evaluation-time post-processing으로 계산했다. 학습 중 normalization을 적용하지 않았다. 따라서 `regime_metrics.csv`와 `summary.csv`의 diagonal/banded 비교는 동일한 read-only probe policy를 사용한다.

### Current clean 결과

| Metric | Diagonal | Banded lag32 | Difference |
| --- | ---: | ---: | ---: |
| NMSE 20 ns (dB) | -17.711844 | -17.365664 | +0.346179 |
| NMSE 80 ns (dB) | -17.250190 | -16.946389 | +0.303800 |
| NMSE 120 ns (dB) | -15.373366 | -15.029499 | +0.343867 |
| NMSE 1 ms (dB) | +1.554752 | +1.391619 | -0.163133 |
| NormAle 20 ns | 0.02221064 | 0.01113269 | -0.01107795 |
| NormAle 80 ns | 0.02206991 | 0.01087383 | -0.01119608 |
| Delta NormAle 80-20 | -0.00014072 | -0.00025885 | -0.00011813 |
| NormEpi 20 ns | 0.00280987 | 0.00181916 | -0.00099071 |
| NormEpi 80 ns | 0.00278565 | 0.00177488 | -0.00101077 |
| NormEpi 120 ns | 0.00284691 | 0.00178119 | -0.00106572 |
| NormEpi 1 ms | 0.00412265 | 0.00170815 | -0.00241450 |
| Validation NLL | -2073.240522 | -5586.065938 | -3512.825415 |

Validation NLL의 절대값은 diagonal과 banded 구현의 density scaling 및 covariance parameterization이 달라 직접적인 품질 비교에는 주의가 필요하다. primary ordering은 banded에서도 `Delta NormAle=-2.5885e-4`로 음수였다. reconstruction ordering `NMSE80 > NMSE20`은 유지되지만 두 ID NMSE 모두 diagonal보다 약 `0.3 dB` 악화되었다.

### Covariance diagnostic과 판정

새 run의 `covariance_stats.csv`에서 off/total energy ratio는 20/80/120 ns가 각각 `0.581180`, `0.581271`, `0.581248`이고 1 ms는 `0.582441`이다. lag-1/8/16/32 평균 absolute covariance는 ID-Easy `0.022759/0.021301/0.016066/0.001737`, ID-Hard `0.022761/0.021303/0.016068/0.001736`로 거의 동일하다. 즉 nonzero local covariance는 실제 사용되었지만 regime-dependent covariance decay는 학습되지 않았다.

판정은 **FAILURE**이다. 새 current-valid run에서도 normalized Aleatoric ordering이 복구되지 않았고, ID reconstruction은 악화되었으며, Near/Far Epistemic separation도 diagonal reference보다 약해졌다. prior banded FAIL의 핵심인 “nonzero covariance는 생기지만 uncertainty semantics와 OOD signal은 개선되지 않는다”는 현상이 현재 run에서도 재현되었다. 다만 5k/10 epoch, bandwidth 32, direct-CFR observation은 모두 implementation assumptions이므로 이 결과만으로 full covariance 일반론을 부정하지 않는다.

### Artifact 목록

- model/checkpoint: `runs/current_valid_baseline/structured_covariance/banded_20260912_current_valid_clean/checkpoint_with_provenance.pt`
- training/evaluation: `training_history.json`, `final_results.csv`, `epoch_probe_results.csv`
- comparison: `summary.csv`, `regime_metrics.csv`, `covariance_stats.csv`
- audit: `implementation_audit.md`, `sanity_checks.json`, `provenance.json`
- plots: `nmse_comparison.png`, `normalized_aleatoric_comparison.png`, `normalized_epistemic_comparison.png`, `covariance_lag_profile.png`
- post-processing helper: `scripts/audit_banded_covariance.py`

### 다음 실험

다음 실험은 하나만 권장한다: **channel/observation-model audit**로 논문의 channel, SNR/feedback stage, normalization 및 현재 direct-CFR AWGN 가정의 차이를 검증한다.

## Statistical Aleatoric Ordering and Training-Diversity Audit (2026-09-14)

### 목적과 직전 Pilot-LS 실험과의 연결

직전 Pilot-LS audit은 orthogonal full-rank pilot과 white AWGN에서 현재
direct-CFR AWGN이 수학적으로 동등함을 보였다. 따라서 이번에는 observation,
mask, loss, covariance, normalization을 바꾸지 않고, 먼저 작은 200-sample
평가에서 보인 `Ale(80) <= Ale(20)`가 실제 population 현상인지 확인했다.
그 뒤 head/loss의 paper 의미를 코드 수준에서 다시 확인하고, mismatch가 없어서
training channel diversity만 바꾸는 단일 ablation을 실행했다.

### Current-valid baseline 고정 설정

| 항목 | 값 | 분류 |
| --- | --- | --- |
| channel | 2x2 MIMO, K=1024, 3.5 GHz, 30 kHz | paper setting 일부 + 세부는 UNKNOWN |
| training channel | TDL-A, zero mobility, normalize=false | `IMPLEMENTATION-ASSUMPTION` |
| train spread | Uniform `[10,100] ns` | paper setting |
| observation | reported CFR complex AWGN, 15 dB, clean target | `IMPLEMENTATION-ASSUMPTION` noise stage |
| mask | train random grouping `[4,8,16,32]`; eval uniform `Ng=16` | implementation protocol |
| model | 32 residual blocks, 192 hidden, pair-scalar kappa/nu, diagonal Psi | paper backbone + diagonal `APPROXIMATION` |
| loss | diagonal multivariate Student-t NLL + pair evidence regularizer | diagonal specialization |
| optimizer | Adam, lr `1e-4`, batch 8 | baseline setting |
| lambda | `1e-3` | `PAPER-SPECIFIED` numeric value |
| seed | `20260819` | `IMPLEMENTATION-ASSUMPTION` |

### STEP 1 - independent-test statistical check

실행 파일은 `scripts/statistical_ordering_audit.py`이며, 동일한 Sionna
generation pipeline으로 20 ns와 80 ns 각각 10,000개의 independent CFR을
생성했다. current-valid checkpoint, direct AWGN 15 dB, `Ng=16`을 그대로
사용했다. sample-level 원자료와 Aleatoric 분포 그림은
`runs/current_valid_baseline/diagnostics/statistical_ordering_audit_20260914/`에
저장했다.

| Regime | NMSE omitted mean / median / std (dB) | Aleatoric mean / median / std | Epistemic mean / median / std |
| --- | --- | --- | --- |
| 20 ns | `-17.7486 / -17.7563 / 0.6609` | `0.041042 / 0.039880 / 0.005890` | `0.005210 / 0.005002 / 0.000976` |
| 80 ns | `-17.2345 / -17.2515 / 0.5796` | `0.041302 / 0.040718 / 0.003724` | `0.005232 / 0.005139 / 0.000577` |

`Delta Ale = Ale(80)-Ale(20) = +0.000259635`. 독립 bootstrap 5,000회
95% CI는 `[+0.000124285, +0.000396869]`로 0보다 완전히 컸다. 따라서
현재 200-sample 결과의 음의 ordering은 재현되지 않았고, `80 ns에서
Aleatoric이 실제로 낮다`고 결론낼 수 없다. 판정은 **SMALL-SAMPLE
ORDERING-ARTIFACT-POSSIBLE**이며, 이번 large test에서는 약한 양의 ordering이
관측되었다. 이는 Aleatoric이 완전히 insensitive하다는 뜻도 아니며, 기존
작은 평가의 reversed ordering이 population 현상으로 확정되지 않는다는
뜻이다.

### STEP 2 - evidential head / loss audit

실행 파일은 `scripts/evidential_head_loss_audit.py`이고 결과는
`runs/current_valid_baseline/diagnostics/evidential_head_loss_audit_20260914/audit.json`에
있다.

| 항목 | 실제 코드 결과 | 판정 |
| --- | --- | --- |
| gamma | `[B,8,1024]`, pair마다 `[2048]` frequency-dependent vector | `MATCH` |
| kappa | `[B,4,1]`, sample당 4개 pair-independent scalar; frequency broadcast | `MATCH` |
| nu | `[B,4,1]`, sample당 4개 pair-independent scalar; frequency broadcast | `MATCH` |
| Psi | `[B,8,1024]` → pair `[B,4,2048]` diagonal scale | `APPROXIMATION`, full `Psi=L L^T` 아님 |
| mapping | `[Re(pair0..3), Im(pair0..3)]`, pairs `(0,4),(1,5),(2,6),(3,7)` | `PASS` |
| active NLL | `d=2048`, `df=nu-d+1`, `S=((kappa+1)/(kappa*df))*Psi`, diagonal Mahalanobis/logdet | diagonal specialization `MATCH` |
| regularizer | `mean(||h-gamma||²*(kappa+nu))`, clean full target | formula scope aligned |

Representative batch에서 raw `L_NLL=-2015.9045`, raw `L_reg=39755.2148`,
`lambda_reg*L_reg=39.7552`, ratio `0.01972`였다. 기존 epoch log의
epoch 2--10 ratio도 약 `0.0155--0.0661` 범위였다. 논문의 Eq.10/11이
정확한 implementation reduction을 공개하지 않아 batch/pair mean은
`IMPLEMENTATION-ASSUMPTION`으로 기록한다. 이 단계에서 kappa/nu semantics나
active math의 중대한 mismatch는 발견되지 않았으므로 STEP 3을 진행했다.

### STEP 3 - Training Channel Diversity Ablation

가설은 동일한 총 exposure와 optimizer update에서 unique CFR 수가 evidential
uncertainty 학습에 영향을 주는지였다. A는 기존 current-valid checkpoint를
재사용했고, B만 새로 학습했다.

- A: 5,000 unique CFR × 10 epochs = `50,000` exposures,
  `625 × 10 = 6,250` optimizer updates
- B: 50,000 unique CFR × 1 epoch = `50,000` exposures,
  `6,250 × 1 = 6,250` optimizer updates
- B의 유일한 의도적 변경: unique training CFR 수와 그에 따른 epoch 수
- B dataset: `runs/current_valid_baseline/diversity_ablation/unique_50k_1ep/generated_data/train_5k_pilot.npz`
- B checkpoint: `uacp_predictor_step4a.pt`
- B 학습은 CUDA `cuda:0`, NVIDIA GB10에서 수행했고 시작/종료 CUDA를 확인했다.

실행 명령:

```bash
.venv-uacp/bin/python scripts/baseline_reproduction.py step4a-scaled-training-pilot \
  --model-config configs/current_valid_baseline_condition_c.json \
  --dataset-config configs/dataset_prototype.json \
  --train-samples 50000 --epochs 1 --probe-samples 200 \
  --observation-noise-snr-db 15 \
  --experiment-name DIVERSITY-50k-1ep-current-valid \
  --output-dir runs/current_valid_baseline/diversity_ablation/unique_50k_1ep
```

동일 evaluation set에 대한 A/B 결과는
`runs/current_valid_baseline/diversity_ablation/comparison_same_eval_v3/`에
저장했다.

| Model | NMSE 20 / 80 (dB) | Ale 20 / 80 | Delta Ale 80-20 | Epi 20 / 80 | Near gap | Far gap |
| --- | --- | --- | ---: | --- | ---: | ---: |
| A: 5k×10 | `-17.7467 / -17.2458` | `0.0209002 / 0.0208206` | `-0.00007955` | `0.0026763 / 0.0026423` | `-0.00000216` | `+0.0014624` |
| B: 50k×1 | `-17.5704 / -16.8370` | `0.0216877 / 0.0217604` | `+0.00007272` | `0.0027846 / 0.0027973` | `+0.00005919` | `+0.0019611` |

두 조건 모두 `NMSE(80) > NMSE(20)`을 유지했다. B에서는 `Delta Ale`가
음수에서 양수로 이동했고 Near/Far epistemic gap도 개선되었다. 따라서
**training channel diversity는 현재 failure의 주요 원인 후보**로 올라갔다.
그러나 이는 한 seed의 단일 A/B training run이므로 원인 확정이 아니라
유력 후보라는 수준이다. B에서 NMSE는 A보다 악화되었으므로 reconstruction
capacity와 uncertainty behavior가 동일하게 개선된 것은 아니다.

### 최종 판단과 다음 실험

- STEP 1: 기존 reversed ordering은 10k independent test에서 통계적으로
  지지되지 않음. large test에서는 작은 양의 `Delta Ale`가 관측됨.
- STEP 2: kappa/nu pair-scalar와 active loss 수식은 구조적으로 aligned;
  full Psi 부재는 기존 `APPROXIMATION`, reduction은
  `IMPLEMENTATION-ASSUMPTION`으로 남음.
- STEP 3: diversity-only ablation에서 Aleatoric ordering과 OOD epistemic
  separation이 개선됨. 따라서 training diversity가 가장 유력한 다음 원인
  후보지만, 단일 run으로 확정하지 않음.

이번 실험에서 수정/추가한 파일은 `scripts/statistical_ordering_audit.py`,
`scripts/evidential_head_loss_audit.py`와 이 누적 README 기록이다. 기존
checkpoint/result/log는 삭제하거나 덮어쓰지 않았다. 다음 실험은 하나만
권장한다: **동일한 50k×1 diversity 조건을 고정하고 독립 seed 2개를 추가해
`Delta Ale`의 부호와 OOD gap 개선이 재현되는지 확인한다.**

## 2026-09-14 Training Channel Diversity 재현성 검증

직전 Pilot-LS audit에서 orthogonal Pilot-LS가 current Direct CFR AWGN과
수학적으로 동등하여 observation stage가 원인으로 보기 어렵다는 결론을
얻었다. 10,000-sample audit에서는 baseline도 `Ale(80)>Ale(20)`이었다.
이번에는 동일 exposure에서 5k CFR 반복보다 50k independent CFR가
uncertainty separation을 seed에 걸쳐 안정화하는지만 검증했다.

| 조건 | Unique CFR | Epoch | Exposure | Batch | Updates | Seed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A baseline | 5,000 | 10 | 50,000 | 8 | 6,250 | 20260819 |
| B1 | 50,000 | 1 | 50,000 | 8 | 6,250 | 20260819 |
| B2 | 50,000 | 1 | 50,000 | 8 | 6,250 | 20260915 |
| B3 | 50,000 | 1 | 50,000 | 8 | 6,250 | 20260916 |

모델, loss, `lambda_reg=1e-3`, Adam `lr=1e-4`, mask, observation 15 dB,
channel, `Ng=16` 및 나머지 baseline 설정은 동일했다. 논문에 없는 TDL-A/zero
mobility/`normalize=false`, direct CFR AWGN, mask grouping 및 batch/pair mean
reduction은 **IMPLEMENTATION-ASSUMPTION**이다. diagonal-only Psi는 기존
**APPROXIMATION**이며 변경하지 않았다. B2/B3 학습과 평가 시작·종료 시
`CUDA=True`, `cuda:0`, `NVIDIA GB10`을 확인했다.

직전 결과를 덮어쓰지 않고 새 공통 set을 생성했다. 각 regime(20, 80, 120 ns,
1 ms) 10,000개 CFR에 동일 realization, `Ng=16` mask, 15 dB noise seed를
A/B1/B2/B3에 사용했다. OOD gap은 기존 정의인
`mean(Epi(regime)) - max(mean(Epi(20)), mean(Epi(80)))`이다.

실행 파일은 `scripts/evaluate_diversity_reproducibility.py`이며 결과는
`runs/current_valid_baseline/diversity_ablation/reproducibility_20260914_retry/`
(`summary.csv`, `gaps.csv`, `per_sample.csv`, `results.json`)에 저장했다.
학습 checkpoint는 `.../diversity_ablation/seed_20260915/` 및
`seed_20260916/`에 추가되었고 B1은 기존 checkpoint를 재사용했다.

| Model | NMSE 20 / 80 dB | Ale 20 / 80 | Epi 20 / 80 | Delta Ale | Near gap | Far gap |
| --- | --- | --- | --- | ---: | ---: | ---: |
| A 5k×10 | -17.7330 / -17.2435 | .041045 / .041432 | .005212 / .005252 | +.0003875 | +.0001136 | +.0029476 |
| B1 seed20260819 | -17.6156 / -16.8642 | .042695 / .043432 | .005480 / .005584 | +.0007370 | +.0001305 | +.0038982 |
| B2 seed20260915 | -18.0981 / -17.2613 | .040545 / .041330 | .005051 / .005168 | +.0007852 | +.0001368 | +.0033014 |
| B3 seed20260916 | -18.2865 / -17.6217 | .041094 / .041900 | .005279 / .005392 | +.0008061 | +.0001443 | +.0037447 |

| B1/B2/B3 metric | Mean ± std | Min–max |
| --- | ---: | ---: |
| Delta Ale | .0007761 ± .0000289 | .0007370–.0008061 |
| Near gap | .0001372 ± .0000057 | .0001305–.0001443 |
| Far gap | .0036481 ± .0002530 | .0033014–.0038982 |

10k bootstrap 95% CI도 네 모델 모두 양수였다. Delta Ale CI는 A
`[.0002467,.0005159]`, B1 `[.0005909,.0008755]`, B2 `[.0006586,.0009068]`,
B3 `[.0006766,.0009244]`이다. Near gap CI는 A `[.0000981,.0001291]`,
B1 `[.0001128,.0001482]`, B2 `[.0001213,.0001523]`, B3
`[.0001288,.0001602]`; Far gap CI는 A `[.0029152,.0029812]`, B1
`[.0038530,.0039442]`, B2 `[.0032629,.0033394]`, B3
`[.0037041,.0037871]`이다.

네 조건 모두 `NMSE(80)>NMSE(20)`을 유지했다. 50k×1의 NMSE는 seed에 따라
baseline보다 좋아지거나 나빠졌지만, uncertainty separation 방향은 모두
같았다. 따라서 이번 기준은 **SUCCESS**: training channel diversity가
Aleatoric ordering과 OOD epistemic separation을 더 크고 안정적으로 만드는
유력하고 재현 가능한 원인 후보다. baseline도 이미 양의 ordering을 보였으므로
유일한 원인 또는 논문 reproduction 완성으로 해석하지 않는다.

이번 단계에서는 epoch/sample 확대, batch/mask/loss/Psi/normalization 변경을
실행하지 않았다. 다음 실험은 하나만 권장한다: **50k unique와 50,000 exposure를
유지한 10k×5 조건을 추가해 diversity와 반복 epoch의 trade-off를 분리한다.**

## 2026-09-14 Final pretrained baseline: canonical Fig.8/Fig.9 validation

### Scope and reuse

랩미팅 중단 이후 전체 7-checkpoint evaluation을 다시 실행하지 않았다. 기존
공통 10k evaluation set
(`runs/current_valid_baseline/diversity_ablation/reproducibility_20260914/common_eval/`)
과 기존 A/B1/B2/B3 comparison result를 재사용했다. 최종 pretrained recipe는
`50,000 unique CFR × 1 epoch`, canonical seed `20260819`로 고정했다.

Canonical checkpoint:
`runs/current_valid_baseline/diversity_ablation/unique_50k_1ep/uacp_predictor_step4a.pt`.

### Recipe comparison used for selection

| Recipe | NMSE 20 / 80 dB | NMSE 120 / 1 ms dB | Delta Ale | Near gap | Far gap |
| --- | --- | --- | ---: | ---: | ---: |
| 5k×10 | -17.733 / -17.244 | -15.322 / 1.581 | .000388 | .000114 | .002948 |
| 10k×5 mean | -17.539 / -17.152 | -15.346 / 1.617 | .000604 | .000128 | .003768 |
| 50k×1 mean | -18.000 / -17.249 | -15.085 / 1.530 | .000776 | .000137 | .003648 |

50k×1은 세 seed 모두 `NMSE(80)>NMSE(20)`, `Delta Ale>0`, Near/Far
epistemic gap>0을 만족했다. 또한 10k×5보다 Delta Ale 평균이 크고 seed
변동이 작아 uncertainty separation 안정성을 우선하는 선택 규칙에 따라
최종 pretrained recipe로 고정했다. 기존 checkpoint/result/log는 삭제하거나
덮어쓰지 않았다.

### Final Fig.8: Epistemic ID/OOD separation

동일한 공통 10k set, `Ng=16`, 동일 mask와 15 dB noise realization으로
canonical 50k×1을 평가했다. Eq.(12)/(13) current aggregation에 따른 raw
epistemic score를 사용했다.

| Regime | Mean | Median | Std | P05–P95 |
| --- | ---: | ---: | ---: | ---: |
| ID-Easy 20 ns | .002740 | .002639 | .000470 | .002197–.003630 |
| ID-Hard 80 ns | .002792 | .002728 | .000341 | .002364–.003441 |
| OOD-Near 120 ns | .002857 | .002803 | .000316 | .002446–.003451 |
| OOD-Far 1 ms | .004736 | .004497 | .001165 | .003371–.006895 |

AUROC는 ID=(20 ns+80 ns), OOD=(120 ns+1 ms) equal-count pooling 기준으로
`0.7933`이었다. 추가 진단은 ID vs Near `0.6036`, ID vs Far `0.9830`이었다.
따라서 ID difficulty 두 조건은 상대적으로 유사하고 Near-OOD는 평균 수준에서
증가하며 Far-OOD는 강하게 증가한다. AUROC pooling 세부사항은 논문에 충분히
공개되지 않았으므로 **IMPLEMENTATION-ASSUMPTION**이다. Bootstrap CI는 기존
canonical 산출물에 sample-level score가 저장되지 않아 계산하지 않았다.

결과:
`runs/current_valid_baseline/final_pretrained_fig8_fig9_20260914/canonical_B1_50k1/fig8_distribution.csv`,
`fig8_epistemic_distribution.png`, `results.json`.

### Final Fig.9-style calibration

현재 diagonal-Psi approximation에서 Eq.(5)의 marginal Student-t를 사용했다.
각 omitted-subcarrier real/imag component에 대해 `df=nu-d+1` 및
`scale=((kappa+1)/(kappa*df))*Psi_diag`를 사용해 nominal interval을 만들고,
ID=(20,80), OOD=(120,1 ms) component coverage를 집계했다. 이는 full-covariance
논문의 exact Fig.9 재현이 아니라 **Fig.9-style calibration under diagonal-Psi
approximation**이다.

Calibration MAE(0.1–0.9 nominal grid의 mean absolute error)는 ID `0.0386`,
OOD `0.2324`였다.

| Nominal | ID empirical | OOD empirical |
|---:|---:|---:|
| .5 | .5487 | .2558 |
| .8 | .8483 | .4416 |
| .9 | .9353 | .5244 |

ID는 nominal보다 coverage가 높아 약한 under-confidence/conservative interval을
보였고, OOD는 nominal보다 크게 낮아 over-confidence가 남았다. 따라서 calibration은
완벽하지 않지만 uncertainty signal 자체가 붕괴한 것은 아니다.

결과:
`runs/current_valid_baseline/final_pretrained_fig8_fig9_20260914/canonical_B1_50k1/fig9_calibration.csv`,
`fig9_calibration_curve.png`, `results.json`.

### Final decision

Canonical 50k×1은 `NMSE(80)>NMSE(20)`, `Ale(80)>Ale(20)`, Near/Far
epistemic 증가를 모두 만족했다. Fig.8에서 Far-OOD separation은 명확하고
Near-OOD도 평균 증가가 확인되었다. Fig.9 OOD calibration error는 높지만
완전 붕괴는 아니므로, 명시한 기준에 따라 **Partial Fine-Tuning GO**로 판정한다.

제한사항은 diagonal-Psi approximation, 공개되지 않은 ID/OOD pooling 및 coverage
aggregation 세부사항이다. 다음 단계에서는 새 training/loss/covariance 실험을
자동 실행하지 않고, GO에 따라 Partial FT를 별도 계획으로 시작한다. Fig.10/11과
Full FT는 이번 검증에서 실행하지 않았다.

## 2026-09-15 PAPER-ALIGNED SAMPLE-COUNT ABLATION: 100k×1 Fig.8

### 목적과 설정

최종 pretrained 후보인 50k unique CFR×1 epoch에서 training CFR 수를
100,000개로 늘렸을 때 논문 Figure 8의 Epistemic ID/OOD separation이
개선되는지 검증했다. 50k→100k는 unique CFR와 total exposure가 함께
증가하므로 diversity 단독 효과가 아니다. 이 실험은
**PAPER-ALIGNED SAMPLE-COUNT ABLATION**이며 exact paper training
reproduction이 아니다.

현재 Sionna pipeline, seed `20260819`, delay spread `U[10,100] ns`, 2×2
MIMO, K=1024, 3.5 GHz, 30 kHz를 유지했다. 생성된 train CFR은
`[100000,1024,2,2]`, `complex64`, delay spread 범위 `10.0026–99.9995 ns`였다.
Epoch=1, batch=8이므로 실제 optimizer update는 `12,500`이다. 시작·종료 시
`CUDA=True`, `cuda:0`, `NVIDIA GB10`을 확인했다.

### Fig.8 score와 dB 정의

Eq.(8)의 `Sigma_epi=Sigma_ale/kappa`를 계산하고, Eq.(12)의 pair Real/Imag
trace 및 4-pair 평균과 Eq.(13)의 omitted-subcarrier 평균으로 sample-level
linear `U_epi`를 만들었다. 논문 Fig.8의 dB conversion 식은 공개 기록에서
확인되지 않아 `U_epi_dB=10 log10(U_epi)`를 사용하는 것을
**IMPLEMENTATION-ASSUMPTION: dB conversion**으로 표시한다. 변환은 반드시
sample별로 수행했으며 linear mean을 변환하지 않았다.

### 100k×1 결과

기존 common evaluation set(각 10,000 CFR), 동일 `Ng=16` mask와 15 dB noise를
사용했다.

| Regime | Linear mean | dB mean | dB median | dB std | P05–P95 dB |
| --- | ---: | ---: | ---: | ---: | ---: |
| ID-Easy 20 ns | .001173 | -29.356 | -29.458 | .638 | -30.203–-28.161 |
| ID-Hard 80 ns | .001190 | -29.269 | -29.337 | .464 | -29.910–-28.398 |
| OOD-Near 120 ns | .001226 | -29.136 | -29.189 | .418 | -29.729–-28.378 |
| OOD-Far 1 ms | .002416 | -26.295 | -26.385 | 1.024 | -27.795–-24.501 |

AUROC:

| Comparison | AUROC | Bootstrap 95% CI |
| --- | ---: | ---: |
| ID pooled vs OOD pooled | .8087 | .8052–.8129 |
| ID pooled vs OOD-Near | .6204 | .6136–.6260 |
| ID pooled vs OOD-Far | .9970 | .9966–.9973 |
| ID-Hard 80 vs OOD-Near 120 | .5962 | .5881–.6050 |

80 ns와 120 ns의 평균 차이는 `0.1327 dB`로 작고 overlap이 크지만, 120 ns는
평균 수준에서 오른쪽으로 이동했다. 1 ms는 ID와 명확히 분리된다.

### 50k×1과 직접 비교

기존 50k sample-level 결과를 재사용해 동일한 dB 변환을 적용했다.

| Metric | 50k×1 | 100k×1 | Change |
| --- | ---: | ---: | ---: |
| NMSE 20 ns (dB) | -17.616 | -18.984 | -1.368 |
| NMSE 80 ns (dB) | -16.864 | -17.910 | -1.046 |
| NMSE 120 ns (dB) | -14.802 | -15.154 | -0.352 |
| NMSE 1 ms (dB) | 1.529 | 1.461 | -0.068 |
| ID vs Near AUROC | .6032 | .6204 | +.0172 |
| ID vs Far AUROC | .9831 | .9970 | +.0138 |
| Pooled ID vs OOD AUROC | .7932 | .8087 | +.0155 |
| 80 vs 120 AUROC | .5719 | .5962 | +.0243 |

50k dB mean은 `-22.668/-22.561/-22.456/-20.344`, 100k dB mean은
`-29.356/-29.269/-29.136/-26.295`였다(순서: 20/80/120 ns/1 ms).
논문 Fig.8의 표시 범위 약 `-37.5~-22.5 dB`와 비교하면 100k 실제 전체 범위는
`-30.696~-20.817 dB`(1–99%: `-30.240~-24.352`)이고 50k는
`-24.022~-14.775 dB`였다. 임의 rescaling은 적용하지 않았다.

### 판정 및 제한

**PARTIAL SUCCESS.** 100k×1은 50k×1보다 ID-vs-Near, pooled, 80-vs-120
AUROC를 개선했고 Far-OOD separation도 유지·강화했다. 그러나 80/120 overlap은
여전히 크므로 training sample count만으로 Near-OOD 문제가 해결되지는 않았다.
AUROC의 ID=(20,80), OOD=(120,1 ms) pooling은 논문 세부 공개가 부족한
**IMPLEMENTATION-ASSUMPTION**이다. TDL-A/zero mobility, normalize=false,
direct CFR AWGN, mask grouping 및 reduction도 **IMPLEMENTATION-ASSUMPTION**이다.

결과 경로:
`runs/current_valid_baseline/paper_sample_alignment_100k_1ep_fig8_20260915/`
(`fig8_eval_retry/results.json`, `distribution.csv`, `auroc.csv`,
`per_sample.csv`, `fig8_100k_epistemic_db.png`,
`fig8_50k_vs_100k_epistemic_db.png`). Fig.9, Fig.10/11, Partial FT는 실행하지
않았다. 다음 실험은 하나만 권장한다: **Fig.8 score scale/dB conversion과
diagonal-Psi 제한을 논문 정의와 대조 audit한다.**

## 2026-09-15 Fig.8 Epistemic score definition audit (100k×1)

### 목적과 보존된 입력

100k×1 canonical checkpoint와 기존 common evaluation set을 재사용하여
Fig.8의 score quantity, aggregation, dB 변환이 논문 공개 수식과 의미상
일치하는지 audit했다. 새 training, forward 재실행, checkpoint/result 덮어쓰기는
하지 않았다. 입력 checkpoint는
`runs/current_valid_baseline/paper_sample_alignment_100k_1ep_fig8_20260915/uacp_predictor_step4a.pt`,
common set은
`runs/current_valid_baseline/diversity_ablation/reproducibility_20260914/common_eval/`
이며, 저장된 `fig8_eval_retry/per_sample.csv`의 40,000 sample을 사용했다.

### 논문 수식과 코드 mapping

논문 원문은 다음 흐름을 공개한다.

1. Eq.(7): `Sigma_ale = Psi / (nu - 2K - 1)`.
2. Eq.(8): `Sigma_epi = Sigma_ale / kappa`.
3. Eq.(12): `U_epi[k] = (1/(Nr Nt)) sum_(r,t) tr(P_k Sigma_epi_(r,t) P_k^T)`.
   `P_k`는 pair의 2차원 Real/Imag subcarrier block을 선택한다.
4. Eq.(13): `U_epi(Sbar) = (1/|Sbar|) sum_(k in Sbar) U_epi[k]`.

현재 구현은 `EvidentialOutput.aleatoric`와 `.epistemic`
(`src/models/evidential.py:135-143`)에서 Eq.(7),(8)을 계산한다. `nu`는
`2K+1+softplus`로 제한되고 `K=1024`이므로 Eq.(7)의 denominator와
Student-t dimension `2K=2048` 의미가 유지된다. `kappa`와 `nu`는
`[B,4,1]` pair-level scalar이고 `expand_pair_scalar()`가 frequency 및
Real/Imag channel로 broadcast한다.

`src/training/uncertainty.py:8-22`와
`scripts/final_pretrained_fig8_fig9_validation.py:47-50`은
`[Re(pair0..3), Im(pair0..3)]`를 `[4 pairs, Re/Imag, K]`로 바꾼 뒤
Real/Imag 두 diagonal을 합하고 네 pair를 평균하며, `omitted = 1-mask`인
subcarrier만 sample별로 평균한다. 따라서 현재 diagonal-Psi 표현에서
Eq.(12),(13)의 직접적인 diagonal specialization과는 **MATCH**한다.

전체 논문 quantity, 특히 full `Psi = L L^T`와 논문에 공개되지 않은 Fig.8
dB 변환까지 포함한 최종 판정은 **PARTIAL MATCH**이다. full covariance를
계산하는 구현이 아니며 dB 공식도 논문 본문/caption에서 확인되지 않았다.

### mask, mapping, aggregation 검사

- `observed=1`, `omitted=0`이며 evaluator는 `1-mask`만 aggregate한다.
- observed subcarrier는 score에 섞이지 않는다.
- antenna pair mapping은 `(0,4),(1,5),(2,6),(3,7)`이다.
- pair-level `kappa/nu`는 subcarrier별 독립 scalar로 잘못 분할되지 않는다.
- pair aggregation은 sum이 아니라 `1/(Nr Nt)` 평균이다.
- omitted aggregation은 Eq.(13)대로 omitted count로 나눈 mean이다.
- `K=1024` factor를 별도 잘못 나누거나 곱하지 않는다.

따라서 명백한 evaluator bug/mismatch는 발견하지 못했고 코드 수정도 하지
않았다. 별도 검산 스크립트는
`scripts/audit_fig8_score_definition_100k.py`이며 결과는
`runs/current_valid_baseline/paper_sample_alignment_100k_1ep_fig8_20260915/score_definition_audit/`
(`audit.json`, `conversion_variants.csv`,
`conversion_auroc.csv`, `sample_scores_with_conversion_variants.csv`)에 저장했다.

### dB conversion audit

논문은 Fig.8 축을 `Epistemic Uncertainty (dB)`로 표시하지만, 공개된 식에서
linear-to-dB 변환 공식을 명시하지 않는다. 현재 score는 covariance trace인
variance/power-like quantity이므로 `10 log10`이 일반적인 의미상 선택이다.
따라서 다음은 **IMPLEMENTATION-ASSUMPTION: dB conversion**이며, sample별
변환 `U_epi -> 10log10(U_epi) -> distribution`을 유지한다.

저장 결과 기반 비교는 다음과 같다.

| Variant | ID-Easy mean dB | ID-Hard mean dB | Near mean dB | Far mean dB |
| --- | ---: | ---: | ---: | ---: |
| A: sample `10log10` | -29.356 | -29.269 | -29.136 | -26.295 |
| B: sample `20log10` | -58.713 | -58.538 | -58.272 | -52.591 |
| C: regime linear mean 후 `10log10` | -29.307 | -29.243 | -29.116 | -26.169 |
| D: A의 sample dB 평균 | -29.356 | -29.269 | -29.136 | -26.295 |

C는 distribution이 아니라 regime-level summary이고, D의 mean은 A와 같지만
sample quantile 정보가 없다. B는 amplitude-like quantity일 때의 일반론적
대안일 뿐 논문 설정이 아니다. A는 현재 Fig.8 distribution에 필요한
sample-level quantity와 variance-like score 의미를 동시에 만족한다.

논문 표시 범위 `약 -37.5~-22.5 dB`와 비교하면 현재 100k sample 전체 범위는
`-30.696~-20.817 dB`, 1--99 percentile은 `-30.240~-24.352 dB`이다.
절대 offset 후보는 pair/Real-Imag sum-vs-mean, omitted mean-vs-sum,
channel normalization, learned `Psi/kappa/nu` scale, 그리고 10-vs-20 log이다.
Pair 또는 component averaging과 omitted mean은 고정 배율이면 주로 절대
offset에 영향을 주고 AUROC/order는 보존한다. normalization 및 learned
evidential scale은 절대값과 80/120 separation 모두에 영향을 줄 수 있다.
논문 범위에 맞춘 임의 rescaling은 하지 않았다.

### diagonal Psi의 영향과 기존 banded 참고

Eq.(12)는 `trace(P_k Sigma_epi P_k^T)`이므로 최종 scalar에서 사용하는 것은
각 subcarrier의 Real/Imag 2×2 principal block의 trace이다. full covariance의
off-diagonal 항은 trace에 직접 나타나지 않는다. 따라서 현재 diagonal-Psi가
Fig.8 scalar 계산식에 빠뜨리는 off-diagonal 항은 직접적으로는 없다.
다만 diagonal Psi는 학습 중 multivariate likelihood와 학습된
`Psi/kappa/nu` 상태를 바꾸므로 간접적으로 score distribution과 separation에
영향을 줄 수 있다. 기존 banded 결과는 full paper covariance가 아닌
**APPROXIMATION**이며, 현재 100k와 동일 조건의 재평가가 아니므로 정량적인
원인 판정에는 사용하지 않았다.

### 100k 결과의 audit 후 판정

현재 100k 결과는 audit 전 결과와 동일하다.

| Regime | Linear U_epi mean | dB mean | dB median | dB std | P05--P95 dB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20 ns | 0.001173 | -29.356 | -29.458 | 0.638 | -30.203-- -28.161 |
| 80 ns | 0.001190 | -29.269 | -29.337 | 0.464 | -29.910-- -28.398 |
| 120 ns | 0.001226 | -29.136 | -29.189 | 0.418 | -29.729-- -28.378 |
| 1 ms | 0.002416 | -26.295 | -26.385 | 1.024 | -27.795-- -24.501 |

AUROC는 pooled ID-vs-OOD `0.8087` (bootstrap 95% CI
`0.8052--0.8129`), ID-vs-Near `0.6204` (`0.6136--0.6260`),
ID-vs-Far `0.9970` (`0.9966--0.9973`), 80-vs-120 `0.5962`
(`0.5881--0.6050`)이다. 80 ns와 120 ns 평균 차이는 `0.1327 dB`로 여전히
작고 P05--P95가 크게 겹친다. 즉 이 overlap은 현재 확인 범위에서
evaluator bug로 설명되지 않으며, 남은 후보는 diagonal-Psi의 간접 학습 영향,
training procedure 및 논문에 공개되지 않은 구현 세부사항이다. 다음 실험은
하나만 권장한다: **full-Psi를 주장하지 않고, 동일 checkpoint에 대해 논문
Fig.8의 공개 범위 내 score/scale provenance를 추가 확인한 뒤 필요한 경우
작은 synthetic covariance unit test로 trace mapping을 고정하는 것**. 새 학습,
Fig.9/10/11, Partial FT는 실행하지 않았다.

## 2026-09-15 Fig.9 Predictive Calibration audit (100k×1)

### 목적과 재사용 조건

100k unique CFR×1 epoch, seed `20260819`의 기존 checkpoint를 유지한 채
Student-t predictive interval과 calibration evaluator를 논문 Eq.(5) 기준으로
감사하고 Fig.9-style curve를 재생성했다. 기존 checkpoint와 기존 Fig.9 결과는
덮어쓰지 않았다. checkpoint는
`runs/current_valid_baseline/paper_sample_alignment_100k_1ep_fig8_20260915/uacp_predictor_step4a.pt`,
evaluation set은 기존 common 10k/regime set
`runs/current_valid_baseline/diversity_ablation/reproducibility_20260914/common_eval/`이다.

### Eq.(5)와 현재 코드 mapping

논문 Eq.(5)의 per-antenna-pair predictive distribution은 location `gamma`,
degrees of freedom `nu-2K+1`, scale matrix
`((kappa+1)/(kappa*(nu-2K+1)))*Psi`를 사용한다. 현재 evaluator
`scripts/evaluate_100k_fig9_audit.py`와 `EvidentialOutput`은 다음과 같이
매핑된다.

- `df = nu_expanded - 2*K + 1` (`K=1024`, 즉 `df=nu-2047`).
- `scale_sq = ((kappa_expanded+1)/(kappa_expanded*df))*psi`.
- `scale = sqrt(scale_sq)`를 univariate marginal Student-t scale로 사용한다.
- two-sided nominal coverage `c`에 대해 `q=F^{-1}((1+c)/2, df)`를 계산한다.
- component interval은 `gamma ± q*scale`이다.
- target layout은 prediction과 동일한 8-channel Real/Imag layout이며, omitted
  subcarrier의 component만 coverage에 포함한다.

이는 variance와 Student-t scale을 혼동하지 않고 sqrt를 정확히 한 번 적용한
구현이다. Eq.(5) marginal interval 계산은 **MATCH**한다. 다만 논문은
multivariate interval을 component-wise interval로 변환하는 방법, observed/
omitted coverage scope, ID/OOD pooling, nominal grid, CE reduction을 공개하지
않았으므로 아래 부분은 **IMPLEMENTATION-ASSUMPTION**이다.

### Synthetic Student-t sanity test

known `Student-t(df=17, loc=0, scale=1.7)`에서 300,000개 sample을 생성하고
동일한 two-sided quantile/interval helper를 적용했다.

| Nominal | Empirical | Absolute error |
|---:|---:|---:|
| 0.50 | 0.50028 | 0.00028 |
| 0.80 | 0.79958 | 0.00042 |
| 0.90 | 0.89959 | 0.00041 |
| 0.95 | 0.94948 | 0.00052 |
| 0.99 | 0.99000 | 0.00000 |

`PASS`이다. 따라서 quantile, df, scale, sqrt 또는 two-sided interval의
일반적인 evaluator bug는 발견되지 않았다.

### 100k canonical Fig.9-style calibration

각 regime 10,000 CFR, `Ng=16`, 15 dB observation noise를 사용했다. primary
pooling은 ID=(20,80 ns), OOD=(120 ns,1 ms)이며 이는
**IMPLEMENTATION-ASSUMPTION**이다.

| Nominal | ID empirical | OOD empirical | OOD-Near | OOD-Far |
|---:|---:|---:|---:|---:|
| 0.50 | 0.5131 | 0.2174 | 0.3651 | 0.0696 |
| 0.80 | 0.8126 | 0.3858 | 0.6380 | 0.1336 |
| 0.90 | 0.9091 | 0.4675 | 0.7623 | 0.1728 |
| 0.95 | 0.9561 | 0.5262 | 0.8446 | 0.2077 |
| 0.99 | 0.9921 | 0.6114 | 0.9444 | 0.2785 |

CE는 `mean_c |empirical(c)-nominal(c)|`이다. 논문의 exact aggregation이
공개되지 않았으므로 이 역시 **IMPLEMENTATION-ASSUMPTION: CE aggregation**으로
표시한다.

| Pool | CE MAE (0.1--0.9) | CE MAE (full grid: 0.1--0.9, .95, .99) |
|---|---:|---:|
| ID | 0.0103 | 0.0092 |
| OOD | 0.2686 | 0.2927 |
| OOD-Near | 0.1154 | 0.1081 |
| OOD-Far | 0.4218 | 0.4773 |

ID curve는 ideal line에 가깝고 약한 보수적 경향이 있다. OOD curve는 nominal이
증가할수록 empirical coverage도 증가하지만 ideal line 아래에 크게 위치한다.
즉 OOD interval이 지나치게 좁은 **실제 OOD over-confidence behavior**로
해석된다. 0.90에서 0.4675, 0.99에서도 0.6114까지 증가하므로 interval 또는
quantile이 비정상적으로 고정된 현상은 아니다.

기존 보고의 OOD `0.543`은 이전 50k canonical Fig.9 artifact에서의 값이며,
이번 100k common-set 재평가의 OOD `0.4675`와 동일한 조건의 수치가 아니다.
이번 결과는 `Fig.9-style calibration under diagonal-Psi approximation`이며
논문의 exact full-covariance calibration reproduction으로 주장하지 않는다.

### Artifact, GPU, 판정

실행 명령:
`python scripts/evaluate_100k_fig9_audit.py --output-dir runs/current_valid_baseline/paper_sample_alignment_100k_1ep_fig9_audit_20260915_final --common-eval-dir runs/current_valid_baseline/diversity_ablation/reproducibility_20260914/common_eval --samples-per-regime 10000 --batch-size 256`

실행 시작·종료 시 `torch.cuda.is_available()=True`, `cuda:0`, `NVIDIA GB10`을
확인했다. 결과는
`runs/current_valid_baseline/paper_sample_alignment_100k_1ep_fig9_audit_20260915_final/`
의 `results.json`, `calibration_curve.csv`, `calibration_curve.png`,
`calibration_error.csv`, `coverage_by_regime.csv`,
`synthetic_student_t_sanity.json`에 저장했다. 수정 파일은 audit evaluator
`scripts/evaluate_100k_fig9_audit.py`와 본 README이며, 기존 model/loss/
checkpoint는 수정하지 않았다.

최종 판정은 `CALIBRATION IMPLEMENTATION BUG NOT FOUND`이다. 현재 OOD
under-coverage는 evaluator 산술 오류보다 100k×1 diagonal-Psi model의 실제
OOD over-confidence 후보로 보는 것이 타당하다. 다음 실험은 하나만 권장한다:
**동일 checkpoint에서 diagonal-Psi의 component-wise marginal calibration과
논문이 공개하지 않은 multivariate-region calibration의 차이를 별도 정의로
분리 분석하되, 새 학습은 하지 않는다.** Fig.11/runtime, Full-Psi, training,
Partial FT는 실행하지 않았다.

### 100k×1 Fig.9 OOD scale-parameter decomposition (2026-09-16)

목적은 1 ms Far-OOD에서 실제 error가 증가하는데 predictive scale이 거의
증가하지 않는 원인을 `Psi`, `kappa`, `nu` 반응으로 분해하는 것이었다. 기존
100k×1 checkpoint
`runs/current_valid_baseline/paper_sample_alignment_100k_1ep_fig8_20260915/uacp_predictor_step4a.pt`
와 common 10k evaluation set
`runs/current_valid_baseline/diversity_ablation/reproducibility_20260914/common_eval`
을 재사용했다. 새 training, Full-Psi, Fig.11/runtime, Partial Fine-Tuning은
실행하지 않았다. 모든 regime에서 동일한 `Ng=16`, 15 dB noise, corrected
Real/Imag antenna-pair mapping, omitted-component scope를 사용했다.

계산식은 Eq.(5)의 diagonal-Psi approximation이다: `df=nu-2K+1`,
`scale_sq=Psi*((kappa+1)/kappa)/df`, `scale=sqrt(scale_sq)`이며 sqrt는
정확히 한 번만 적용했다. 평균은 omitted real/imag component 전체의 exact
online sum/count weighted aggregation이다. q50/q90/q95/q99는 RAM 제한을 위한
parameter별 fixed-size reservoir(8192)의 근사 quantile이다.

| Regime | Error | Psi | kappa | df | Scale | Error/Scale |
|---|---:|---:|---:|---:|---:|---:|
| 20 ns | 0.0617 | 0.1851 | 11.7548 | 29.1086 | 0.08217 | 0.7489 |
| 80 ns | 0.0699 | 0.1869 | 11.6966 | 28.9685 | 0.08277 | 0.8411 |
| 120 ns | 0.0962 | 0.1896 | 11.6031 | 28.7467 | 0.08372 | 1.1490 |
| 1 ms | 0.6654 | 0.2548 | 9.7203 | 24.3182 | 0.10669 | 6.3063 |

20 ns 대비 1 ms mean 변화 배수는 `Error=10.78x`, `Psi=1.376x`,
`kappa=0.827x`, `df=0.835x`, `kappa_factor=((kappa+1)/kappa)=1.017x`,
`1/df=1.200x`, `scale_sq=1.688x`, `scale=1.298x`이다. 따라서 kappa 감소와
df 감소는 scale을 억제하지 않고 각각 `kappa_factor`와 `1/df`를 통해 scale을
증가시키는 방향으로 작용한다. 그러나 Psi의 증가와 두 factor의 증가는
최종 scale을 약 1.30배만 키운 반면 실제 error는 약 10.78배 증가했다.

최종 판정은 **CASE C**이다. `Psi/kappa/df` 모두 OOD 방향으로 반응하고
predictive scale도 증가하지만, actual Far-OOD error가 훨씬 더 빠르게
증가한다. 따라서 현재 가장 직접적인 원인은 parameter coupling에 의한
상쇄가 아니라, diagonal-Psi predictive scale이 training 범위 밖의 Far-OOD
error magnitude를 충분히 일반화하지 못하는 것이다. 이는 full-covariance
논문의 exact calibration이 아닌 **IMPLEMENTATION-ASSUMPTION: diagonal-Psi
approximation** 결과이다.

결과 artifact:
`runs/current_valid_baseline/diagnostics/fig9_scale_parameter_decomposition_20260916/`
의 `results.json`, `parameter_summary.csv`, `relative_change.csv`,
`error_vs_scale.png`, `parameter_factor_change.png`. 실행 명령:
`python scripts/fig9_scale_parameter_decomposition.py --output-dir runs/current_valid_baseline/diagnostics/fig9_scale_parameter_decomposition_20260916 --common-eval-dir runs/current_valid_baseline/diversity_ablation/reproducibility_20260914/common_eval --samples-per-regime 10000 --batch-size 256 --eval-seed 20262000`
Peak RSS는 약 `1.736 GiB`, swap은 `0`이었다. 수정 파일은
`scripts/fig9_scale_parameter_decomposition.py`와 본 README이다.

다음 추천 실험은 새 학습 없이 동일 checkpoint에서 `Psi`, `kappa`, `df`를
각각 20 ns baseline 값으로 counterfactual 고정해 1 ms scale 변화를 비교하는
단일-factor intervention이다.

### Figure 11-style dynamic validation and runtime (2026-09-16)

#### Scope and frozen inputs

이번 검증은 현재 reproduction baseline에서 channel regime 변화가 uncertainty와
다음 sounding action으로 연결되는지, 그리고 online control path 및 별도
full-model single-update 비용이 어느 정도인지 확인하기 위한 것이다. canonical
100k×1 checkpoint
`runs/current_valid_baseline/paper_sample_alignment_100k_1ep_fig8_20260915/uacp_predictor_step4a.pt`
를 그대로 사용했고, 새 predictor training, Partial Fine-Tuning, Full-Psi,
Uncertainty-aware Fine-Tuning은 실행하지 않았다.

Paper-aligned settings: `Nr=2`, `Nt=2`, `K=1024`, carrier `3.5 GHz`,
subcarrier spacing `30 kHz`, `SNR=15 dB`. Dynamic order는 논문 Figure 11과
같이 `20 ns -> 80 ns -> 10 ns -> 40 ns -> 60 ns -> 120 ns`로 고정했다.
현재 baseline은 논문과 달리 `100k unique CFR / 1 epoch / batch 8 /
diagonal-Psi / current direct-CFR observation / current training-mask
distribution`이다. 논문은 `150 epochs / batch 4096 / full covariance Psi /
online randomized mask`를 사용하므로 이번 결과는 exact reproduction이 아니라
**Figure 11-style dynamic validation using the current reproduction baseline**이다.

#### Implementation assumptions and controller

Repository에는 Figure 11의 temporal-correlated sequence generator, 실제
Algorithm 1 controller, BER/EVM/precoding/postcoding pipeline, online adaptation
optimizer가 없었다. 따라서 기존 delay-sweep CFR artifact
`runs/baseline_reproduction/step2_delay_sweep_repro/generated_data/`에서 각
regime의 time-ordered CFR 40개를 이어 붙여 240-step sequence를 만들었다.
이는 **IMPLEMENTATION-ASSUMPTION**이며 논문의 `60,000 sequences x 64
temporally-correlated realizations` 또는 Figure 11 x축 단위와 동일시하지 않는다.

Candidate Ng는 논문 exact 공개값이 없으므로
`[128, 64, 32, 16, 8, 4, 1]`로 두었다. `Ng=1`은 full feedback이다. 현재
모델이 학습에서 보지 않은 `Ng=128/64`는 **mask-pattern extrapolation
limitation**이다. `U_epi_max`는 dynamic data를 보지 않고 기존 ID 20/80 ns
자료에서 모든 candidate Ng의 Eq.(13) score 99th percentile로 고정했다:
`U_epi_max=0.242031`. Aleatoric target은 기존 ID 20/80 ns Eq.(13) 평균,
`U_ale^opt=0.0139182`, hysteresis는 `0.1*U_ale^opt`로 두었다. 이는
**IMPLEMENTATION-ASSUMPTION**이다.

Algorithm 1 causality는 유지했다. round `u`의 uncertainty를 계산한 뒤
`S_(u+1)`/`Ng_(u+1)`를 결정했으며, epistemic threshold 초과 시에만
`adaptation_trigger=True`와 다음 round full feedback을 설정했다. 실제 online
fine-tuning은 수행하지 않았다. BER/EVM/precoding downstream pipeline은
없으므로 해당 수치는 만들지 않았다.

#### Dynamic trace result

| Regime | Current Ng values | Mean NMSE omitted [dB] | Mean U_ale Eq.(13) | Mean U_epi Eq.(13) | Trigger |
|---|---|---:|---:|---:|---:|
| 20 ns | 128, 64, 32, 16, 8 | -18.008 | 0.02493 | 0.00465 | 0 |
| 80 ns | 32, 16, 8 | -17.988 | 0.01430 | 0.00127 | 0 |
| 10 ns | 32, 16, 8 | -18.906 | 0.01626 | 0.00158 | 0 |
| 40 ns | 32, 16, 8 | -19.078 | 0.01665 | 0.00165 | 0 |
| 60 ns | 32, 16, 8 | -18.768 | 0.01461 | 0.00131 | 0 |
| 120 ns OOD | 16, 8 | -15.339 | 0.01335 | 0.00113 | 0 |

초기 20 ns에서 실제 action은 `Ng=128 -> 64 -> 32 -> 16`으로 dense해졌고,
80 ns에서는 `Ng=16` 중심으로 유지됐다. 따라서 aleatoric feedback path는
일부 연결되지만, 20 ns의 안정적 고 Ng 수렴은 재현되지 않았다. 120 ns에서는
NMSE가 악화됐지만 epistemic score가 threshold `0.242031`에 도달하지 않아
adaptation trigger와 full-feedback fallback이 모두 0회였다. 결과는
**PARTIAL MATCH**: uncertainty/action 연결은 관찰되지만 Figure 11의 OOD
fallback qualitative behavior는 **MISMATCH**이다.

#### Runtime

CUDA Event/synchronize를 사용해 batch size 1에서 200회 측정했다.

| Stage | p50 [ms] | p95 [ms] | p99 [ms] |
|---|---:|---:|---:|
| Predictor forward | 134.049 | 138.692 | 140.094 |
| Evidential uncertainty | 0.779 | 1.599 | 2.123 |
| Eq.(12)/(13) aggregation | 1.981 | 4.137 | 5.239 |
| Controller decision | 0.009 | 0.009 | 0.016 |
| Total online control | 135.853 | 141.041 | 142.148 |
| End-to-end software | 137.008 | 141.942 | 144.797 |

별도 in-memory model instance에서 current loss, batch size 8, full-model
optimizer를 사용해 50회 single-update microbenchmark를 수행했다. 이는 전체
adaptation latency가 아니다.

| Stage | p50 [ms] | p95 [ms] |
|---|---:|---:|
| Forward + loss | 192.568 | 197.472 |
| Backward | 252.464 | 259.272 |
| Optimizer step | 7.685 | 10.877 |
| Total single update | 452.562 | 462.364 |

Peak system RSS는 `1.736 GiB` 수준이었고 peak GPU allocated memory는 약
`906 MiB`였다. `torch.cuda.is_available=True`, `cuda:0`, `NVIDIA GB10`을
시작/종료에서 확인했다. BER/EVM은 **unavailable: repository PHY/
precoding/postcoding pipeline 없음**이다.

#### Interpretation

Q1에 대한 결론은 **PARTIAL MATCH**이다. 현재 model uncertainty는 aleatoric
score를 통해 next-round Ng를 변경하는 control signal로 연결되지만, candidate
Ng=128/64에서 mask extrapolation uncertainty가 커서 자연스러운 steady-state
수렴이 불안정하다. ID-only candidate-wide threshold 기준에서는 120 ns OOD
epistemic trigger가 발생하지 않아 OOD fallback signal로는 현재 baseline을
사용할 수 없다.

Q2에 대한 결론은 uncertainty aggregation/controller만 보면 약 `2.8 ms`
(p50 기준)지만 predictor forward가 약 `134 ms`여서 일반적인 `10 ms`
sounding interval과 직접 비교할 때 online full path는 충분히 빠르지 않다.
Single full-model optimizer update는 약 `453 ms`이므로 inference/control보다
약 3.3배 무겁다. 단, 논문이 adaptation update 횟수와 stopping rule을 공개하지
않았으므로 전체 adaptation 시간을 외삽하지 않는다.

Artifact는
`runs/current_valid_baseline/fig11_dynamic_runtime_20260916_v2/`의
`results.json`, `config_used.json`, `dynamic_trace.csv`,
`runtime_summary.csv`, `adaptation_runtime.csv`, `fig11_style.png`,
`uncertainty_diagnostic.png`이다. 실행 명령:
`python scripts/fig11_dynamic_runtime_validation.py --output-dir runs/current_valid_baseline/fig11_dynamic_runtime_20260916_v2 --segment-length 40 --runtime-repeats 200 --adaptation-repeats 50 --batch-size 8 --eval-seed 20262000`

다음 추천 실험은 새 training 없이, 현재 checkpoint의 `Ng=128/64/32/16`에
대한 mask-pattern calibration을 ID validation에서 별도로 측정해 candidate
별 `U_epi_max`를 비교하는 것이다. 이는 threshold 문제와 model OOD 반응 문제를
분리한다.

## Sequential diagnosis of missing 120 ns OOD fallback (2026-09-16)

### Experiment 1: Ng-conditioned epistemic threshold

목적은 candidate-wide global `U_epi_max=0.242031`가 서로 다른 Ng의 ID
분포를 섞어 120 ns trigger를 억제하는지 검증하는 것이었다. 기존 canonical
100k×1 checkpoint
`runs/current_valid_baseline/paper_sample_alignment_100k_1ep_fig8_20260915/uacp_predictor_step4a.pt`
를 그대로 사용했고, 새 training은 없었다. 기존 200-sample delay-sweep
파일의 20/80 ns를 ID calibration, 120 ns와 1 ms를 OOD 평가에 사용했다.
각 Ng의 threshold는 `tau_Ng = q99(U_epi_ID(20 ns + 80 ns))`로 계산했다.
Eq.(8)의 현재 diagonal-Psi 계산을 corrected antenna-pair mapping으로
Eq.(12)/(13)에 적용했다.

`Ng={4,8,16,32}`는 training-supported, `{64,128}`는 mask-pattern
extrapolation, `Ng=1`은 omitted set이 없어 score가 정의되지 않는 full-feedback
diagnostic이다. 이는 implementation assumption이다.

| Ng | status | tau_Ng | ID false trigger | 120 ns trigger | 1 ms trigger | Near AUROC | Far AUROC |
|---:|---|---:|---:|---:|---:|---:|---:|
| 4 | supported | 0.000512 | 1.0% | 3.0% | 99.5% | 0.662 | 0.999 |
| 8 | supported | 0.000731 | 1.0% | 2.5% | 97.0% | 0.647 | 0.998 |
| 16 | supported | 0.001597 | 1.0% | 1.0% | 98.0% | 0.631 | 0.999 |
| 32 | supported | 0.006364 | 1.0% | 1.0% | 93.5% | 0.637 | 0.998 |
| 64 | extrapolation | 0.295796 | 1.0% | 0.5% | 0.0% | 0.512 | 0.532 |
| 128 | extrapolation | 0.728487 | 1.0% | 0.0% | 0.0% | 0.507 | 0.517 |

판정: candidate-specific threshold에서도 training-supported Ng의 120 ns
trigger는 1–3%에 그쳤다. 따라서 global threshold mixing은 주요 원인이
아니다. Far-OOD는 잘 분리되지만 Near-OOD는 score overlap이 크다.

Artifact: `runs/current_valid_baseline/diagnostics/exp1_ng_conditioned_epi_threshold_20260916/`
의 `results.json`, `ng_threshold_summary.csv`, `score_summary.csv`.
실행 명령:
`python scripts/exp1_ng_conditioned_epi_threshold.py --output-dir runs/current_valid_baseline/diagnostics/exp1_ng_conditioned_epi_threshold_20260916 --batch-size 8 --samples-per-regime 200`

### Experiment 2: backbone feature-space OOD detection

가설은 evidential U_epi가 120 ns를 놓쳐도 shared backbone feature가 shift를
표현하는지였다. 마지막 residual block 출력(`residual_blocks[-1]`), 즉
evidential heads 직전의 `[B,192,1024]`를 frequency mean-pooling해 192차원
표현으로 고정했다. Ng=16을 고정해 mask-pattern confounding을 피했고, ID
20/80 ns feature만으로 diagonal covariance와 mean을 추정했다. 이 고정 Ng,
frequency pooling, diagonal covariance 및 variance floor `1e-6`은
IMPLEMENTATION-ASSUMPTION이며 OOD feature로 fit하지 않았다.

| Score | ID vs 120 ns AUROC | ID vs 1 ms AUROC |
|---|---:|---:|
| U_epi | 0.630 | 0.999 |
| feature Mahalanobis | 0.326 | 0.524 |

Feature Mahalanobis mean/median/q95는 20 ns `18.506/18.129/26.017`,
80 ns `4.340/4.295/6.253`, 120 ns `5.053/4.947/7.591`, 1 ms
`10.267/10.144/13.698`이었다. 즉 120 ns가 ID보다 일관되게 멀어지지
않았고, backbone representation 자체의 Near-OOD separation은 확인되지
않았다. U_epi와 Mahalanobis의 120 ns Spearman correlation은 0.452였다.

Artifact: `runs/current_valid_baseline/diagnostics/exp2_feature_ood_mahalanobis_20260916/`
의 `results.json`, `mahalanobis_summary.csv`, `correlation_summary.csv`.
실행 명령:
`python scripts/exp2_feature_ood_mahalanobis.py --output-dir runs/current_valid_baseline/diagnostics/exp2_feature_ood_mahalanobis_20260916 --batch-size 8 --samples-per-regime 200`

### Gated Experiment 3 decision

`lambda_reg=1e-2` 새 training은 수행하지 않았다. Experiment 1은 threshold만의
문제를 지지하지 않았고 Experiment 2도 backbone이 120 ns를 구분한다는
근거를 제공하지 않았다. 따라서 현재 evidence만으로 evidential regularizer
strength를 원인으로 분리하는 것은 부적절하다. 이 실험은 RESEARCH-ABLATION으로
보류했다. Partial Fine-Tuning, Full-Psi, 새로운 pretrained training도 수행하지
않았다.

두 실험 모두 inference는 `torch.inference_mode()`, batch size 8, scalar score
보존 방식으로 실행했다. `cuda:0`, NVIDIA GB10을 사용했고 peak RSS는 Exp1
1.273 GiB, Exp2 1.272 GiB였다. 기존 결과/로그는 삭제하거나 덮어쓰지 않았다.

종합 판정은 **C (backbone representation 문제) + 약한 evidential mapping의
복합 양상(D에 가까운 보조 요인)**이다. 현재 가장 직접적인 원인은 120 ns
Near-OOD의 shift가 backbone과 U_epi에 충분히 크고 일관된 evidence separation으로
전달되지 않아 ID q99 threshold를 넘지 못하는 것이다.

다음 추천 실험 1개: 새 training 전, ID validation과 120 ns를 동일 Ng=16에서
frequency별 shared-feature drift와 per-subcarrier U_epi를 online scalar
통계로 비교해 Near-OOD 정보가 어느 주파수 영역에서 소실되는지 확인한다.

## Figure 8 paper-style distribution plot (2026-09-16)

논문 Fig.8과 직접 비교할 수 있도록 canonical 100k×1 checkpoint의 기존
sample-level evaluation artifact를 다시 plot했다. 새 training이나 model
forward는 수행하지 않았다. Source checkpoint:
`runs/current_valid_baseline/paper_sample_alignment_100k_1ep_fig8_20260915/uacp_predictor_step4a.pt`;
sample scores:
`runs/current_valid_baseline/paper_sample_alignment_100k_1ep_fig8_20260915/fig8_eval_retry/per_sample.csv`.
20/80/120 ns/1 ms 각각 10,000 sample을 사용했다.

Score는 기존 corrected Eq.(7)/(8)/(12)/(13) sample-level linear `U_epi`를
사용하고 각 sample별로 `10*log10(U_epi)` 변환했다. 이 dB 변환은 논문 식이
아닌 IMPLEMENTATION-ASSUMPTION이며 offset/rescaling은 적용하지 않았다.
기존 Fig.8 script의 density 표현인 histogram을 따라 네 regime에 같은
0.25 dB bin 폭을 썼다. Paper bin 설정은 공개되지 않아 bin 폭은
IMPLEMENTATION-ASSUMPTION이다. Density는 모든 원본 sample 수로 정규화하고,
paper-axis clipping 후 재정규화하지 않았다.

Main plot의 x축은 `[-37.5,-22.5]`, y축 `[0,0.4]`; x ticks는 2.5 dB 간격,
y ticks는 0.1 간격이다. 흰 배경, dashed light grid, black border, filled
semi-transparent histogram을 사용했으며 색은 ID-Easy blue `#1f77b4`,
ID-Hard orange `#ff7f0e`, OOD-Near green `#2ca02c`, OOD-Far red `#d62728`이다.
별도 full-range plot은 같은 bins/density로 표시 범위만 자동 설정했다.

| Regime | N | Mean dB | Median dB | Std dB | Min–max dB | Paper x-range 밖 |
|---|---:|---:|---:|---:|---:|---:|
| ID-Easy 20 ns | 10,000 | -29.3563 | -29.4581 | 0.6382 | -30.6955 to -24.9445 | 0% |
| ID-Hard 80 ns | 10,000 | -29.2689 | -29.3371 | 0.4638 | -30.2718 to -26.3184 | 0% |
| OOD-Near 120 ns | 10,000 | -29.1361 | -29.1886 | 0.4185 | -30.2623 to -27.2794 | 0% |
| OOD-Far 1 ms | 10,000 | -26.2954 | -26.3847 | 1.0240 | -28.9486 to -20.8170 | 0.21% right of axis |

Recomputed AUROC matches existing artifact exactly: ID vs pooled OOD `0.8086853`,
ID vs Near `0.6204087`, ID vs Far `0.9969620`, ID-Hard 80 vs Near 120
`0.5962121`. The regime means also match to <1e-9 dB. ID-Easy/ID-Hard/Near
substantially overlap; Near shifts only slightly right, while Far shifts clearly
right, consistent with original sample statistics. The fixed y limit of 0.4
clips the tallest histogram bins; this is retained to honor the paper-axis
comparison and noted as a visual limitation.

Generated artifacts:
`runs/current_valid_baseline/paper_sample_alignment_100k_1ep_fig8_20260916/paper_style_plot/`
contains `fig8_paper_style_axis.png/.pdf/.svg`, `fig8_full_range.png`,
`fig8_sample_scores.csv`, and `fig8_plot_summary.json`.
Reproduction command:
`python scripts/fig8_paper_style_plot.py --output-dir runs/current_valid_baseline/paper_sample_alignment_100k_1ep_fig8_20260916/paper_style_plot`

## Figure 9 paper-style calibration plot (2026-09-16)

원 논문 Fig.9와 동일한 좌표계/curve 표현으로 현재 canonical 100k×1
calibration을 비교 plot했다. 새 training이나 reevaluation은 하지 않고 기존
최종 Fig.9 audit artifact의 actual empirical counts를 사용했다. Checkpoint는
`runs/current_valid_baseline/paper_sample_alignment_100k_1ep_fig8_20260915/uacp_predictor_step4a.pt`,
evaluation set은 `runs/current_valid_baseline/diversity_ablation/reproducibility_20260914/common_eval`이며
regime당 10,000 sample이다.

Coverage는 검증된 marginal Student-t evaluator 그대로 계산된 audit 결과를
사용한다: `df=nu-2K+1`, `scale²=((kappa+1)/(kappa*df))*Psi`,
`scale=sqrt(scale²)`, 양측 quantile `t_df^-1((1+c)/2)`, interval은
`gamma ± q*scale`. 현재 diagonal-Psi approximation에서 omitted-subcarrier
Real/Imag components를 집계했다. ID는 20+80 ns의 component counts를 pooling,
OOD pooled는 120 ns+1 ms count pooling이며 Near/Far 가중을 바꾸지 않았다.
실제 plotting grid는 0.1–0.9 (0.1 간격), 0.95, 0.99이며
IMPLEMENTATION-ASSUMPTION이다. 점 사이 interpolation은 하지 않았다.

CE는 `mean_c |empirical(c)-nominal(c)|`이며 논문의 exact reduction과
동일하다고 주장하지 않는다. 기존 0.1–0.9 grid CE는 ID `0.010303`, pooled
OOD `0.268596` (Near `0.115351`, Far `0.421841`)이다. 추가 0.95/0.99를
포함하는 실제 11-point plot-grid CE는 ID `0.009169`, pooled OOD `0.292709`
(Near `0.108106`, Far `0.477311`)이다. Main legend에는 plotted grid CE를
표시했다. 별도의 5 anchor point (0.50/0.80/0.90/0.95/0.99)만으로 계산한
보조 CE는 ID `0.008583`, pooled OOD `0.386342`이다.

Main plot은 x/y `[0,1]`, ticks `0.0,0.2,...,1.0`, Ideal dark-gray dashed,
ID `#1f77b4` solid triangle, OOD pooled `#ff7f0e` dash-dot circle, upper-left
legend로 표시한다. Diagnostic plot은 별도이며 ID/Near/Far를 분리한다.
0.50/0.80/0.90/0.95/0.99 coverage는 요청된 기존 값과 반올림 수준에서
일치한다 (최대 절대 차이 `0.0000503`). ID curve는 ideal에 가깝고 약간
위쪽이며, pooled OOD curve는 전 nominal range에서 크게 아래쪽이다.

Artifacts:
`runs/current_valid_baseline/paper_sample_alignment_100k_1ep_fig9_20260916/paper_style_plot/`
에 `fig9_paper_style.png/.pdf/.svg`, `fig9_near_far_diagnostic.png`,
`calibration_curve_full.csv`, `calibration_plot_summary.json`이 있다.
기존 Fig.9 audit artifact는 변경하지 않았다. 실행 명령:
`python scripts/fig9_paper_style_plot.py --output-dir runs/current_valid_baseline/paper_sample_alignment_100k_1ep_fig9_20260916/paper_style_plot`

## Fig.8/Fig.9 protocol audit and conditional 100k×5 gate (2026-09-16)

새 training 전에 canonical 100k×1 evaluator를 audit했다. Checkpoint는
`runs/current_valid_baseline/paper_sample_alignment_100k_1ep_fig8_20260915/uacp_predictor_step4a.pt`
(시작/종료 SHA256 확인값 `0e5b0472fd559d4e08f7e6200c512c10dfc208f7a4d507bf613dbd69958b6382`),
common evaluation은 regime별 10,000 sample이다. 관련 수식과 구현의
Real/Imag pair mapping, pair-scalar kappa/nu broadcast, omitted mask 방향,
sample-level dB 변환 및 Student-t scale은 기존 구현과 대조했다.

Fig.8 sensitivity (모든 regime sample 수 동일; ID 20+80 pooled, pooled OOD
120+1ms pooled; paper의 exact pooling 비율은 미공개이므로
IMPLEMENTATION-ASSUMPTION):

| Aggregation diagnostic | ID vs 120 ns | ID vs 1 ms | 80 vs 120 ns |
| --- | ---: | ---: | ---: |
| omitted-only Eq.(13), canonical | 0.6204 | 0.9970 | 0.5962 |
| all-subcarrier (diagnostic) | 0.6211 | 0.9970 | 0.5968 |
| observed-only (diagnostic) | 0.6329 | 0.9980 | 0.6068 |

Linear U_epi와 sample별 `10 log10(U_epi)` AUROC 차이는 0이었다. Omitted
점수는 저장된 기존 Fig.8 sample score와 정확히 일치했다 (max difference 0).
합리적인 aggregation 변경에서 Near AUROC의 최대 회복은 약 0.0125뿐이므로
protocol mismatch만으로 Near separation 부족을 설명하지 못한다.

Fig.9는 기존 synthetic-validated Student-t 식을 유지했다:
`df=nu-2K+1`, `scale²=((kappa+1)/(kappa*df))*Psi`, sqrt 1회,
interval `gamma ± t_df^-1((1+c)/2)*scale`. Omitted/all/observed diagnostic의
coverage@0.9는 각각 다음과 같고, 모두 OOD에서 ideal 아래다.

| Aggregation | ID | Near 120 ns | Far 1 ms | pooled OOD |
| --- | ---: | ---: | ---: | ---: |
| omitted-only canonical | 0.9091 | 0.7623 | 0.1728 | 0.4675 |
| all-subcarrier diagnostic | 0.9089 | 0.7623 | 0.1798 | 0.4710 |
| observed-only diagnostic | 0.9063 | 0.7621 | 0.2847 | 0.5234 |

Fig.8 score / Fig.9 omitted coverage는 이전 결과와 정확히 일치했고
(Fig.9 최대 차이 0), pooling/aggregation은 OOD curve 방향을 반전하지 않았다.
따라서 protocol-only 설명은 NO이며, 현 baseline에서 실제 Near-OOD epistemic
separation 및 OOD predictive calibration 부족이 남는다. 세 aggregation은
sensitivity diagnostic이며 observed/all을 논문 정의로 주장하지 않는다.

Audit artifacts:
`runs/current_valid_baseline/diagnostics/fig8_fig9_protocol_audit_20260916_v2/`
(`protocol_audit_results.json`, `fig8_protocol_comparison.csv`,
`fig9_protocol_comparison.csv`). 실행 명령:
`python scripts/fig8_fig9_protocol_audit.py --output-dir runs/current_valid_baseline/diagnostics/fig8_fig9_protocol_audit_20260916_v2 --common-eval-dir runs/current_valid_baseline/diversity_ablation/reproducibility_20260914/common_eval --batch-size 256`.
감사 중 peak RAM은 약 1.69 GiB였다.

### Conditional 100k×5 RNG validation and controlled run (historical)

protocol audit에서 문제가 설명되지 않아 100k×5 epoch-convergence 비교를
시작했다. 의도한 변경은 epoch 수(1→5)뿐이고, 기존 100k CFR archive,
batch 8, seed 20260819, architecture/diagonal-Psi, SNR 15 dB, mask policy,
Adam/LR 1e-4, lambda 1e-3를 사용했다. CUDA device는 NVIDIA GB10 / cuda:0였다.

첫 full-epoch 시도는 training CFR generation의 RNG side effect를 재현하지
않아 epoch 1 gate에서 실패했다: total loss `-2042.3039` 대 `-2038.2043`,
omitted NMSE `-17.0774` 대 `-17.0517 dB`. 추적 결과 canonical Step-4A에서는
model 초기화 직전 `_step4_probe_paths()`가 1 ms Far-OOD probe 200개를 생성하며
매 CFR마다 RNG를 재설정했다. 이 마지막 probe generation이 initialization RNG
상태를 결정하므로 training CFR 재생성만으로는 충분하지 않았다. 실패 artifact는
보존했다:
`runs/current_valid_baseline/training_convergence_100k_5ep_20260916/`.
해당 실패 run peak RAM은 약 4.67 GiB였고, canonical checkpoint SHA256는
변경되지 않았다.

다음 시도는 사용자 요청에 따라 epoch 1 이전(1,000 batch)에서 중단했다.
`.venv-uacp`(PyTorch 2.13.0+cu130, Sionna 2.0.1, GB10/cuda:0)에서 짧은
preflight를 수행했다. canonical 200-probe generation path와 마지막 probe 1개만
replay하는 경로의 model initialization SHA256가 모두
`9c4ef340d8ad565fdf8669e2725db608317fb0fcd49fdc1fa4a2661dbeee8247`로 일치했고,
first-three-batch train loss/NMSE aggregates도 정확히 일치했다. Canonical initial
weights 자체는 저장되어 있지 않으므로 이는 code-path replay 간 검증이며,
가장 이른 저장된 deterministic reference인 epoch-1 aggregate로 다시 gate한다.

이 검증 후 시작한 최종 controlled run은
`runs/current_valid_baseline/training_convergence_100k_5ep_20260916_final_controlled/`
에 저장되었다. Epoch 1은 canonical history와 PASS했다: total
`-2042.303875336914`, NLL `-2100.9545548980714`, reg `58650.67674273437`,
NMSE-all `-17.081016470531523 dB`, NMSE-omitted `-17.077443819029032 dB` 모두
저장값과 일치한다. final checkpoint 및 matched comparison도 해당 디렉터리에
보존되어 있다. Epoch-1 peak RSS는 약
5.10 GiB, peak allocated GPU memory 약 861 MiB; canonical checkpoint는 보존된다.

### Final 100k×5 batch400 BF16 run (2026-09-18)

목적은 기존 100k×1 batch8 및 100k×5 batch8과 비교하여 batch를 400으로
늘렸을 때 Fig.8/9/11 uncertainty behavior가 복구되는지 확인하는 것이었다.
이전 10k×5 batch400 실행은 100k training set 조건이 아니므로 최종 실험으로
해석하지 않았고 artifact를 보존했다.

최종 설정은 unique CFR 100,000, epochs 5, batch 400, steps/epoch 250,
optimizer steps 1,250, exposure 500,000, seed 20260819, LR 1e-4,
lambda_reg 1e-3, SNR 15 dB, cuda:0/NVIDIA GB10이다. architecture,
diagonal-Psi, mask policy, observation pipeline 및 canonical optimizer를
유지했다. BF16 AMP는 backbone에만 적용하고 evidential transform/loss는
FP32로 계산했다(IMPLEMENTATION-ASSUMPTION). torch.compile은 사용하지 않았다.

실행 방법:

`python scripts/train_100k5_batch400.py --expected-n 100000 --output-dir runs/current_valid_baseline/batch400_100k5ep_20260918_bf16_final`

학습은 약 93분(5,579초), peak allocated GPU memory는 약 17.22 GiB였다. 최종 결과와
checkpoint는 `runs/current_valid_baseline/batch400_100k5ep_20260918_bf16_final/`
에 있다. Fig.8/9는 regime별 10,000 samples를 streaming/chunk evaluator로
처리했고 Fig.9 coverage anchor는 0.5/0.8/0.9/0.95/0.99를 저장했다.

| model | NMSE 20/80/120/1ms (dB) | ID→Near AUROC | ID→Far AUROC | pooled AUROC | Fig.9 ID/OOD MAE | ν/inf |
| --- | --- | ---: | ---: | ---: | --- | --- |
| 100k×1 batch8 | -18.98/-17.86/-15.15/1.25 | 0.621 | 0.997 | 0.809 | 0.0086/0.3863 | prior result |
| 100k×5 batch8 | -21.80/-19.48/-16.47/1.31 | 0.948 | NaN | NaN | 0.0008/0.0500 | Far inf |
| 100k×5 batch400 BF16 | -17.35/-17.09/-15.85/1.42 | 0.542 | 0.803 | 0.673 | 0.2698/0.0859 | finite, nonfinite=0 |

100k×5 batch400은 80 ns NMSE가 20 ns보다 0.26 dB 높아 조금 더 어려웠고,
aleatoric mean도 0.14152→0.14192로 소폭 증가했다. Far AUROC 0.803은 남았지만
Near separation은 0.542로 개선되지 않았다. 기존 100k×5 batch8의 높은 Near
AUROC는 Far ν/inf가 있어 전체적으로 유효한 기준으로 보기 어렵다. 현재 run은
ν가 유한하다는 점은 개선됐지만 ID calibration MAE는 커졌다.

Fig.11-lite 순서(20→80→10→40→60→120)에서는 세 모델 모두 20 ns 이후
Ng=1로 collapse했으므로 controller linkage 성공 기준은 실패했다. EVM/BER는
요청대로 실행하지 않았다. ordered delay-sweep 및 ID-only threshold는
IMPLEMENTATION-ASSUMPTION이다.

비교표는 `comparison_100k1_100k5batch8_100k5batch400.csv/json`이며, Fig.8/9와
Fig.11-lite 산출물은 `evaluation_streaming_v5/`, `fig11_lite/`,
`fig11_comparison/`에 있다. 다음 추천은 uncertainty calibration과 ν
parameterization을 먼저 점검하는 것이다.

### Final report evaluation: efficient reproduction baseline (2026-09-18)

추가 training 없이 `runs/current_valid_baseline/batch400_100k5ep_20260918_bf16_final/`
의 최종 checkpoint를 고정하여 보고서용 Fig.8, Fig.9, Fig.11-lite와 runtime을
새 `report_final_20260918/` 디렉터리에서 생성했다. 이 결과는 논문의
100k/150 epochs/batch4096 완전 재현이 아니라, GB10에서 실제로 완료되고 GPU
효율과 안정성이 확인된 **현재 GPU 환경에서 학습 효율과 안정성을 고려한 재현
baseline**이다.

평가 설정은 100k samples×5 epochs/direct batch400/BF16 AMP backbone이며,
evidential transform/loss는 FP32로 유지했다. 기존 Eq.(7),(8),(12),(13)
pipeline, 20/80/120 ns 및 1 ms regime, common evaluation set을 그대로
사용했다. BF16 AMP, batch400, 5 epochs, diagonal-Psi, reported CFR 15 dB
complex AWGN, current mask/observation pipeline, pair-scalar Student-t
aggregation, ordered delay sweep와 ID-only controller threshold는
`IMPLEMENTATION-ASSUMPTION`이다.

| 항목 | 결과 |
| --- | --- |
| Training | 100,000×5, direct batch400, 1,250 steps, 500,000 exposure |
| Training runtime | 5,579 s (약 93분), 실제 89.6 samples/s |
| GPU | NVIDIA GB10/cuda:0, benchmark util 92.8%, peak 17.22 GiB |
| Fig.8 AUROC | ID→Near 0.5423, ID→Far 0.8031, pooled 0.6727 |
| Fig.9 MAE | ID 0.2698, Near 0.2409, Far 0.2920, pooled OOD 0.0859 |
| Fig.11-lite | mean Ng: 4.175→1→1→1→1→1; EVM/BER unavailable |
| Runtime | BF16 forward 38.39 ms, uncertainty 6.12 ms, controller 0.056 ms |
| End-to-end sounding | 56.76 ms mean, 29.69 ms median, 118.18 ms p95 |
| Evaluator wall-clock | Fig.8+Fig.9 streaming 446.09 s, Fig.11-lite 60.56 s |

Fig.8/9 재생성은 10,000 samples/regime streaming 평가이며, `ν` nonfinite는
0이었다. Far-OOD epistemic은 증가했지만 Near-OOD separation은 약하고 ID
calibration은 나쁘다. Fig.11 controller는 첫 20 ns sample에서 threshold를
trigger한 뒤 Ng=1로 collapse했고 120 ns 진입 시에는 이미 Ng=1이어서 별도
epistemic trigger가 발생하지 않았다. repository에는 PHY/precoding/postcoding
BER/EVM 계산 경로가 없으므로 두 metric은 보고하지 않았다.

상세 보고서:
`runs/current_valid_baseline/batch400_100k5ep_20260918_bf16_final/report_final_20260918/REPORT.md`
와 `runtime/runtime_summary.csv`, `fig11_lite/dynamic_trace.csv`.
이 baseline 검증 이후에만 Fig.8/9/11 해석을 확정하고, 그 다음 Full FT vs
Partial FT를 검토한다.

### Random-subset mask scratch experiment (2026-09-18)

기존 predictor가 periodic/uniform grouping placement에 과도하게 적응했을
가능성을 검증하기 위해 기존 checkpoint recovery가 아닌 scratch training을
수행했다. 각 sample에서 `Ng={4,8,16,32,64,128}`을 균등하게 선택하고,
1024개 subcarrier 중 `1024/Ng`개를 without replacement random subset으로
선택했다. `Ng=1`은 training에서 제외했다.

이는 `IMPLEMENTATION-ASSUMPTION: random subset sampling conditioned on Ng in
{4,8,16,32,64,128}, uniformly sampled`이다. 논문 설정으로 단정하지 않는다.
기존 architecture, Adam LR=1e-4, loss, diagonal-Psi, delay/noise/target/layout,
BF16 backbone 및 FP32 evidential path는 유지했다.

100,000 samples × 2 epochs / direct batch400으로 500 optimizer steps,
200,000 exposure를 수행했다. 학습 시간은 2,236.18초(37.27분), NVIDIA
GB10/cuda:0, peak allocated 17,640 MiB였다. 실제 Ng draw count는
`33468/32993/33350/33292/33310/33587` (Ng4/8/16/32/64/128)로 균등했다.

Quick Gate는 20/80/120 ns × Ng16/32/64/128, 조합당 50 samples로 random
subset evaluation을 수행했다. 모든 조합에서 `nu` nonfinite는 0이었다.

| 조건 | 20 ns | 80 ns | 20→80 Aleatoric |
| --- | ---: | ---: | ---: |
| Ng16 all-NMSE | -8.746 dB | -8.876 dB | AUROC 0.432, 감소 |
| Ng32 all-NMSE | -4.415 dB | -4.340 dB | AUROC 0.591, 증가 |
| Ng64 all-NMSE | -1.977 dB | -2.020 dB | AUROC 0.426, 감소 |
| Ng128 all-NMSE | -0.847 dB | -0.897 dB | AUROC 0.457, 감소 |

Ng64/128은 기존 periodic baseline의 약 `-3.6/-1.3 dB`보다 좋아지지
않았고, Aleatoric 방향성도 Ng32에서만 나타났다. 따라서 Gate A와 Gate B
모두 실패했다. Early-stop 규칙에 따라 epoch 3~5, static sweep, Fig.8,
Fig.9, Fig.11은 실행하지 않았다.

최종 판단은 **C. Random subset mask로도 두 문제가 충분히 해결되지 않음**이다.
즉 periodic placement 하나가 유일한 원인이라는 가설은 이 2 epoch scratch
조건에서 지지되지 않았다. 다음 분석 우선순위는 diagonal-Psi의
frequency-correlation 부재, evidential loss/regularization scale,
training scale/optimization exposure 순이다. 추가 training은 승인 전
실행하지 않는다. 상세 결과와 원자료는
`runs/current_valid_baseline/random_subset_mask_100k2ep_batch400_bf16_20260918/`
의 `RANDOM_SUBSET_SUMMARY.md`와 `quick_gate/`에 있다.

### Ng=64/128 mask-coverage recovery diagnostic (2026-09-18)

추가 training 없이 종료할 수 있는 가장 짧은 진단으로, 기존 최종 checkpoint
`runs/current_valid_baseline/batch400_100k5ep_20260918_bf16_final/uacp_predictor_100k_5ep_batch400_bf16.pt`
에서 이어서 1 epoch만 학습했다. 목적은 기존 training mask가 `Ng={4,8,16,32}`에만
노출되었기 때문에 `Ng=64,128` sparse reconstruction이 무너졌다는 가설을
검증하는 것이었다. 새 recovery mask는 `Ng={4,8,16,32,64,128}`을 각 `1/6`
확률로 선택하고 기존 random offset을 유지했다. 이는
`IMPLEMENTATION-ASSUMPTION: uniform Ng sampling over {4,8,16,32,64,128}`이며,
`Ng=1`은 training에 넣지 않았다.

설정은 100k samples, direct batch400, BF16 backbone/FP32 evidential path,
기존 architecture/loss/data/noise/target/layout을 유지했다. 원 checkpoint는
weights-only state-dict라 optimizer state를 복원할 수 없었고, 기존과 동일한
Adam/LR=1e-4 optimizer를 새로 생성했다(IMPLEMENTATION-ASSUMPTION). 1 epoch은
250 steps, 100k 추가 exposure, 1,323.16초(22.05분), peak allocated 17,685 MiB,
NVIDIA GB10/cuda:0였다. recovery artifact는
`runs/current_valid_baseline/ng64_128_recovery_1ep_20260918/`에 보존했다.

Quick Gate는 delay 20/80/120 ns × Ng 16/32/64/128, 조합당 50 samples로
수행했다. 모든 `nu`는 finite였지만 결과는 다음과 같다.

| 조건 | 20 ns | 80 ns | 1 epoch 판정 |
| --- | ---: | ---: | --- |
| Ng64 all-NMSE (dB) | -4.661 | -4.295 | 기존 약 -3.6 dB 대비 약 1 dB만 개선, 여전히 열화 |
| Ng128 all-NMSE (dB) | -1.415 | -1.200 | 기존 약 -1.3 dB와 실질적으로 동일 |
| Aleatoric, Ng16 | 0.03327 | 0.03350 | 미세 증가 |
| Aleatoric, Ng32 | 0.10232 | 0.10165 | 감소 |
| Aleatoric, Ng64 | 6.33905 | 6.02186 | 감소 |
| Aleatoric, Ng128 | 680.842 | 662.633 | 감소 |

따라서 Gate A(Ng64/128 reconstruction recovery)와 Gate B(20→80 ns
Aleatoric 방향성)를 모두 통과하지 못했다. Early-stop 규칙에 따라 2번째
recovery epoch, full static sweep, Fig.8/9, dynamic Fig.11은 실행하지 않았다.
결론은 **C. expanded mask exposure로도 sparse reconstruction이 충분히
회복되지 않음**이다. 이는 1 epoch/100k 추가 exposure가 부족했거나 optimizer
state 부재 등 continuation 조건의 영향을 포함하므로, mask coverage가 전혀
원인이 아니라고 단정하지 않는다. 추가 recovery/retraining은 별도 승인 후
설계해야 한다. 상세 표와 원자료는 recovery 디렉터리의
`RECOVERY_SUMMARY.md`, `quick_gate/quick_gate.csv`에 있다.

### Random-subset mask 100k×1 batch8 comparison (2026-09-18)

동일한 random-subset mask 조건에서 `100,000 unique CFR × 1 epoch / direct
batch8 / BF16 AMP` scratch training을 수행했다. 기존 periodic 100k×1 batch8
artifact는 보존하고 비교 기준으로만 사용했다. 새 모델은
`Ng={4,8,16,32,64,128}`을 각 `1/6` 확률로 선택하고, sample마다 1024개
subcarrier 중 `1024/Ng`개를 without replacement random sampling했다. 이는
`IMPLEMENTATION-ASSUMPTION`이다.

총 12,500 optimizer steps, 100,000 exposure, 학습 시간 2,252.87초(37.55분),
peak allocated 558 MiB, NVIDIA GB10/cuda:0였다. Ng draw count는
`16627/16708/16559/16867/16697/16542`로 균등했다.

| 조건 | Ng64 20 ns | Ng128 20 ns | Ng64 80 ns | Ng128 80 ns |
| --- | ---: | ---: | ---: | ---: |
| Random subset 100k×1 batch8 | -8.617 dB | -5.029 dB | -6.790 dB | -3.639 dB |
| Random subset 100k×2 batch400 | -1.977 dB | -0.847 dB | -2.020 dB | -0.897 dB |
| 기존 periodic 100k×5 batch400 | 약 -3.6 dB | 약 -1.3 dB | 약 -3.6 dB | 약 -1.3 dB |

batch8 random-subset 모델은 batch400 random-subset보다 Ng64/128
reconstruction이 크게 좋았다. Aleatoric 20→80 ns AUROC도 Ng16/32/128에서
`0.790/0.754/0.528`로 나타났고, Ng64는 `0.469`였다. 따라서 이전 batch400
random-subset 2 epoch보다 개선 신호가 있지만 모든 Ng에서 일관되지는 않는다.
`ν` nonfinite는 0이었다.

이번에는 비교 목적의 Quick Gate만 수행했으며 full Fig.8/9/11은 실행하지
않았다. 원자료는
`runs/current_valid_baseline/random_subset_mask_100k1ep_batch8_bf16_20260918/`
및 `runs/current_valid_baseline/random_subset_mask_comparison_20260918.csv`에
있다.

### Random-subset batch8 vs batch400 Fig.11-lite comparison (2026-09-18)

두 random-subset checkpoint를 추가 training 없이 동일한 Fig.11-lite 조건으로
비교했다. Sequence는 `20→80→10→40→60→120 ns`, 각 regime 40 steps,
시작 Ng=16, candidate `{4,8,16,32,64,128}`이다. 두 모델 모두 ID-only
calibration procedure를 적용했고, OOD 결과는 threshold에 사용하지 않았다.

| 항목 | 100k×1 batch8 | 100k×2 batch400 |
|---|---:|---:|
| Optimizer steps / exposure | 12,500 / 100,000 | 500 / 200,000 |
| Training time | 2,252.87 s | 2,236.18 s |
| Ng64/128 20 ns NMSE | -8.617 / -5.029 dB | -1.977 / -0.847 dB |
| Ng64/128 80 ns NMSE | -6.790 / -3.639 dB | -2.020 / -0.897 dB |
| Aleatoric AUROC Ng16/32/64/128 | .790/.754/.469/.528 | .432/.591/.426/.457 |
| Fig.11 20/80/120 mean NMSE | -17.145/-15.693/-13.683 dB | -9.035/-8.868/-8.290 dB |
| Epistemic trigger / ν nonfinite | 0 / 0 | 0 / 0 |

batch8은 sparse reconstruction, 일부 Aleatoric separation, dynamic NMSE
안정성에서 더 적합했다. 그러나 두 모델 모두 regime 변화에 대한 Ng 이동이
완전히 안정적이지 않았고, 120 ns에서 epistemic Ng=1 fallback은 발생하지
않았다. 따라서 최종 선택은 **100k×1 batch8 random-subset**이지만, 논문의
Fig.11 완전 재현으로 표현할 수는 없다. 이는 batch size만의 controlled
comparison이 아니다(optimizer steps와 exposure가 다름).

상세 보고서와 동일 축 figure는
`runs/current_valid_baseline/random_subset_batch8_vs_batch400_fig11_report_20260918/REPORT.md`
및 해당 디렉터리에 저장했다.

## Phase 1 repository cleanup (2026-09-18)

정리 목적은 명백한 Python/pytest cache와 참조되지 않는 빈 산출물 디렉터리만
제거하고, 현재 active 연구 pipeline을 문서화하는 것이었다. 삭제한 항목은
`.pyc`, `.pyo`, `__pycache__/`, `.pytest_cache/` 및 원본 문서에서 직접 참조되지
않고 파일이 0개였던 empty output directory뿐이다.

100k×1 batch8 FP32, 100k×5 batch8 FP32, 100k×5 epoch별 checkpoint, common_eval,
Fig.8/9/11 결과와 batch400/2048/2432 benchmark, covariance, observation,
diagnostics는 보존했다. `runs/` 내부 경로는 이동하지 않았다.

Active pipeline index: `docs/ACTIVE_PIPELINE.md`

Script classification: `docs/SCRIPT_CLASSIFICATION.md`

Snapshot과 삭제 전후 목록: `docs/cleanup_snapshot_20260918/`

검증 결과: `python -m pytest -q`는 51 passed, `python -m compileall -q src scripts tests`
와 `git diff --check`도 exit 0이었다. 검증 과정에서 자동 생성된 cache는 다시
제거했다.

## 100k×5 batch8 FP32 bottleneck profiling and epoch checkpoint selection (2026-09-18)

### 목적과 보존 원칙

100k×1 batch8 FP32가 약 44.3분인 반면 100k×5 batch8 FP32가 약 12.24시간인
이유를 분리 측정하고, 기존 epoch별 checkpoint 중 numerical stability와 OOD
separation을 동시에 만족하는 candidate를 찾았다. 기존 checkpoint, result, log,
dataset은 수정하거나 덮어쓰지 않았다. batch400 추가 학습과 Fig.9/Fig.11 반복
실행은 수행하지 않았다.

### Step 1: short profiling

기존 `train_100k5_convergence.py`의 model/loss/observation path를 유지한 채,
새 profiling directory에서 warmup 8 step과 FP32 representative 64 step만
실행했다. GPU는 `NVIDIA GB10 / cuda:0`였고 실제 CUDA 사용을 확인했다.

Artifact:
`runs/current_valid_baseline/profiling_100k5_batch8_20260918/`

| Stage | Mean seconds/step |
|---|---:|
| Data loading | 0.000590 |
| CPU→GPU transfer | 0.001675 |
| Observation/mask/noise preparation | 0.020825 |
| Forward | 0.280730 |
| Evidential loss | 0.004835 |
| Backward + gradient clipping | 0.402247 |
| Optimizer step | 0.012565 |
| Post-step metrics/scalar logging | 0.004855 |
| Total wall time | 0.728334 |

평균 GPU utilization은 91.9%, peak allocated VRAM은 860.5 MiB, throughput은
10.98 samples/s였다. 따라서 병목은 DataLoader나 CPU→GPU transfer가 아니라
forward/backward GPU compute이며, 특히 backward가 가장 큰 단일 stage다.
현재 `num_workers=0`, `pin_memory=False`, `persistent_workers=False`이지만
측정상 이 설정만으로 12.24시간 문제를 설명할 수 없다. LR/loss/model 의미를
바꾸는 최적화는 적용하지 않았다. DataLoader/pin-memory 변경은 후속 단일
변수 benchmark 후보로만 남겼다.

Validation은 4.42초, profiling checkpoint 저장은 0.35초였으므로 epoch 전체
시간의 주원인이 아니다. 이는 **IMPLEMENTATION-ASSUMPTION**인 stage wall-time
경계와 cuda synchronization 기반 측정이다.

### Step 2: existing epoch checkpoint evaluation

새 학습 없이 다음 기존 checkpoint를 평가했다.

`runs/current_valid_baseline/training_convergence_100k_5ep_20260916_final_controlled/uacp_predictor_100k_5ep_epoch_1.pt`
through `epoch_5.pt`.

공통 evaluation data, `Ng=16` mask, direct-CFR AWGN 15 dB, 동일 seed schedule을
사용했다. 표는 각 regime 2,000 samples의 동일 evaluator 결과이며, epoch1은
10,000 samples/regime에서도 Near `0.62112`, Far `0.99696`로 독립 교차 확인했다.
AUROC pooling은 논문 세부사항이 공개되지 않았으므로
**IMPLEMENTATION-ASSUMPTION**이다.

| Epoch | NMSE 20 ns | NMSE 80 ns | NMSE 120 ns | NMSE 1 ms | Near AUROC | Far AUROC | Pooled AUROC | Far first nonfinite |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | -18.986 | -17.933 | -15.204 | 1.454 | 0.6163 | 0.9964 | 0.8063 | none |
| 2 | -20.165 | -18.361 | -16.238 | 1.838 | 0.7821 | 0.9999 | 0.8910 | none |
| 3 | -20.930 | -18.802 | -16.896 | 1.682 | 0.9364 | 1.0000 | 0.9682 | none |
| 4 | -21.377 | -19.345 | -16.938 | 1.633 | 0.9522 | NaN | NaN | aleatoric |
| 5 | -21.802 | -19.472 | -16.554 | 1.524 | 0.9484 | NaN | NaN | aleatoric |

Artifact:
`runs/current_valid_baseline/epoch_checkpoint_evaluation_100k5_batch8_20260918/`

Epoch3는 네 regime에서 NaN/inf가 없고, 20 ns가 80 ns보다 낮은 NMSE이며,
Near/Far epistemic separation을 모두 보였다. Aleatoric median도 epoch3에서
20 ns `0.00368`, 80 ns `0.00473`으로 80 ns가 더 높았다.

### ν/inf 원인

OOD-Far에서 epoch4부터 최초 nonfinite가 `aleatoric` 단계에서 발생했다.
Raw gamma/Psi/kappa/nu head, admissibility output, `nu`, `Psi`, 그리고
`nu - (2K+1)` 자체의 nonfinite count는 0이었다. 그러나 Far-OOD에서
`kappa`가 epoch3 median `0.000747`에서 epoch4 `0.00000929`, epoch5
`0.00000135`로 감소하고, `nu - 2049` median이 epoch3 `13.154`에서
epoch4 `0.0278`, epoch5 `0`으로 수렴했다. 그 결과

`Aleatoric = Psi / (nu - 2K - 1)`

에서 0 denominator가 발생해 `inf`가 되고, 이어서 Epistemic과 Fig.8 Far
AUROC가 nonfinite가 되었다. 이는 raw head overflow보다는 admissibility
margin이 FP32에서 소실되는 late-epoch evidential degeneration으로 판단한다.

### Step 3 decision

기존 epoch3가 finite reconstruction, 정상적인 20/80/120 ns/1 ms 순서,
Near AUROC 0.9364, Far AUROC 0.9999985를 동시에 만족한다. 따라서 불필요한
재학습을 하지 않고 다음 candidate baseline으로 제안한다.

`runs/current_valid_baseline/training_convergence_100k_5ep_20260916_final_controlled/uacp_predictor_100k_5ep_epoch_3.pt`

LR `1e-4 → 5e-5` 단일 변경 재학습은 수행하지 않았다. 따라서 이번 단계의
결론은 LR 가설 검증 결과가 아니라, 기존 epoch3 checkpoint 선택 결과다.
최종 candidate 확정 후 Fig.9/Fig.11을 다시 실행한다.

검증: `python -m pytest -q`는 51 passed, `python -m compileall -q src scripts tests`
와 `git diff --check`는 exit 0이었다. 검증 과정에서 생성된 cache는 repository
active tree에서 제거했다.
## Frozen 100k×5 epoch-3 candidate validation (2026-09-19)

기존 `100k×5 / batch8 / FP32 / LR=1e-4` 학습의 epoch3 checkpoint를 새 학습 없이 대규모 정적 평가하고, canonical Fig.8 → Fig.9 → Fig.11 순서로 candidate baseline 가능성을 점검했다. 기존 checkpoint, result, log, dataset은 수정하거나 덮어쓰지 않았다. 평가는 기존 `common_eval` 10,000 samples/regime, Ng=16, direct-CFR sample-wise AWGN 15 dB, 동일 random-mask/evaluation protocol을 사용했다. GPU는 NVIDIA GB10 / `cuda:0`에서 실제 inference를 수행했다.

결과 산출물은 `runs/current_valid_baseline/epoch3_baseline_validation_20260919/`에 저장되어 있다.

### Fig.8 static uncertainty

| checkpoint | NMSE 20 ns | NMSE 80 ns | NMSE 120 ns | NMSE 1 ms | Near AUROC | Far AUROC | pooled AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| 100k×1 batch8 FP32 | -18.98 | -17.86 | -15.15 | 1.25 | 0.621 | 0.997 | 0.809 |
| 100k×5 epoch3 batch8 FP32 | -20.947 | -18.790 | -16.903 | 1.685 | 0.936263 | 0.9999997 | 0.968131 |
| 100k×5 epoch5 batch8 FP32 | -21.80 | -19.48 | -16.47 | 1.31 | 0.948 | N/A (ν/inf) | N/A |

Epoch3는 10,000 samples/regime에서 네 regime 모두 NaN/inf가 없었다. Aleatoric median은 20 ns 0.003929, 80 ns 0.005024로 80 ns가 더 높았고, epistemic median은 20 ns 0.000347, 80 ns 0.000635, 120 ns 0.000972, 1 ms 22.232였다. OOD-Far의 ν−(2K+1) median은 13.148, minimum은 0.0269로 finite이나 margin이 좁다. 100k×5 epoch4/5의 기존 failure와 같이 ν−(2K+1) → 0이면 `Psi / (nu-(2K+1))`인 aleatoric 단계에서 처음 nonfinite가 발생한다.

### Fig.9 calibration

Canonical Student-t interval과 기존 nominal grid를 사용했다. omitted-only calibration error MAE는 ID 0.001195, OOD-pooled 0.099724, OOD-Near 0.067179, OOD-Far 0.244081이었다. 따라서 ID curve는 ideal diagonal에 가깝지만, OOD 특히 Far에서 coverage가 악화되는 방향이 수치로 확인된다. 결과는 `static_fig8_fig9/fig9_calibration_curve.csv`, `fig9_calibration_error.csv`, `fig9_calibration_curve.png`에 있다.

### Fig.11 dynamic validation

기존 controller와 기존 threshold calibration/hysteresis를 변경하지 않고 20 → 80 → 10 → 40 → 60 → 120 ns 순서로 40 samples/regime을 평가했다. 20 ns에서는 Ng가 평균 31.2, 80 ns에서는 24.0으로 감소하는 방향이 관찰됐다. 그러나 10 ns에서 기존 epistemic-threshold controller가 한 차례 full-feedback fallback을 발생시켜 Ng=1로 고정했고, 이후 40/60/120 ns는 omitted set이 없어 기존 metric이 NaN이 되었다. inference/controller runtime은 predictor forward 평균 199.8 ms, uncertainty aggregation 6.81 ms, controller decision 0.051 ms, end-to-end 평균 204.8 ms였다. EVM/BER는 현재 repository에 PHY/precoding/postcoding 경로가 없어 계산하지 않았다.

Threshold와 hysteresis는 논문에 공개되지 않은 현재 구현 설정이며 `IMPLEMENTATION-ASSUMPTION`으로 기록했다. 기존 delay-sweep samples를 순서형 sequence로 사용하는 것도 구현 가정이다.

#### Updated Fig.11 trace visualization (2026-09-23)

본문에는 false-trigger 주변을 확대한 그림을 우선 사용한다. 저장된 `dynamic_trace.csv`에서 실제 false-trigger는 **step 90 (10 ns)**이며, step 91부터 `Ng=1` full-feedback fallback이 시작되어 omitted NMSE는 정의되지 않는다. 확대 그림의 x축은 `0–95`로 제한했고, 전체 그림에는 `0–239`의 모든 channel-regime 배경 구간을 표시했다. 기존 trace와 기존 PNG는 보존하고 새 출력만 생성했다.

![Epoch3 Fig.11 false-trigger zoom](runs/current_valid_baseline/epoch3_baseline_validation_20260919/fig11/regime_annotated_20260923_v2/epoch3_dynamic_trace_false_trigger_zoom.png)

전체 sequence 참고 그림: [epoch3_dynamic_trace_full.png](runs/current_valid_baseline/epoch3_baseline_validation_20260919/fig11/regime_annotated_20260923_v2/epoch3_dynamic_trace_full.png)

### Baseline decision and next step

Epoch3는 정적 reconstruction/uncertainty/Fig.8/Fig.9 candidate baseline으로 유지한다. 다만 이번 Fig.11 결과는 쉬운 10 ns에서 fallback 후 Ng=1 고정이 발생해, 전체 Fig.8→Fig.9→Fig.11 기준의 최종 baseline으로는 아직 확정하지 않는다. 새 training, LR 변경, clipping, ν clamp, batch400 실험은 수행하지 않았다. 다음 추천 작업은 모델을 재학습하지 않고 controller의 공개되지 않은 threshold provenance와 Ng=1 fallback semantics를 별도 diagnostic으로 검토한 뒤, 필요한 경우에만 명시적인 implementation assumption으로 controller ablation을 수행하는 것이다.
## Epoch3 Fig.11 false OOD fallback diagnosis (2026-09-19)

epoch3 predictor는 변경하지 않고, 기존 Fig.11 controller의 10 ns ID false fallback 원인만 조사했다. 기존 global calibration은 `scripts/fig11_dynamic_runtime_validation.py`의 동작을 그대로 재현했다: ID 20/80 ns, candidate Ng, batch-mean Eq.(13) score, ID-only 99th percentile. 결과 threshold는 `52.45695`로 기존 epoch3 Fig.11 결과와 일치했다.

직접 원인은 threshold가 조금 낮은 것이 아니라 `Ng=64`에서 발생한 단일 거대 epistemic spike였다. 기존 canonical sequence의 정확한 trigger는 time step 90, 10 ns, current/previous Ng=64, omitted subcarrier 1008, epistemic `29,672.99`, threshold `52.45695`, ratio `565.7x`, NMSE `-7.88 dB`였다. 10 ns standalone에서도 Ng=64에서 trigger가 재현되었으므로 80→10 transition만의 state 문제로 볼 수 없다. 이후 Ng=1은 full-feedback/empty omitted set이며 uncertainty metric은 `N/A (full-feedback / empty omitted set)`로 구분한다.

10/20/40/60/80/100 ns ID score를 Ng별로 조사한 결과 Ng dependence가 강했다. 10–100 ns pooled q99는 Ng4/8/16/32/64/128에서 각각 `0.0010/0.0015/0.0028/0.0214/1559.64/50.91`이었다. Ng64의 global threshold 초과율은 `4.75%`였고, Ng4–32는 0%, Ng128은 1.0%였다. 따라서 하나의 global threshold가 모든 mask density를 대표하지 못한다는 가설은 지지된다.

### Ng-conditioned threshold and persistence tests

ID 10/20/40/60/80/100 ns만 사용해 `tau_Ng = per-Ng ID q99`를 계산했다. OOD 120 ns/1 ms는 calibration에서 제외했다. 이 q99 규칙은 논문에 공개된 설정이 아니라 **IMPLEMENTATION-ASSUMPTION**이다. Ng-conditioned threshold만 적용하면 10 ns step90 spike는 `29,672.99 / 1559.64 = 19.0x`로 여전히 trigger되어 false fallback이 남았다.

그 후에만 최소 persistence `N=2`를 별도 시험했다. 이는 역시 **IMPLEMENTATION-ASSUMPTION**이다. N=2는 10 ns false fallback을 제거했지만, 120 ns에서 관찰된 threshold 초과가 단발성 5회로 분산되어 OOD fallback도 발생하지 않았다. 따라서 `Ng-conditioned q99 + N=2`는 10 ns 문제만 숨기고 120 ns detection을 잃으므로 최종 controller 후보로 채택하지 않았다. 추가적인 threshold 인하, 10 ns hard-code 예외, predictor 재학습, ν clamp는 수행하지 않았다.

진단 산출물: `runs/current_valid_baseline/epoch3_threshold_diagnosis_20260919/`.

- `step1_step3_final/step1_canonical_trace.csv`: canonical trigger trace
- `step1_step3_final/step1_standalone_transition_trace.csv`: standalone/transition 비교
- `step1_step3_final/step3_id_ng_distribution.csv` 및 `step3_id_ng_scores.csv`: Ng별 ID distribution
- `step4_ng_conditioned/`: Ng-conditioned threshold 전후 결과
- `step5_persistence_N2/`: N=2 persistence 결과

판정: 문제의 직접 원인은 Ng=64 mask-conditioned epistemic tail/outlier이며 predictor numerical failure가 아니다. 현재 검증한 최소 controller 변경 조합은 10 ns false trigger 제거와 120 ns OOD fallback 유지를 동시에 만족하지 못했다. 다음 실험은 threshold를 임의로 조정하기 전에 Ng64 tail의 sample-level source와 temporal continuity를 진단하는 것이다.
## Epoch3 step90 Ng32 versus Ng64 root-cause audit (2026-09-19)

10 ns channel 자체와 sparse observation 효과를 분리하기 위해 epoch3 checkpoint를 재학습하지 않고 canonical Fig.11 step90의 동일 CFR를 고정해 Ng만 비교했다. 기존 canonical sequence는 20 ns 40 samples, 80 ns 40 samples, 10 ns sample index 10 순서이므로 step90은 `test_delay_10_ns.npz[10]`이며 canonical seed는 `20262000 + 90 = 20262090`이다. 동일 seed로 direct-CFR 15 dB AWGN의 동일한 standard-normal base noise tensor를 재사용했다. pipeline은 mask별 observed signal power로 noise amplitude를 다시 계산하므로 최종 noise power는 Ng32/Ng64에서 약간 다르다.

### Actual epoch3 training Ng range

`runs/current_valid_baseline/training_convergence_100k_5ep_20260916_final_controlled/config_used.json`과 `scripts/train_predictor.py`를 확인한 결과, training은 `random_grouping_mask`에서 `{4, 8, 16, 32}` 중 하나를 sample별로 선택하고 random offset을 사용했다. 따라서 Ng64/Ng128은 epoch3 training 범위 밖의 mask extrapolation이다. Evaluation metadata의 fixed Ng16은 training mask range 안에 있다.

### Matched canonical step90

| Metric | Ng32 | Ng64 | 변화 |
|---|---:|---:|---:|
| Observed subcarriers | 32 | 16 | -50% |
| Omitted subcarriers | 992 | 1008 | +16 |
| NMSE | -18.351 dB | -7.880 dB | +10.471 dB |
| Aleatoric Eq.(13) | 0.02796 | 2.07747 | 74.3× |
| Epistemic Eq.(13) | 0.04056 | 29,672.99 | 731,700× |
| kappa mean | 0.7068 | 7.14e-5 | 약 9,900배 감소 |
| nu-(2K+1) mean | 20.936 | 0.574 | 36.5배 감소 |
| Psi mean | 0.2927 | 0.5914 | 2.02× |
| Raw evidential finite | yes | yes | unchanged |
| Final NaN/inf | no | no | unchanged |

분해하면 `Aleatoric = Psi / (nu-(2K+1))`이므로 nu margin 감소와 Psi 증가가 Aleatoric을 약 74배 키웠다. 이후 `Epistemic = Aleatoric / kappa`에서 kappa가 약 0.707에서 `7.1e-5`로 붕괴하여 추가로 약 9,900배 증폭되었다. 따라서 직접적인 지배 원인은 Ng64 sparse observation에서의 kappa collapse이며, nu margin 감소가 2차적으로 Aleatoric을 키웠다. raw head와 admissibility transform 이후 값은 모두 finite이므로 epoch4/5의 ν/inf numerical failure와는 다른 현상이다.

### Matched 10 ns sample set

canonical 10 ns block에 대응하는 40개 matched CFR를 Ng32/Ng64로 각각 평가했다. 동일 sample별 seed와 동일 standard-normal base noise를 사용했다.

| Ng | Epistemic median | q95 | q99 | max | >52.45695 | >1000 |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.00184 | 0.00407 | 0.02658 | 0.04056 | 0% | 0% |
| 64 | 0.07296 | 78.278 | 18,132.47 | 29,672.99 | 7.5% | 2.5% |

Ng64에서만 extreme Epistemic이 관찰되며, step90은 이 40개 중 최대값이다. 따라서 10 ns 전체가 아니라 `Ng64 + 특정 CFR/sample interaction`이 재현되는 원인이다. 동시에 Ng64의 tail이 반복적으로 커져 epoch3 모델의 학습 범위 밖 operating condition이라는 해석도 지지된다.

새 결과는 `runs/current_valid_baseline/epoch3_step90_ng64_audit_20260919/`에 저장했다. 이번 단계에서는 threshold, persistence, controller, model, loss, training을 수정하지 않았다. 다음 단계는 model을 즉시 변경하기보다, 먼저 Ng64를 실제 training/evaluation operating range에 포함할지와 kappa head의 sparse-mask extrapolation을 별도 설계 가설로 검토하는 것이다.
## Epoch3 Fig.11 restricted Ng candidate ablation (2026-09-19)

단일 가설은 epoch3가 학습하지 않은 Ng64/128을 Fig.11 controller가 사용하기 때문에 10 ns ID에서 Epistemic 폭발과 false fallback이 발생하는지 확인하는 것이었다. predictor checkpoint, threshold `52.45695`, hysteresis target/delta, initial state, seed `20262000`, 40 samples/regime, sequence `20 → 80 → 10 → 40 → 60 → 120 ns`, observation/noise protocol은 기존 canonical Fig.11과 유지했다. 유일한 실험 조건은 adaptive sparse candidate를 `{128,64,32,16,8,4}`에서 `{32,16,8,4}`로 제한한 것이다. Ng=1 full-feedback fallback은 유지했다.

기존 initial Ng=128은 첫 관측 상태로 그대로 유지했다. 128은 restricted candidate에 속하지 않으므로 첫 non-fallback adaptive action에서 허용된 최대 candidate Ng=32로 projection했다. 이는 candidate-set restriction을 정의하기 위한 경계 semantics이며, threshold/persistence/hysteresis 변경은 아니다.

### Existing versus restricted controller

| Regime | Existing mean Ng | Restricted mean Ng | Existing mean NMSE | Restricted mean NMSE | Existing trigger | Restricted trigger |
|---|---:|---:|---:|---:|---:|---:|
| 20 ns | 31.2 | 29.6 | -18.350 | -18.760 | 0 | 0 |
| 80 ns | 24.0 | 24.0 | -17.458 | -17.458 | 0 | 0 |
| 10 ns | 8.725 | 27.6 | N/A after fallback | -19.095 | 1 false | 0 |
| 40 ns | 1.0 | 24.8 | N/A after fallback | -18.974 | 0 | 0 |
| 60 ns | 1.0 | 23.6 | N/A after fallback | -18.254 | 0 | 0 |
| 120 ns | 1.0 | 23.6 | N/A after fallback | -15.610 | 0 | 0 |

Restricted candidate 결과는 모든 regime에서 false OOD trigger가 0회였고, 기존 canonical step90의 10 ns Ng64 fallback은 사라졌다. restricted trajectory는 Ng1로 내려가지 않아 모든 regime의 omitted-set NMSE/uncertainty가 finite하게 유지됐다. 그러나 120 ns OOD에서도 threshold 초과가 발생하지 않아 Ng1 fallback은 0회였다. 따라서 `Ng64/128이 false fallback을 유발한다`는 원인은 지지되지만, candidate restriction만으로 120 ns OOD detection을 유지한다는 성공 기준은 실패했다.

결과는 `runs/current_valid_baseline/epoch3_restricted_ng_fig11_20260919/`의 `dynamic_trace.csv`, `regime_summary.csv`, `results.json`, `fig11_restricted_candidates.png`에 저장했다. 이번 실험에서는 새 training, model/loss 변경, threshold 변경, persistence, hysteresis 변경, observation/noise/mask protocol 변경을 수행하지 않았다. Ng=1 이후의 기존 baseline 값은 `N/A (full-feedback / empty omitted set)`으로 해석했다.

판정: **PARTIAL SUPPORT**. Ng64/128은 epoch3 training range 밖이며 false trigger의 핵심 조건으로 보이지만, restricted candidate controller는 OOD fallback sensitivity도 잃는다. 다음 단계는 model 재학습을 즉시 수행하기보다 restricted operating range에서 120 ns detection을 어떻게 정의할지와 Ng32 이하의 uncertainty separation을 별도 검토하는 것이다.
## Epoch3 restricted-Ng ID-only threshold recalibration (2026-09-19)

목적은 restricted controller `{4,8,16,32}`에서 기존 global threshold를 같은 operating range의 ID 분포에 맞춘 Ng-conditioned threshold로 바꾸면 10 ns false trigger 없이 120 ns OOD fallback을 복구할 수 있는지 검증하는 것이었다. checkpoint, model, loss, seed, hysteresis, observation/noise, sequence, initial state, Ng=1 fallback은 기존 restricted Fig.11과 동일하게 유지했다. OOD 120 ns/1 ms 자료는 threshold calibration에 사용하지 않았다.

Calibration은 기존 ID score artifact `step3_id_ng_scores.csv`를 재사용해 10/20/40/60/80/100 ns × Ng별 200 samples를 앞 100개 calibration과 뒤 100개 holdout으로 나눴다. 주 threshold는 논문에 공개되지 않은 **IMPLEMENTATION-ASSUMPTION**인 `tau_Ng = pooled ID q99` 하나만 사용했다.

### Ng-conditioned thresholds

| Ng | calibration count | median | q95 | q99 (`tau_Ng`) | max | 기존 global |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 600 | 0.000339 | 0.000748 | 0.000988 | 0.001165 | 52.45695 |
| 8 | 600 | 0.000560 | 0.001125 | 0.001394 | 0.001739 | 52.45695 |
| 16 | 600 | 0.001004 | 0.002031 | 0.002866 | 0.003695 | 52.45695 |
| 32 | 600 | 0.003097 | 0.010485 | 0.021396 | 0.041176 | 52.45695 |

Holdout ID 초과율은 10/20/40 ns에서 0%, 60 ns에서 0.5%, 80 ns에서 0%, 100 ns에서 6.25%였다. 따라서 q99가 calibration subset에는 맞지만 ID 전 범위의 finite-sample tail을 완전히 대표하지 못했다.

### Fig.11 result

Ng-conditioned threshold 적용 후 canonical restricted sequence에서 step90, 10 ns, current Ng32, epistemic `0.04056`, `tau_Ng32=0.021396`으로 false trigger가 다시 발생했다. 이후 Ng1 full-feedback으로 전환되어 40/60/120 ns omitted metrics는 `N/A (full-feedback / empty omitted set)`가 되었다. 120 ns OOD trigger는 관찰되지 않았다.

| Condition | 10 ns false trigger | ID false trigger | 120 ns OOD trigger | 120 ns fallback |
|---|---:|---:|---:|---:|
| A. Global + `{4,8,16,32,64,128}` | 1 | 1 | 0 | 1 earlier at 10 ns |
| B. Global + `{4,8,16,32}` | 0 | 0 | 0 | 0 |
| C. Ng-conditioned q99 + `{4,8,16,32}` | 1 | 1 | 0 | 1 at step90 |

| Regime | A mean Ng | B mean Ng | C mean Ng | C mean NMSE |
|---|---:|---:|---:|---:|
| 20 ns | 31.2 | 29.6 | 29.6 | -18.760 dB |
| 80 ns | 24.0 | 24.0 | 24.0 | -17.458 dB |
| 10 ns | 8.725 | 27.6 | 7.925 | -19.406 dB |
| 40 ns | 1.0 | 24.8 | 1.0 | N/A full-feedback |
| 60 ns | 1.0 | 23.6 | 1.0 | N/A full-feedback |
| 120 ns | 1.0 | 23.6 | 1.0 | N/A full-feedback |

판정: **실패**. Ng-conditioned q99는 기존 global threshold보다 operating-range에 맞지만, 10 ns false trigger가 재발했고 120 ns OOD detection도 복구하지 못했다. threshold를 사후 조정하거나 persistence/예외 처리를 추가하지 않았다. 결과는 `runs/current_valid_baseline/epoch3_ng_conditioned_threshold_fig11_20260919_final/`의 `id_calibration_thresholds.csv`, `id_holdout_false_trigger.csv`, `dynamic_trace.csv`, `regime_summary.csv`, `controller_comparison.csv`, `results.json`에 저장했다. 다음 원인 분석은 threshold calibration이 아니라 Ng32 이하에서 ID/OOD epistemic score가 충분히 분리되지 않는지 확인하는 방향이 적절하다.
## Epoch3 ID versus 120 ns Epistemic distribution analysis (2026-09-19)

이번 단계는 controller/threshold를 변경하지 않고, epoch3 checkpoint가 restricted operating range `{4,8,16,32}`에서 ID와 OOD-Near 120 ns를 실제로 분리하는지 분포만 분석했다. ID는 10/20/40/60/80/100 ns 각 200 samples를 pooled했고, 120 ns는 200 samples를 사용했다. 기존 direct-CFR AWGN 15 dB, periodic mask, Eq.(12)/(13) omitted aggregation protocol을 유지했다. GPU는 NVIDIA GB10 / `cuda:0`에서 사용했다.

### Ng별 ID/OOD 분포

| Ng | ID median / q99 | 120 ns median / q99 | AUROC | log10 overlap | ID FPR @ ID q99 | 120 ns TPR @ ID q99 |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.000336 / 0.001030 | 0.000888 / 0.002398 | 0.9580 | 0.185 | 1.0% | 31.0% |
| 8 | 0.000552 / 0.001466 | 0.001209 / 0.002933 | 0.9461 | 0.197 | 1.0% | 24.0% |
| 16 | 0.000985 / 0.002761 | 0.001962 / 0.005054 | 0.9210 | 0.253 | 1.0% | 12.0% |
| 32 | 0.003061 / 0.023434 | 0.009663 / 0.057063 | 0.9100 | 0.291 | 1.0% | 9.0% |

모든 Ng에서 OOD median/mean은 ID보다 높았지만, Ng가 증가할수록 distribution overlap이 커지고 ID q99 threshold 기준 true-positive rate가 감소했다. 가장 좋은 separation은 Ng4, 가장 문제가 큰 operating point는 Ng32였다. 따라서 ranking 자체는 존재하지만 absolute score threshold로 사용할 때 120 ns tail의 상당 부분을 놓친다.

### Parameter direction

Ng16 기준 ID→120 ns median은 Aleatoric `0.00463→0.00555`, kappa `9.40→5.69`, ν margin `35.63→30.10`, Ψ `0.1629→0.1670`이었다. Ng32에서는 Aleatoric `0.00876→0.01084`, kappa `5.71→2.25`, ν margin `27.26→22.32`, Ψ `0.2374→0.2418`이었다. 즉 OOD에서 Aleatoric과 Ψ는 완만하게 증가하고, kappa와 ν margin은 감소한다. 특히 Ng32에서는 kappa/ν margin 감소와 큰 ID tail이 겹쳐 Epistemic separation 품질이 악화된다.

### Fig.11 connection and judgment

이번 결과는 **B와 D가 함께 나타나는 복합 양상**이다. Ng별 AUROC는 모두 0.91 이상으로 ranking separation은 있지만, ID q99 기준 TPR은 9–31%로 낮아 threshold 기반 Fig.11 fallback을 안정적으로 만들 만큼 absolute separation이 충분하지 않다. Ng16/32는 각각 TPR 12%/9%로 특히 약해 restricted controller에서 120 ns OOD fallback이 발생하지 않은 현상과 연결된다. 이는 threshold만의 문제라기보다, Ng 증가에 따라 predictor Epistemic scale과 tail이 변하고 ID/OOD overlap이 커지는 operating-range/uncertainty quality 문제다.

새 결과는 `runs/current_valid_baseline/epoch3_id_vs_120_epistemic_20260919/`의 `ng_distribution_summary.csv`, `per_sample.csv`, `ng_epistemic_id_vs_120ns.png`, `results.json`에 저장했다. 이번 단계에서는 model, loss, threshold, controller, hysteresis, persistence, training을 변경하지 않았다. 다음 추천은 추가 threshold 조정보다 Ng16/32에서 OOD ranking은 유지되지만 absolute calibration이 약한 원인을 predictor mask-coverage/evidential head 관점에서 검토하는 것이다.

## Epoch3 Ng16/32 training coverage and evidential decomposition audit (2026-09-19)

이번 단계는 “Ng16/32 ID mask coverage가 부족/편향된 것인지”와 “coverage와 무관하게 evidential head가 120 ns에서 evidence를 충분히 낮추지 못하는지”를 분리 진단하기 위한 것이었다. 새 학습이나 model/loss/controller/threshold 변경은 수행하지 않았다.

### Training mask coverage

Epoch3 checkpoint의 실제 training 설정과 mask 생성 코드는 `Ng ∈ {4,8,16,32}` 및 sample별 `torch.randint` factor/offset 추출을 확인했다. 그러나 기존 `batch_progress.jsonl`, `epoch_progress.jsonl`, `training_results.json`에는 sample별 Ng draw 또는 random offset이 기록되어 있지 않다. 따라서 첫 3 epoch의 총 sample draw 300,000은 확인할 수 있지만, Ng별 실제 count/ratio와 offset coverage는 정확한 RNG replay 없이 복원할 수 없어 **확인 불가**로 기록했다. 25%씩이라는 값은 코드의 이론적 기대값일 뿐 실제 coverage audit 결과가 아니다.

상세 기록은 `runs/current_valid_baseline/epoch3_ng16_ng32_decomposition_20260919/training_mask_coverage_audit.json`에 저장했다. 기존 checkpoint나 run artifact는 변경하지 않았다.

### Ng16/Ng32 parameter decomposition

기존 GPU 생성 artifact `epoch3_id_vs_120_epistemic_20260919/per_sample.csv`를 동일 protocol로 재사용했다. 각 조건은 200 samples이며, 아래는 median과 80→120 ns median ratio다.

| Metric | Ng16 ID20 | Ng16 ID80 | Ng16 120 | Ng32 ID20 | Ng32 ID80 | Ng32 120 |
|---|---:|---:|---:|---:|---:|---:|
| NMSE (dB) | -20.932 | -18.744 | -16.939 | -18.145 | -15.961 | -14.217 |
| Ψ | 0.1558 | 0.1657 | 0.1670 | 0.2304 | 0.2401 | 0.2418 |
| κ | 11.542 | 7.934 | 5.687 | 8.021 | 4.230 | 2.250 |
| ν−(2K+1) | 39.847 | 33.066 | 30.097 | 31.221 | 24.961 | 22.319 |
| Aleatoric | 0.003913 | 0.005009 | 0.005553 | 0.007402 | 0.009661 | 0.010841 |
| Epistemic | 0.000676 | 0.001262 | 0.001962 | 0.001836 | 0.004596 | 0.009663 |

80→120 ns median ratio는 Ng16에서 Ψ 1.008, κ 0.717, ν margin 0.910, Aleatoric 1.109, Epistemic 1.555였고, Ng32에서 Ψ 1.007, κ 0.532, ν margin 0.894, Aleatoric 1.122, Epistemic 2.102였다. 즉 120 ns에서 Ψ 증가는 거의 없고, κ 감소와 ν margin 감소가 주된 변화다. 다만 Ng32는 ID Epistemic tail 자체가 크며 ID q99가 0.02343으로 Ng16의 0.00276보다 훨씬 높아, OOD score 상승과 ID tail inflation이 함께 overlap을 키운다.

### Diagnosis

실제 training coverage imbalance는 로그 부재로 판정할 수 없으므로 A를 확인하거나 배제하지 않았다. 관측된 parameter 결과만으로는 Ng16/32에서 κ가 ID보다 감소하더라도 ID tail과 겹치며 absolute threshold separation이 약해지는 **B/C 복합 양상**이 가장 유력하다. 특히 Ng32의 낮은 OOD TPR은 Ψ 변화 부족, κ/ν 변화, 그리고 큰 ID tail이 함께 설명한다. 이는 threshold를 다시 조정하면 해결된다고 단정할 수 있는 결과가 아니다.

분석 산출물은 `runs/current_valid_baseline/epoch3_ng16_ng32_decomposition_20260919/`의 `parameter_summary.csv`, `parameter_summary.json`, `comparison_80_to_120.csv`, `training_mask_coverage_audit.json`, `analysis_manifest.json`이다. 다음 단일 변수 실험은 먼저 sample별 Ng/offset을 명시적으로 기록하는 균형 mask-exposure 재학습으로 A를 검증하는 것이 적절하다. 그 실험은 이번 단계에서는 수행하지 않았다.

## Balanced Ng/offset exposure versus baseline epoch3 (2026-09-20)

기존 epoch3의 실제 mask exposure가 로그에 없었던 문제를 검증하기 위해, model/loss/optimizer/LR/λreg/observation/evaluation protocol은 유지하고 Ng/offset mask schedule만 균등화한 새 100k×3 학습을 평가했다. balanced 학습은 각 epoch에서 Ng4/8/16/32를 각각 25,000 samples로 배정했고, offset count 차이는 각 Ng 내부에서 최대 1이었다. 각 batch에서는 canonical `random_grouping_mask()`의 CUDA RNG draw를 먼저 소비하고 버린 뒤, 별도 schedule seed로 만든 balanced mask를 사용했다. 이 방식은 downstream noise/dropout RNG를 최대한 보존하지만 전체 stochastic path의 완전한 동일성을 보장하지 않는 **IMPLEMENTATION-ASSUMPTION**이다.

### Ng16/Ng32 comparison

동일한 200 samples/regime, direct-CFR AWGN 15 dB, periodic mask, omitted aggregation evaluator를 사용했다. ID pooled는 10/20/40/60/80/100 ns, Near는 120 ns, Far는 1 ms이다. pooled OOD AUROC는 120 ns와 1 ms를 합친 값이다.

| Metric | Baseline Ng16 | Balanced Ng16 | Baseline Ng32 | Balanced Ng32 |
|---|---:|---:|---:|---:|
| ID q99 | 0.002761 | 0.002462 | 0.023434 | 0.013659 |
| ID–120 overlap | 0.2533 | 0.2558 | 0.2908 | 0.2975 |
| Near AUROC | 0.9210 | 0.9117 | 0.9100 | 0.8953 |
| Near TPR @ ID q99 | 12.0% | 8.0% | 9.0% | 4.5% |
| Far AUROC | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| pooled OOD AUROC | 0.9605 | 0.9558 | 0.9550 | 0.9476 |

Balanced exposure는 ID q99를 낮췄지만 120 ns OOD score도 함께 낮아졌다. 따라서 ID tail 감소가 absolute OOD separation 개선으로 이어지지 않았고, overlap은 Ng16에서 거의 동일하며 Ng32에서는 증가했다. ID q99 기준 120 ns TPR과 Near AUROC도 두 Ng 모두 악화됐다.

### Static baseline behavior

아래는 median 값이다. 모든 200 samples가 finite였고 NaN/inf는 없었다.

| Model/Ng | NMSE 20 | NMSE 80 | NMSE 120 | NMSE 1 ms | Aleatoric 20 | Aleatoric 80 | Near AUROC | Far AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline/16 | -20.932 | -18.744 | -16.939 | 1.668 | 0.003913 | 0.005009 | 0.9210 | 1.0000 |
| Balanced/16 | -20.540 | -18.509 | -16.838 | 1.637 | 0.003823 | 0.004998 | 0.9117 | 1.0000 |
| Baseline/32 | -18.145 | -15.961 | -14.217 | 1.824 | 0.007402 | 0.009661 | 0.9100 | 1.0000 |
| Balanced/32 | -17.850 | -15.824 | -14.087 | 1.746 | 0.007230 | 0.009157 | 0.8953 | 1.0000 |

20 ns reconstruction은 80 ns보다 좋았고 Aleatoric 80 ns는 20 ns보다 높았다. Balanced 1 ms Epistemic median은 Ng16 `2.296`, Ng32 `11.123`으로 finite였지만 baseline의 `41.224`, `704.187`보다 낮아졌다. Far AUROC는 유지됐으나 이는 ranking 기준이며, Near absolute separation 개선을 의미하지 않는다.

### Training efficiency and hypothesis judgment

Balanced run은 총 16,198.3 s, epoch당 5,374–5,440 s, 37,500 optimizer steps, peak VRAM 약 861 MiB, NVIDIA GB10/`cuda:0`, NaN/inf 0이었다. 기존 run의 epoch1–3 기록 합계는 약 26,600.3 s이나 기존 script에는 epoch별 regime probe 평가가 포함되어 있어 balanced training time과 완전히 동일한 범주의 runtime 비교는 아니다.

판정: **mask coverage 가설은 지지되지 않았다.** 실제 balanced exposure는 확인되었지만 Ng16/32에서 ID tail 감소가 120 ns absolute separation 개선으로 연결되지 않았다. ID q99 기준 TPR, Near AUROC, overlap이 개선되지 않았으므로 추가 mask schedule tuning이나 재학습은 중단하고, 다음 원인은 evidential head의 Ng-dependent calibration/separation으로 조사한다.

새 평가 산출물은 `runs/current_valid_baseline/epoch3_balanced_mask_exposure_comparison_20260920/`의 `per_sample.csv`, `summary.csv`, `results.json`이며 기존 checkpoint/result/log/dataset은 변경하지 않았다.

## Balanced-vs-baseline evidential parameter causality audit (2026-09-20)

이번 분석은 balanced mask exposure에서 ID Epistemic tail과 120 ns Epistemic이 함께 감소한 이유를 parameter level에서 분해했다. 기존 matched evaluation artifact를 재사용했으며, 두 checkpoint에 동일 CFR sample, Ng, periodic mask, batch seed-derived AWGN protocol을 적용한 결과다. 새 training, inference protocol, threshold, controller, 후처리는 수행하지 않았다.

### 80→120 ns parameter ratios

| Model | Ng | Ψ ratio | κ ratio | ν margin ratio | Aleatoric ratio | Epistemic ratio |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 16 | 1.008 | 0.717 | 0.910 | 1.109 | 1.555 |
| Balanced | 16 | 1.009 | 0.765 | 0.918 | 1.099 | 1.441 |
| Baseline | 32 | 1.007 | 0.532 | 0.894 | 1.122 | 2.102 |
| Balanced | 32 | 1.003 | 0.688 | 0.908 | 1.109 | 1.587 |

Baseline에서는 Ng16/32 모두 80→120 ns에서 Ψ 변화가 거의 없고, κ 감소와 ν margin 감소가 OOD Epistemic 상승을 만든다. Balanced 모델에서는 특히 κ의 80→120 감소폭이 작아졌다. 따라서 OOD에서 evidence가 충분히 낮아지지 않아 Epistemic 증가가 억제되었다.

### Baseline→Balanced matched parameter shift

| Condition | Δ Aleatoric median | Δ κ median | Δ ν margin median | Δ Ψ median | Δ Epistemic median |
|---|---:|---:|---:|---:|---:|
| Ng16 / 20 ns | -0.000085 | +0.125 | +0.261 | -0.00303 | -0.000021 |
| Ng16 / 80 ns | +0.000005 | +0.403 | +0.028 | +0.00020 | -0.000063 |
| Ng16 / 120 ns | -0.000055 | +0.714 | +0.321 | +0.000004 | -0.000228 |
| Ng32 / 20 ns | -0.000185 | +0.104 | +0.334 | -0.00384 | -0.000076 |
| Ng32 / 80 ns | -0.000439 | +0.959 | +0.836 | -0.00362 | -0.001070 |
| Ng32 / 120 ns | -0.000622 | +1.168 | +0.932 | -0.00430 | -0.003876 |

Balanced에서는 모든 주요 조건에서 κ가 증가했으며, 120 ns에서 증가폭이 특히 컸다. `Epistemic = Aleatoric / κ` 관점에서 κ 증가가 Aleatoric 감소보다 지배적이므로 ID tail과 OOD score가 동시에 내려갔다. Ng32에서는 Ψ와 Aleatoric도 함께 감소해 score 억제가 더 강해졌다.

### Matched sample groups

sample-level 그룹은 controller threshold가 아닌 설명용 고정 규칙으로 정의했다. A는 Baseline ID(20/80 ns) Epistemic q95 이상, B는 Baseline 120 ns score가 Baseline ID q99 미만, C는 Baseline 120 ns upper-quartile 중 Balanced−Baseline Epistemic 감소폭이 가장 큰 10개다.

| Group | Ng | N | Median Δκ | Median Δν margin | Median ΔΨ | Median ΔAleatoric | Median ΔEpistemic |
|---|---:|---:|---:|---:|---:|---:|---:|
| A ID upper-tail | 16 | 20 | +0.497 | +0.305 | +0.00001 | -0.000045 | -0.000188 |
| A ID upper-tail | 32 | 20 | +0.989 | +1.216 | -0.00510 | -0.000961 | -0.004230 |
| B OOD below ID q99 | 16 | 136 | +0.661 | +0.260 | +0.00020 | -0.000027 | -0.000186 |
| B OOD below ID q99 | 32 | 144 | +1.172 | +0.855 | -0.00394 | -0.000560 | -0.002535 |
| C largest OOD decrease | 16 | 10 | +0.944 | +0.722 | -0.00159 | -0.000207 | -0.000875 |
| C largest OOD decrease | 32 | 10 | +1.093 | +1.391 | -0.00422 | -0.001052 | -0.031185 |

### Direct-cause judgment

Ng32의 ID tail이 큰 직접 원인은 sparse observation에서 Baseline κ가 이미 낮고 분산이 넓어진 **Ng-dependent κ calibration**이다. 120 ns separation을 제한하는 가장 직접적인 parameter도 κ다. Baseline의 80→120 κ ratio는 Ng16 `0.717`, Ng32 `0.532`로 OOD에서 κ가 낮아지지만, ID 쪽 κ tail도 낮기 때문에 overlap이 남는다. ν margin은 보조적인 변화이고, Ψ는 120 ns separation의 주된 원인이 아니다.

Balanced exposure는 ID κ를 높여 ID tail을 줄였지만 120 ns κ도 더 크게 높여 OOD Epistemic을 낮췄다. 따라서 balanced mask가 evidential head의 Ng-dependent global calibration을 바꾸었을 뿐, ID/OOD contrast를 개선하지 못했다. 결론적으로 **evidential head/calibration 문제 가설을 지지**하며, 다음 단일 실험은 mask가 아닌 κ/evidence calibration만 검증해야 한다.

상세 산출물은 `runs/current_valid_baseline/epoch3_balanced_mask_parameter_causality_20260920/`의 `distribution_summary.csv`, `ratio_120_over_80.csv`, `matched_sample_comparison.csv`, `matched_groups.csv`, `matched_delta_summary.csv`, `group_definitions.json`이다.

## Epoch3 κ loss-gradient diagnosis (2026-09-20)

현재 실제 training loss가 Near-OOD κ separation을 어떤 방향으로 학습시키는지 진단했다. 대상은 기존 epoch3 checkpoint, Ng16/32, 80 ns ID-Hard와 120 ns OOD-Near 각 200 samples다. 기존 observation, direct-CFR 15 dB AWGN, periodic mask, CFR representation을 유지했고 optimizer step은 수행하지 않았다.

### Actual loss implementation

현재 config는 `nll_mode=diagonal_multivariate`, `reg_mode=pair`, `lambda_reg=1e-3`이다. 실제 `src/models/evidential.py`의 다음 구현을 그대로 사용했다.

```text
L_total = L_NLL + 0.001 * L_reg
L_NLL = diagonal_multivariate_student_t_nll
L_reg = pair_level_evidence_regularizer
```

κ gradient는 transformed κ와 softplus 이전 raw κ head 모두 계산했다. gradient descent 해석은 `gradient > 0 → κ 감소`, `gradient < 0 → κ 증가`이다. 각 sample을 별도 loss로 계산했으므로 training batch 평균과 비교할 때 양의 batch scaling만 차이 나며 sign은 유지된다.

### Per-sample transformed-κ gradient summary

아래 gradient는 sample별 4개 κ pair gradient의 평균 median이며, `|gradient|`은 sample/pair absolute gradient 평균이다.

| Ng | Regime | κ median | NLL median grad | Reg median grad | Total median grad | Total lower-κ fraction |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 80 ns | 7.960 | +0.00705 | +0.00314 | +0.01012 | 76.0% |
| 16 | 120 ns | 5.726 | +0.03494 | +0.00491 | +0.03978 | 96.6% |
| 32 | 80 ns | 4.153 | +0.00136 | +0.00610 | +0.00695 | 60.5% |
| 32 | 120 ns | 2.270 | +0.03903 | +0.00916 | +0.04726 | 72.1% |

Regularizer는 모든 조건에서 κ를 낮추는 방향이었다. Ng16에서는 NLL도 120 ns에서 80 ns보다 명확히 강해져 total signal이 일관됐다. Ng32에서는 80 ns NLL gradient의 lower-κ fraction이 55.9%로 sign cancellation이 컸고, 120 ns도 lower-κ fraction은 69.4%에 그쳤다. 다만 median 방향은 양수였으며, 일부 큰 음의 NLL outlier 때문에 Ng32 120 ns의 batch-like mean gradient는 음수가 되었다. 이는 sample-specific residual/evidence interaction이 존재함을 뜻한다.

### 120/80 gradient magnitude ratio

| Ng | NLL κ | λreg·Lreg κ | Total κ | Total raw-κ |
|---:|---:|---:|---:|---:|
| 16 | 2.71× | 1.50× | 2.64× | 2.61× |
| 32 | 7.51× | 1.46× | 7.19× | 3.53× |

따라서 120 ns에서 loss signal 자체는 존재하며, 특히 NLL은 80 ns보다 κ를 더 낮추는 방향으로 작용한다. Regularizer도 같은 방향이어서 NLL–regularizer 상쇄가 주원인은 아니다. 그러나 Ng32 ID에서도 κ를 낮추는 regularizer gradient가 100%, total lower-κ fraction이 60.5%로 나타나 sparse observation이 ID κ confidence를 함께 낮추는 Ng-dependent calibration 문제가 확인된다.

### Judgment

판정은 **A와 D의 결합**이다. Ng16에서는 loss가 120 ns κ를 80 ns보다 강하게 낮추므로 Near-OOD signal은 충분히 존재한다. Ng32에서도 120 ns total gradient magnitude는 크지만 ID-Hard κ gradient가 이미 낮은-evidence 방향으로 넓게 분포하고 sign cancellation이 크다. 따라서 Near-OOD separation을 제한하는 직접 원인은 loss signal의 완전한 부재가 아니라, sparse Ng32에서의 κ head calibration/optimization과 sample-dependent gradient cancellation이다. 이번 단계에서는 λreg, κ head, model, optimizer를 수정하지 않았다.

다음 단일 실험은 loss 전체를 바꾸지 않고 κ/evidence head에 대한 calibration 또는 gradient contribution만 독립적으로 검증해야 한다. 구체적인 변경은 이 진단 결과 검토 후 별도 승인 대상으로 남긴다.

상세 산출물은 `runs/current_valid_baseline/epoch3_kappa_loss_gradient_diagnosis_20260920/`의 `per_sample_gradients.csv`, `gradient_summary.csv`, `gradient_ratio_120_over_80.csv`, `analysis_manifest.json`이다.

## Epoch3 κ delay-sweep trajectory diagnosis (2026-09-20)

이번 단계는 κ calibration 문제가 ID training range `10–100 ns` 안에서 이미 시작되는지, 아니면 OOD boundary `100→120 ns`에서만 발생하는지 확인했다. 기존 epoch3 checkpoint에 대해 10/20/40/60/80/100 ns ID와 120 ns OOD-Near를 평가했으며, 120 ns gradient는 실제 training gradient가 아니라 checkpoint에 OOD sample을 입력해 계산한 **diagnostic gradient**다. 새 training, optimizer step, loss/model/mask 변경은 없었다.

### κ and Epistemic trajectory

| Ng | Delay | κ median | κ q05–q95 | Epistemic median | Epistemic q99 | Total κ gradient median | κ 감소 방향 비율 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 20 | 11.484 | 7.928–13.415 | 0.000340 | 0.000768 | -0.00462 | 39.9% |
| 16 | 80 | 7.974 | 5.837–9.408 | 0.000623 | 0.001211 | +0.01063 | 72.4% |
| 16 | 100 | 6.739 | 4.724–8.318 | 0.000771 | 0.001759 | +0.01890 | 86.0% |
| 16 | 120 | 5.549 | 3.807–7.155 | 0.001016 | 0.002236 | +0.04230 | 97.5% |
| 32 | 20 | 8.049 | 4.126–9.980 | 0.000925 | 0.004214 | -0.00825 | 39.4% |
| 32 | 80 | 4.210 | 2.156–5.876 | 0.002320 | 0.008186 | +0.00759 | 63.8% |
| 32 | 100 | 3.237 | 1.326–4.865 | 0.003177 | 0.024958 | +0.02348 | 66.3% |
| 32 | 120 | 2.237 | 0.747–3.789 | 0.004938 | 0.030057 | +0.05201 | 73.4% |

ID 내부에서도 κ가 지속적으로 감소하고 Epistemic이 증가한다. Ng32는 Ng16보다 낮은 κ와 훨씬 큰 Epistemic q99 tail을 보이며, 100 ns에서 이미 q99 `0.02496`으로 120 ns `0.03006`에 근접한다.

### 100→120 ns boundary

| Ng | κ 120/100 | ν margin 120/100 | Ψ 120/100 | Aleatoric 120/100 | Epistemic 120/100 | Total gradient magnitude 120/100 | Epistemic AUROC | Epistemic overlap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.823 | 0.949 | 1.003 | 1.054 | 1.318 | 1.781 | 0.731 | 0.595 |
| 32 | 0.691 | 0.945 | 1.004 | 1.065 | 1.554 | 1.476 | 0.708 | 0.630 |

100→120 ns에서 추가 κ 감소와 Aleatoric 증가가 발생하지만 Ψ는 거의 변하지 않는다. 따라서 boundary extrapolation도 보조적으로 separation을 악화시키지만, 문제의 시작점은 100 ns 이후가 아니다.

### ID gradient transition and judgment

Ng16의 total κ gradient median은 10/20 ns에서 음수(κ 증가 방향)였지만 40 ns 부근부터 양수로 전환되고 80–100 ns에서 κ 감소 방향 비율이 72.4%/86.0%로 증가한다. Ng32는 10–60 ns에서 median gradient가 음수 또는 거의 0이지만 80 ns부터 양수로 전환되며, ID 80/100 ns에서도 κ 감소 방향 비율이 63.8%/66.3%다.

판정은 **C를 주원인, B를 보조원인**으로 한다. Ng32 sparse observation에서 κ calibration 저하는 ID 범위 내, 특히 80–100 ns에서 이미 시작된다. 100→120 ns는 κ 감소와 Epistemic 증가를 추가하지만 새로운 단절점이라기보다 이미 진행된 Ng-dependent calibration drift의 연장이다. ν margin은 약 5–6% 감소하고 Ψ는 약 0.3–0.4% 증가에 그쳐, 가장 직접적인 parameter는 여전히 κ다.

상세 결과는 `runs/current_valid_baseline/epoch3_kappa_delay_sweep_diagnosis_20260920/`의 `per_sample_gradients.csv`, `delay_trajectory_summary.csv`, `comparison_20_80_100_120.csv`, `boundary_100_vs_120.csv`, `trajectory_analysis_manifest.json`에 저장했다. 다음 단일 실험은 OOD boundary만 조정하기보다 ID 80–100 ns에서의 Ng32 κ calibration을 독립적으로 검증해야 한다.

## Epoch3 Ng×delay factorial κ diagnosis (2026-09-20)

기존 epoch3 checkpoint만 사용해 Ng sparsity effect와 delay/channel-difficulty effect를 분리했다. ID grid는 Ng `{4,8,16,32}` × delay `{20,40,60,80,100}` ns이고, 120 ns는 ID 분석 이후 reference로 추가했다. 각 grid point는 200 samples, canonical periodic mask, direct-CFR AWGN 15 dB, 동일 seed policy를 사용했다. 새 training이나 model/loss/mask/controller 변경은 없었다.

### κ median grid

| Delay | Ng4 | Ng8 | Ng16 | Ng32 |
|---:|---:|---:|---:|---:|
| 20 ns | 12.267 | 12.312 | 11.535 | 8.046 |
| 40 ns | 11.412 | 11.509 | 10.480 | 6.780 |
| 60 ns | 10.038 | 10.243 | 9.174 | 5.471 |
| 80 ns | 8.414 | 8.753 | 7.967 | 4.250 |
| 100 ns | 6.712 | 7.337 | 6.784 | 3.187 |
| 120 ns reference | 5.122 | 6.092 | 5.632 | 2.248 |

### Epistemic median / q99 grid

| Delay | Ng4 | Ng8 | Ng16 | Ng32 |
|---:|---:|---:|---:|---:|
| 20 ns | 0.000241 / 0.000450 | 0.000393 / 0.000782 | 0.000678 / 0.001541 | 0.001828 / 0.007564 |
| 40 ns | 0.000281 / 0.000656 | 0.000465 / 0.001105 | 0.000827 / 0.002390 | 0.002423 / 0.017078 |
| 60 ns | 0.000354 / 0.000682 | 0.000568 / 0.001150 | 0.001031 / 0.002335 | 0.003311 / 0.017370 |
| 80 ns | 0.000463 / 0.000784 | 0.000719 / 0.001202 | 0.001266 / 0.002328 | 0.004559 / 0.019533 |
| 100 ns | 0.000622 / 0.001473 | 0.000906 / 0.002055 | 0.001552 / 0.003394 | 0.006366 / 0.039059 |
| 120 ns reference | 0.000889 / 0.002311 | 0.001169 / 0.002811 | 0.001979 / 0.004685 | 0.009587 / 0.070626 |

### Sparsity and difficulty effects

| Delay | κ Ng32/Ng4 | κ Ng32/Ng16 | Epistemic Ng32/Ng16 | NMSE Ng32−Ng4 (dB) |
|---:|---:|---:|---:|---:|
| 20 ns | 0.656 | 0.698 | 2.70× | +7.071 |
| 40 ns | 0.594 | 0.647 | 2.93× | +7.104 |
| 60 ns | 0.545 | 0.596 | 3.21× | +7.184 |
| 80 ns | 0.505 | 0.533 | 3.60× | +7.439 |
| 100 ns | 0.475 | 0.470 | 4.10× | +7.286 |

At fixed Ng, the 100/20 ratios were:

| Ng | κ 100/20 | Epistemic 100/20 | Aleatoric 100/20 | NMSE 100−20 (dB) |
|---:|---:|---:|---:|---:|
| 4 | 0.547 | 2.579× | 1.423× | +2.689 |
| 8 | 0.596 | 2.305× | 1.377× | +2.737 |
| 16 | 0.588 | 2.291× | 1.352× | +2.837 |
| 32 | 0.396 | 3.482× | 1.389× | +2.905 |

Ng sparsity alone already reduces κ at easy 20 ns: Ng32/Ng4 is `0.656`. Delay difficulty alone also reduces κ at Ng4: κ 100/20 is `0.547`. Ng32 combines both effects and has the strongest difficulty ratio `0.396`, with Epistemic increasing `3.482×` from 20 to 100 ns.

### Two-factor and reconstruction relationship

Descriptive two-factor decomposition of `log(median κ)` over the ID grid gave:

| Effect | Fraction of grid sum of squares |
|---|---:|
| Ng main effect | 53.6% |
| Delay main effect | 43.5% |
| Ng×delay interaction | 2.8% |

Thus the dominant decomposition is additive Ng sparsity plus delay difficulty, with a smaller but visible interaction in the Ng32×80/100 tail. Across 1,000 ID samples, Spearman correlations were:

| Ng | NMSE vs κ | NMSE vs Epistemic |
|---:|---:|---:|
| 4 | -0.681 | +0.645 |
| 8 | -0.670 | +0.649 |
| 16 | -0.630 | +0.614 |
| 32 | -0.592 | +0.572 |

Higher reconstruction error (less negative NMSE) is associated with lower κ and higher Epistemic, supporting the hard-ID-sample → low-evidence pathway.

### 100→120 ns reference and judgment

| Ng | κ 120/100 | Epistemic 120/100 | κ overlap | Epistemic overlap | Epistemic AUROC |
|---:|---:|---:|---:|---:|---:|
| 4 | 0.763 | 1.430 | 0.475 | 0.475 | 0.803 |
| 8 | 0.830 | 1.291 | 0.510 | 0.565 | 0.775 |
| 16 | 0.830 | 1.275 | 0.505 | 0.575 | 0.725 |
| 32 | 0.705 | 1.506 | 0.595 | 0.620 | 0.708 |

판정: **A와 B가 모두 존재하지만, 주원인은 A+B의 결합이며 Ng32에서 C가 tail을 악화시킨다.** Ng 증가만으로도 20 ns에서 κ가 낮아지고, delay 증가만으로도 Ng4에서 κ가 낮아진다. Ng×delay interaction의 grid-level 크기는 2.8%로 main effect보다 작지만, Ng32의 80–100 ns에서 practical tail 문제를 강화한다. 100→120 ns는 추가 OOD effect이나 ID 내부 효과보다 독립적인 주원인은 아니다.

새 결과는 `runs/current_valid_baseline/epoch3_ng_delay_factorial_diagnosis_20260920/`의 `per_sample.csv`, `summary.csv`, `sparsity_effect.csv`, `difficulty_effect.csv`, `two_factor_effects.json`, `reconstruction_uncertainty_correlations.csv`, `boundary_100_vs_120.csv`, `kappa_heatmap.png`, `epistemic_heatmap.png`, `reconstruction_vs_uncertainty.png`에 저장했다. 다음 단일 실험은 모델/loss를 바꾸기 전에 Ng32의 sparse-observation-conditioned κ calibration을 단독으로 검증하는 것이 적절하다.

## Epoch3 reconstruction-error matched Ng16/Ng32 κ diagnosis (2026-09-20)

이번 단계는 Ng32 κ 저하가 reconstruction difficulty만으로 설명되는지, error를 맞춘 뒤에도 Ng 자체 효과가 남는지 확인했다. 기존 factorial evaluation의 epoch3 sample artifact를 재사용했으며 checkpoint, model, loss, mask, observation, controller는 변경하지 않았다.

### Matching protocol

NMSE 기준 greedy sorted nearest-neighbor matching을 사용했고, sample을 재사용하지 않았다. 고정 tolerance는 **0.25 dB**였다. 동일 delay matching을 우선했으며, overlap이 부족한 경우 ID 20–100 ns pooled matching을 보조로 사용했다. 120 ns는 별도 reference로만 처리했다.

동일 delay의 overlap은 매우 제한적이었다.

| Delay | Matched N @ 0.25 dB | Error difference mean (Ng32−Ng16) |
|---:|---:|---:|
| 20 ns | 5 | -0.113 dB |
| 40 ns | 6 | -0.149 dB |
| 60 ns | 6 | -0.054 dB |
| 80 ns | 0 | N/A |
| 100 ns | 7 | -0.120 dB |

동일 delay pair 수가 너무 적어 delay별 matching만으로는 강한 결론을 내리지 않았다. ID pooled 20–100 ns에서는 328쌍이 형성되었고 error difference 평균 `-0.166 dB`, median `-0.231 dB`, 최대 절대 차이 `0.250 dB`였다.

### Error-matched κ comparison

| Group | N | κ16 median | κ32 median | κ32/κ16 | κ32<κ16 | Epistemic16 | Epistemic32 | Epistemic32>16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ID 20–100 ns pooled | 328 | 7.552 | 6.787 | 0.903 | 60.7% | 0.001330 | 0.002383 | 86.6% |
| 20 ns direct | 5 | 11.225 | 6.711 | 0.592 | 100% | 0.000706 | 0.002359 | 100% |
| 40 ns direct | 6 | 9.748 | 6.634 | 0.742 | 100% | 0.000923 | 0.002642 | 100% |
| 60 ns direct | 6 | 9.940 | 5.266 | 0.512 | 100% | 0.000903 | 0.003512 | 100% |
| 100 ns direct | 7 | 7.501 | 2.226 | 0.364 | 100% | 0.001286 | 0.009783 | 100% |

ID pooled paired κ difference의 bootstrap 95% CI는 `[-1.033, -0.528]`이고, Epistemic difference CI는 `[0.00138, 0.00211]`이었다. 이는 reconstruction error가 거의 같은 matched pair에서도 Ng32 κ가 낮고 Epistemic이 높은 방향이 유지됨을 보여준다.

### Difficulty-bin limitation

각 delay에서 combined NMSE quartile bin을 만들었지만, Ng16/Ng32 error distribution이 크게 이동해 같은 bin에 양쪽 sample이 충분히 들어가지 않는 경우가 많았다. 특히 80 ns에서는 직접 matching이 0쌍이었다. 따라서 difficulty-bin 결과는 보조 descriptive 결과로만 저장했고, 모든 bin을 Ng effect 검정에 사용하지 않았다.

### Conditional association

ID 전체 1,000 samples에 대해 단순 진단식 `log(κ) ~ NMSE + delay_ns + I(Ng=32)`를 적합했다. 이는 causal proof가 아닌 conditional association이다.

| Term | Coefficient |
|---|---:|
| NMSE | +0.1220 |
| delay_ns | -0.01359 |
| Ng32 indicator | -0.9192 |
| R² | 0.6804 |

NMSE와 delay를 통제한 뒤에도 Ng32 indicator가 음수로 남았다. 따라서 reconstruction difficulty leakage만으로 Ng32 κ 저하를 설명하기 어렵고, Ng 자체의 sparse-observation-conditioned κ calibration effect가 추가로 존재한다.

### Judgment

판정은 **A가 주원인, C가 보조원인**이다. error-matched pooled ID pair에서도 Ng32 κ가 낮고 Epistemic이 높아 Ng 자체 effect를 지지한다. 동시에 Ng32 error distribution이 Ng16보다 어려운 방향으로 이동하고, 동일 delay matching 표본이 적다는 점에서 sparsity×difficulty interaction도 존재한다. 다만 80 ns direct matching이 0쌍이므로 모든 delay에서 동일한 크기의 Ng effect가 확정됐다고 과장하지 않는다.

120 ns reference matching에서는 43쌍이 형성되었고 κ32/κ16 `0.194`, Epistemic32/Epistemic16 `12.67×`였다. 이는 OOD에서 Ng effect가 더 커짐을 보여주지만 training ID calibration 판단에는 사용하지 않았다.

상세 결과는 `runs/current_valid_baseline/epoch3_ng_error_matched_kappa_20260920/`의 `matched_pairs.csv`, `matched_summary.csv`, `difficulty_bins.csv`, `conditional_regression.json`, `analysis_manifest.json`이다. 다음 단일 실험은 전체 reconstruction error를 다시 맞추는 방식보다 Ng-conditioned κ calibration 자체를 독립적으로 검증해야 한다.

## Diagnostic-only post-hoc Ng κ scalar calibration (2026-09-20)

이번 단계는 Ng-dependent κ scale bias가 하나의 positive scalar로 완화되는지 진단했다. 이는 논문에 공개되지 않은 **IMPLEMENTATION-ASSUMPTION / DIAGNOSTIC ONLY**이며, checkpoint/model/loss/mask/controller/threshold를 실제로 수정하지 않았다.

### Split and calibration rule

각 Ng에서 ID delay `10/20/40/60/80/100 ns`의 sample `0–99`를 calibration split, sample `100–199`를 held-out evaluation split으로 사용했다. 120 ns는 calibration에 사용하지 않았다. 각 split은 Ng당 600 samples이며, 120 ns held-out reference는 Ng당 100 samples다.

사용한 단일 scalar는 다음과 같다.

```text
m_Ng  = median(log κ_Ng) on ID calibration split
m_ref = median(log κ) over pooled ID calibration samples
c_Ng  = exp(m_ref - m_Ng)
κ_cal = c_Ng κ
Epistemic_cal = Aleatoric / κ_cal
```

### Fitted Ng scale factors

| Ng | m_Ng | c_Ng |
|---:|---:|---:|
| 4 | 2.3262 | 0.8707 |
| 8 | 2.3312 | 0.8664 |
| 16 | 2.2305 | 0.9583 |
| 32 | 1.7353 | 1.5723 |

Ng32에 가장 큰 보정 factor가 필요해 기존 κ scale bias가 확인되었다.

### Held-out ID scale alignment

| Ng | κ median before | κ median after | Epi median before | Epi median after | Epi q95 before | Epi q95 after | Epi q99 before | Epi q99 after |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 10.437 | 9.088 | 0.000333 | 0.000191 | 0.000739 | 0.000423 | 0.001058 | 0.000605 |
| 8 | 10.489 | 9.088 | 0.000548 | 0.000316 | 0.001094 | 0.000631 | 0.001418 | 0.000817 |
| 16 | 9.524 | 9.127 | 0.000968 | 0.000505 | 0.002019 | 0.001052 | 0.002716 | 0.001415 |
| 32 | 5.765 | 9.065 | 0.003020 | 0.000960 | 0.010563 | 0.003343 | 0.026006 | 0.008197 |

κ median은 Ng 간 약간의 차이로 정렬되었다. Epistemic scale spread도 감소했다.

- median spread: `9.07× → 5.02×`
- q95 spread: `14.30× → 7.90×`
- q99 spread: `24.57× → 13.55×`

그러나 Aleatoric 자체가 Ng별로 다르기 때문에 Epistemic distribution alignment는 불완전했다.

### Global threshold and within-Ng sanity check

Pooled ID calibration q99 threshold는 `0.011979 → 0.003793`으로 변했다. Held-out ID 결과는 다음과 같다.

| Ng | ID FPR before | ID FPR after | 120 ns TPR before | 120 ns TPR after | AUROC before | AUROC after |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.0% | 0.0% | 0.0% | 0.0% | 0.95753 | 0.95745 |
| 8 | 0.0% | 0.0% | 0.0% | 0.0% | 0.94627 | 0.94623 |
| 16 | 0.0% | 0.0% | 0.0% | 0.0% | 0.92210 | 0.92203 |
| 32 | 3.5% | 3.5% | 39.0% | 39.0% | 0.90890 | 0.90887 |

Positive scalar scaling은 Ng 내부 ranking을 보존하므로 within-Ng AUROC와 자체 threshold 기준 TPR이 거의 변하지 않는 것이 정상이다. 실제 변화도 수치 오차 수준이었다.

Ng별 held-out ID FPR range/std는 before와 after 모두 `0.035 / 0.0152`로 동일했다. 따라서 scalar κ alignment만으로 global threshold의 Ng별 false-trigger 편차가 줄어들지는 않았다.

### Offline Fig.11 threshold crossing

기존 restricted Fig.11 trace에 calibrated score와 새 pooled threshold를 적용한 결과는 controller 재실행이 아닌 **offline diagnostic only**다. 기존 trace의 threshold crossing은 before 11회, after 31회였으며, after에는 20/60/80 ns ID crossing도 추가되었다. 이 결과는 trace와 held-out split의 score distribution 차이 때문에 post-hoc scalar만으로 실제 Fig.11 controller 개선을 주장할 수 없음을 보여준다. 기존 trace의 10 ns crossing은 calibrated score에서도 남았다.

### Judgment

판정은 **B: 단순 scale bias는 존재하지만 intrinsic distribution-shape/separation 문제도 큼**이다.

- Ng32 κ scale bias: 명확히 존재
- ID κ/Epistemic scale alignment: 개선
- global threshold FPR 편차: 개선되지 않음
- 120 ns TPR: held-out 기준 변화 없음
- within-Ng AUROC: 본질적으로 변화 없음

따라서 단순 Ng scalar calibration은 cross-Ng score scale 일부를 설명하지만, Ng32의 ID tail, Aleatoric 차이, ID/OOD overlap 자체를 해결하지 않는다. 이번 단계에서는 calibration factor를 실제 controller나 evaluator에 적용하지 않았다.

상세 결과는 `runs/current_valid_baseline/epoch3_ng_scalar_kappa_calibration_20260920/`의 `scale_alignment.csv`, `global_threshold_results.csv`, `offline_fig11_threshold_crossings.csv`, `offline_fig11_summary.csv`, `results.json`, `analysis_manifest.json`이다. 다음 단일 실험은 scalar 보정이 아니라 Ng별 Epistemic distribution shape/heteroscedasticity를 독립적으로 검증하는 것이 적절하다.

### Correction audit: κ-only calibration (2026-09-20)

위 calibration 결과는 수식 audit에서 **INVALID DIAGNOSTIC**로 표시했다. 원래 script는 `epistemic_cal = persisted_aleatoric / kappa_cal`을 계산했지만, 입력 artifact의 `aleatoric`과 `epistemic`은 서로 다른 pair/component aggregation 단위로 저장되어 있었다. 실제로 `epistemic / (aleatoric/kappa)`가 약 2였으므로 의도하지 않은 추가 scale factor가 들어갔다.

수정된 κ-only 계산은 기존 `epistemic`을 보존하고 `epistemic_cal = epistemic_original / c_Ng`만 적용했다. sample-wise `epistemic_after / epistemic_before`는 모든 finite sample에서 정확히 `1/c_Ng`였다. 수정 결과는 `runs/current_valid_baseline/epoch3_ng_scalar_kappa_calibration_20260920_corrected/`에 저장했으며, 원래 결과 파일은 삭제하거나 덮어쓰지 않았다.

Corrected pooled ID q99 threshold는 `0.011979 → 0.007619`이다. Held-out ID FPR과 120 ns TPR은 Ng4/8/16/32에서 각각 기존과 동일하게 `0/0/0/3.5%` 및 `0/0/0/39%`였고, within-Ng AUROC도 동일했다. Corrected held-out Epi median은 Ng4/8/16/32에서 각각 `0.000382/0.000632/0.001010/0.001921`이다. 따라서 이전 결론의 방향(단순 scale bias는 있으나 separation/분포 shape 문제도 남음)은 유지되지만, 이전 수치와 threshold `0.003793`은 폐기해야 한다.

## Ng32 ID Epistemic tail diagnosis (2026-09-20)

이번 분석은 기존 epoch3 checkpoint에 대한 **diagnostic only** 평가이며 training, optimizer step, model/loss/controller/threshold 변경을 수행하지 않았다. Ng32 ID `20/40/60/80/100 ns` 각각 200 samples, seed `20262000`, canonical direct-CFR 15 dB AWGN protocol을 사용했다. Canonical evaluator의 mask는 `mask[:,::Ng]`로 offset 0 하나만 사용하므로 offset별 원인 비교는 식별할 수 없고, 이를 별도 offset 결과로 해석하지 않는다.

Ng32 pooled ID Epistemic 기준은 p50=`0.003495`, p95=`0.011137`, p99=`0.024761`이다. Normal은 0–50 percentile 500 samples, upper는 p95–p99 40 samples, extreme은 p99 이상 10 samples다.

| Group | κ median | ν margin median | Aleatoric median | Epistemic median | NMSE median |
|---|---:|---:|---:|---:|---:|
| Normal 0–50 | 6.893 | 29.101 | 0.01610 | 0.002311 | -17.539 dB |
| Upper p95–p99 | 1.676 | 21.396 | 0.02431 | 0.014313 | -15.969 dB |
| Extreme p99+ | 0.866 | 20.037 | 0.02664 | 0.032400 | -16.414 dB |

Extreme tail에서는 κ 감소뿐 아니라 Aleatoric 증가와 ν margin 감소가 동시에 나타났다. 따라서 판정은 **B: κ + Aleatoric joint amplification**, 보조적으로 **C: ν-margin contribution**이다. raw κ도 같은 방향으로 감소했으며 raw ν는 admissibility transform 이전 값으로 저장했다.

q99 tail은 100 ns에 5/10 samples로 가장 많이 집중되지만 60 ns 2개, 80 ns 2개, 20 ns 1개도 존재한다. 따라서 high-delay concentration은 있으나 100 ns만의 현상은 아니다.

동일 delay 내 NMSE nearest-neighbor matching은 9쌍을 만들었고, matched NMSE 절대차 median은 `0.215 dB`였다. 모든 matched pair에서 extreme sample의 κ는 더 낮고, Aleatoric과 Epistemic은 더 높았으며, ν margin도 더 작았다. 다만 q99 tail 자체가 10 samples뿐이므로 강한 일반화 결론이 아니라 일관된 진단 신호로 해석한다.

Ng16 q99 tail median은 κ=`4.327`, ν margin=`28.307`, Aleatoric=`0.01331`, Epistemic=`0.003125`였고, Ng32 q99 tail은 각각 `0.866/20.037/0.02664/0.03240`이었다. Ng32에서 같은 tail mechanism이 훨씬 증폭된다. Ng32 120 ns reference median은 `κ=2.324`, `ν margin=22.277`, `Aleatoric=0.02157`, `Epistemic=0.009304`로, ID q99 tail이 오히려 더 extreme한 Epistemic/evidence pattern을 보였다. 이는 ID tail을 단순한 OOD-like score로 사용할 수 없음을 의미한다.

상세 산출물은 `runs/current_valid_baseline/epoch3_ng32_id_epistemic_tail_diagnosis_20260920/`의 `per_sample.csv`, `group_summary.csv`, `delay_concentration.csv`, `offset_concentration.csv`, `error_matched_extreme_normal.csv`, `error_matched_effects.json`, `ng16_ng32_ood_reference.csv`, `analysis_manifest.json`이다. 다음 단일 진단은 특정 offset을 조정하는 것이 아니라, Ng32에서 κ/ν/Aleatoric joint tail을 만드는 sample-dependent evidence coupling을 직접 분해하는 것이 적절하다.

## Ng32 CFR feature diagnosis for κ collapse (2026-09-20)

이번 단계는 기존 sample-level artifact의 후처리만 수행한 **diagnostic only** 분석이다. normalization은 적용하지 않았다. 현재 dataset generation 설정의 `normalize_channel=false`는 논문 공개 설정으로 확인되지 않은 **IMPLEMENTATION-ASSUMPTION**이다. 따라서 이번 결과는 normalization 효과를 검증한 것이 아니다.

사용 feature는 저장된 CFR magnitude mean/std, RMS `sqrt(mean^2+std^2)`, adjacent-subcarrier difference mean, peak-to-mean ratio다. Ng32 ID 20–100 ns 1,000 samples를 사용했고, 이전에 고정한 Ng32 pooled ID q99 tail 정의를 그대로 재사용했다. 120 ns는 reference로만 사용했다.

### Correlation and partial effect

Ng32에서 RMS magnitude와 κ의 Spearman correlation은 `-0.578`, adjacent variation과 κ는 `-0.811`이었다. Epistemic correlation은 각각 `+0.598`, `+0.797`이었다. Ng16 reference에서도 각각 κ `-0.565`, `-0.821`로 frequency variation과 κ의 강한 연관이 유지됐다.

표준화한 진단 회귀 `log(κ) ~ RMS magnitude + adjacent variation + NMSE + delay_ns`의 conditional association은 다음과 같다.

| Term | Standardized coefficient |
|---|---:|
| RMS magnitude | -0.260 |
| Adjacent variation | -0.549 |
| NMSE | +0.012 |
| Delay | +0.206 |

R²는 `0.898`이지만 이는 causal proof가 아니다. 관측된 feature 중 adjacent frequency variation이 amplitude보다 κ와 더 강하게 연결된다.

### Matched analyses

Matching은 sample 재사용 없이 pooled IQR의 25% caliper를 사용했다.

| Matching | N | κ tail-normal median difference | Epi difference | 판단 |
|---|---:|---:|---:|---|
| RMS amplitude | 2 | -4.448 | +0.02598 | overlap 부족, 결론 보류 |
| Adjacent variation | 5 | -4.468 | +0.02359 | κ 차이 유지 |
| RMS + variation + NMSE | 0 | N/A | N/A | joint overlap 부족 |

Frequency variation을 엄격히 맞춘 5쌍에서도 모든 pair에서 tail κ가 더 낮고 Epistemic이 더 높았다. 그러나 NMSE까지 함께 맞출 수 있는 pair가 없어 “모든 CFR feature를 통제한 뒤에도 head 문제가 남는다”고 확정할 수는 없다.

3×3 amplitude×frequency tail-rate 표에서는 high-amplitude/high-variation bin의 q99 rate가 `6.09%`로 가장 높았고, 전체 기준 1%보다 높았다. 이는 joint CFR condition을 지지하지만 bin별 표본 수를 고려해 진단적 결과로만 해석한다.

Ng32 ID q99와 120 ns reference의 feature median은 다음과 같다.

| Group | RMS | Adjacent variation | κ | Aleatoric | Epistemic |
|---|---:|---:|---:|---:|---:|
| Ng16 ID q99 | 1.472 | 0.02451 | 4.327 | 0.01331 | 0.003125 |
| Ng32 ID q99 | 1.518 | 0.02289 | 0.866 | 0.02664 | 0.032400 |
| Ng32 120 ns | 0.956 | 0.02691 | 2.324 | 0.02157 | 0.009304 |

Ng32 ID q99는 120 ns보다 amplitude가 더 크고, adjacent variation은 비슷한 수준이다. 따라서 ID tail과 OOD를 CFR feature만으로 동일시할 수 없다.

### Judgment

현재 증거상 **B/C에 가까운 결과**다. Frequency-selective/shape feature가 κ collapse와 가장 강하게 연결되고, amplitude가 큰 동시에 variation도 큰 joint condition에서 tail rate가 증가한다. 하지만 strict amplitude+frequency+NMSE overlap이 0이므로 amplitude와 evidential-head 자체의 잔여 효과를 완전히 분리하지 못했다. 다음 단일 실험은 normalization을 실제 적용하지 말고, 동일 feature overlap을 확보할 수 있는 matched/conditional evidence response 진단으로 제한해야 한다.

상세 결과는 `runs/current_valid_baseline/epoch3_cfr_feature_tail_diagnosis_20260920/`의 `spearman_correlations.csv`, `matched_summary.csv`, `tail_rate_2d_bins.csv`, `partial_effect_diagnostic.json`, `reference_summary.csv`, `analysis_manifest.json`이다.

## 2026-09-20 Overnight Phase 1/2 and Partial FT Readiness

상세 morning handoff는 `docs/PARTIAL_FT_READINESS_20260921.md`에 기록했다. 새 결과는 모두 `runs/current_valid_baseline/overnight_20260920_*` 아래에 있다.

- Phase 1 conditional residual: held-out R² `0.884`, Ng32 coefficient `-0.558`, Ng32 ID q99 held-out residual κ ratio median `0.463`. 따라서 평균 Ng effect는 줄지만 q99 evidential-head/distribution-shape tail은 남는다. 이는 conditional association diagnostic이지 causal proof가 아니다.
- Phase 1 provenance limitation: 20/80 ns는 각 2,000개 common-eval sample을 사용했지만 40/60/100 ns는 기존 archive의 각 200개만 사용했다. 후자의 표본 부족은 결론 제한으로 기록했다.
- Phase 2 amplitude diagnostic: Ng32에서 α=0.75/1.25의 κ·Epistemic ratio가 delay에 따라 크게 변했다. `normalize_channel=false`는 중요한 implementation-calibration 후보지만 `IMPLEMENTATION-ASSUMPTION / DIAGNOSTIC ONLY`이며 재학습하지 않았다.
- Partial FT readiness: actual model inventory, four freeze scopes, comparison evaluator, adaptation config, and four one-step dry-runs are complete. Full/Partial 본 학습은 시작하지 않았다.

## 2026-09-21 Controlled Partial vs Full Fine-Tuning

이번 실험은 `last_block_plus_head` 3.1625%와 `full` 100%만 비교한 `RESEARCH-EXTENSION / IMPLEMENTATION-ASSUMPTION`이다. 두 run은 동일 epoch3 checkpoint, 1,000/200 adaptation split, seed `20260921`, Adam/LR `1e-4`, λreg `1e-3`, batch 8, 3 epochs/375 steps와 동일 deterministic sparse mask/noise schedule을 사용했다. Full-CFR은 clean target 확보에만 사용했고 predictor input은 Ng `{4,8,16,32}` sparse CFR + 15 dB AWGN + mask channel이었다.

Data leakage 검사에서 split 간 CFR byte-hash 중복은 0건이었다. Partial/Full schedule digest는 동일하고, 기존 epoch3 checkpoint SHA-256 `d3c864788ee7e0e58bbe9f683aa47615039a470258ff054ec2828179b1f23986`도 보존되었다. Partial frozen tensor 126개는 max absolute diff `0.0`이었다.

Epoch3 120 ns omitted-NMSE improvement (`pre - post`)은 Ng16에서 Partial `1.243 dB`, Full `1.417 dB`, Ng32에서 Partial `0.918 dB`, Full `1.033 dB`였다. Partial retention은 Full 대비 `87.7%/88.9%`였다. Partial의 ID forgetting은 Full보다 작았고, Far-OOD Epistemic도 Partial이 더 많이 보존했다. 반면 120 ns Epistemic 감소 폭은 Full이 더 컸다.

판정은 **부분지지**다. Partial은 reconstruction adaptation·ID 보존·효율성 측면에서 유효한 후보지만, Full과 동일한 uncertainty-head adaptation 효과까지 입증하지는 못했다. 재튜닝이나 추가 scope 실험은 수행하지 않았다.

상세 결과는 `docs/PARTIAL_FT_RESULTS_20260921.md`, `runs/current_valid_baseline/partial_ft_20260921_sparse_evaluation/`에 기록했다. Checkpoint는 `runs/current_valid_baseline/partial_ft_20260921_sparse_3ep/` 및 `runs/current_valid_baseline/full_ft_20260921_sparse_3ep/`에 있다.

## 2026-09-21 Controlled 25% Partial Fine-Tuning Extension

`last_8_blocks_plus_head` (`ResidualBlock 24–31` + `gamma/psi/kappa/nu_head`)를 신규 학습하고 기존 3.16% Partial 및 Full 결과와 비교했다. Partial layer policy, adaptation sample count, 3 epochs, batch 8, Adam/LR `1e-4`, seed `20260921`은 모두 **RESEARCH-EXTENSION / IMPLEMENTATION-ASSUMPTION**이며 논문 재현 설정이 아니다. 논문에서 따르는 것은 OOD 감지 후 full feedback 흐름뿐이다.

25% scope는 실제 `2,956,824` parameters, `25.0253%`였다. 기존 epoch3 checkpoint, 120 ns 1,000/200 adaptation split, untouched 20/80/120 ns/1 ms test, sparse CFR input, clean full-CFR target, mask/noise schedule, optimizer/LR/epoch/seed를 기존 3.16%/Full과 동일하게 유지했다. Full CFR은 target 확보에만 사용했고 identity fine-tuning은 하지 않았다. leakage는 0건, starting checkpoint SHA-256은 `d3c864788ee7e0e58bbe9f683aa47615039a470258ff054ec2828179b1f23986`, frozen tensor max diff는 `0.0`이었다.

25%는 120 ns reconstruction improvement에서 Full 대비 Ng16 `107.4%`, Ng32 `109.3%` retention을 보였고, 120 ns Epistemic 감소도 Full에 가까워졌다. 20/80 ns ID forgetting은 Full보다 작았다. 그러나 1 ms Epistemic이 크게 낮아져 Far-OOD separation이 약화됐다. 따라서 판정은 **부분지지**다: 25%는 reconstruction/near-OOD adaptation 후보로 강하지만, uncertainty-preserving 관점에서 3.16%보다 일괄적으로 우월하지 않다. 추가 scope/LR/epoch tuning은 수행하지 않았다.

- 상세 결과표: `docs/PARTIAL_FT_25PCT_RESULTS_20260921.md`
- 25% training: `runs/current_valid_baseline/partial_ft_20260921_sparse_25pct_3ep/`
- 25% evaluation: `runs/current_valid_baseline/partial_ft_20260921_sparse_25pct_evaluation/`
- 25% evaluator: `scripts/evaluate_partial_ft_25pct.py`

## 2026-09-21 Far-OOD Epistemic Collapse Diagnosis

The 25% Partial FT Far-OOD collapse was diagnosed using fixed sparse evaluation samples, raw Eq.(7)/(8) evidential decomposition, in-memory component swaps, and Pre-vs-25% activation/parameter drift. This was a **RESEARCH-EXTENSION / IMPLEMENTATION-ASSUMPTION**; no paper setting was inferred.

The direct cause was late feature drift in ResidualBlock 24–31. H-BLOCK (adapted blocks only, Pre heads) reproduced the 1 ms collapse, while H-HEAD/H-UNC (adapted heads only, Pre backbone) preserved the Pre distribution. At 1 ms, 25%/Pre median ratios were approximately `kappa ×120.5/×213.5`, `nu margin ×1.42/×1.85`, and `Psi ×0.98/×0.93` for Ng16/32. Thus kappa was the dominant direct Eq.(8) factor, with nu-margin amplification; Psi was secondary. The 1 ms Epistemic distribution moved toward ID, not only its mean.

Exactly one confirmatory training followed: `last_4_blocks_plus_head` (ResidualBlock 28–31 + all heads), selected to reduce late-block drift without sweeping scopes. It used the same checkpoint, split, sparse input, clean target, seed, schedule, optimizer, LR, epochs, and evaluation protocol. It trained `1,480,728` parameters (`12.5323%`), achieved Full-like 120 ns reconstruction, lower ID forgetting than Full, and restored the 1 ms/120 ns Epistemic ratio from roughly `2.8e3/3.1e3` at 25% to `6.1e4/3.6e4`.

Final diagnosis is **late residual feature drift with feature-to-evidential-head interaction**, not head-only confidence drift. Detailed tables, AUROC, quantiles, activation drift, and raw statistics are in `docs/PARTIAL_FT_FAR_OOD_DIAGNOSIS_20260921.md` and `runs/current_valid_baseline/partial_ft_far_ood_diagnosis_20260921/`. Confirmatory artifacts are in `runs/current_valid_baseline/partial_ft_far_ood_confirmatory_last4_20260921_3ep/` and its evaluation directory.

## 2026-09-22 Final Partial FT Confirmatory Validation

The final validation repeated the four fixed scopes (`3.16%`, `last_4_blocks_plus_head` `12.5323%`, `25.0253%`, and Full) for seeds `20260921/22/23` and 10 epochs. Epoch10 was fixed as the primary endpoint; epochs 1/3/5/10 were retained for duration trajectories. All Partial policies, the 10-epoch extension, and adaptation schedule are **RESEARCH-EXTENSION / IMPLEMENTATION-ASSUMPTION**.

The last4 scope retained Full-like 120 ns reconstruction at epoch10 (Ng16 `100.6%`, Ng32 `101.2%` of Full improvement), and used substantially less compute (12.53% parameters, 60.8% of Full mean training time, 22.6% of Full peak VRAM). However, its 1 ms Epistemic median fell to `1.7%`/`0.9%` of Pre for Ng16/32 by epoch10. The direct mechanism was kappa inflation (`×43.3`/`×67.5`) with nu-margin growth (`×1.32`/`×1.67`); Psi was secondary. Late feature drift increased at the trainable boundary and tracked the collapse.

Therefore the confirmatory hypothesis is **not supported**: last4 is not adopted as a robust final Partial FT operating point under the current 10-epoch adaptation setting. The prior 3-epoch result is treated as a transient short-duration effect. No new scope sweep or hyperparameter tuning was performed. Dynamic controller validation was skipped because the existing Fig.11 implementation is hard-wired to a different checkpoint/sequence/threshold provenance; static uncertainty is the primary evidence.

- Final report: `docs/PARTIAL_FT_FINAL_VALIDATION_20260921.md`
- Training checkpoints: `runs/current_valid_baseline/partial_ft_final_validation_20260921_retry1/`
- Evaluation/statistics: `runs/current_valid_baseline/partial_ft_final_validation_20260921_analysis_retry5/`
- Evaluator/postprocessor: `scripts/final_partial_ft_validation.py`, `scripts/summarize_final_partial_ft_validation.py`

## 2026-09-22 Feature-Anchored Uncertainty-Preserving Partial FT

The fixed `last_4_blocks_plus_head` scope (`1,480,728` parameters, `12.5323%`) was trained for 10 epochs with a frozen Pre teacher and a normalized ID feature anchor at the ResidualBlock31/evidential-head input. The anchor used a separate 1,000-sample subset from the original 10–100 ns training distribution, `lambda_anchor=1.0`, and the same 120 ns sparse-CFR-to-clean-full-CFR adaptation task. This is **RESEARCH-EXTENSION / IMPLEMENTATION-ASSUMPTION**; the paper does not specify this layer policy, anchor loss, subset, coefficient, or duration.

The experiment used seeds `20260921/22/23`, identical adaptation/evaluation files and schedules, FP32 on NVIDIA GB10 `cuda:0`, and preserved checkpoints at epochs 1/3/5/10. Leakage checks remained zero. Reconstruction passed the pre-specified gate: epoch10 Full-retention was `100.6%` (Ng16) and `101.2%` (Ng32). The anchor did not preserve Far-OOD uncertainty: 1 ms Epistemic median/Pre was only `0.0171` and `0.0087`, essentially unchanged from unregularized last4 (`0.0171`/`0.0086`) and below the practical gates `0.17`/`0.18`. Epoch trajectories showed continuing kappa inflation and head-input drift; AUROC stayed high and was therefore not sufficient evidence of calibration preservation.

Final judgment: **CASE 3 — Not supported**. Feature anchoring alone, with the fixed coefficient and scope, did not solve long-duration Far-OOD uncertainty forgetting. No lambda/scope sweep or automatic retuning was performed. The one next experiment is documented as freezing the uncertainty heads while retaining the same reconstruction adaptation protocol.

- Report: `docs/PARTIAL_FT_UNCERTAINTY_PRESERVING_20260922.md`
- Training/checkpoints/results: `runs/current_valid_baseline/partial_ft_uncertainty_preserving_feature_anchor_20260922/`
- Scripts: `scripts/partial_ft_feature_anchor.py`, `scripts/prepare_id_anchor_data.py`, `scripts/evaluate_feature_anchor.py`, `scripts/evaluate_feature_anchor_auroc.py`, `scripts/summarize_feature_anchor.py`

## 2026-09-22 Boundary-Distilled Partial FT

The fixed `last_4_blocks_plus_head` scope was trained with three separate roles: 120 ns sparse-to-clean reconstruction, old-ID uncertainty distillation from the original 10–100 ns distribution, and uncertainty-only boundary rehearsal on a new Uniform(160,500) ns proxy-OOD split. The proxy split had 1,000 samples and zero hash overlap with all existing train/validation/test splits. The 1 ms set was blind until final evaluation. This is **RESEARCH-EXTENSION / IMPLEMENTATION-ASSUMPTION**.

The objective used fixed normalized auxiliary terms with `lambda_ID=lambda_boundary=1.0`. Proxy-OOD received no reconstruction target loss. The reconstruction gate passed: epoch10 Full-retention was `99.1%` (Ng16) and `99.9%` (Ng32). However, uncertainty preservation was not successful: Ng16 1 ms Epistemic median/Pre rose to `8.85`, while Ng32 reached zero ν-margin for some samples and produced NaN Aleatoric/Epistemic values across all seeds. The boundary curve increased from 120 ns through 160/250/500/1000 ns, but was excessively steep and numerically unstable.

The direct mechanism was not κ inflation; the one-sided penalty drove κ too low and ν-margin toward zero, creating over-uncertainty/instability instead of calibrated Far-OOD preservation. Final judgment: **CASE 3 — Not supported**. No scope, lambda, or proxy-range sweep was performed.

- Report: `docs/PARTIAL_FT_UNCERTAINTY_BOUNDARY_20260922.md`
- Results/checkpoints: `runs/current_valid_baseline/partial_ft_uncertainty_boundary_distillation_20260922/`
- Proxy manifest: `runs/current_valid_baseline/partial_ft_uncertainty_boundary_distillation_20260922/proxy_ood/manifest.json`
- Scripts: `scripts/prepare_proxy_ood_boundary_data.py`, `scripts/partial_ft_boundary_distill.py`, `scripts/evaluate_boundary_distillation.py`, `scripts/evaluate_boundary_curve.py`, `scripts/summarize_boundary_distillation.py`
