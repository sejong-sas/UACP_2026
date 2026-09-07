
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
