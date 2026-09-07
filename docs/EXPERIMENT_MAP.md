# UACP 최종 실험 지도

이 문서는 저장소의 실제 코드와 결과를 최종 보고서의 논리 순서에 맞춰 연결한다.
숫자는 결과 파일에서 확인한 값만 기록한다. 학습을 다시 한 실험과 checkpoint를
읽기만 한 진단을 구분한다.

표기: PAPER-SPECIFIED는 논문 직접 기재, IMPLEMENTATION-ASSUMPTION은 논문에
세부 정보가 없어 현재 구현에서 정한 내용, APPROXIMATION은 논문 구조를 줄인 부분,
HISTORICAL / PRE-FIX는 현재 기준보다 이전 결과, CURRENT VALID는 pair mapping 수정
후 다시 학습한 현재 기준 모델이다.

전체 흐름:

    데이터/채널 -> sparse input/Predictor -> evidential loss
    -> CURRENT VALID BASELINE -> Figure 5 -> power normalization
    -> Figure 7 -> aggregation/Psi/matching -> gradient/parameter audit
    -> formula/semantic audit -> 다음 원인 분리 실험

## PART A. UACP Baseline 구성

### 1. Dataset / Channel Generation

PDF 1페이지의 데이터·채널 생성 정리에 해당한다. clean full CFR을 만들고 delay
spread가 다른 시험 집합으로 채널 난이도와 분포 이동을 나눈다. RMS delay spread가
커지면 주파수축 CFR 변화가 빨라질 수 있어 일부 subcarrier만 보고 나머지를 복원하기
어려워진다.

실험 질문: 데이터 자체가 20 ns < 80 ns < 120 ns 난이도를 만드는가?

종류: TRAINING DATA GENERATION 및 EVALUATION-ONLY DATA GENERATION.

- 코드: src/channel/sionna_channel.py의 load_sectioned_config(),
  generate_cfr_for_delay_spreads(); scripts/generate_dataset.py의 main()
- 조건: 2x2 MIMO, K=1024, 3.5 GHz, 30 kHz
- 학습 spread: Uniform [10,100] ns; 현재 train sample 5,000
- clean full CFR shape: [sample,K,2,2] complex
- TDL-A, mobility=0, normalize=false: IMPLEMENTATION-ASSUMPTION

| Regime | Linear interpolation omitted NMSE (dB) |
| --- | ---: |
| 20 ns | -16.8870 |
| 80 ns | -16.5537 |
| 120 ns | -15.9823 |

20 < 80 < 120가 확인되어 PASS다. 모든 pair가 각각 같은 순서를 보인다는 뜻은
아니다. 데이터가 난이도를 만들지 못한 원인은 배제하고 sparse Predictor로 진행했다.

### 2. Sparse Input / Predictor

src/training/data.py의 uniform_grouping_mask()는 mask[:, ::grouping_factor]를
1로 만든다. Ng=16이면 observed 64개, omitted 960개다. 학습에서는
random_grouping_mask()가 [4,8,16,32] 중 하나와 offset을 sample별로 선택한다.

cfr_to_real_imag()는 [B,K,2,2] complex CFR을 [B,8,K]로 바꾼다.
build_sparse_input()은 observed만 남기고 omitted를 0으로 채운 뒤 mask를 붙여
[B,9,1024]를 만든다. build_noisy_sparse_input()은 reported CFR에만 15 dB
AWGN을 넣고 clean full CFR을 target으로 반환한다. noise 단계는 UNKNOWN IN PAPER,
직접 CFR AWGN은 IMPLEMENTATION-ASSUMPTION이다.

src/models/uacp_predictor.py의 UACPEvidentialPredictor는 1x1 projection,
192 feature, 32 ResidualBlock, evidential head를 사용한다.

| Parameter | 논문 pair shape | 현재 shape | 의미 |
| --- | --- | --- | --- |
| gamma_p | [2048] | [B,8,1024] -> [B,4,2048] | CFR 평균 예측 |
| kappa_p | scalar | [B,4,1] | 평균 evidence |
| Psi_p | full [2048,2048] | diagonal [B,4,2048] | covariance scale |
| nu_p | scalar | [B,4,1] | 자유도/evidence parameter |

layout은 [Re(pair0..3), Im(pair0..3)]이며 pair는 (0,4),(1,5),(2,6),(3,7)이다.
논문의 full Psi=L L^T는 PAPER-SPECIFIED이고 현재 diagonal은 APPROXIMATION이다.

### 3. Loss

src/models/evidential.py의 active 함수는 diagonal_multivariate_student_t_nll()다.
pair 하나에서 d=2K=2048, residual r=h-gamma를 사용한다.

    df = nu - d + 1
    S  = ((kappa+1)/(kappa*df)) Psi

| 논문 항 | 실제 코드 |
| --- | --- |
| df | degrees = nu - dimension + 1 |
| scale | scale_diag = ((kappa+1)/(kappa*degrees))*psi_diag |
| log determinant | sum(log(scale_diag)) |
| Mahalanobis | sum(residual.square()/scale_diag) |
| normalization | torch.lgamma 두 항과 dimension log(df*pi) |
| reduction | -log_prob.mean() |

NLL은 2048개 univariate NLL의 단순 합이 아니라 pair 하나의 diagonal
multivariate Student-t density다. student_t_nll()은 historical elementwise path다.

Student-t scale S와 Eq.7 covariance는 다르다.

    S             = ((kappa+1)/(kappa*(nu-d+1))) Psi
    Sigma_ale     = Psi/(nu-d-1)
    Sigma_epi     = Sigma_ale/kappa

NLL은 S를, uncertainty는 Eq.7/8을 사용하며 두 값을 바꿔 쓰는 active 경로는
확인되지 않았다.

pair_level_evidence_regularizer()는 clean target 전체에 대해

    L_reg = mean_batch,pair[ ||h_p-gamma_p||_2^2 * (kappa_p+nu_p) ]

를 계산한다. observed와 omitted를 따로 제한하지 않으며 Psi는 식에 없으므로
dL_reg/d(raw_Psi)=0이다. evidential_loss()가 NLL + lambda_reg * L_reg를 만든다.

## PART B. Current Valid Baseline

### 4. Current Valid Baseline Training

종류: TRAINING. 이후 진단은 이 checkpoint를 고정한 EVALUATION ONLY,
READ-ONLY DIAGNOSTIC, AUTOGRAD DIAGNOSTIC 또는 POST-HOC CALCULATION이다.

scripts/train_current_valid_baseline.py의 main()은
pair_mapping_assert() -> train_one_epoch() -> evaluate() -> diagnose_regime()를
호출한다. train_one_epoch() 내부 순서는 mask -> build_noisy_sparse_input ->
model -> evidential_loss -> backward -> gradient clip -> optimizer.step()이다.

- checkpoint: runs/current_valid_baseline/condition_c_5k_10ep_lambda1e3_corrected/
- SHA-256: c9df50913b5b48688c8576bbb174b5645a842b269667cacf2822dc049cd626fc
- 5,000 samples, 10 epochs, batch 8, Adam, lr 1e-4, lambda 1e-3
- train spread Uniform [10,100] ns; train grouping [4,8,16,32]; eval Ng=16
- seed 20260819; CUDA cuda:0; NVIDIA GB10
- 15 dB reported-CFR AWGN -> noisy sparse input -> clean target:
  IMPLEMENTATION-ASSUMPTION

| Regime | NMSE all | NMSE omitted | Aleatoric | Epistemic |
| --- | ---: | ---: | ---: | ---: |
| 20 ns | -17.6702 | -17.6709 | 0.020904 | 0.002677 |
| 80 ns | -17.2629 | -17.2605 | 0.020815 | 0.002642 |
| 120 ns | -15.2965 | -15.2930 | 0.021007 | 0.002673 |
| 1 ms | 1.3591 | 1.5622 | 0.028924 | 0.004138 |

Predictor는 난이도를 반영하고 1 ms Epistemic도 증가시키지만, 80 ns Aleatoric과
120 ns Near-OOD Epistemic 분리는 약하다.

## PART C. Figure 5 - Sample-level Uncertainty / Error

### 5. Figure 5 raw diagnostic

논문 Section 5.1 Figure 5와 대응한다. 종류는 EVALUATION ONLY다.
scripts/paper_uncertainty_diagnostics.py의 evaluate_samples()가 sample별 omitted
NMSE와 Eq.12/13 uncertainty를 만들고 run_p1_p2()가 correlation/risk-coverage를
저장한다. 점 하나는 pair가 아니라 sample 하나다.

    NMSE_omitted = 10log10(sum_om(error^2)/sum_om(target^2))
    U_total = U_ale(Eq.12 -> Eq.13) + U_epi(Eq.12 -> Eq.13)

| Regime | Pearson | Spearman |
| --- | ---: | ---: |
| 20 ns | -0.2846 | -0.3265 |
| 80 ns | -0.6201 | -0.6257 |
| 120 ns | -0.6356 | -0.6698 |
| 1 ms | -0.1197 | -0.1184 |

### 6. Figure 5 Power-Normalization Diagnostic

종류: COUNTERFACTUAL/POST-HOC CALCULATION.
scripts/p1_power_normalization_diagnostic.py의 evaluate_regime()가
a=sqrt(mean(|H_true|^2))를 계산하고 H/gamma는 /a, covariance variance는
/a^2로 바꾼다.

| Regime | raw U/NMSE P | normalized U/NMSE P | U/power P |
| --- | ---: | ---: | ---: |
| 20 ns | -0.2846 | 0.4409 | 0.7815 |
| 80 ns | -0.6201 | 0.6991 | 0.9598 |
| 120 ns | -0.6356 | 0.7984 | 0.9357 |
| 1 ms | -0.1197 | 0.1342 | 0.8843 |

power mismatch는 Figure 5 inversion의 주요 원인 중 하나였지만 Figure 7 실패의
유일한 원인은 아니다.

## PART D. Figure 7 - ID Delay-Spread Aleatoric

### 7. Normalized Figure 7 Delay Sweep

논문 Section 5.2 Figure 7 대응, EVALUATION ONLY다.
scripts/normalized_id_delay_sweep.py의 eval_delay()가 10/20/40/60/80/100 ns에서
같은 checkpoint와 Eq.12/13을 사용한다.

| ns | NMSE dB | normalized Ale | normalized Epi |
| ---: | ---: | ---: | ---: |
| 10 | -17.6697 | 0.024138 | 0.003032 |
| 20 | -17.7283 | 0.023310 | 0.002924 |
| 40 | -17.7392 | 0.022690 | 0.002845 |
| 60 | -17.6313 | 0.022936 | 0.002887 |
| 80 | -17.2591 | 0.022293 | 0.002809 |
| 100 | -16.4009 | 0.022465 | 0.002839 |

NMSE는 40 ns 이후 악화해 PARTIAL, Aleatoric은 flat/감소해 FAIL, Epistemic은
대체로 안정적이나 완전하지 않아 PARTIAL이다.

## PART E. 원인 추적

### 8. Eq.12 -> Eq.13 Aggregation Audit

종류: READ-ONLY DIAGNOSTIC. scripts/audit_normalized_aleatoric_aggregation.py의
evaluate_regime()가 pair -> subcarrier map -> mask -> score를 저장한다.
observed=64, omitted=960이다. Eq.12는 ale_sub[k]=mean_pair(Re[k]+Im[k]),
Eq.13은 ale_sub[omitted].mean()이다.

현재 “Eq.13 이전 Aleatoric”이라는 별도 scalar는 없다. Eq.13 직전은 1024개
subcarrier의 Eq.12 map ale_sub다. eq12_ale_raw_observed와
eq12_ale_raw_omitted는 이미 mask 선택 후 평균한 값이다.

| Regime | pair error omitted | final Aleatoric | normalized final Ale |
| --- | ---: | ---: | ---: |
| 20 ns | 0.0163714 | 0.0204326 | 0.0233126 |
| 80 ns | 0.0191251 | 0.0207934 | 0.0223008 |

Eq.13이 inversion을 새로 만들지 않았으며 mask, pair mapping, Re/Im indexing
오류는 audit에서 배제된다. final normalized Ale와 normalized Psi의 상관은
regime별 sample 200개다.

| Regime | Pearson | Spearman |
| --- | ---: | ---: |
| 20 ns | 0.9919 | 0.9871 |
| 80 ns | 0.9990 | 0.9980 |

### 9. Psi - Error / Power Alignment

scripts/psi_error_alignment_diagnostic.py의 evaluate(), make_corr(),
make_power_bins()를 사용한 sample×pair 진단이다.

| Regime | raw Psi/power | raw Psi/absolute error | normalized Psi/normalized error |
| --- | ---: | ---: | ---: |
| 20 ns | 0.8615 | 0.7006 | 0.2389 |
| 80 ns | 0.9304 | 0.7758 | 0.3766 |

raw Psi는 power와 강하게 정렬되고, normalization 후 error 관계는 개선되지만
80 ns regime mean의 Aleatoric 증가를 보장하지 않는다.

### 10. 1:1 Paired-Power Matching

종류: POST-HOC CALCULATION. paired_power_matching.py의 aggregate()는 네 pair를
sample 평균으로 줄이고 match()는 Hungarian assignment로 총 absolute power
distance를 최소화하며 replacement 없이 200:200 매칭한다.

| Quantity | Mean 80-20 | Median | Positive ratio | Wilcoxon p |
| --- | ---: | ---: | ---: | ---: |
| normalized error | +0.002337 | +0.002429 | 0.800 | 1.05e-18 |
| normalized Psi | -0.011117 | -0.001957 | 0.430 | 0.0178 |
| normalized Aleatoric | -0.001012 | -0.000215 | 0.475 | 0.0819 |

같은 power에서도 80 ns error는 증가하지만 Psi/Aleatoric은 일관되게 증가하지
않는다. power만으로 정리할 수 없다.

## PART F. Gradient / Evidential Parameter Analysis

### 11. Loss -> Psi Gradient Audit

종류: AUTOGRAD DIAGNOSTIC. loss_psi_gradient_alignment.py의
forward_with_raw(), pair_nll(), torch.autograd.grad()가 raw parameter gradient를
계산한다.

aggregation audit의 0.016896은 omitted subcarrier당 Re^2+Im^2를 pair 평균한
값이다. gradient audit의 0.008448은 omitted Re/Im 1,920개 real component 평균이다.
따라서 둘은 서로 다른 reduction이며 전자가 약 2배다.

Psi pressure P=-dL/d(rawPsi)는 gradient descent에서 양수일 때 Psi 증가 방향이다.

| Regime | low error pressure | high error pressure |
| --- | ---: | ---: |
| 20 ns | -5.84e-5 | +1.16e-5 |
| 80 ns | -2.61e-5 | +3.43e-6 |

80 ns는 less negative인 상대적 개선은 있으나 평균 양의 Psi pressure는 아니다.

### 12. Psi - Nu Parameter Coupling

논문의 u는 sounding round, r/t는 receive/transmit antenna, Zu는 sparse
feedback/context다. 현재 A=Psi/(nu-d-1), Psi=softplus(rawPsi)+eps,
nu=d+1+softplus(rawNu)+eps다.

    dA/drawPsi = sigmoid(rawPsi)/(nu-d-1)
    dA/drawNu  = -Psi*sigmoid(rawNu)/(nu-d-1)^2

| Regime | |dA/drawPsi| | |dA/drawNu| | Psi/Nu ratio |
| --- | ---: | ---: | ---: |
| 20 ns | 4.849e-6 | 6.410e-4 | 0.00756 |
| 80 ns | 4.558e-6 | 5.740e-4 | 0.00794 |

Nu raw 변화에 훨씬 민감하지만, 이를 곧바로 잘못된 parameter라고 단정하지 않는다.

### 13. Signed Aleatoric Contribution

    S_Ale_Psi = -(dA/drawPsi)(dL/drawPsi)
    S_Ale_Nu  = -(dA/drawNu)(dL/drawNu)
    S_Ale_Total = S_Ale_Psi + S_Ale_Nu

양수는 한 gradient step 기준 Aleatoric 증가, 음수는 감소 방향이다.

| Regime | S_Ale_Psi | S_Ale_Nu | S_Ale_Total | positive total ratio |
| --- | ---: | ---: | ---: | ---: |
| 20 ns | -1.055e-6 | -2.041e-6 | -3.096e-6 | 0.174 |
| 80 ns | -0.609e-6 | -0.971e-6 | -1.580e-6 | 0.273 |

80 ns는 less negative지만 두 평균 모두 음수다. Nu가 모든 조정을 담당한다고
단정할 수 없으며, NLL의 Nu 방향이 regularizer보다 커서 total 방향을 결정한다.

## PART G. Formula / Semantic Audit

### 14. Formula / Semantic Consistency Audit

종류: AUTOGRAD 및 작은 차원 POST-HOC audit.
scripts/evidential_formula_semantic_audit.py의 reduction_audit(),
sensitivity_audit(), operating_ranges(), toy_audit()를 사용한다.

| 항목 | 판정 |
| --- | --- |
| active Eq.5 diagonal Student-t | algebraically consistent |
| Eq.7/8 pair-first 계산 | diagonal 조건에서 consistent |
| Eq.10/11 mean reduction | IMPLEMENTATION-ASSUMPTION |
| Eq.11 direct Psi gradient | 없음 |
| Nu sensitivity | Psi보다 약 130배 큼 |
| full -> diagonal Psi | APPROXIMATION, semantic risk |

대표 B=8, pair 4에서 paper pair sum과 current pair mean은 32배 차이 났다.
lambda=1e-3의 같은-reduction reg/NLL은 20 ns 2.164%, 80 ns 2.015%다.
따라서 numeric lambda만으로 paper-equivalent strength라고 할 수 없다.

clean CFR lag-32 correlation은 20/80/120에서 0.993236/0.904606/0.830189였다.
full Psi는 이를 determinant와 Mahalanobis에 넣지만 diagonal은 marginal variance만
남긴다. diagonal은 위험 후보지만 단독 원인으로 확정하지 않는다.

## PART H. Supporting / Historical Experiments

### Empirical frequency covariance

current_baseline_diagnostic_ab.py의 run_experiment_b()가 clean CFR covariance와
lag correlation을 계산했다. 결과는
runs/current_valid_baseline/diagnostics/expB_diagonal_psi_covariance_diagnostic/다.

### Banded Psi

train_banded_covariance.py가 build_banded_cholesky()와
banded_multivariate_student_t_nll()로 bandwidth 32 L을 학습했다.
20/80 Aleatoric은 0.009979/0.009986으로 거의 같았고 Near/Far epistemic 및
sample calibration이 약해졌다. 따라서 diagonal sole-cause 가설은 약화된다.
결과는 runs/current_valid_baseline/covariance_experiments/banded_lag32_5k_10ep/다.

### Pilot / LS observation

requested_pilot_gate.py와 compare_observation_pipeline.py가 requested mask 이후
Hadamard pilot/LS와 direct CFR AWGN을 비교했다. white AWGN과 full-rank
orthogonal pilot에서는 effective additive Gaussian error가 되어 두 관측이 사실상
동등했다. 결과는 runs/current_valid_baseline/observation_experiments/다.

### Lambda / formulation history

runs/baseline_reproduction/step6_lambda_calibration/은 pre-fix lambda sweep이고
runs/ 아래 formulation으로 시작하는 폴더는 A/B/C/D 비교다. 모두 HISTORICAL / PRE-FIX이며 current valid
uncertainty baseline을 대체하지 않는다.

## PART I. 현재 종합 결론

| Baseline behavior | Status | Evidence |
| --- | --- | --- |
| CFR reconstruction learns | PASS | delay 증가와 1 ms에서 NMSE 악화 |
| 80 ns harder than 20 ns | PASS | interpolation과 Predictor |
| sample uncertainty tracks error | PASS after power normalization | ID Pearson 0.4409/0.6991 |
| ID-Hard Aleatoric increase | FAIL | normalized Figure 7 및 aggregation audit |
| Near-OOD Epistemic increase | weak / partial | gap이 거의 0 |
| Far-OOD Epistemic increase | PASS | 1 ms가 ID보다 높음 |

최신 원인 우선순위는 evidential loss/Psi-Nu coupling, power confounding의 일부
영향, diagonal covariance의 theoretical risk 순이다. aggregation/mask/indexing은
현재 audit에서 배제되었고 banded pilot은 diagonal sole-cause 가설을 강화하지
않았다. Full/Partial Fine-Tuning은 아직 시작하지 않는다.

## 다음 실험: Nu Compensation Counterfactual

Nu 민감도가 크다는 사실만으로 오류라 하지 않고 Psi와 Nu의 combined effect를
분리한다.

    A20       = Psi20/D20
    A_PsiOnly = Psi80/D20
    A_NuOnly  = Psi20/D80
    A80       = Psi80/D80
    D = nu-2K-1

Psi effect, Nu compensation effect, combined effect를 비교한다. 이 문서 작성에서
새 실험과 재학습은 수행하지 않았다.

## Metric / Quantity Glossary

| 이름 | 실제 정의 | reduction | 의미 |
| --- | --- | --- | --- |
| omitted NMSE dB | 10log10(sum_om(error^2)/sum_om(target^2)) | sample 후 dataset | 상대 복원오차 |
| aggregation pair error | sum_om(Re^2+Im^2)/K_om | 4 pair 평균 | subcarrier당 Re/Im 합 |
| gradient normalized error | sum_om(Re^2+Im^2)/1920 | real component 평균 | 위 값의 약 1/2 |
| Eq.12 map | mean_pair(Sigma[Re,k]+Sigma[Im,k]) | pair 평균, k별 | subcarrier uncertainty |
| Eq.13 score | mean omitted Eq.12 map | omitted 평균 | sample uncertainty |
| normalized Psi | Psi/a^2 | 실험별 pair/vector 평균 | variance scale 보정 |
| Psi pressure | -dL/d(rawPsi) | raw component/pair 요약 | Psi 증가 방향 |
| S_Ale_Psi/Nu | -(dA/drawParam)(dL/drawParam) | pair 요약 | local Aleatoric 기여 |

## Code Index

- Dataset: src/channel/sionna_channel.py -> generate_cfr_for_delay_spreads()
- Input/mask: src/training/data.py -> cfr_to_real_imag(),
  uniform_grouping_mask(), random_grouping_mask(), add_complex_awgn(),
  build_noisy_sparse_input(), linear_interpolate_real_imag()
- Model: src/models/uacp_predictor.py -> UACPEvidentialPredictor.forward(),
  ResidualBlock.forward()
- Loss: src/models/evidential.py -> diagonal_multivariate_student_t_nll(),
  pair_level_evidence_regularizer(), evidential_loss()
- Aggregation: src/training/uncertainty.py ->
  paper_subcarrier_uncertainty_map(), paper_omitted_uncertainty_score()
- Training: scripts/train_current_valid_baseline.py -> main(), pair_mapping_assert();
  scripts/train_predictor.py -> train_one_epoch(), evaluate()
- Figure 5~8: scripts/paper_uncertainty_diagnostics.py ->
  evaluate_samples(), run_p1_p2(), run_p3_p4()
- 원인 audit: audit_normalized_aleatoric_aggregation.py,
  psi_error_alignment_diagnostic.py, paired_power_matching.py,
  loss_psi_gradient_alignment.py, evidential_parameter_contribution.py,
  evidential_formula_semantic_audit.py

## Result Index

아래 표는 실제 결과 폴더와 대표 파일을 연결한 목록이다. `current_valid_baseline`은
앞으로의 진단에 사용하는 현재 기준이고, `baseline_reproduction`은 pair mapping
수정 전후 기록을 포함한 역사적 재현 실험이다.

| 실험 | 종류/분류 | 실행 script | 결과 폴더 | 대표 결과 파일 |
| --- | --- | --- | --- | --- |
| Current Valid Baseline | TRAINING / CURRENT VALID | scripts/train_current_valid_baseline.py | runs/current_valid_baseline/condition_c_5k_10ep_lambda1e3_corrected/ | summary.csv, final_results.json, provenance.json |
| Figure 5 P1/P2 | EVALUATION ONLY / CORE | scripts/paper_uncertainty_diagnostics.py | runs/paper_style_uncertainty_diagnostics_20260904/ | p1_summary.csv, p1_total_uncertainty_vs_nmse.png, p2_risk_coverage.csv |
| Figure 5 power normalization | POST-HOC / CORE | scripts/p1_power_normalization_diagnostic.py | runs/paper_style_uncertainty_diagnostics_20260904/p1_power_normalization/ | summary.csv, summary.md |
| Normalized Figure 7 sweep | EVALUATION ONLY / CORE | scripts/normalized_id_delay_sweep.py | runs/paper_style_uncertainty_diagnostics_20260904/p3_normalized_delay_sweep/ | normalized_delay_sweep.csv, normalized_fig7_delay_sweep.png |
| Eq.12 -> Eq.13 audit | READ-ONLY DIAGNOSTIC / CORE | scripts/audit_normalized_aleatoric_aggregation.py | runs/current_valid_baseline/diagnostics/normalized_aleatoric_aggregation_audit_20260904_v2/ | summary.csv, stage_values.csv, mask_audit.json |
| Psi-error alignment | READ-ONLY DIAGNOSTIC / CORE | scripts/psi_error_alignment_diagnostic.py | runs/current_valid_baseline/diagnostics/psi_error_alignment_20260904_v2/ | summary.csv, correlations.csv, sample_pair_metrics.csv |
| 1:1 power matching | POST-HOC / CORE | scripts/paired_power_matching.py | runs/current_valid_baseline/diagnostics/psi_error_alignment_20260904_v2/paired_power_matching/ | paired_statistics.csv, matched_pairs.csv |
| Loss -> Psi gradient | AUTOGRAD DIAGNOSTIC / CORE | scripts/loss_psi_gradient_alignment.py | runs/current_valid_baseline/diagnostics/loss_psi_gradient_alignment_20260904_v4/ | summary.csv, correlations.csv |
| Evidential contribution | AUTOGRAD DIAGNOSTIC / CORE | scripts/evidential_parameter_contribution.py | runs/current_valid_baseline/diagnostics/evidential_parameter_contribution_20260904_v6/ | summary.csv, gradient_summary.csv |
| Formula/semantic audit | READ-ONLY + TOY AUDIT / CORE | scripts/evidential_formula_semantic_audit.py | runs/current_valid_baseline/diagnostics/evidential_formula_semantic_audit_20260904_v3/ | equation_audit.json, reduction_gradient_comparison.csv |
| Empirical covariance | READ-ONLY DIAGNOSTIC / SUPPORTING | scripts/current_baseline_diagnostic_ab.py | runs/current_valid_baseline/diagnostics/expB_diagonal_psi_covariance_diagnostic/ | covariance_energy.csv, pair_averaged_lag_correlation.csv |
| Banded covariance pilot | TRAINING / HISTORICAL SUPPORTING | scripts/train_banded_covariance.py | runs/current_valid_baseline/covariance_experiments/banded_lag32_5k_10ep/ | final_results.json, learned_covariance_summary.csv |
| Requested-pilot LS audit | EVALUATION ONLY / SUPPORTING | scripts/requested_pilot_gate.py | runs/current_valid_baseline/observation_experiments/requested_subcarrier_hadamard_ls_15db/gate1_complete/ | sanity_comparison.json, gate1_summary.csv |
| Pre-fix STEP 1~8 | HISTORICAL / PRE-FIX | 각 단계별 reproduction script | runs/baseline_reproduction/ | 각 단계의 summary.csv, summary.md |

현재 기준 모델에 직접 연결되는 하위 진단은 다음 위치에서 찾을 수 있다.

    runs/current_valid_baseline/
    ├── condition_c_5k_10ep_lambda1e3_corrected/
    ├── diagnostics/
    │   ├── expA_sample_pair_evidence_calibration/
    │   ├── expB_diagonal_psi_covariance_diagnostic/
    │   ├── normalized_aleatoric_aggregation_audit_20260904_v2/
    │   ├── psi_error_alignment_20260904_v2/
    │   ├── loss_psi_gradient_alignment_20260904_v4/
    │   ├── evidential_parameter_contribution_20260904_v6/
    │   └── evidential_formula_semantic_audit_20260904_v3/
    ├── covariance_experiments/
    └── observation_experiments/

## 문서 사용 규칙

새로운 학습은 baseline 폴더를 덮어쓰지 않고 별도 폴더에 저장한다. checkpoint를
사용하는 진단은 결과 폴더의 config/provenance와 함께 기록한다. 숫자는 대표 CSV와
JSON을 우선 확인하고, 비슷한 이름의 metric은 Metric / Quantity Glossary의 reduction을
함께 읽는다. `baseline_reproduction` 및 `formulation_*` 결과는 현재 valid 결과와
섞지 않고 과거 비교 자료로만 사용한다. `formulation_` 또는 `prototype_` 접두사 폴더도
같은 원칙으로 다룬다.

## Historical / Archive 경계

runs/baseline_reproduction/ 및 runs/ 아래 formulation/prototype으로 시작하는 폴더는 초기
prototype, formulation 비교, pre-fix STEP 결과다. pre-fix STEP 6은 pair mapping bug 이전 결과이고,
STEP 7 rank-4와 banded pilot은 supporting evidence다. 모두 보존하되 CURRENT VALID
uncertainty 결과와 섞지 않는다.
