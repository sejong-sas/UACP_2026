# 주요 기능 코드 조회표

## 모델 구조 어디에 있나?

- 파일: `src/models/uacp_predictor.py`
- 클래스: `UACPEvidentialPredictor`
- 기능 요약: 9채널 sparse 입력을 받아 32개 residual block을 통과시킨 뒤 CFR 평균과
  evidential parameter를 출력합니다.

## Psi는 어디서 나오나?

- 파일: `src/models/uacp_predictor.py`
- 함수/부분: `psi_head`
- 기능 요약: backbone feature에서 frequency별 raw 값을 만들고
  `constrain_pair_scalar_evidential`의 softplus를 거쳐 양수 Psi가 됩니다.
  현재 Psi는 논문의 full matrix가 아니라 diagonal 근사입니다.

## Kappa와 Nu는 어디서 나오나?

- 파일: `src/models/uacp_predictor.py`
- 부분: `pool`, `kappa_head`, `nu_head`
- 기능 요약: 전체 frequency feature를 pooling한 뒤 antenna pair마다 scalar 하나를
  출력합니다. shape은 `[B,4,1]`입니다.

## Aleatoric 계산은 어디에 있나?

- 파일: `src/models/evidential.py`
- 위치: `EvidentialOutput.aleatoric`
- 기능 요약: `Psi/(nu-2K-1)`로 계산합니다. 데이터 자체의 예측 어려움을 나타내는
  값으로 논문 Eq.7과 연결됩니다.

## Epistemic 계산은 어디에 있나?

- 파일: `src/models/evidential.py`
- 위치: `EvidentialOutput.epistemic`
- 기능 요약: Aleatoric을 Kappa로 나눕니다. evidence가 부족할 때 커지는 모델 불확실성으로
  논문 Eq.8과 연결됩니다.

## NLL은 논문 식의 어디인가?

- 파일: `src/models/evidential.py`
- 함수: `diagonal_multivariate_student_t_nll`
- 기능 요약: pair 하나의 2048차원 CFR를 하나의 multivariate Student-t로 보고
  자유도, scale, log determinant, Mahalanobis 항을 계산합니다.

## Regularizer는 왜 필요한가?

- 파일: `src/models/evidential.py`
- 함수: `pair_level_evidence_regularizer`
- 기능 요약: `||h-gamma||^2*(kappa+nu)`를 사용해 많이 틀린 예측에 높은 evidence를
  주지 않도록 합니다. 논문 Eq.11과 연결됩니다.

## Eq.12와 Eq.13은 어디에서 계산하나?

- 파일: `src/training/uncertainty.py`
- 함수: `paper_subcarrier_uncertainty_map`, `paper_omitted_uncertainty_score`
- 기능 요약: pair별 Real/Imag trace를 subcarrier score로 만들고, 마지막에는
  보고하지 않은 omitted 위치만 평균합니다.

## 20/80/120 ns 비교 코드는?

- 파일: `scripts/paper_uncertainty_diagnostics.py`
- 함수: `evaluate_samples`
- 기능 요약: 같은 checkpoint와 `Ng=16` 조건으로 각 regime을 평가하고 NMSE,
  uncertainty, correlation을 저장합니다.

## Power normalization은 어디에서 했나?

- 파일: `scripts/p1_power_normalization_diagnostic.py`
- 기능 요약: sample별 clean CFR RMS power로 CFR와 prediction을 나누고, covariance는
  분산이므로 power scale의 제곱으로 나눠 비교합니다.

## Figure 7 delay sweep은?

- 파일: `scripts/normalized_id_delay_sweep.py`
- 기능 요약: 10~100 ns ID 구간을 sweep하면서 NMSE와 normalized Aleatoric/Epistemic curve를 만듭니다.

## aggregation이 맞는지는 어떻게 확인했나?

- 파일: `scripts/audit_normalized_aleatoric_aggregation.py`
- 기능 요약: pair -> Real/Imag trace -> pair 평균 -> omitted 평균의 각 단계를 raw CSV로
  저장해 mask와 target alignment를 검사했습니다.

## 왜 Psi 문제를 의심하나?

- 파일: `scripts/psi_error_alignment_diagnostic.py`,
  `scripts/paired_power_matching.py`
- 기능 요약: power를 맞춘 뒤에도 80 ns error는 더 컸지만 normalized Psi가 더 커지지
  않았습니다. 따라서 단순 CFR power 문제만은 아닙니다.

## Loss가 Psi를 키우는지 어떻게 확인했나?

- 파일: `scripts/loss_psi_gradient_alignment.py`
- 기능 요약: optimizer 없이 autograd로 NLL과 regularizer gradient를 분리했습니다.
  Eq.11 regularizer의 raw Psi gradient가 0임을 확인했습니다.

## Nu가 영향을 주는지 어떻게 확인했나?

- 파일: `scripts/evidential_parameter_contribution.py`
- 기능 요약: Psi, Nu, Kappa를 통한 Aleatoric 변화량을 local contribution으로 나눴고,
  Nu가 중요한 경로임을 확인했습니다.

## 논문 식과 구현이 맞는지는?

- 파일: `scripts/evidential_formula_semantic_audit.py`
- 결과: `runs/current_valid_baseline/diagnostics/evidential_formula_semantic_audit_20260904_v3/`
- 기능 요약: active diagonal Student-t 경로는 식과 맞지만, full Psi를 diagonal로 줄인
  점과 mean reduction은 구현상의 가정입니다.

## 현재 기준 checkpoint는?

- 파일: `runs/current_valid_baseline/condition_c_5k_10ep_lambda1e3_corrected/checkpoint_with_provenance.pt`
- 기능 요약: 이후 진단은 이 checkpoint를 읽기만 하며, historical pre-fix STEP 6 checkpoint와
  구분합니다.
