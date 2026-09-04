# UACP 실험 전체 흐름

이 문서는 현재까지의 실험 순서와 결과를 설명할 때 사용하는 목차입니다.
실제 결과 파일은 그대로 두고, 여기서는 무엇을 먼저 확인할지와
각 결과가 다음 질문으로 어떻게 이어졌는지만 정리합니다.

## 0. 현재 Valid Baseline

**실험 질문:** 수정된 antenna-pair mapping으로 만든 기준 모델이 있는가?

**왜 필요한가:** 과거 STEP 6 결과는 pair mapping 수정 전 checkpoint라서
uncertainty 비교의 기준으로 사용할 수 없습니다. 현재 baseline은 수정된
코드로 다시 학습했고, 이후 모든 진단이 같은 checkpoint를 읽습니다.

- 실행 스크립트: `scripts/train_current_valid_baseline.py`
- 핵심 함수: `train_one_epoch`, `evaluate`, `diagnose_regime`
- 입력: CFR 데이터, sparse mask, reported CFR에 대한 15 dB AWGN
- 출력: Predictor의 `gamma`, `Psi`, `kappa`, `nu`, reconstruction과 uncertainty
- checkpoint: `runs/current_valid_baseline/condition_c_5k_10ep_lambda1e3_corrected/checkpoint_with_provenance.pt`
- 결과: `runs/current_valid_baseline/condition_c_5k_10ep_lambda1e3_corrected/`
- 현재 판단: 이후 진단의 **CURRENT VALID BASELINE**

핵심 설정은 5,000 samples, 10 epochs, `Ng=16`, seed `20260819`입니다.
TDL-A, 직접 AWGN 위치, 5k/10 epoch는 논문에 모두 공개된 설정이 아니므로
`IMPLEMENTATION-ASSUMPTION`으로 설명합니다.

## 1. Figure 5: uncertainty와 실제 error 관계

**실험 질문:** uncertainty가 큰 sample에서 실제 reconstruction error도 큰가?

- 실행 스크립트: `scripts/paper_uncertainty_diagnostics.py`
- 핵심 함수: sample-level NMSE 계산, Eq.12 -> Eq.13 score, correlation
- 입력: current valid checkpoint와 20/80/120 ns/1 ms probe
- 출력: sample별 NMSE·Aleatoric·Epistemic, scatter, Pearson/Spearman
- 결과: `runs/paper_style_uncertainty_diagnostics_20260904/`
- 핵심 결과: raw Total uncertainty와 NMSE의 관계는 음수였고, power 정규화 후
  관계가 개선되었습니다.
- 해석: uncertainty가 error proxy로 완전히 실패한 것은 아니지만 CFR power가
  결과를 크게 섞고 있었습니다.
- 다음 이유: power scale을 제거한 뒤에도 남는 문제를 분리해야 했습니다.

## 2. CFR power 정규화

**실험 질문:** Figure 5의 반대 상관이 sample별 CFR power 차이 때문인가?

- 실행 스크립트: `scripts/p1_power_normalization_diagnostic.py`
- 핵심 함수: sample RMS power 계산, `H/gamma`의 `/a`, covariance의 `/a^2`
- 입력: 기존 sample probe와 current valid checkpoint
- 출력: 정규화 전·후 correlation과 scatter
- 결과: `runs/paper_style_uncertainty_diagnostics_20260904/p1_power_normalization/`
- 핵심 결과: correlation이 개선되었지만 모든 uncertainty ordering이 복구되지는
  않았습니다.
- 해석: power variation은 실제 원인이지만 유일한 원인은 아닙니다.

## 3. Figure 7: ID delay-spread sweep

**실험 질문:** training range 안에서 delay spread가 커질 때 Aleatoric은 증가하고
Epistemic은 안정적인가?

- 실행 스크립트: `scripts/normalized_id_delay_sweep.py`
- 핵심 함수: `run_sweep`, sample-wise covariance 정규화, Eq.12/13 aggregation
- 입력: 10/20/40/60/80/100 ns probe와 baseline checkpoint
- 출력: NMSE·normalized Aleatoric·normalized Epistemic curve
- 결과: `runs/paper_style_uncertainty_diagnostics_20260904/p3_normalized_delay_sweep/`
- 핵심 결과: NMSE는 대체로 악화했지만 Aleatoric은 flat/불안정했습니다.
- 해석: predictor는 난이도를 보지만 uncertainty 의미 분리는 약합니다.

## 4. Eq.12 -> Eq.13 aggregation audit

**실험 질문:** uncertainty ordering 실패가 mask 또는 omitted aggregation의 오류인가?

- 실행 스크립트: `scripts/audit_normalized_aleatoric_aggregation.py`
- 핵심 함수: `paper_subcarrier_uncertainty_map`, `paper_omitted_uncertainty_score`
- 입력: 20/80/120/1 ms probe, mask, pair covariance
- 출력: pair·subcarrier·sample 단계별 값과 mask audit
- 결과: `runs/current_valid_baseline/diagnostics/normalized_aleatoric_aggregation_audit_20260904_v2/`
- 핵심 결과: observed/omitted 구분과 Eq.12/13 계산은 정상입니다.
- 해석: 최종 평균을 잘못 계산해서 생긴 문제는 아닙니다.

## 5. Psi와 error의 정렬

**실험 질문:** learned Psi가 실제 error나 CFR power를 제대로 반영하는가?

- 실행 스크립트: `scripts/psi_error_alignment_diagnostic.py`
- 핵심 함수: `sample_pair_metrics`, power-bin 비교
- 결과: `runs/current_valid_baseline/diagnostics/psi_error_alignment_20260904_v2/`
- 추가 결과: `paired_power_matching/`
- 핵심 결과: power를 맞춘 뒤에도 80 ns의 normalized error가 더 크지만 Psi와
  Aleatoric은 일관되게 더 커지지 않았습니다.
- 다음 이유: sample power가 아닌 evidential parameter 학습을 직접 확인했습니다.

## 6. 1:1 paired-power matching

**실험 질문:** CFR power를 거의 같게 맞춰도 80 ns가 더 어려운가?

- 실행 스크립트: `scripts/paired_power_matching.py`
- 결과: Psi/Aleatoric separation은 여전히 안정적으로 회복되지 않았습니다.
- 해석: power confounding만으로는 설명이 부족합니다.

## 7. Loss -> Psi gradient audit

**실험 질문:** 어려운 sample이 Psi를 키우는 학습 신호를 받는가?

- 실행 스크립트: `scripts/loss_psi_gradient_alignment.py`
- 핵심 함수: `torch.autograd.grad`, NLL·regularizer 분리
- 결과: `runs/current_valid_baseline/diagnostics/loss_psi_gradient_alignment_20260904_v4/`
- 핵심 결과: regularizer의 raw Psi gradient는 정확히 0입니다. direct Psi pressure는
  NLL에서만 나오며 평균 pressure는 음수였습니다.
- 해석: objective가 harder sample을 Psi 증가로 직접 연결하는 경로가 약합니다.

## 8. Evidential parameter contribution

**실험 질문:** Aleatoric 조절이 Psi, Nu, Kappa 중 어느 경로로 일어나는가?

- 실행 스크립트: `scripts/evidential_parameter_contribution.py`
- 결과: `runs/current_valid_baseline/diagnostics/evidential_parameter_contribution_20260904_v6/`
- 핵심 결과: Nu 경로가 Psi보다 더 큰 상대 영향력을 보였고, total contribution은
  평균적으로 여전히 음수였습니다.
- 해석: parameter coupling이 중요한 후보입니다.

## 9. Formula / Semantic Consistency Audit

**실험 질문:** 식이 맞더라도 reduction·parameterization이 논문의 의미를 바꾸는가?

- 실행 스크립트: `scripts/evidential_formula_semantic_audit.py`
- 핵심 함수: Eq.5 active NLL, Eq.7/8 sensitivity, sum/mean gradient 비교
- 결과: `runs/current_valid_baseline/diagnostics/evidential_formula_semantic_audit_20260904_v3/`
- 핵심 결과: active diagonal 식은 algebraically 맞지만, regularizer는 Psi에 직접
  gradient를 주지 않고 Nu sensitivity가 훨씬 큽니다.
- 해석: 명백한 active 수식 bug보다는 parameter coupling과 diagonal Psi가 남은
  핵심 원인 후보입니다.

## Supporting 실험

empirical covariance, pilot observation, banded covariance, lambda sweep,
formulation B/C/D와 각 단계의 smoke test는 핵심 결론의 근거를 보강합니다.
상세 위치는 `runs/README.md`에서 찾을 수 있습니다.

## Archive / Historical 실험

`runs/baseline_reproduction/`, `runs/formulation_*`, `runs/prototype_*`는 초기
prototype, formulation 비교, STEP 1~8의 역사적 결과입니다. 삭제하지 않고
과거 판단과 bug 영향 추적에만 사용합니다. 특히 pre-fix STEP 6은 현재 uncertainty
baseline으로 사용하지 않습니다.
