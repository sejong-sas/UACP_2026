# 스크립트 안내

위쪽부터 CORE를 먼저 읽습니다. `YES`는 현재 연구 이야기를 설명할 때 직접
사용하는 파일이고, 나머지는 보조 또는 과거 실험용입니다.

## CORE

| 스크립트 | 역할 | 현재 흐름 |
| --- | --- | --- |
| `train_current_valid_baseline.py` | 수정된 baseline을 학습하고 provenance를 저장 | YES |
| `paper_uncertainty_diagnostics.py` | Figure 5~8 목적의 read-only 평가 | YES |
| `p1_power_normalization_diagnostic.py` | CFR power 차이의 영향을 분리 | YES |
| `normalized_id_delay_sweep.py` | ID delay spread와 normalized uncertainty 비교 | YES |
| `audit_normalized_aleatoric_aggregation.py` | Eq.12 -> Eq.13과 omitted mask 검증 | YES |
| `psi_error_alignment_diagnostic.py` | Psi가 error와 power를 어떻게 따라가는지 확인 | YES |
| `paired_power_matching.py` | 20/80 ns를 비슷한 power로 1:1 비교 | YES |
| `loss_psi_gradient_alignment.py` | loss가 Psi를 키우는 방향인지 확인 | YES |
| `evidential_parameter_contribution.py` | Psi/Nu/Kappa의 uncertainty 기여 분리 | YES |
| `evidential_formula_semantic_audit.py` | 논문 식과 active 구현의 의미·gradient 비교 | YES |
| `current_baseline_diagnostics.py` | linear interpolation과 parameter trace 실행 | YES |
| `current_baseline_diagnostic_ab.py` | sample/pair calibration과 empirical covariance | YES |

## SUPPORTING

| 스크립트 | 역할 | 현재 흐름 |
| --- | --- | --- |
| `compare_banded_covariance.py` | diagonal과 banded 결과를 비교 | NO |
| `train_banded_covariance.py` | frequency-local covariance pilot 학습 | NO |
| `step7_covariance.py` | STEP 7 equation audit와 covariance 결과 정리 | NO |
| `compare_observation_pipeline.py` | direct AWGN과 pilot-LS 결과 비교 | NO |
| `requested_pilot_gate.py` | requested-subcarrier pilot equivalence gate | NO |
| `train_pilot_ls_observation.py` | pilot-LS observation treatment 학습 | NO |
| `audit_step6_condition_c.py` | historical STEP 6 설정을 read-only 검증 | NO |
| `baseline_reproduction.py` | STEP 1~8 staged experiment runner | NO |
| `step8_revalidation.py` | corrected lambda 결과 요약 | NO |
| `loss_psi_gradient_alignment.py` | loss pressure 상세 확인 | 보조 재사용 |
| `diagnose_predictor.py` | 공통 regime 진단 함수 제공 | 공통 모듈 |
| `train_predictor.py` | prototype 학습 공통 loop 제공 | 공통 모듈 |
| `generate_dataset.py` | dataset 생성 | 준비 단계 |
| `inspect_dataset.py` | dataset 구조와 분포 확인 | 준비 단계 |
| `sionna_smoke.py` | Sionna 설치·채널 smoke test | 준비 단계 |

## ARCHIVE / HISTORICAL

| 스크립트 | 역할 | 현재 흐름 |
| --- | --- | --- |
| `train_predictor.py`의 prototype 실행 경로 | 초기 element-wise prototype 학습 | 과거 참고 |
| `baseline_reproduction.py`의 STEP 1~8 명령 | formulation 수정 과정의 기록 | 과거 참고 |

스크립트 자체는 삭제하지 않았습니다. 위 표에서 CORE 파일을 먼저 보고, 특정
실험의 세부 구현이 필요할 때 SUPPORTING 파일로 내려가면 됩니다.
