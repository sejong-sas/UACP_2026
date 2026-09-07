# 결과 폴더 안내

결과 파일은 실험 재현에 필요하므로 이동하거나 삭제하지 않았습니다. 아래 분류는
기능 요약 순서만 정한 것입니다.

## 현재 연구 흐름

```text
runs/current_valid_baseline/
├── condition_c_5k_10ep_lambda1e3_corrected/   현재 기준 checkpoint
├── diagnostics/
│   ├── exp1_linear_interpolation_no_predictor/
│   ├── exp2_evidential_parameter_trace/
│   ├── expA_sample_pair_evidence_calibration/
│   ├── expB_diagonal_psi_covariance_diagnostic/
│   ├── psi_error_alignment_20260904_v2/
│   ├── loss_psi_gradient_alignment_20260904_v4/
│   ├── evidential_parameter_contribution_20260904_v6/
│   └── evidential_formula_semantic_audit_20260904_v3/
├── covariance_experiments/banded_lag32_5k_10ep/
└── observation_experiments/
```

## CORE

| 결과 폴더 | 대응 스크립트 | 핵심 결과 |
| --- | --- | --- |
| `current_valid_baseline/condition_c_5k_10ep_lambda1e3_corrected` | `train_current_valid_baseline.py` | 수정 mapping으로 만든 기준 모델 |
| `paper_style_uncertainty_diagnostics_20260904` | `paper_uncertainty_diagnostics.py` | Figure 5~8 목적의 raw sample 평가 |
| `current_valid_baseline/diagnostics/exp1_linear_interpolation_no_predictor` | `current_baseline_diagnostics.py` | dataset 난이도 정상 확인 |
| `current_valid_baseline/diagnostics/exp2_evidential_parameter_trace` | `current_baseline_diagnostics.py` | gamma/Psi/Nu/Kappa 경로 추적 |
| `current_valid_baseline/diagnostics/normalized_aleatoric_aggregation_audit_20260904_v2` | `audit_normalized_aleatoric_aggregation.py` | mask와 Eq.12/13 정상 |
| `current_valid_baseline/diagnostics/psi_error_alignment_20260904_v2` | `psi_error_alignment_diagnostic.py` | power matching 후에도 Psi 정렬 부족 |
| `current_valid_baseline/diagnostics/loss_psi_gradient_alignment_20260904_v4` | `loss_psi_gradient_alignment.py` | regularizer의 Psi gradient 0 |
| `current_valid_baseline/diagnostics/evidential_parameter_contribution_20260904_v6` | `evidential_parameter_contribution.py` | Nu/Psi uncertainty 기여 비교 |
| `current_valid_baseline/diagnostics/evidential_formula_semantic_audit_20260904_v3` | `evidential_formula_semantic_audit.py` | 식은 맞지만 coupling/diagonal risk 확인 |

## SUPPORTING

| 결과 폴더 | 대응 스크립트 | 기능 요약 |
| --- | --- | --- |
| `current_valid_baseline/diagnostics/expB_diagonal_psi_covariance_diagnostic` | `current_baseline_diagnostic_ab.py` | 실제 CFR frequency correlation 차이 |
| `current_valid_baseline/covariance_experiments/banded_lag32_5k_10ep` | `train_banded_covariance.py` | banded covariance pilot 실패 참고 |
| `current_valid_baseline/observation_experiments/` | `train_pilot_ls_observation.py`, `requested_pilot_gate.py` | observation pipeline 보조 검증 |
| `baseline_reproduction/step6_lambda_calibration` | `baseline_reproduction.py` | pre-fix lambda sweep |
| `baseline_reproduction/step6_condition_c_audit` | `audit_step6_condition_c.py` | historical Condition C audit |

## ARCHIVE / HISTORICAL

| 결과 폴더 | 기능 요약 |
| --- | --- |
| `prototype_predictor`, `prototype_predictor_repro`, `prototype_diagnostics` | 최초 작은 prototype과 재현성 결과 |
| `formulation_b_pair_scalar`, `formulation_c_diag_mvnll`, `formulation_d_pair_reg` | A/B/C/D formulation 비교 |
| `formulation_*_diagnostics`, `formulation_comparison` | 위 비교의 상세 diagnostic |
| `baseline_reproduction/step1_ood_far` ~ `step8_corrected_revalidation` | 단계별 baseline reproduction 기록 |

주의: pre-fix STEP 6와 formulation A~D는 현재 valid uncertainty baseline이
아닙니다. 다만 pair mapping bug가 결과에 미친 영향을 기능 요약할 때 필요하므로
역사적 참고자료로 보존합니다.
