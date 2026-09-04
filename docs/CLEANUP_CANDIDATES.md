# 정리 후보 목록

이 문서는 삭제 허가 목록이 아닙니다. 실제 파일과 폴더는 그대로 두었고,
나중에 연구 기록을 정리할 때 검토할 후보만 적었습니다.

| 경로 | 분류 | 불필요해 보이는 이유 | 현재 참조 | 나중 삭제 가능성 |
| --- | --- | --- | --- | --- |
| `runs/prototype_predictor/` | 과거 실험 | 초기 element-wise prototype 결과 | README와 과거 명령에서 참조 | 낮음, 역사 설명에 필요 |
| `runs/prototype_predictor_repro/` | 중복 가능성 | prototype 재현성 확인 결과 | 과거 기록에서 참조 | 중간 |
| `runs/formulation_b_pair_scalar/` | 과거 실험 | corrected baseline 이전 formulation | formulation 비교 기록에서 참조 | 낮음 |
| `runs/formulation_c_diag_mvnll/` | 과거 실험 | 현재 valid baseline 이전 checkpoint | STEP 기록에서 참조 | 낮음 |
| `runs/formulation_d_pair_reg/` | 과거 실험 | pair mapping bug 이전 결과 | historical 비교에 참조 | 낮음 |
| `runs/formulation_*_diagnostics/` | 중복 가능성 | 각 formulation의 상세 보조 결과가 많음 | README의 과거 단계에서 참조 | 중간 |
| `runs/baseline_reproduction/step1_ood_far_repro/` | 중복 가능성 | 동일 조건 재현 run | 재현성 기록에서 참조 | 중간 |
| `runs/baseline_reproduction/step2_delay_sweep_repro/` | 중복 가능성 | 동일 조건 재현 run | 재현성 기록에서 참조 | 중간 |
| `runs/baseline_reproduction/step3_uncertainty_aggregation_repro/` | 중복 가능성 | aggregation 재현 run | 과거 기록에서 참조 | 중간 |
| `runs/current_valid_baseline/diagnostics/evidential_formula_semantic_audit_20260904/` | 중복 가능성 | audit 중간 버전 | v2/v3가 후속 보정판 | 높음, 단 provenance 확인 후 |
| `runs/current_valid_baseline/diagnostics/evidential_formula_semantic_audit_20260904_v2/` | 중복 가능성 | audit 중간 버전 | v3가 최종 보정판 | 높음, 단 provenance 확인 후 |
| `scripts/baseline_reproduction.py` | 목적 혼합 | 여러 STEP 실행 기능이 한 파일에 있음 | README 명령에서 직접 참조 | 낮음, 기능 분리 후 |
| `scripts/train_predictor.py` | 공통 모듈/과거 경로 | prototype 학습과 공통 함수가 섞임 | baseline script가 import | 없음, 지금 삭제 불가 |
| `src/models/evidential_reference.py` | 보조 코드 | 작은 차원 reference 전용 | equation test와 audit에서 사용 | 중간 |
| `src/models/__pycache__/`, `scripts/__pycache__/` | 임시 파일 | Python 실행이 만든 캐시 | 소스 참조 없음 | 높음, 환경 재생성 가능 |
| `data/prototype_repro/` | 재현성 데이터 | prototype dataset 복제본 | repro 결과가 참조 | 중간 |

현재 사용 여부는 `rg` 검색과 README/스크립트 import를 기준으로 적었습니다.
특히 checkpoint와 결과 폴더는 삭제 전에 README, provenance, 재현 명령을 먼저
확인해야 합니다.
