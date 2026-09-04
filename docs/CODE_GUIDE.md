# 코드 구조 가이드

코드 구조를 확인할 때 아래 순서로 이동하면 됩니다.

## A. CFR 데이터 생성

- 파일: `src/channel/sionna_channel.py`
- 함수: 채널과 CFR 생성 함수
- 하는 일: Sionna 채널 모델에서 주파수별 복소 CFR을 만듭니다.
- 한 문장 설명: “먼저 각 sample의 실제 채널 응답을 만듭니다.”

데이터 저장과 작은 dataset 생성은 `scripts/generate_dataset.py`의
`save_split_npz`와 `load_dataset_config`에서 확인합니다.

## B. Sparse observation과 mask

- 파일: `src/training/data.py`
- 함수: `uniform_grouping_mask`, `random_grouping_mask`, `build_sparse_input`,
  `build_noisy_sparse_input`
- 입력: 복소 CFR `[B,K,2,2]`, subcarrier mask
- 출력: CFR의 Real/Imag 8채널과 mask 1채널, 총 `[B,9,K]`
- 하는 일: 보고된 위치만 남기고 나머지는 0으로 채운 뒤 mask를 함께 넣습니다.
- 한 문장 설명: “모델이 어느 주파수를 보았는지 알 수 있도록 값과 mask를 같이 줍니다.”

현재 baseline의 관측 noise는 보고된 CFR에만 sample-wise 15 dB AWGN을 넣는
`build_noisy_sparse_input`에 있습니다. 정확한 논문 noise 위치는 공개되지 않아
`IMPLEMENTATION-ASSUMPTION`입니다.

## C. Predictor 구조

- 파일: `src/models/uacp_predictor.py`
- 클래스: `UACPEvidentialPredictor`
- 입력 shape: `[B,9,1024]`
- 출력: `gamma`, `Psi` 계열은 `[B,8,1024]`, `kappa`, `nu`는 pair scalar
  `[B,4,1]`
- 하는 일: 1개 입력층과 32개 `ResidualBlock`을 통과시켜 CFR 평균과 uncertainty
  parameter를 한 번에 출력합니다.
- 한 문장 설명: “Sparse CFR와 mask를 받아 전체 CFR와 uncertainty를 동시에 예측합니다.”

## D. Evidential parameter 생성

- 파일: `src/models/evidential.py`
- 함수: `constrain_pair_scalar_evidential`, `channels_to_pair_vectors`
- `gamma`: 각 antenna pair의 Real/Imag CFR 예측 평균
- `Psi`: 데이터 자체의 변동을 나타내는 covariance scale; 현재는 diagonal 근사
- `kappa`: 평균 예측을 얼마나 믿는지 나타내는 pair별 scalar
- `nu`: covariance 추정에 대한 evidence와 자유도 조건을 나타내는 pair별 scalar

채널 순서는 `[Re(pair0..3), Im(pair0..3)]`이고, pair는 `(0,4)`, `(1,5)`,
`(2,6)`, `(3,7)`입니다. `Psi=L L^T` full matrix는 논문 식이고 현재 baseline의
diagonal `Psi`는 `APPROXIMATION`입니다.

## E. Student-t NLL

- 파일: `src/models/evidential.py`
- 함수: `diagonal_multivariate_student_t_nll`, `evidential_loss`
- 하는 일: 한 antenna pair의 2048차원 CFR 전체를 하나의 multivariate Student-t
  분포로 보고 예측 가능성을 계산합니다.
- 논문 연결: `degrees=nu-d+1`, `scale_diag=((kappa+1)/(kappa*degrees))*Psi`
- 한 문장 설명: “예측 평균뿐 아니라 예측 분포의 크기까지 target CFR에 맞추도록 학습합니다.”

## F. Evidence regularizer

- 파일: `src/models/evidential.py`
- 함수: `pair_level_evidence_regularizer`
- 논문 연결: Eq.11의 `||h-gamma||^2 * (kappa+nu)`
- 하는 일: 틀린 예측에 높은 evidence를 주지 않도록 전체 pair error와 evidence를
  함께 벌점으로 줍니다.
- 한 문장 설명: “많이 틀린데 자신만만한 출력을 줄이는 항입니다.”

## G. Aleatoric과 Epistemic 계산

- 파일: `src/models/evidential.py`
- 위치: `EvidentialOutput.aleatoric`, `EvidentialOutput.epistemic`
- Eq.7: `Sigma_ale = Psi/(nu-2K-1)`
- Eq.8: `Sigma_epi = Sigma_ale/kappa`
- 한 문장 설명: “데이터 자체의 어려움과 모델이 낯선 상황을 두 값으로 나눕니다.”

## H. Eq.12 / Eq.13 score

- 파일: `src/training/uncertainty.py`
- 함수: `paper_subcarrier_uncertainty_map`, `paper_omitted_uncertainty_score`
- 하는 일: pair별 covariance에서 같은 subcarrier의 Real/Imag trace를 더하고,
  4개 pair를 평균한 뒤, omitted 위치만 다시 평균합니다.
- 한 문장 설명: “다음에 보고하지 않은 주파수에서 얼마나 위험한지를 계산합니다.”

## I. Training loop

- 파일: `scripts/train_predictor.py`, `scripts/train_current_valid_baseline.py`
- 함수: `train_one_epoch`, `evaluate`, `main`
- 흐름: input 생성 -> predictor forward -> NLL와 regularizer 계산 -> backward ->
  `optimizer.step()`
- 한 문장 설명: “예측하고, 틀린 정도를 계산하고, 그 차이를 줄이는 방향으로 weight를 업데이트합니다.”

## J. Evaluation

- 파일: `scripts/paper_uncertainty_diagnostics.py`, `scripts/current_baseline_diagnostics.py`
- 함수: `evaluate_samples`, `diagnose_regime`, `paper_omitted_uncertainty_score`
- 흐름: 같은 checkpoint로 20/80/120 ns와 1 ms를 각각 평가하고 NMSE, parameter,
  uncertainty, correlation을 저장합니다.
- 한 문장 설명: “학습은 고정하고 환경 난이도만 바꾸어 error와 uncertainty가 함께 움직이는지 봅니다.”

## 우선 확인할 파일

1. `scripts/train_current_valid_baseline.py`
2. `src/models/uacp_predictor.py`
3. `src/models/evidential.py`
4. `src/training/data.py`
5. `src/training/uncertainty.py`
6. `scripts/paper_uncertainty_diagnostics.py`
7. `scripts/evidential_formula_semantic_audit.py`
