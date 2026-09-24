
## 6. Training Scale 확대를 통한 Baseline 재검증

기존 보고서는 2026-09-18 시점의 Sections 1–3과 batch400 비교까지를 기록하고, 이후 Fig.11 및 후속 진단은 추가 예정으로 남겼다. 아래 Sections 6–15는 그 이후 실제로 저장된 결과를 근거로 이어서 정리한 최신 내용이다.

앞선 소규모 학습에서는 reconstruction 자체는 개선되었지만, ID 내부의 난이도와 OOD를 구분하는 Aleatoric/Epistemic behavior가 원 논문에서 기대한 형태와 충분히 유사하지 않았다. 원 논문이 100,000 training samples를 사용하므로, 이번 단계에서는 sample scale을 100k로 맞추고 batch8 FP32 조건에서 먼저 학습량의 영향을 분리해 확인했다. 이번 절의 중심 비교는 batch400이 아니라 `100k × 1 epoch / batch8`과 `100k × 5 epoch / batch8`이다.

현재 구현의 공통 흐름은 다음과 같다. `src/channel/sionna_channel.py`가 Sionna에서 clean CFR을 만들고, `scripts/generate_dataset.py`가 dataset으로 저장한다. `src/training/data.py`는 sparse feedback mask를 만들고 관측 subcarrier에 15 dB sample-wise complex AWGN을 넣은 뒤, omitted 위치를 0으로 만든 predictor input을 구성한다. `src/models/uacp_predictor.py`는 CFR reconstruction mean과 evidential parameter인 γ, κ, Ψ, ν를 출력하며, `src/models/evidential.py`가 Student-t NLL과 evidence regularizer를 계산한다. 학습과 checkpoint 저장은 `scripts/train_predictor.py` 및 `scripts/train_100k5_convergence.py`가 담당하고, Fig.8 평가는 `scripts/evaluate_100k_fig8.py`와 epoch checkpoint 평가 스크립트가 수행한다. 이 noise stage와 diagonal Ψ는 기존 보고서와 동일하게 `IMPLEMENTATION-ASSUMPTION` 또는 `APPROXIMATION`으로 해석한다.

### 6.1 100k × 1 epoch / batch8 — 안정적인 기준 Baseline 확보

#### 실험 목적 및 설계

먼저 논문과 같은 100k sample scale에서 한 번의 epoch만 학습했다. 처음부터 150 epochs를 장시간 실행하기보다, 1 epoch에서 reconstruction이 실제로 학습되는지와 evidential parameter 계산이 수치적으로 안정적인지를 확인하는 것이 목적이다. 학습은 FP32, batch8, Adam, learning rate `1e-4`, `lambda_reg=1e-3` 조건으로 수행되었다.

#### 결과

| Regime | NMSE omitted (dB) | Epistemic | Fig.8 평가 |
|---|---:|---:|---:|
| ID-Easy 20 ns | -18.975 | 0.000596 | finite |
| ID-Hard 80 ns | -17.851 | 0.000598 | finite |
| OOD-Near 120 ns | -15.142 | 0.000613 | finite |
| OOD-Far 1 ms | 1.436 | 0.001206 | finite |

Fig.8 AUROC는 Near `0.6204`, Far `0.9970`, pooled OOD `0.8087`이었다. 모든 주요 regime에서 output과 uncertainty가 finite였고, 20 ns보다 80 ns에서 reconstruction이 어려워지며 1 ms에서 Epistemic이 증가했다. 따라서 이 run은 최고 성능 모델이라기보다 NaN/inf 없이 동작하는 control baseline으로 적합하다. 이후 학습량을 늘렸을 때 separation이 좋아지는지, 그리고 그 개선이 수치 불안정성과 교환되는지를 비교할 수 있는 기준점이 되었다.

### 6.2 100k × 5 epoch / batch8 — Separation 향상과 ν Instability

#### 실험 목적 및 설계

100k training samples는 유지하고 epoch만 1에서 5로 늘렸다. `scripts/train_100k5_convergence.py`는 epoch 수 외의 주요 조건이 바뀌지 않았는지 확인하고, 각 epoch checkpoint를 저장한다. 이후 `scripts/evaluate_100k5_epoch_checkpoints.py`가 동일한 common evaluation set, uniform `Ng=16` mask, direct-CFR 15 dB AWGN 조건으로 epoch별 Fig.8 결과를 계산했다.

#### Epoch별 변화

| Epoch | 20 ns NMSE | 80 ns NMSE | 120 ns NMSE | 1 ms NMSE | Near AUROC | Far AUROC | Stability |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | -18.975 | -17.851 | -15.142 | 1.436 | 0.6163 | 0.9964 | finite |
| 2 | -20.135 | -18.333 | -16.207 | 1.823 | 0.7821 | 0.9999 | finite |
| 3 | -20.873 | -18.795 | -16.854 | 1.658 | 0.9364 | 1.0000 | finite |
| 4 | -21.370 | -19.299 | -16.877 | 1.623 | 0.9522 | N/A | Far non-finite |
| 5 | -21.800 | -19.472 | -16.463 | 1.504 | 0.9484 | N/A | Far non-finite |

epoch가 증가하면서 reconstruction NMSE는 전반적으로 낮아졌고, Near-OOD separation은 epoch1의 `0.6163`에서 epoch3의 `0.9364`로 크게 향상되었다. epoch3은 Far-OOD에서도 `0.9999985`의 separation을 보였다. 그러나 epoch4와 epoch5에서는 Far-OOD의 전체 score에 non-finite 값이 섞여 공식 AUROC를 계산할 수 없었다. 따라서 이 결과를 “100k×5 실패”라고 단순화하는 것은 적절하지 않다. 학습량 증가가 reconstruction과 Near-OOD separation을 실제로 개선했지만, 후반 epoch에서 evidential numerical stability가 무너지는 trade-off가 발견된 것이다.

<div style="page-break-after: always;"></div>

## 7. Late Epoch에서 발생한 ν Margin Collapse

후반 epoch의 문제는 ν 자체가 단순히 `inf`가 된 현상으로 보기 어렵다. 현재 구현에서 `K=1024`일 때 Aleatoric은 다음 denominator를 사용한다.

`Aleatoric = Ψ / [ν − (2K+1)]`

즉 중요한 값은 ν 그 자체가 아니라 `ν - (2K+1)`인 ν margin이다. 이 margin이 0에 가까워지면 같은 Ψ라도 Aleatoric이 급격히 커지고, Epistemic도 함께 폭증한다.

| Epoch | Far-OOD ν margin median | Far-OOD ν margin minimum | Epistemic score median | Non-finite score |
|---:|---:|---:|---:|---:|
| 3 | 13.1543 | 0.0659 | 47.7274 | 0 / 2,000 |
| 4 | 0.0278 | 0.0000 | 1,702.5169 | 732 / 2,000 |
| 5 | 0.0000 | 0.0000 | 6,503.6699 | 1,347 / 2,000 |

변화의 흐름은 `epoch 증가 → Far-OOD ν margin 감소 → denominator가 0에 접근 → Aleatoric 폭증 → Epistemic inf → Far-OOD AUROC 계산 불가`이다. 따라서 “ν가 inf가 되었다”는 표현보다 “ν margin collapse가 uncertainty 계산을 불안정하게 만들었다”고 쓰는 것이 정확하다. 성능 개선과 수치 안정성 사이의 이 trade-off 때문에, 이후 baseline 개선은 reconstruction만 낮추는 방향이 아니라 ν margin을 보존하는 방향이어야 한다.

## 8. Epoch3를 현재 Static Baseline 후보로 선정

epoch3는 학습량 증가에 따른 reconstruction 개선과 Near-OOD separation을 확보하면서도, 주요 Fig.8 uncertainty가 모두 finite한 마지막 안정 구간이다. 저장된 epoch3 static validation에서 주요 수치는 다음과 같다.

| Regime | NMSE omitted (dB) | Aleatoric median | Epistemic median |
|---|---:|---:|---:|
| ID-Easy 20 ns | -20.947 | 0.003929 | 0.000347 |
| ID-Hard 80 ns | -18.790 | 0.005024 | 0.000635 |
| OOD-Near 120 ns | -16.903 | 0.005539 | 0.000972 |
| OOD-Far 1 ms | 1.685 | 0.016604 | 22.232 |

ID-Easy, ID-Hard, Near, Far의 static Fig.8 AUROC는 각각 `ID-pooled→Near=0.9363`, `ID-pooled→Far=1.0000`, pooled OOD `0.9681`이었다. 80 ns는 20 ns보다 NMSE와 Aleatoric이 커서 ID 내부의 난이도 증가가 확인되며, Far-OOD는 Epistemic이 크게 증가한다. 이 때문에 epoch3 checkpoint를 이후 dynamic controller 진단과 adaptation 실험의 공통 starting checkpoint로 사용했다.

![Epoch3 static Fig.8 Epistemic density](../runs/current_valid_baseline/epoch3_baseline_validation_20260919/static_fig8_fig9/fig8_epistemic_density.png)

그림 8. Epoch3 checkpoint의 static Fig.8 Epistemic 분포. 수치는 `epoch3_baseline_validation_20260919/static_fig8_fig9/`의 저장 결과를 사용했다.

## 9. Static Separation이 좋아도 Dynamic Control에서 발생한 문제

static Fig.8에서 epoch3 separation이 좋아졌기 때문에, 다음으로는 실제 channel regime가 변할 때 Fig.11과 같은 uncertainty 기반 feedback/controller가 작동하는지 확인했다. 초기 controller 후보는 `Ng={4,8,16,32,64,128}`으로 두었지만, 실제 training exposure는 `Ng={4,8,16,32}`에만 있었다.

초기 후보를 그대로 둔 dynamic trace에서는 학습 중 보지 않은 `Ng=64,128`에서 정상적인 10 ns ID sample도 OOD로 잘못 판단하는 false trigger가 발생했다. sparse mask가 더 희소해지는 구간으로 외삽하면서 κ가 collapse하고 Epistemic이 급격히 커진 것이 원인으로 진단되었다. controller 후보를 training support인 `{4,8,16,32}`로 제한하면 10 ns false trigger는 사라졌다. 다만 이 경우 120 ns Near-OOD에서 Epistemic fallback이 안정적으로 발생하지 않았다.

저장된 restricted controller 결과에서 20 ns와 80 ns는 각각 평균 `Ng=29.6`, `24.0`으로 동작했지만, 120 ns도 평균 `Ng=23.6`에서 fallback/trigger가 발생하지 않았다. 반대로 full candidate trace에서는 10 ns에서 1회 fallback이 발생했고, 40/60/120 ns 구간은 `Ng=1`에 도달한 뒤 omitted set이 없어 NMSE가 정의되지 않는 구간이 남았다. 이 결과는 static AUROC가 높다는 사실만으로 하나의 absolute threshold가 dynamic ID/near-OOD 제어를 보장하지 않음을 보여준다.

![Epoch3 Fig.11-style dynamic trace](../runs/current_valid_baseline/epoch3_baseline_validation_20260919/fig11/fig11_style.png)

그림 9. Epoch3 checkpoint의 Fig.11-style dynamic validation. 이 평가는 240 step의 저장 trace이며, 완전한 논문 Fig.11 재현이 아니라 현재 temporal/data protocol의 runtime validation이다.

## 10. Near-OOD Epistemic Separation 실패 원인 분석

많은 작은 진단 실험은 각각 별도의 결과로 나누기보다, “어떤 가설이 남았는가”를 비교하는 표로 압축했다.

| 의심한 원인 | 확인 방법 | 결과 |
|---|---|---|
| Ng/offset training imbalance | balanced 100k×3 재학습 및 동일 평가 | 개선 없음. Near AUROC와 tail overlap의 근본 문제는 남음 |
| 단순 threshold 문제 | global 및 Ng-conditioned threshold 비교 | threshold를 바꿔도 근본 해결 안 됨 |
| loss 방향 문제 | κ에 대한 NLL/regularizer gradient analysis | 두 gradient 모두 예상된 방향. loss 부호 오류의 증거 없음 |
| 단순 Ng scale bias | post-hoc scalar κ calibration | scale 차이는 줄지만 FPR/TPR 개선 없음 |
| reconstruction error | error-matched Ng16/32 비교 | 같은 error에서도 Ng32 κ가 더 낮음 |
| CFR feature 차이 | amplitude 및 adjacent-frequency variation 분석 | frequency variation과 κ/Epistemic의 강한 연관 확인 |

따라서 Near-OOD 실패는 단일 threshold나 단일 loss 부호의 문제가 아니라, sparse observation이 만든 입력 난이도와 CFR shape가 evidential head의 κ, ν margin, Ψ에 함께 반영되는 calibration 문제로 범위를 좁혔다.

### 10.1 Ng와 delay가 κ에 미치는 영향

factorial diagnostic은 `Ng={4,8,16,32}`, ID delay `20–100 ns`의 4×5 grid에서 log median κ를 비교했다. 분산 분해 결과 Ng main effect는 약 `53.6%`, delay main effect는 약 `43.5%`, interaction은 약 `2.8%`였다.

통신 관점에서 Ng가 커진다는 것은 관측 subcarrier 간격이 넓어져 실제로 보는 subcarrier 수가 줄어든다는 뜻이다. 관측 정보가 줄면 predictor가 channel을 복원하기 어려워지고 κ가 감소한다. 동시에 같은 ID 범위 안에서도 delay spread가 커지면 frequency-selective channel이 더 어려워져 κ가 추가로 감소한다. Epistemic은 `Sigma_epi = Sigma_ale / κ` 방향으로 κ의 영향을 받으므로, 현재 모델은 OOD mismatch뿐 아니라 hard-ID difficulty에도 반응한다.

이 점이 현재 calibration 문제의 핵심이다. OOD인지 여부와 “관측 조건에서 channel이 어려운지”가 같은 uncertainty 축에 섞이므로, global threshold 하나로 ID와 120 ns를 안정적으로 분리하기 어렵다.

### 10.2 Ng32의 ID Epistemic Tail

Ng32 ID sample을 정상 범위와 상위 tail로 나누면 다음과 같은 차이가 나타난다.

| Group | κ median | ν margin median | Aleatoric median | Epistemic median | NMSE median (dB) |
|---|---:|---:|---:|---:|---:|
| Ng32 normal ID (0–q95) | 6.893 | 29.101 | 0.01610 | 0.002311 | -17.539 |
| Ng32 ID q95–q99 | 1.676 | 21.396 | 0.02431 | 0.014313 | -15.969 |
| Ng32 ID q99+ | 0.866 | 20.037 | 0.02664 | 0.032400 | -16.414 |
| Ng32 120 ns OOD median | 2.324 | 22.278 | 0.02157 | 0.009303 | -14.235 |

Ng32 ID의 q99 tail은 단순 κ 하나의 문제가 아니다. κ가 감소하고, ν margin도 감소하며, Aleatoric이 증가한다. 이 세 효과가 함께 작동하면서 Epistemic이 크게 증폭된다. 특히 Ng32 ID q99 Epistemic median은 `0.032400`으로, Ng32 120 ns OOD의 median `0.009303`보다 높다. 즉 일부 정상 ID sample이 평균적인 120 ns OOD보다 더 높은 uncertainty를 가질 수 있다. 이것이 global threshold가 ID false positive를 만들거나 Near-OOD를 놓치는 이유를 직관적으로 설명한다.

## 11. CFR Shape와 Input Scale이 Uncertainty에 미치는 영향

CFR feature 진단에서는 magnitude, magnitude의 sample-level spread, adjacent-subcarrier variation을 비교했다. 특히 주파수축에서 인접 subcarrier 간 변화량이 커질수록 κ가 낮아지고 Epistemic이 커지는 association이 강하게 나타났다. 이는 sparse observation에서 frequency-selective shape를 복원하는 과정이 evidential response에 직접 연결되어 있음을 뜻한다.

동일한 CFR shape를 유지하고 amplitude만 바꾼 overnight sensitivity 결과도 이 해석을 뒷받침한다. Ng32에서 α=0.75일 때 κ/Epistemic median ratio는 20 ns `1.227/0.723`, 60 ns `1.464/0.568`, 100 ns `1.927/0.422`였고, α=1.25일 때는 각각 `0.733/1.593`, `0.521/2.410`, `0.253/5.048`이었다. 즉 absolute CFR scale만 바꿔도 shape가 같을 때 κ와 Epistemic의 크기가 크게 변한다.

현재 `normalize_channel=false`는 논문에서 확인된 설정이 아니므로 반드시 `IMPLEMENTATION-ASSUMPTION`으로 표시해야 한다. 다만 이 진단만으로 normalize=false가 문제의 원인이라고 확정할 수는 없다. 현재 uncertainty tail에는 channel frequency-selectivity, absolute CFR scale, sample-dependent evidential response가 함께 영향을 줄 가능성이 있으며, normalization은 그중 중요한 implementation/calibration candidate다.

## 12. 현재 Baseline 상태 및 남은 Reproduction Gap

### 현재 확보한 behavior

- CFR reconstruction은 epoch3 control path에서 finite하게 동작한다.
- 80 ns ID-Hard는 20 ns ID-Easy보다 reconstruction이 어렵다.
- delay와 관측 난이도가 커질 때 Aleatoric이 증가하는 경향이 확인된다.
- 1 ms Far-OOD에서는 reconstruction 저하와 Epistemic 증가가 강하게 나타난다.
- epoch3에서는 static Near/Far separation과 주요 uncertainty 계산이 모두 finite하다.

### 남아 있는 gap

- Near-OOD와 hard-ID의 Epistemic tail overlap
- Fig.11 threshold/controller instability
- 논문의 full Ψ 대신 사용하는 diagonal Ψ approximation
- reported-CFR AWGN stage에 대한 implementation assumption
- `normalize_channel=false`의 미공개/미확인 설정
- late epoch ν margin collapse

따라서 핵심 baseline behavior는 확보했지만, exact paper reproduction에는 uncertainty calibration과 일부 implementation gap이 남아 있다. 다음 baseline 개선 방향은 `100k×1 batch8`을 안정적인 control로 사용해 hyperparameter optimization을 수행하고, `100k×5`에서 얻은 좋은 Near-OOD separation을 유지하면서 ν collapse를 방지하는 것이다.

## 13. Uncertainty 기반 Partial Fine-Tuning 연구 확장

**RESEARCH-EXTENSION / IMPLEMENTATION-ASSUMPTION**

원 논문은 높은 Epistemic에서 full feedback과 model adaptation을 수행하지만, 어떤 layer만 Partial Fine-Tuning하는지는 공개하지 않았다. 따라서 아래 Partial policy와 adaptation protocol은 논문 재현이 아니라 연구 확장이다.

실제 model inventory에는 `input_projection`, 32개 `ResidualBlock`, 그리고 `gamma_head`, `psi_head`, `kappa_head`, `nu_head`가 있다.

| Fine-tuning 범위 | Trainable params | 비율 |
|---|---:|---:|
| Head only | 4,632 | 0.0392% |
| Last block + head | 373,656 | 3.1625% |
| Last 8 blocks + head | 2,956,824 | 25.0253% |
| Full | 11,815,320 | 100% |

첫 가설은 “Last block + evidential/output heads의 3.16%만 fine-tuning해도 Full FT 100%와 유사한 120 ns adaptation 효과를 얻을 수 있는가?”였다.

### 13.1 실험 설계

두 branch는 동일한 epoch3 checkpoint에서 독립적으로 시작한다.

```text
Epoch3 checkpoint
       ├── Partial 3.16%  ── 120 ns adaptation ── untouched evaluation
       └── Full 100%      ── 120 ns adaptation ── untouched evaluation
```

공통 조건은 새 120 ns train 1,000개, validation 200개, 3 epochs, batch8, Adam, learning rate `1e-4`, `lambda_reg=1e-3`, FP32, seed `20260921`, 동일 data order, 동일 mask/noise schedule이다. 두 branch의 유일한 실험 변수는 trainable parameter scope다.

### 13.2 Full Feedback과 Sparse Adaptation

여기서 Full feedback은 120 ns 환경에서 모든 subcarrier의 CFR을 받아 clean complete CFR target을 확보한다는 뜻이다. 그러나 complete CFR을 입력과 target으로 그대로 복사하는 identity training은 하지 않는다. complete CFR로 clean target을 만든 뒤, 다시 canonical `Ng={4,8,16,32}` sparse mask와 masked 15 dB AWGN을 적용해 predictor input을 구성한다.

따라서 실제 학습 흐름은 다음과 같다.

```text
complete 120 ns CFR
  → clean full CFR target 확보
  → sparse mask + 15 dB AWGN
  → sparse 120 ns CFR predictor input
  → predictor
  → clean full 120 ns CFR reconstruction target
```

프로토콜은 `configs/adaptation_120ns_protocol_20260921.json`으로 정의하고, `scripts/partial_ft_adapt.py`가 freeze/unfreeze scope와 adaptation training을 수행한다. 공통 평가 비교는 `scripts/evaluate_partial_ft_comparison.py`가 담당하며, input construction의 기본 함수는 `src/training/data.py`, 모델과 loss는 `src/models/uacp_predictor.py` 및 `src/models/evidential.py`에 있다.

### 13.3 평가 방법

adaptation train/validation과 untouched test는 분리했다. test는 20 ns, 80 ns, 120 ns, 1 ms이며 Ng16/Ng32에서 Pre epoch3, Partial, Full을 비교한다. 120 ns NMSE 개선만 보지 않고 Epistemic 변화, 20/80 ns forgetting, 1 ms Far-OOD uncertainty 유지, trainable parameters, training time, peak VRAM을 함께 기록한다.

20 ns와 80 ns를 다시 보는 이유는 120 ns 성능만 좋아지고 기존 ID 성능이 무너지면 좋은 adaptation이라고 할 수 없기 때문이다. 1 ms를 보는 이유는 120 ns를 학습한 뒤에도 아직 보지 못한 Far-OOD에 대해 높은 uncertainty를 유지하는지 확인하기 위해서다.

## 14. Partial Fine-Tuning 실험 결과 및 현재 상태

**RESEARCH-EXTENSION / IMPLEMENTATION-ASSUMPTION**

이번에는 readiness smoke test를 넘어 동일 epoch3 checkpoint에서 실제 Partial-vs-Full adaptation 결과가 저장되어 있다. 두 branch는 checkpoint hash `d3c864788ee7e0e58bbe9f683aa47615039a470258ff054ec2828179b1f23986`에서 시작했고, 동일한 120 ns adaptation schedule을 사용했다. split 간 CFR byte-hash duplicate는 0개였으며, 56,000개 평가 record가 finite였다.

### 주요 결과

| Model | Ng | 20 ns | 80 ns | 120 ns | 1 ms |
|---|---:|---:|---:|---:|---:|
| Pre | 16 | -20.957 | -18.795 | -16.891 | 1.700 |
| Partial | 16 | -19.592 | -18.871 | -18.134 | 1.976 |
| Full | 16 | -18.186 | -18.600 | -18.308 | 2.033 |
| Pre | 32 | -18.130 | -16.000 | -14.215 | 1.842 |
| Partial | 32 | -16.821 | -16.112 | -15.133 | 2.088 |
| Full | 32 | -15.616 | -15.774 | -15.248 | 2.165 |

120 ns NMSE improvement은 Ng16에서 Partial `1.243 dB`, Full `1.417 dB`, Ng32에서 Partial `0.918 dB`, Full `1.033 dB`였다. 따라서 Partial은 Full improvement의 각각 `87.7%`, `88.9%`를 얻었다. 다만 Epistemic 감소는 Partial이 작았다. Ng16에서 Partial/Full 변화는 `-0.000130/-0.000633`, Ng32에서는 `-0.006678/-0.026831`이었다. 즉 Partial은 reconstruction adaptation에는 근접했지만 evidential head까지 Full과 같은 정도로 적응했다고 말할 수 없다.

20 ns forgetting도 Partial이 작았다. Ng16의 post−pre NMSE 변화는 Partial `+1.365 dB`, Full `+2.771 dB`였고, Ng32에서는 각각 `+1.309 dB`, `+2.514 dB`였다. 반면 80 ns에서는 Partial이 Ng16 `-0.076 dB`, Ng32 `-0.113 dB`로 유지/개선되었고, Full은 각각 `+0.195 dB`, `+0.226 dB`였다.

### 비용과 uncertainty 보존

| Metric | Partial 3.1625% | Full 100% |
|---|---:|---:|
| Trainable parameters | 373,656 | 11,815,320 |
| Training time | 137.3 s | 285.9 s |
| Samples/second | 21.85 | 10.49 |
| Peak allocated VRAM | 117.8 MiB | 816.2 MiB |
| Estimated Adam state | 2.851 MiB | 90.144 MiB |

Partial은 Full보다 wall-clock이 48.0%, peak VRAM이 14.4% 수준이었다. 또한 1 ms/120 ns Epistemic ratio는 Partial 후에도 Ng16 `3,352,129`, Ng32 `48,435,804`로 매우 크게 유지된 반면, Full에서는 각각 `45,529`, `45,364`로 낮아졌다. 이 run에서 Partial은 Far-OOD uncertainty를 더 많이 보존했지만, absolute uncertainty scale 자체가 regime에 따라 크게 달라지므로 이 ratio를 일반 법칙으로 확대 해석하지 않는다.

### 판단

현재 결과는 Partial 가설을 부분적으로 지지한다. Last block + head 3.16%는 Full의 120 ns NMSE improvement 중 약 88–89%를 얻었고, ID forgetting이 더 작으며, 비용도 크게 낮았다. 그러나 Full은 120 ns Epistemic을 더 크게 줄였으므로, Partial이 Full과 동일한 uncertainty-head adaptation을 수행한다고 결론내릴 수는 없다. 다음 비교에서는 layer scope 자체를 튜닝하기보다, 이 결과를 고정된 첫 연구 확장 결과로 보존하고 normalization, ν margin, adaptation budget의 영향을 별도로 분리해야 한다.

아직 논문에 공개되지 않은 Partial layer policy, adaptation sample count, epoch/LR, diagonal Ψ, 현재 AWGN pipeline, `normalize_channel=false`는 모두 `RESEARCH-EXTENSION / IMPLEMENTATION-ASSUMPTION` 또는 `APPROXIMATION`으로 남아 있다.

## 15. 결론

이번 확장 결과는 다음의 연구 흐름을 확립했다. 100k sample scale에서는 학습량 증가가 reconstruction과 Near-OOD separation을 개선했지만, epoch4–5에서 Far-OOD ν margin collapse가 발생했다. 따라서 epoch3를 static baseline 후보로 선택했다. epoch3의 static separation은 좋았지만 dynamic controller에서는 unseen Ng extrapolation false trigger와 Near-OOD fallback 부재가 나타났다. 진단 실험은 이 문제가 threshold 하나로 해결되지 않고, Ng, delay, CFR frequency shape, absolute amplitude scale이 κ와 uncertainty에 함께 영향을 주기 때문임을 보여주었다.

그 위에서 수행한 Partial Fine-Tuning 연구 확장은 3.16% trainable scope만으로 Full의 약 88–89% reconstruction adaptation을 얻을 수 있음을 보였고, ID forgetting·VRAM·시간 측면의 이점을 확인했다. 다만 uncertainty adaptation magnitude는 Full보다 작았다. 따라서 현재 결론은 “핵심 baseline behavior는 확보했지만 exact paper reproduction에는 uncertainty calibration 및 일부 implementation gap이 남아 있으며, Partial Fine-Tuning은 비용 효율적인 연구 확장 후보”이다.

### 주요 source/result 경로

- `README.md`, `docs/ACTIVE_PIPELINE.md`, `docs/PARTIAL_FT_READINESS_20260921.md`
- 100k×1: `runs/current_valid_baseline/paper_sample_alignment_100k_1ep_fig8_20260915/`
- 100k×5 및 epoch checkpoints: `runs/current_valid_baseline/training_convergence_100k_5ep_20260916_final_controlled/`
- epoch별 Fig.8: `runs/current_valid_baseline/epoch_checkpoint_evaluation_100k5_batch8_20260918/`
- epoch3 static/Fig.11: `runs/current_valid_baseline/epoch3_baseline_validation_20260919/`
- balanced mask·factorial·κ tail·CFR 진단: `runs/current_valid_baseline/epoch3_balanced_mask_exposure_comparison_20260920/`, `epoch3_balanced_mask_parameter_causality_20260920/`, `epoch3_ng_delay_factorial_diagnosis_20260920/`, `epoch3_ng32_id_epistemic_tail_diagnosis_20260920/`, `epoch3_cfr_feature_tail_diagnosis_20260920/`
- Partial FT 결과: `docs/PARTIAL_FT_RESULTS_20260921.md`, `runs/current_valid_baseline/partial_ft_20260921_sparse_evaluation/`

이 문서는 위 저장 artifact만을 사용해 작성했으며, 이번 보고서 작성을 위해 새 실험이나 재학습을 수행하지 않았다. 기존 `report/UACP_reproduction_report.docx`는 덮어쓰지 않았다.
