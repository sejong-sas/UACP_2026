# UACP Partial Fine-Tuning 연구 이야기

## 1. 기존 연구에서 무엇이 아직 해결되지 않았는가?

UACP는 sparse Channel Frequency Response (CFR)로부터 전체 CFR을 복원하면서 Aleatoric Uncertainty와 Epistemic Uncertainty를 추정하고, Epistemic Uncertainty가 높아지면 학습 분포를 벗어난 채널로 판단하여 full feedback과 model adaptation을 유도한다. 그러나 원 논문은 OOD 감지 이후 실제 모델을 어느 범위까지 업데이트해야 하는지, 전체 parameter를 업데이트하는 방식과 일부 parameter만 업데이트하는 방식이 reconstruction·uncertainty·기존 성능·계산 비용에 어떤 차이를 만드는지는 구체적으로 비교하지 않는다. 따라서 adaptation이 필요하다는 판단과 adaptation을 실제로 수행하는 방법 사이에 분석 공백이 남아 있다.

이번 연구에서 확인하는 adaptation 정책은 원 UACP가 제안한 고정 정책이 아니라 현재 repository에서 구현한 연구 확장이다. 특히 Partial Fine-Tuning의 layer 범위, 120 ns adaptation sample 수, 3 epochs, batch size, learning rate, mask/noise schedule은 원 논문 설정으로 간주하지 않는다.

## 2. 이번 연구에서 정확히 어떤 질문을 검증하는가?

Epoch 3 pretrained evidential channel predictor가 120 ns OOD 채널을 만났을 때, 전체 parameter를 fine-tuning하지 않고 일부 parameter만 업데이트해도 새로운 환경의 CFR reconstruction 성능을 회복할 수 있는가? 동시에 partial update가 20 ns 및 80 ns ID 성능을 덜 훼손하고, 학습하지 않은 1 ms Far-OOD에서 높은 Epistemic Uncertainty를 유지하는지, 그리고 trainable parameter 수·학습 시간·GPU memory 측면의 비용이 어떻게 달라지는지를 비교한다.

비교 범위는 Pretrained model, 마지막 block과 evidential heads를 업데이트하는 3.16%, ResidualBlock 24–31과 evidential heads를 업데이트하는 25.03%, 전체 parameter를 업데이트하는 Full Fine-Tuning이다. 핵심은 Partial Fine-Tuning이 항상 Full Fine-Tuning보다 좋다는 주장을 검증하는 것이 아니라, adaptation 성능과 비용 및 uncertainty 보존 사이의 trade-off를 확인하는 것이다.

## 3. 어떤 실험이 그 질문에 답하는가?

모든 adaptation 비교는 동일한 Epoch 3 checkpoint, 120 ns adaptation train/validation split, 동일한 sample order·mask·15 dB complex AWGN schedule, Adam optimizer, learning rate `1e-4`, batch size 8, 3 epochs/375 steps, FP32, NVIDIA GB10에서 수행된 결과를 사용한다. adaptation input은 sparse noisy CFR과 mask이고 target은 clean full CFR이다. 120 ns test set은 adaptation train/validation에 재사용하지 않았으며 cross-split CFR hash 중복은 0이다.

각 scope에 대해 다음을 비교한다.

- 120 ns omitted-subcarrier NMSE와 Pre 대비 개선량
- 120 ns Epistemic Uncertainty 변화
- 20 ns 및 80 ns NMSE의 post-minus-pre 변화로 본 ID forgetting
- adaptation에 사용하지 않은 1 ms의 Epistemic Uncertainty와 120 ns 대비 분리
- trainable parameter 수, training time, step time, peak allocated GPU memory

Epoch 3 공통 비교에서 Ng=16 기준 120 ns NMSE 개선량은 3.16% `1.2428 dB`, 25% `1.5210 dB`, Full `1.4167 dB`이고, Ng=32에서는 각각 `0.9180 dB`, `1.1287 dB`, `1.0329 dB`이다. 3.16%는 Full 개선량의 약 87.7%/88.9%를 얻었고, 25%는 약 107.4%/109.3% 수준이었다. 그러나 120 ns Epistemic 감소는 Full이 가장 컸다. 1 ms Epistemic은 3.16%에서 Ng16 `27,697.6`, Ng32 `2.2521e6`으로 유지된 반면, 25%에서는 각각 `22.10`, `82.30`으로 크게 낮아졌다. 따라서 reconstruction 회복만으로 adaptation의 성공을 판단할 수 없다.

## 4. 결과로 무엇까지 주장할 수 있는가?

현재 결과로는 다음 정도까지 주장할 수 있다.

- 현재 구현과 고정된 3-epoch adaptation protocol에서는 3.16% Partial Fine-Tuning도 120 ns reconstruction 개선을 얻었고, Full 대비 더 낮은 trainable parameter 수·training time·GPU memory를 사용했다.
- 25% Partial Fine-Tuning은 이 protocol에서 3.16%보다 120 ns reconstruction 개선이 컸고 Full과 유사하거나 더 큰 개선량을 보였지만, 1 ms Far-OOD Epistemic Uncertainty를 크게 낮췄다.
- Full Fine-Tuning은 120 ns Epistemic Uncertainty 감소가 가장 컸지만, 3.16%보다 높은 ID forgetting과 더 큰 adaptation cost를 보였다.
- 따라서 parameter-efficient adaptation의 가능성은 확인되지만, parameter 비율이 클수록 reconstruction과 near-OOD uncertainty adaptation이 좋아진다는 관찰이 Far-OOD uncertainty 보존까지 의미하지는 않는다. 현재 결과는 scope·data·optimizer·epoch·observation pipeline에 한정된 trade-off 분석이다.

다음은 주장하지 않는다. UACP를 완전히 재현했다는 주장, Partial Fine-Tuning이 Full Fine-Tuning보다 전반적으로 우수하다는 주장, 모든 OOD 환경에 일반화된다는 주장, 실시간 활용 가능성을 증명했다는 주장이다. 또한 현재 구현의 diagonal Ψ, 보고된 CFR에만 적용한 sample-wise AWGN, `normalize_channel=false`, 고정 sparse mask/evaluation protocol은 원 논문에 완전히 공개되거나 동일하게 확인된 설정이 아니므로 구현 가정으로 표시한다. ID 내부 delay 변화에 따른 Aleatoric Uncertainty 증가 경향도 원 논문만큼 모든 실험에서 안정적으로 재현되었다고 쓰지 않는다.
