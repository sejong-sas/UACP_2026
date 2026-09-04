# 현재 연구 이야기

## 연구 목표

UACP 논문의 channel prediction과 evidential uncertainty를 현재 환경에서
가능한 한 같은 의미로 재현하는 것이 목표입니다. 숫자를 그대로 맞추기보다,
uncertainty가 실제로 위험한 channel을 알려주는지 확인하고 있습니다.

## 현재 baseline

수정된 Real/Imag antenna-pair mapping으로 5,000 samples, 10 epochs를 다시
학습한 checkpoint를 현재 기준으로 고정했습니다. 입력은 `Ng=16` sparse CFR와
mask이고, 보고된 CFR에는 15 dB noise를 넣습니다. 이 noise 위치와 TDL-A는
논문에 정확히 공개되지 않은 `IMPLEMENTATION-ASSUMPTION`입니다.

## 문제를 좁힌 과정

처음에는 20/80/120 ns에서 reconstruction 난이도는 증가하는데 uncertainty가
반대로 움직였습니다. Linear interpolation으로 dataset 자체의 난이도는 정상임을
확인했고, mask와 Eq.12 -> Eq.13 omitted aggregation도 다시 검사했습니다.

CFR power 차이가 첫 번째 혼란 요인이어서 sample power normalization과
1:1 power matching을 수행했습니다. correlation은 좋아졌지만 80 ns Aleatoric
증가는 안정적으로 복구되지 않았습니다.

그 다음 Psi-error alignment, loss gradient, parameter contribution을 보았습니다.
80 ns의 normalized error는 더 컸지만 Psi/Aleatoric은 일관되게 커지지 않았고,
Eq.11 regularizer는 Psi에 직접 gradient를 주지 않았습니다. Aleatoric의 local
sensitivity는 Psi보다 Nu 경로가 훨씬 컸습니다.

Empirical clean CFR를 보면 delay spread에 따라 frequency correlation이 크게
달라졌습니다. 하지만 현재 baseline은 full covariance가 아니라 diagonal Psi만
사용하므로 이 구조를 직접 표현하지 못합니다.

## 현재 결론

현재 active Student-t 식과 Eq.7/8은 diagonal specialization으로 algebraically
맞습니다. 따라서 단순 식의 오타보다는 다음 두 후보가 남아 있습니다.

1. `Psi`, `nu`, `kappa`가 uncertainty를 나누어 조절하는 방식의 coupling
2. 논문의 full frequency covariance를 diagonal로 줄인 approximation

현재 uncertainty는 Far-OOD와 sample-level risk를 어느 정도 구분하지만,
ID-Hard Aleatoric과 Near-OOD Epistemic의 의미 분리는 충분하지 않습니다.

## 다음 실험

가장 우선순위가 높은 다음 실험은 **Psi-aware evidence objective의 짧은 통제
ablation**입니다. 현재 데이터, 모델 구조, 관측 조건은 고정하고 Psi에 직접
어떤 학습 신호가 생기는지만 비교해야 합니다. 아직 실행하지 않았습니다.
