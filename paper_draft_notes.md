# paper_draft_v1 작성 메모

## 핵심 연구 질문

학습 분포 밖의 120 ns 채널에서 adaptation이 필요할 때, 전체 parameter를 업데이트하지 않고 3.16% 또는 25.03%만 업데이트해도 reconstruction 회복을 얻을 수 있는가? Partial update의 비용 절감이 20 ns/80 ns ID 성능 보존과 1 ms Far-OOD Epistemic Uncertainty 보존을 함께 만족하는지, Full Fine-Tuning과 비교해 trade-off를 확인한다.

## Section별 역할

- 서론: MIMO CSI feedback overhead와 sparse CFR reconstruction의 필요성을 설명하고, UACP가 OOD를 감지한 뒤 adaptation을 trigger하지만 update 범위와 cost를 구체화하지 않았다는 문제를 제시한다. 본 연구를 새 알고리즘 제안이 아닌 parameter-efficient adaptation의 실험적 분석으로 위치시킨다.
- 실험 디자인 2.1: training distribution, ID-Easy 20 ns, ID-Hard 80 ns, adaptation target 120 ns, blind Far-OOD 1 ms와 channel/observation 설정을 원 논문과 현재 구현으로 구분한다.
- 실험 디자인 2.2: sparse CFR+mask 입력, 32 residual blocks, gamma/kappa/Psi/nu 출력, current diagonal-Psi approximation, Aleatoric/Epistemic 계산식을 정리한다.
- 실험 디자인 2.3: 3.16%, 25.03%, Full의 실제 layer 범위와 동일 adaptation schedule을 설명한다.
- 실험 디자인 2.4: 120 ns NMSE recovery, 120 ns Epistemic change, ID forgetting, 1 ms preservation, parameter/time/VRAM을 정의한다.
- 결과 3.1: Epoch 3 checkpoint가 CFR reconstruction, 80 ns>20 ns reconstruction difficulty, OOD Epistemic 증가, adaptation 가능성을 보이는지 짧게 확인한다. UACP 완전 재현으로 표현하지 않는다.
- 결과 3.2: 동일 protocol에서 3.16%/25%/Full의 120 ns NMSE improvement를 Ng=16/32로 비교한다.
- 결과 3.3: 120 ns Epistemic 감소, 20/80 ns forgetting, 1 ms Epistemic collapse/preservation을 함께 해석한다.
- 결과 3.4: trainable parameter, training time, peak VRAM과 성능을 trade-off로 해석한다.
- 결론: 결과 범위, 한계, 추가 실험(다중 seed, longer adaptation, uncertainty-preserving objective, layer selection)을 분리한다.

## 사용한 실험 결과와 source file 경로

### 공통 adaptation 비교의 기준

- 시작 checkpoint: `runs/current_valid_baseline/training_convergence_100k_5ep_20260916_final_controlled/uacp_predictor_100k_5ep_epoch_3.pt`
- checkpoint SHA-256: `d3c864788ee7e0e58bbe9f683aa47615039a470258ff054ec2828179b1f23986` (실제 파일 hash와 문서 기록값 일치 확인)
- protocol 설명: `docs/PARTIAL_FT_RESULTS_20260921.md`
- 25% 결과: `docs/PARTIAL_FT_25PCT_RESULTS_20260921.md`
- 3.16%/Full 결과: `runs/current_valid_baseline/partial_ft_20260921_sparse_evaluation/summary.csv`, `primary_performance_epoch3.csv`, `uncertainty_epoch3.csv`, `adaptation_forgetting.csv`, `efficiency.csv`
- 25% 결과: `runs/current_valid_baseline/partial_ft_20260921_sparse_25pct_evaluation/`
- adaptation data/config: `configs/adaptation_120ns_protocol_20260921.json`, `runs/current_valid_baseline/overnight_20260920_adaptation_data/provenance_manifest.json`
- implementation: `scripts/partial_ft_adapt.py`, `src/training/data.py`, `src/models/uacp_predictor.py`, `src/models/evidential.py`, `src/training/uncertainty.py`

### 주요 수치

- Epoch 3 common evaluation, Ng=16:
  - Pre: 20 ns `-20.9567`, 80 ns `-18.7954`, 120 ns `-16.8910`, 1 ms `1.6995` dB.
  - 3.16%: 20 ns `-19.5916`, 80 ns `-18.8713`, 120 ns `-18.1338`, 1 ms `1.9763` dB.
  - 25%: 20 ns `-18.4849`, 80 ns `-18.7680`, 120 ns `-18.4120`, 1 ms `1.9742` dB.
  - Full: 20 ns `-18.1859`, 80 ns `-18.6005`, 120 ns `-18.3077`, 1 ms `2.0330` dB.
- Epoch 3 common evaluation, Ng=32:
  - Pre: 20 ns `-18.1301`, 80 ns `-15.9997`, 120 ns `-14.2149`, 1 ms `1.8416` dB.
  - 3.16%: 20 ns `-16.8207`, 80 ns `-16.1124`, 120 ns `-15.1330`, 1 ms `2.0880` dB.
  - 25%: 20 ns `-15.8773`, 80 ns `-15.9308`, 120 ns `-15.3437`, 1 ms `2.1423` dB.
  - Full: 20 ns `-15.6162`, 80 ns `-15.7736`, 120 ns `-15.2478`, 1 ms `2.1653` dB.
- 120 ns NMSE improvement: Ng16 `1.2428 / 1.5210 / 1.4167` dB, Ng32 `0.9180 / 1.1287 / 1.0329` dB for 3.16% / 25% / Full.
- 120 ns Epistemic change: Ng16 `-0.000130 / -0.000469 / -0.000633`, Ng32 `-0.006678 / -0.026232 / -0.026831`.
- Epoch 3 Epistemic, Ng16: Pre `0.008393` at 120 ns and `32985.8` at 1 ms; 3.16% `0.008263` and `27697.6`; 25% `0.007924` and `22.10`; Full `0.007759` and `353.28`.
- Epoch 3 Epistemic, Ng32: Pre `0.053175` at 120 ns and `3.0569e6` at 1 ms; 3.16% `0.046497` and `2.2521e6`; 25% `0.026943` and `82.30`; Full `0.026344` and `1195.07`.
- Cost: 3.16% `373,656` params, `137.32 s`, `117.84 MiB`; 25% `2,956,824`, `149.34 s`, `275.55 MiB`; Full `11,815,320`, `285.94 s`, `816.16 MiB`.

### Baseline/Fig.8/Fig.9/Fig.11 evidence

- Epoch 3 baseline summary and interpretation: `docs/CURRENT_RESEARCH_STORY.md`, `report/latest_sections_20260921.md`.
- Baseline epoch3 static validation: `runs/current_valid_baseline/epoch3_baseline_validation_20260919/` and related `epoch3_*` evaluation folders.
- Baseline report source map: `report/source_data/summary_tables.csv`, `report/source_data/figure_sources.json`.
- Fig.11 limitation: `docs/PARTIAL_FT_FINAL_VALIDATION_20260921.md` records that the repository controller is tied to an older checkpoint/delay sweep and is not directly comparable to the fixed adaptation protocol. Do not use Fig.11 as a primary quantitative adaptation result.

## 아직 확인이 필요한 부분

- 저자명, 소속, 이메일은 repository에서 확인되지 않으므로 placeholder로 남겼다.
- checkpoint SHA-256은 `docs/PARTIAL_FT_RESULTS_20260921.md`의 기록값과 실제 파일 hash가 일치함을 확인했다.
- 원 논문의 최종 출판 상태와 DOI는 repository PDF가 `Submitted to ACM MobiHoc '26`으로 표시하므로 확정하지 않았다. DOI를 만들지 않는다.
- 원 논문에서 noise가 적용되는 정확한 단계, TDL 세부 설정, `normalize_channel` 값, full Ψ 구현 세부는 현재 repository에서 확인된 구현과 동일하다고 쓰지 않는다.
- 3.16%/25%/Full의 주 비교는 3 epoch, Ng=16/32 fixed evaluation이다. 10 epoch multi-seed final validation은 동일한 비교표로 합치지 않고 한계/후속 결과로 분리했다.
- GPU memory는 peak allocated VRAM이며, system-wide reserved memory나 inference memory가 아니다.

## 참고문헌 목록과 링크/DOI

본문 인용 순서는 초안 기준이며, 제출 전 KIPS 양식에 맞춰 번호를 다시 확인한다.

1. UACP 원 논문: `mobihoc26-paper289.pdf` (repository copy, `Submitted to ACM MobiHoc '26`; DOI 확인 불가, 생성하지 않음).
2. A. Goldsmith, *Wireless Communications*, Cambridge University Press, 2005. [Cambridge listing](https://www.cambridge.org/core/books/wireless-communications/).
3. C.-K. Wen, W.-T. Shih, and S. Jin, “Deep Learning for Massive MIMO CSI Feedback,” *IEEE Wireless Communications Letters*, 7(5), 748–751, 2018. DOI: [10.1109/LWC.2018.2818160](https://doi.org/10.1109/LWC.2018.2818160).
4. J. Wang et al., “Compressive Sampled CSI Feedback Method Based on Deep Learning for FDD Massive MIMO Systems,” *IEEE Transactions on Communications*, 69(9), 5873–5885, 2021. DOI: [10.1109/TCOMM.2021.3086525](https://doi.org/10.1109/TCOMM.2021.3086525).
5. A. Amini et al., “Deep Evidential Regression,” *NeurIPS 33*, 14927–14937, 2020. [arXiv:1910.02600](https://arxiv.org/abs/1910.02600).
6. N. Meinert and A. Lavin, “Multivariate Deep Evidential Regression,” arXiv:2104.06135, 2021. [arXiv:2104.06135](https://arxiv.org/abs/2104.06135).
7. N. Houlsby et al., “Parameter-Efficient Transfer Learning for NLP,” *ICML*, PMLR 97, 2790–2799, 2019. [PMLR](https://proceedings.mlr.press/v97/houlsby19a.html).
8. E. J. Hu et al., “LoRA: Low-Rank Adaptation of Large Language Models,” *ICLR*, 2022. [OpenReview](https://openreview.net/forum?id=nZeVKeeFYf9).
9. J. Kirkpatrick et al., “Overcoming Catastrophic Forgetting in Neural Networks,” *PNAS*, 114(13), 3521–3526, 2017. DOI: [10.1073/pnas.1611835114](https://doi.org/10.1073/pnas.1611835114).
10. J. Hoydis et al., “Sionna: An Open-Source Library for Next-Generation Wireless Communications,” 2022. [Sionna documentation](https://nvlabs.github.io/sionna/). 원 논문 PDF의 reference는 software/documentation citation으로 기록되어 있어 최종 서지 형식은 확인 필요.

## 현재 결과로 주장 가능한 것

- 고정된 현재 구현과 3-epoch protocol에서 Partial Fine-Tuning은 120 ns reconstruction을 개선할 수 있다.
- 3.16%는 Full보다 낮은 cost로 Full improvement에 87.7%/88.9% 근접했다.
- 25%는 120 ns reconstruction 및 120 ns Epistemic adaptation에서 Full-like 결과를 보였지만 1 ms Far-OOD Epistemic이 크게 줄었다.
- Full은 120 ns Epistemic 감소가 가장 크지만 parameter/time/VRAM과 ID forgetting도 컸다.
- 결론은 구현·dataset·schedule에 한정된 trade-off evidence다.

## 주장하면 안 되는 것

- UACP 완전 재현, 모든 OOD 일반화, real-time deployment 증명, Partial이 Full보다 전반적으로 우수하다는 주장.
- `diagonal Ψ`, reported-CFR AWGN, `normalize=false`, adaptation sample/epoch/LR/layer policy를 원 논문 설정으로 쓰는 것.
- 25% 결과의 1 ms uncertainty collapse를 숨기고 reconstruction만으로 “성공”이라고 결론내리는 것.
- Fig.11 controller 결과를 현재 fixed adaptation comparison과 동일한 protocol 결과로 쓰는 것.
