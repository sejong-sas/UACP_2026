# STEP 4A Scaled-Training Pilot

## Runtime

- Estimated training seconds: `1330.0`
- Dataset generation seconds: `15.4`
- Actual training/probe seconds: `1230.6`

## Final Epoch Probe

| Regime | NMSE_all | NMSE_omitted | Aleatoric | Epistemic | psi | kappa | df_cov | Err-Ale P/S | Err-Epi P/S |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ID-Easy 20 ns | -18.6127 | -18.6130 | 0.013579 | 0.002292 | 0.199490 | 5.989818 | 14.812567 | 0.2526/0.1674 | 0.2496/0.1565 |
| ID-Hard 80 ns | -17.8221 | -17.8195 | 0.013549 | 0.002277 | 0.199214 | 5.961555 | 14.737772 | 0.3097/0.1364 | 0.3112/0.1407 |
| OOD-Far 1 ms | 1.3200 | 1.5199 | 0.020495 | 0.003979 | 0.252344 | 5.216050 | 12.465777 | 0.0756/0.0795 | 0.0825/0.0810 |
| OOD-Near 120 ns | -15.6232 | -15.6224 | 0.013813 | 0.002338 | 0.201456 | 5.920317 | 14.617096 | 0.2982/0.0890 | 0.2983/0.0918 |

## Interpretation

- case: Case A
- latest_epoch: 10
- Aleatoric(80) > Aleatoric(20): False
- Epistemic(120) > Epistemic(ID): True
- Epistemic(1 ms) > Epistemic(ID): True
- Early epoch OOD-Far epistemic > ID: True
- OOD-Far NMSE worse than ID-Easy: True
- Final error-uncertainty Pearson correlations positive: True
- proceed_to_step4b: True
