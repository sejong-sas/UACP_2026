# STEP 8A Corrected Lambda 1e-2

## Runtime

- Estimated training seconds: `1330.0`
- Dataset generation seconds: `18.5`
- Actual training/probe seconds: `1257.5`

## Final Epoch Probe

| Regime | NMSE_all | NMSE_omitted | Aleatoric | Epistemic | psi | kappa | df_cov | Err-Ale P/S | Err-Epi P/S |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ID-Easy 20 ns | -18.6653 | -18.6654 | 0.013234 | 0.002242 | 0.193701 | 5.943318 | 14.711778 | 0.2524/0.1938 | 0.2538/0.1914 |
| ID-Hard 80 ns | -17.8805 | -17.8788 | 0.013166 | 0.002223 | 0.192869 | 5.931975 | 14.683245 | 0.3195/0.1442 | 0.3216/0.1506 |
| OOD-Near 120 ns | -15.6705 | -15.6704 | 0.013444 | 0.002287 | 0.195377 | 5.889027 | 14.563555 | 0.3033/0.0967 | 0.3048/0.1012 |
| OOD-Far 1 ms | 1.3339 | 1.5340 | 0.020038 | 0.003912 | 0.245949 | 5.188488 | 12.431871 | 0.0783/0.0798 | 0.0852/0.0824 |

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
