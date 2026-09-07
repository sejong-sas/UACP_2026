# STEP 7A Corrected Diagonal Control

## Runtime

- Estimated training seconds: `1330.0`
- Dataset generation seconds: `22.2`
- Actual training/probe seconds: `1225.6`

## Final Epoch Probe

| Regime | NMSE_all | NMSE_omitted | Aleatoric | Epistemic | psi | kappa | df_cov | Err-Ale P/S | Err-Epi P/S |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ID-Easy 20 ns | -18.4379 | -18.4342 | 0.010542 | 0.001290 | 0.193981 | 8.235355 | 18.506250 | 0.2392/0.1704 | 0.2412/0.1670 |
| ID-Hard 80 ns | -17.7491 | -17.7426 | 0.010528 | 0.001287 | 0.193554 | 8.188599 | 18.412962 | 0.2887/0.1242 | 0.2917/0.1341 |
| OOD-Near 120 ns | -15.6096 | -15.6018 | 0.010654 | 0.001310 | 0.194851 | 8.143311 | 18.314020 | 0.2802/0.0931 | 0.2831/0.0998 |
| OOD-Far 1 ms | 1.3384 | 1.5445 | 0.014483 | 0.002003 | 0.233414 | 7.300094 | 16.245275 | 0.0620/0.0806 | 0.0697/0.0836 |

## Interpretation

- case: Case A
- latest_epoch: 10
- Aleatoric(80) > Aleatoric(20): False
- Epistemic(120) > Epistemic(ID): True
- Epistemic(1 ms) > Epistemic(ID): True
- Early epoch OOD-Far epistemic > ID: False
- OOD-Far NMSE worse than ID-Easy: True
- Final error-uncertainty Pearson correlations positive: True
- proceed_to_step4b: True
