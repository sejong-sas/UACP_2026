# STEP 4A Scaled-Training Pilot

## Runtime

- Estimated training seconds: `1330.0`
- Dataset generation seconds: `14.0`
- Actual training/probe seconds: `1176.3`

## Final Epoch Probe

| Regime | NMSE_all | NMSE_omitted | Aleatoric | Epistemic | psi | kappa | df_cov | Err-Ale P/S | Err-Epi P/S |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ID-Easy 20 ns | -21.3436 | -21.2917 | 0.003307 | 0.000351 | 0.079102 | 9.608957 | 24.415354 | 0.3505/0.1779 | 0.3538/0.1504 |
| ID-Hard 80 ns | -21.2558 | -21.2026 | 0.003178 | 0.000333 | 0.077149 | 9.611016 | 24.491085 | 0.3344/0.1729 | 0.3341/0.1465 |
| OOD-Near 120 ns | -20.4320 | -20.3570 | 0.003184 | 0.000334 | 0.077364 | 9.603812 | 24.481995 | 0.3743/0.1515 | 0.3732/0.1279 |
| OOD-Far 1 ms | 2.3117 | 2.5856 | 0.004149 | 0.000464 | 0.094128 | 9.048777 | 22.937614 | 0.0192/0.1279 | 0.0218/0.1294 |

## Interpretation

- case: Case A
- latest_epoch: 10
- Aleatoric(80) > Aleatoric(20): False
- Epistemic(120) > Epistemic(ID): False
- Epistemic(1 ms) > Epistemic(ID): True
- Early epoch OOD-Far epistemic > ID: False
- OOD-Far NMSE worse than ID-Easy: True
- Final error-uncertainty Pearson correlations positive: True
- proceed_to_step4b: True
