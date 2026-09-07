# STEP 4A Scaled-Training Pilot

## Runtime

- Estimated training seconds: `1330.0`
- Dataset generation seconds: `14.9`
- Actual training/probe seconds: `1237.0`

## Final Epoch Probe

| Regime | NMSE_all | NMSE_omitted | Aleatoric | Epistemic | psi | kappa | df_cov | Err-Ale P/S | Err-Epi P/S |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ID-Easy 20 ns | -17.9255 | -17.9311 | 0.010976 | 0.001428 | 0.196028 | 7.733562 | 17.950399 | 0.2573/0.1679 | 0.2571/0.1599 |
| ID-Hard 80 ns | -17.4048 | -17.4089 | 0.010968 | 0.001429 | 0.195668 | 7.691234 | 17.874102 | 0.2717/0.1323 | 0.2712/0.1261 |
| OOD-Far 1 ms | 1.4023 | 1.6109 | 0.015592 | 0.002330 | 0.240933 | 6.777931 | 15.609756 | 0.0641/0.0787 | 0.0727/0.0823 |
| OOD-Near 120 ns | -15.5205 | -15.5175 | 0.011140 | 0.001462 | 0.197366 | 7.635877 | 17.747883 | 0.2644/0.0898 | 0.2644/0.0864 |

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
