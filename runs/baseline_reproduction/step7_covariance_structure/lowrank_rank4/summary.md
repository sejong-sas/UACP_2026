# STEP 7B Low-Rank + Diagonal Rank-4 Pilot

## Runtime

- Estimated training seconds: `1330.0`
- Dataset generation seconds: `19.4`
- Actual training/probe seconds: `1292.0`

## Final Epoch Probe

| Regime | NMSE_all | NMSE_omitted | Aleatoric | Epistemic | psi | kappa | df_cov | Err-Ale P/S | Err-Epi P/S |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ID-Easy 20 ns | -15.6180 | -15.6119 | 0.002177 | 0.018429 | 0.003069 | 0.118669 | 4.462734 | 0.1965/0.1115 | 0.1990/0.1630 |
| ID-Hard 80 ns | -15.4950 | -15.4903 | 0.002171 | 0.018268 | 0.003086 | 0.119098 | 4.462406 | 0.2185/0.0854 | 0.2458/0.1866 |
| OOD-Near 120 ns | -13.7203 | -13.7204 | 0.002164 | 0.018170 | 0.003107 | 0.119377 | 4.458216 | 0.2133/0.0555 | 0.2379/0.1279 |
| OOD-Far 1 ms | 1.0064 | 1.1698 | 0.002121 | 0.017694 | 0.003231 | 0.120184 | 4.394356 | 0.0460/0.0515 | 0.0430/0.0476 |

## Interpretation

- case: Case B
- latest_epoch: 10
- Aleatoric(80) > Aleatoric(20): False
- Epistemic(120) > Epistemic(ID): False
- Epistemic(1 ms) > Epistemic(ID): False
- Early epoch OOD-Far epistemic > ID: False
- OOD-Far NMSE worse than ID-Easy: True
- Final error-uncertainty Pearson correlations positive: True
- proceed_to_step4b: False
