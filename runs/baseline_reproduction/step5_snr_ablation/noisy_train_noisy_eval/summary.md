# STEP 5 Condition C Noisy Train / Noisy Eval

## Runtime

- Estimated training seconds: `1330.0`
- Dataset generation seconds: `17.4`
- Actual training/probe seconds: `1289.3`

## Final Epoch Probe

| Regime | NMSE_all | NMSE_omitted | Aleatoric | Epistemic | psi | kappa | df_cov | Err-Ale P/S | Err-Epi P/S |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ID-Easy 20 ns | -18.2451 | -18.2480 | 0.011062 | 0.001384 | 0.200627 | 8.039047 | 18.228395 | 0.2663/0.1845 | 0.2708/0.1844 |
| ID-Hard 80 ns | -17.6480 | -17.6463 | 0.011011 | 0.001378 | 0.199742 | 8.006546 | 18.172579 | 0.2922/0.1433 | 0.2956/0.1513 |
| OOD-Near 120 ns | -15.6245 | -15.6202 | 0.011164 | 0.001406 | 0.201328 | 7.956999 | 18.060585 | 0.2728/0.0988 | 0.2760/0.1047 |
| OOD-Far 1 ms | 1.3893 | 1.5948 | 0.015788 | 0.002271 | 0.246241 | 7.035846 | 15.750890 | 0.0685/0.0844 | 0.0767/0.0868 |

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
