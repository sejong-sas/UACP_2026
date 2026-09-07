# STEP 8A Corrected Lambda 0

## Runtime

- Estimated training seconds: `1330.0`
- Dataset generation seconds: `19.0`
- Actual training/probe seconds: `1234.8`

## Final Epoch Probe

| Regime | NMSE_all | NMSE_omitted | Aleatoric | Epistemic | psi | kappa | df_cov | Err-Ale P/S | Err-Epi P/S |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ID-Easy 20 ns | -18.0944 | -18.0936 | 0.010998 | 0.001466 | 0.193581 | 7.568457 | 17.714615 | 0.2541/0.1581 | 0.2511/0.1455 |
| ID-Hard 80 ns | -17.5241 | -17.5201 | 0.010984 | 0.001463 | 0.193382 | 7.518594 | 17.640642 | 0.2883/0.1294 | 0.2898/0.1354 |
| OOD-Near 120 ns | -15.5701 | -15.5629 | 0.011177 | 0.001501 | 0.195249 | 7.458014 | 17.503244 | 0.2735/0.0907 | 0.2751/0.0940 |
| OOD-Far 1 ms | 1.3796 | 1.5861 | 0.015269 | 0.002311 | 0.234604 | 6.682291 | 15.510604 | 0.0615/0.0758 | 0.0695/0.0788 |

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
