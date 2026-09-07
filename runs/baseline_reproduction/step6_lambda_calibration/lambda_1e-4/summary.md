# STEP 4A Scaled-Training Pilot

## Runtime

- Estimated training seconds: `1330.0`
- Dataset generation seconds: `21.9`
- Actual training/probe seconds: `1235.9`

## Final Epoch Probe

| Regime | NMSE_all | NMSE_omitted | Aleatoric | Epistemic | psi | kappa | df_cov | Err-Ale P/S | Err-Epi P/S |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ID-Easy 20 ns | -18.1472 | -18.1487 | 0.010515 | 0.001331 | 0.191973 | 7.932406 | 18.331881 | 0.2547/0.1721 | 0.2556/0.1726 |
| ID-Hard 80 ns | -17.6052 | -17.6035 | 0.010480 | 0.001330 | 0.191230 | 7.895282 | 18.275812 | 0.2853/0.1289 | 0.2866/0.1297 |
| OOD-Far 1 ms | 1.3643 | 1.5695 | 0.014254 | 0.002038 | 0.229568 | 7.054420 | 16.215729 | 0.0619/0.0818 | 0.0696/0.0846 |
| OOD-Near 120 ns | -15.6493 | -15.6451 | 0.010595 | 0.001352 | 0.192303 | 7.848219 | 18.175774 | 0.2732/0.0898 | 0.2745/0.0907 |

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
