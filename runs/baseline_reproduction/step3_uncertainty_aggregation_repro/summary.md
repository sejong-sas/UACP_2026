# STEP 3 Uncertainty Score Aggregation

| Regime | Ale current | Ale paper | Epi current | Epi paper | Total current | Total paper |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ID-Easy 20 ns | 0.411422 | 0.822844 | 0.416726 | 0.833453 | 0.828148 | 1.656297 |
| ID-Hard 80 ns | 0.404211 | 0.808421 | 0.400847 | 0.801694 | 0.805058 | 1.610115 |
| OOD-Near 120 ns | 0.401234 | 0.802469 | 0.397144 | 0.794289 | 0.798379 | 1.596757 |
| OOD-Far 1 ms | 0.388641 | 0.777281 | 0.387491 | 0.774982 | 0.776132 | 1.552263 |

## Interpretation

- Aggregation scale: For diagonal covariance, paper-style trace aggregation is 2x the current 8-channel mean.
- Aleatoric(80) > Aleatoric(20) under paper aggregation: False
- Epistemic(120) > Epistemic(ID) under paper aggregation: False
- Epistemic(1 ms) > Epistemic(ID) under paper aggregation: False
- Next step: Aggregation does not change ordering if it is only a constant scale; proceed to scaled training if trends still fail.
