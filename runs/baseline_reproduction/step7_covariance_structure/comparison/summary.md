# STEP 7 Covariance Comparison

| Model | Regime | NMSE omitted | Aleatoric | Epistemic | off-diagonal ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| Diagonal corrected control | ID-Easy 20 ns | -18.4342 | 0.010542 | 0.001290 | 0.0000 |
| Diagonal corrected control | ID-Hard 80 ns | -17.7426 | 0.010528 | 0.001287 | 0.0000 |
| Diagonal corrected control | OOD-Near 120 ns | -15.6018 | 0.010654 | 0.001310 | 0.0000 |
| Diagonal corrected control | OOD-Far 1 ms | 1.5445 | 0.014483 | 0.002003 | 0.0000 |
| Low-Rank rank4 | ID-Easy 20 ns | -15.6119 | 0.002177 | 0.018429 | 0.9977 |
| Low-Rank rank4 | ID-Hard 80 ns | -15.4903 | 0.002171 | 0.018268 | 0.9974 |
| Low-Rank rank4 | OOD-Near 120 ns | -13.7204 | 0.002164 | 0.018170 | 0.9974 |
| Low-Rank rank4 | OOD-Far 1 ms | 1.1698 | 0.002121 | 0.017694 | 0.9975 |

## Gaps

| Model | Ale 80-20 | Epi 120-ID | Epi 1ms-ID |
| --- | ---: | ---: | ---: |
| Diagonal corrected control | -0.00001405 | 0.00002025 | 0.00071337 |
| Low-Rank rank4 | -0.00000665 | -0.00025827 | -0.00073487 |
