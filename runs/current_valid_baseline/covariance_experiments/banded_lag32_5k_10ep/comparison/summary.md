# Diagonal vs Banded lag32

Paired read-only evaluation uses the same deterministic probe seeds.

| Model | Regime | NMSE omitted | Aleatoric | Epistemic |
| --- | --- | ---: | ---: | ---: |
| Diagonal current-valid control | ID-Easy 20 ns | -17.684033 | 0.020907141 | 0.002677559 |
| Diagonal current-valid control | ID-Hard 80 ns | -17.222405 | 0.020818593 | 0.002642030 |
| Diagonal current-valid control | OOD-Near 120 ns | -15.271637 | 0.021020940 | 0.002675008 |
| Diagonal current-valid control | OOD-Far 1 ms | 1.557583 | 0.028841565 | 0.004121700 |
| Banded lag32 | ID-Easy 20 ns | -17.321945 | 0.009978971 | 0.001630803 |
| Banded lag32 | ID-Hard 80 ns | -16.941495 | 0.009986444 | 0.001629854 |
| Banded lag32 | OOD-Near 120 ns | -14.922742 | 0.009986794 | 0.001629742 |
| Banded lag32 | OOD-Far 1 ms | 1.393975 | 0.010126974 | 0.001659934 |

| Model | Ale 80-20 | Epi 120-ID | Epi 1ms-ID | Mean ID NMSE |
| --- | ---: | ---: | ---: | ---: |
| Diagonal current-valid control | -8.85482132e-05 | -2.5515072e-06 | 0.00144414142 | -17.453219 |
| Banded lag32 | 7.47315586e-06 | -1.0608742e-06 | 2.91310856e-05 | -17.131720 |
