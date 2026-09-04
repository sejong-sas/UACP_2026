# Frequency-Local Banded Psi Pilot

Bandwidth: `32`; device: `cuda:0`; runtime: `3338.439` s.

| Regime | NMSE omitted | Aleatoric | Epistemic |
| --- | ---: | ---: | ---: |
| ID-Easy 20 ns | -17.321945 | 0.009978971 | 0.001630803 |
| ID-Hard 80 ns | -16.941495 | 0.009986444 | 0.001629854 |
| OOD-Near 120 ns | -14.922742 | 0.009986794 | 0.001629742 |
| OOD-Far 1 ms | 1.393975 | 0.010126974 | 0.001659934 |

## Learned covariance

- ID-Easy 20 ns: off/total=0.581180, lag32=0.001736991
- ID-Hard 80 ns: off/total=0.581271, lag32=0.001736464
- OOD-Near 120 ns: off/total=0.581248, lag32=0.001737583
- OOD-Far 1 ms: off/total=0.582441, lag32=0.001707363
