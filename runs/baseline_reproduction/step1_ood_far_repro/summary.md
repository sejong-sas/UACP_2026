# STEP 1 OOD-Far 1 ms

| Regime | NMSE_all | NMSE_omitted | Aleatoric | Epistemic | psi | kappa | df_cov | Err-Ale P/S | Err-Epi P/S |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ID-Easy 20 ns | -18.6034 | -18.5640 | 0.411422 | 0.416726 | 0.451061 | 0.998764 | 1.188094 | 0.3076/0.0692 | 0.3067/0.0769 |
| ID-Hard 80 ns | -18.5585 | -18.5147 | 0.404211 | 0.400847 | 0.448519 | 1.002625 | 1.190580 | 0.3082/0.0638 | 0.3136/0.0668 |
| OOD-Near 120 ns | -17.8214 | -17.7412 | 0.401234 | 0.397144 | 0.446655 | 1.005048 | 1.194963 | 0.3281/0.0700 | 0.3337/0.0726 |
| OOD-Far 1 ms | 2.5615 | 2.8407 | 0.390166 | 0.389986 | 0.433749 | 0.996152 | 1.199741 | -0.0040/0.0118 | -0.0046/0.0113 |

## Interpretation

- Aleatoric(80) > Aleatoric(20): False
- Epistemic(120) > Epistemic(ID): False
- Epistemic(1 ms) > Epistemic(ID): False
- Step 1 case: Case B: OOD-Far epistemic does not increase
- Next step: Proceed to delay sweep to locate trend shape before changing training objective.
