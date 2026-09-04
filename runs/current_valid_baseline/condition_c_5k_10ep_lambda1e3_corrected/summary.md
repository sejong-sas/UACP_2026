# CURRENT VALID BASELINE - Condition C

This run was trained from scratch with the corrected real/imag antenna-pair mapping. It is the current control for later diagnostics.

## Conditions

- 5,000 train samples, 10 epochs, batch size 8, Adam, learning rate 1e-4
- Uniform `[10,100] ns` training delay spread, TDL-A channel (`IMPLEMENTATION-ASSUMPTION`)
- 15 dB reported-CFR complex AWGN for both train and evaluation (`IMPLEMENTATION-ASSUMPTION`); clean target
- Experiment D: pair-scalar kappa/nu, diagonal Psi, diagonal multivariate NLL, pair regularizer
- Fixed evaluation grouping `Ng=16`, seed `20260819`, device `cuda:0`

`SNR=15 dB` is PAPER-SPECIFIED; its exact paper application stage is UNKNOWN. The 5k/10 epoch pilot and TDL-A are not paper-specified.

## Final Probe (Eq.12 -> Eq.13 scores)

| Regime | NMSE all | NMSE omitted | Aleatoric | Epistemic | Err-Ale P | Err-Epi P |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ID-Easy 20 ns | -17.6702 | -17.6709 | 0.020904 | 0.002677 | 0.2358 | 0.2415 |
| ID-Hard 80 ns | -17.2629 | -17.2605 | 0.020815 | 0.002642 | 0.2646 | 0.2685 |
| OOD-Near 120 ns | -15.2965 | -15.2930 | 0.021007 | 0.002673 | 0.2534 | 0.2558 |
| OOD-Far 1 ms | 1.3591 | 1.5622 | 0.028924 | 0.004138 | 0.0613 | 0.0698 |

## Provenance

- Git commit: `f5870fb735009d62d7237e3574d669e92dd5e9d8`
- Config SHA-256: `bce6092b64c4c171bb8395b131c921be25d3131134a4b718c2ecefaa9ccf1c63`
- GPU: `NVIDIA GB10`; CUDA available: `True`
- Training/probe seconds: `1380.565`
- Wrapped checkpoint: `checkpoint_with_provenance.pt`
- Compatibility state dict: `model_state_dict.pt`

## Known Approximation

The paper specifies full pair covariance `Psi=L L^T`; this baseline uses diagonal `Psi` and must be treated as an APPROXIMATION. Historical STEP 6 Condition C is pre-fix and remains invalid for uncertainty comparison.
