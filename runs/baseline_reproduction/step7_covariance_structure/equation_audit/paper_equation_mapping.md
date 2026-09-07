# STEP 7A Equation Mapping

| Paper item | Current implementation | Status |
| --- | --- | --- |
| Eq. (5) df | `nu - d + 1`, with `d=2K` | Exact for diagonal Psi |
| Eq. (5) scale | `((kappa+1)/(kappa*df))*Psi` | Exact for diagonal Psi |
| Eq. (5) density | Multivariate gamma, log determinant, Mahalanobis term | Exact diagonal specialization |
| Eq. (7) | `Psi/(nu-d-1)` | Exact diagonal specialization |
| Eq. (8) | `Sigma_ale/kappa` | Exact |
| Eq. (9) | softplus constraints; full `Psi=LL^T` absent | Diagonal approximation |
| Eq. (10) | Mean of pair log densities | Reduction assumption; paper writes a sum |
| Eq. (11) | Pair squared error times `kappa+nu`, all target frequencies | Formula aligned; reduction is implementation-defined |

The channel layout is real channels for all four antenna pairs followed by imaginary
channels for all four pairs. Pair vectors must therefore be formed by permutation,
not a direct reshape. The previous direct reshape and pair-scalar interleaved
broadcast were layout mismatches and are corrected in the current implementation.

`Psi=D+UU^T` is an implementation approximation for STEP 7B, not a paper claim.
