# Paper Eq.5~11 vs Current Evidential Implementation

This is a read-only semantic audit. No optimizer step or checkpoint write was performed.

## Status vocabulary

- **PAPER-SPECIFIED**: stated by the supplied paper text/equations.
- **IMPLEMENTATION-ASSUMPTION**: selected in this prototype where the paper does not specify details.
- **APPROXIMATION**: a deliberate computational simplification of the paper formulation.
- **UNKNOWN IN PAPER**: not exposed by the supplied paper.

## Mapping

| Item | Paper formulation | Current implementation | Status |
| --- | --- | --- | --- |
| Latent dimension | `d=2K`, one pair vector in `R^d` | `d=2048`, `K=1024` | PAPER-SPECIFIED / configuration fixed |
| `gamma_p` | pair CFR mean, `[d]` | channels `[8,K]`, converted to pair `[4,2K]` | PAPER-SPECIFIED semantics, layout is implementation |
| `kappa_p` | one scalar per antenna pair | `[B,4,1]`, softplus constrained | PAPER-SPECIFIED semantics |
| `Psi_p` | full `[d,d]`, `Psi=L L^T` | diagonal `[B,4,d]` | PAPER-SPECIFIED full structure; diagonal is APPROXIMATION |
| `nu_p` | one scalar per antenna pair, `nu>d+1` | `[B,4,1]`, `d+1+softplus(raw_nu)+eps` | PAPER-SPECIFIED semantics, transform implementation |
| Predictive df | `nu-d+1` | same in active diagonal NLL | MATCH |
| Predictive scale | `((kappa+1)/(kappa*df))*Psi` | same, diagonal specialization | MATCH under diagonal approximation |
| Eq.7 | `Sigma_ale=Psi/(nu-d-1)` | same diagonal marginal | MATCH under diagonal approximation |
| Eq.8 | `Sigma_epi=Sigma_ale/kappa` | same, pair scalar broadcast | MATCH |
| Eq.9 | positive covariance, `Psi=L L^T` | positive diagonal via softplus | APPROXIMATION |
| Eq.10 | pair predictive log-density aggregation | mean over batch and pairs | IMPLEMENTATION-ASSUMPTION / reduction difference |
| Eq.11 scope | pair error norm times `Phi=kappa+nu`, observed + omitted | full clean target, pair norm, pair evidence | MATCH in scope; reduction is assumption |
| Eq.11 reduction | sum notation in paper | mean over batch and pairs | IMPLEMENTATION-ASSUMPTION |

## Eq.5 expansion

For one pair, `r=h-gamma`, `S=((kappa+1)/(kappa*df))*Psi`, `df=nu-d+1`:

`log p(h) = lgamma((df+d)/2) - lgamma(df/2) - 0.5*(d*log(df*pi)+log|S|) - ((df+d)/2)*log(1 + r^T S^-1 r / df)`.

For diagonal `Psi`, `log|S| = sum_i log(S_i)` and `r^T S^-1 r = sum_i r_i^2/S_i`. The active code implements these as `scale_diag`, `torch.log(scale_diag).sum(dim=-1)`, and `torch.sum(residual.square()/scale_diag, dim=-1)` in `diagonal_multivariate_student_t_nll`. The returned value is the negative mean log density.

The Student-t scale `S` is not the covariance `Sigma_ale`. Eq.7 is the aleatoric covariance component and uses the different denominator `nu-d-1`.

## Reduction and semantic cautions

Paper sum versus current mean changes the absolute scalar and gradient magnitude by the number of batch/pair terms. If NLL and regularizer use the same reduction, their common factor cancels in a simple ratio; however, mixing reductions or using the inactive elementwise path changes effective balance. The active baseline uses diagonal multivariate NLL plus pair regularizer.

The pair regularizer contains `kappa+nu` but no `Psi`, so `d L_reg / d raw_Psi = 0` is intrinsic to this Eq.11 implementation, not a numerical accident. Aleatoric can therefore be changed by `Psi` through NLL and by `nu` through both NLL and regularization; `kappa` affects Epistemic directly but not Aleatoric directly.

The paper's full `Psi` can encode frequency-domain covariance in the quadratic and determinant. Diagonal `Psi` retains marginal variances but removes those off-diagonal terms. This is a semantic risk when delay spread changes frequency correlation, but the audit does not establish it as the sole cause.

## Paper unknowns and prototype assumptions

The supplied paper does not expose the exact network head layout, noise placement, TDL-A choice, 5k/10 epoch pilot scale, or reduction convention used here. Those are IMPLEMENTATION-ASSUMPTIONS, not paper claims.
