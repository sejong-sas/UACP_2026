# Evidential Formula / Semantic Consistency Audit

Read-only audit of the current valid baseline. No training or optimizer step was performed.

- Device: `cuda:0`; GPU: `NVIDIA GB10`
- Checkpoint SHA-256 preserved: `True`
- All audit outputs finite: `True`

## Main conclusions

1. The active diagonal multivariate Student-t path matches the diagonal specialization of the paper's Eq.5, Eq.7, and Eq.8 algebraically.
2. The current pair mean reduction is an implementation assumption relative to paper sum notation; exact values and gradient norms are in the CSV files.
3. The active Eq.11 regularizer has zero direct raw-Psi gradient because its implemented factor is `(kappa+nu)`. Its direct pressure is on kappa/nu and gamma.
4. Full-Psi frequency covariance is paper-specified, while diagonal Psi is an approximation. Existing empirical lag-correlation separation makes semantic loss of off-diagonal information plausible, but this audit does not prove causality.

## Files

- `paper_code_mapping.md`
- `equation_audit.json`
- `reduction_gradient_comparison.csv`
- `regularizer_gradient_comparison.csv`
- `parameter_sensitivity.csv`
- `parameter_operating_range.csv`
- `raw_parameter_operating_range.csv`
- `toy_full_vs_diagonal.csv`
