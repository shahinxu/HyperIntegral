# Supplementary Tables

## Table S1 — Default hyperparameters of all methods

Every number below is read from the code that produced the results, not from
the paper draft. The source file is given for each block.

### IntegralPINN (ours)

| Hyperparameter | Value |
|---|---|
| learning rate | 1e-3 |
| epochs | 20 000 |
| data-only stage | 2 500 epochs |
| physics ramp | 2 000 epochs |
| λ_physics after the ramp | 1.0 |
| sampled intervals per step | 64 |
| Gauss–Legendre nodes per interval | 5 |
| maximum window length | 0.5 T |
| pre-activation clip c | 2.0 |
| pre-activation init w₀ | −1.5 |
| L1 weight | 0 |
| surrogate | residual MLP, 8 layers × 64 units |

### HyperPINN

| Hyperparameter | Value |
|---|---|
| learning rate | 5e-4 |
| epochs | 14 000 (3 000 on neuronal) |
| physics weight schedule | 0.1 → 0.01 + 0.99·progress → 1.0 |
| L1, orders 2/3/4/5 | 0.03 / 0.05 / 0.04 / 0.03 |
| L0, orders 2/3/4/5 | 0.01 / 0.02 / 0.015 / 0.01 |
| Gumbel-sigmoid temperature | 1.0, ×0.995 every 100 epochs, floor 0.5 |

### Kernel-based

| Hyperparameter | Value |
|---|---|
| bandwidth σ | chosen on a validation split from {0.2, 0.5, 1.0, 2.0} |
| ridge γ | chosen on the same split from {1e-6, 1e-5, 1e-4, 1e-3} |
| kernel ridge | 1e-4 |
| θ ridge | 1e-8 |

### THIS

| Hyperparameter | Value |
|---|---|
| sparsity penalty λ | 0.05 |
| ADMM penalty ρ | 0.001 |
| iterations | 10 |

### SINDy variants

| Variant | form | estimator | threshold | window | ridge |
|---|---|---|---|---|---|
| Integral SINDy | integral | STLSQ | 0.01 | 20 samples | 1e-8 |
| Derivative SINDy | derivative (central difference) | STLSQ | 0.01 | — | 1e-8 |
| Linear least squares | integral | lstsq (no thresholding) | — | 20 samples | 1e-8 |

### Shared across every method

| Setting | Value |
|---|---|
| samples per trajectory | 300 |
| observation noise σ | 0.02 |
| seeds | 11–20 (ten runs) |
| candidate pool, ecological / social | complete: 372 / 210 |
| candidate pool, neuronal | all true hyperedges + 3 000 negatives per order = 9 435 |

### Note

1. **Kernel-based does not run at a fixed configuration.** It selects σ and γ
   on a validation split from a 4 × 4 grid, so it receives a layer of tuning
   the other methods do not get at their defaults.

2. **HyperPINN carries L1 and L0 sparsity terms while the proposed method
   defaults to L1 = 0.** At default settings the two therefore differ by more
   than the integral-versus-derivative residual. Table R2.5b in the response
   sets out the full audit.

3. **HyperPINN runs 3 000 epochs on neuronal rather than 14 000.** At n = 30 a
   single run of the full budget does not finish inside the job timeout. The
   reduction is a constraint, not a choice, and should be declared.
