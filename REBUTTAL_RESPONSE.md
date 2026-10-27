# Response Tables

Scenes: ecological (n=9, 372 candidates), social contagion (n=8, 210), neuronal (n=30, 174406).
All AUC entries: mean ± std over seeds. Empty cells are not yet run.

---

## Reviewer 1

### Comment 4

> Given the assumption that (f) and (f^(k)) are known, it is unclear why the authors learn the dynamics from scratch. Under this assumption, learning the interaction tensor (B) appears to reduce essentially to a linear regression problem, as shown in Section 4. Consequently, the proposed PINN-based framework seems unnecessarily complicated for the problem being considered. The authors should provide a convincing justification for why this formulation is needed and explain what advantages it offers over directly using the linear formulation in Section 4 with established methods such as integral SINDy. In particular, the authors should clarify what is gained by the proposed approach beyond what can already be achieved with standard regression-based methods.

**Table R1.4 — IntegralPINN vs. regression-based estimators on the same candidate library and the same known f, f^(k)**

| Scene | Method | 2-edges | 3-edges | 4-edges | 5-edges | Macro-AUC |
|---|---|---|---|---|---|---|
| Ecological (n=9) | Ours (IntegralPINN) | 0.457 ± 0.031 | 0.628 ± 0.050 | 0.401 ± 0.111 | 0.489 ± 0.190 | 0.494 ± 0.086 |
| Ecological (n=9) | Integral SINDy (STLSQ) | 0.435 ± 0.088 | 0.500 ± 0.047 | 0.493 ± 0.038 | 0.490 ± 0.158 | 0.479 ± 0.054 |
| Ecological (n=9) | Derivative SINDy (STLSQ) | 0.391 ± 0.075 | 0.460 ± 0.084 | 0.497 ± 0.099 | 0.609 ± 0.165 | 0.489 ± 0.056 |
| Ecological (n=9) | Linear least squares | 0.435 ± 0.088 | 0.500 ± 0.047 | 0.493 ± 0.038 | 0.490 ± 0.158 | 0.479 ± 0.054 |
| Social (n=8) | Ours (IntegralPINN) | 0.589 ± 0.065 | 0.674 ± 0.085 | 0.631 ± 0.098 | 0.570 ± 0.143 | 0.616 ± 0.064 |
| Social (n=8) | Integral SINDy (STLSQ) | 0.462 ± 0.042 | 0.498 ± 0.019 | 0.503 ± 0.004 | 0.501 ± 0.003 | 0.491 ± 0.012 |
| Social (n=8) | Derivative SINDy (STLSQ) | 0.501 ± 0.038 | 0.470 ± 0.066 | 0.513 ± 0.011 | 0.521 ± 0.015 | 0.501 ± 0.020 |
| Social (n=8) | Linear least squares | 0.462 ± 0.042 | 0.498 ± 0.019 | 0.503 ± 0.004 | 0.501 ± 0.003 | 0.491 ± 0.012 |
| Neuronal (n=30) | Ours (IntegralPINN) | 0.494 ± 0.022 | 0.504 ± 0.009 | 0.501 ± 0.027 | 0.544 ± 0.101 | 0.511 ± 0.027 |
| Neuronal (n=30) | Integral SINDy (STLSQ) | 0.546 ± 0.022 | 0.498 ± 0.040 | 0.498 ± 0.012 | 0.496 ± 0.027 | 0.510 ± 0.016 |
| Neuronal (n=30) | Derivative SINDy (STLSQ) | 0.514 ± 0.028 | 0.500 ± 0.058 | 0.502 ± 0.012 | 0.502 ± 0.001 | 0.505 ± 0.020 |
| Neuronal (n=30) | Linear least squares | 0.546 ± 0.022 | 0.498 ± 0.040 | 0.498 ± 0.012 | 0.496 ± 0.027 | 0.510 ± 0.016 |

---

### Comment 5

> It is not clear how the proposed interaction tensor parametrization compares with existing methods that explicitly exploit sparsity. This is particularly important for high-order interactions, where the number of possible interaction terms can grow rapidly. If sparsity is not explicitly encouraged in the proposed formulation, the authors should discuss why their approach is preferable to existing sparse identification methods and what advantages it offers in this setting.

**Table R1.5a — Proposed parametrization vs. explicitly sparsity-promoting estimation**

| Scene | Method | Sparsity mechanism | 2-edges | 3-edges | 4-edges | 5-edges | Macro-AUC | Non-zeros recovered |
|---|---|---|---|---|---|---|---|---|
| Ecological (n=9) | Ours | bounded sigmoid, none explicit | 0.457 ± 0.031 | 0.628 ± 0.050 | 0.401 ± 0.111 | 0.489 ± 0.190 | 0.494 ± 0.086 | 25 ± 32 |
| Ecological (n=9) | Ours + L1 | explicit L1 | 0.517 ± 0.007 | 0.648 ± 0.022 | 0.462 ± 0.041 | 0.617 ± 0.004 | 0.561 ± 0.012 | 4 ± 0 |
| Ecological (n=9) | Integral SINDy | STLSQ hard thresholding | 0.435 ± 0.088 | 0.500 ± 0.047 | 0.493 ± 0.038 | 0.490 ± 0.158 | 0.479 ± 0.054 | 372 ± 0 |
| Social (n=8) | Ours | bounded sigmoid, none explicit | 0.589 ± 0.065 | 0.674 ± 0.085 | 0.631 ± 0.098 | 0.570 ± 0.143 | 0.616 ± 0.064 | 0 ± 0 |
| Social (n=8) | Ours + L1 | explicit L1 | 0.502 ± 0.088 | 0.580 ± 0.138 | 0.520 ± 0.061 | 0.503 ± 0.057 | 0.526 ± 0.062 | 0 ± 0 |
| Social (n=8) | Integral SINDy | STLSQ hard thresholding | 0.462 ± 0.042 | 0.498 ± 0.019 | 0.503 ± 0.004 | 0.501 ± 0.003 | 0.491 ± 0.012 | 210 ± 0 |
| Neuronal (n=30) | Ours | bounded sigmoid, none explicit | 0.494 ± 0.022 | 0.504 ± 0.009 | 0.501 ± 0.027 | 0.544 ± 0.101 | 0.511 ± 0.027 | 7923 ± 437 |
| Neuronal (n=30) | Ours + L1 | explicit L1 | - | - | - | - | - | - |
| Neuronal (n=30) | Integral SINDy | STLSQ hard thresholding | 0.546 ± 0.022 | 0.498 ± 0.040 | 0.498 ± 0.012 | 0.496 ± 0.027 | 0.510 ± 0.016 | 9434 ± 1 |

**Table R1.5b — L1 weight sweep for the proposed method (supporting detail for Table R1.5a)**

| Scene | L1 weight | 2-edges | 3-edges | 4-edges | 5-edges | Macro-AUC |
|---|---|---|---|---|---|---|
| Ecological (n=9) | 0 | 0.457 ± 0.031 | 0.628 ± 0.050 | 0.401 ± 0.111 | 0.489 ± 0.190 | 0.494 ± 0.086 |
| Ecological (n=9) | 1e-4 | 0.506 ± 0.043 | 0.632 ± 0.021 | 0.446 ± 0.036 | 0.600 ± 0.046 | 0.546 ± 0.031 |
| Ecological (n=9) | 1e-3 | 0.517 ± 0.007 | 0.648 ± 0.022 | 0.462 ± 0.041 | 0.617 ± 0.004 | 0.561 ± 0.012 |
| Ecological (n=9) | 1e-2 | 0.513 ± 0.026 | 0.641 ± 0.017 | 0.453 ± 0.031 | 0.610 ± 0.011 | 0.554 ± 0.005 |
| Social (n=8) | 0 | 0.589 ± 0.065 | 0.674 ± 0.085 | 0.631 ± 0.098 | 0.570 ± 0.143 | 0.616 ± 0.064 |
| Social (n=8) | 1e-4 | 0.516 ± 0.036 | 0.585 ± 0.082 | 0.555 ± 0.048 | 0.503 ± 0.033 | 0.540 ± 0.035 |
| Social (n=8) | 1e-3 | 0.502 ± 0.088 | 0.580 ± 0.138 | 0.520 ± 0.061 | 0.503 ± 0.057 | 0.526 ± 0.062 |
| Social (n=8) | 1e-2 | 0.561 ± 0.074 | 0.685 ± 0.088 | 0.644 ± 0.174 | 0.627 ± 0.151 | 0.629 ± 0.114 |
| Neuronal (n=30) | 0 | 0.494 ± 0.022 | 0.504 ± 0.009 | 0.501 ± 0.027 | 0.544 ± 0.101 | 0.511 ± 0.027 |
| Neuronal (n=30) | 1e-4 | - | - | - | - | - |
| Neuronal (n=30) | 1e-3 | - | - | - | - | - |
| Neuronal (n=30) | 1e-2 | - | - | - | - | - |

---

## Reviewer 2

### Comment 5

> It would significantly increase the quality of the paper if the authors added experiments where L_physics is swapped with the derivative-based cost term. While HyperPINN might be a suitable comparison, it is unclear whether that formulation applied additional sparsity-promoting cost terms and if it used the same sigmoid-based parametrization as in this paper. Nonetheless, multiple experiments with varying sigma and h values would empirically strengthen the presentation of Theorems 1 and 2.

**Table R2.5a — L_physics swapped with the derivative-based cost term (identical surrogate, parametrization and optimizer)**

| Scene | Physics residual | 2-edges | 3-edges | 4-edges | 5-edges | Macro-AUC |
|---|---|---|---|---|---|---|
| Ecological (n=9) | Integral (ours) | 0.469 ± 0.032 | 0.609 ± 0.017 | 0.333 ± 0.037 | 0.376 ± 0.037 | 0.447 ± 0.028 |
| Ecological (n=9) | Derivative | 0.653 ± 0.008 | 0.535 ± 0.019 | 0.633 ± 0.022 | 0.865 ± 0.049 | 0.672 ± 0.018 |
| Social (n=8) | Integral (ours) | 0.561 ± 0.090 | 0.665 ± 0.066 | 0.641 ± 0.101 | 0.580 ± 0.107 | 0.612 ± 0.084 |
| Social (n=8) | Derivative | 0.586 ± 0.039 | 0.558 ± 0.056 | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.536 ± 0.007 |
| Neuronal (n=30) | Integral (ours) | - | - | - | - | - |
| Neuronal (n=30) | Derivative | - | - | - | - | - |

**Table R2.5b — HyperPINN configuration audit**

| Variant | Additional sparsity-promoting cost terms | Weights | Same sigmoid-based parametrization as this paper |
|---|---|---|---|
| hyperpinn / hyperpinn_base | L1 and L0, all orders | L1: 0.03 / 0.05 / 0.04 / 0.03; L0: 0.01 / 0.02 / 0.015 / 0.01 | Yes (sigmoid on per-hyperedge logits, Gumbel-sigmoid relaxation with temperature annealing) |
| hyperpinn_cp | L1 on 2/3-edges + factor penalty | 0.03 / 0.05 outer, 0.01 / 0.01 inner | No (softplus low-rank CP factors) |
| hyperpinn_tucker | L1 on 2/3-edges + factor penalty | 0.03 / 0.05 outer, 0.01 / 0.01 inner | No (softplus low-rank factors) |
| hyperpinn_tensor_train | L1 on 2/3-edges + factor penalty | 0.03 / 0.05 outer, 0.01 / 0.01 inner | No (softplus low-rank factors) |
| hyperpinn_ITC_0 | L1 on 2/3-edges + factor penalty | 0.01 / 0.01 | No (softplus low-rank factors from node descriptors) |
| hyperpinn_ITC_sup | L1 on 2/3-edges + factor penalty | 0.01 / 0.01 | No (softplus low-rank factors; additionally uses 50% of true hyperedges as supervision) |

**Table R2.5c — Macro-AUC over observation noise sigma and integral window length h (Theorems 1 and 2). The derivative residual has no window parameter and is listed as a single reference column.**

| Scene | sigma | h=0.1 T | h=0.25 T | h=0.5 T | h=1.0 T | Derivative residual (no h) |
|---|---|---|---|---|---|---|
| Ecological (n=9) | 0.00 | 0.677 ± 0.041 | 0.526 ± 0.088 | 0.551 ± 0.006 | 0.424 ± 0.044 | 0.745 ± 0.001 |
| Ecological (n=9) | 0.01 | 0.691 ± 0.013 | 0.481 ± 0.016 | 0.547 ± 0.006 | 0.430 ± 0.055 | 0.691 ± 0.012 |
| Ecological (n=9) | 0.02 | 0.682 ± 0.026 | 0.466 ± 0.013 | 0.563 ± 0.010 | 0.440 ± 0.080 | 0.664 ± 0.008 |
| Ecological (n=9) | 0.05 | 0.679 ± 0.038 | 0.458 ± 0.018 | 0.423 ± 0.145 | 0.418 ± 0.045 | 0.641 ± 0.009 |
| Ecological (n=9) | 0.10 | 0.681 ± 0.054 | 0.433 ± 0.014 | 0.491 ± 0.029 | 0.444 ± 0.031 | 0.532 ± 0.034 |
| Social (n=8) | 0.00 | 0.522 ± 0.021 | 0.519 ± 0.033 | 0.612 ± 0.129 | 0.661 ± 0.044 | 0.516 ± 0.028 |
| Social (n=8) | 0.01 | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.612 ± 0.120 | 0.690 ± 0.038 | 0.528 ± 0.013 |
| Social (n=8) | 0.02 | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.522 ± 0.085 | 0.630 ± 0.083 | 0.535 ± 0.038 |
| Social (n=8) | 0.05 | 0.505 ± 0.008 | 0.500 ± 0.000 | 0.613 ± 0.098 | 0.576 ± 0.095 | 0.521 ± 0.026 |
| Social (n=8) | 0.10 | 0.537 ± 0.024 | 0.503 ± 0.006 | 0.622 ± 0.083 | 0.639 ± 0.049 | 0.504 ± 0.007 |
| Neuronal (n=30) | 0.00 | - | - | - | - | - |
| Neuronal (n=30) | 0.01 | - | - | - | - | - |
| Neuronal (n=30) | 0.02 | - | - | - | - | - |
| Neuronal (n=30) | 0.05 | - | - | - | - | - |
| Neuronal (n=30) | 0.10 | - | - | - | - | - |

Cells at exactly 0.500 ± 0.000 are degenerate runs: on the binary social scene a short window makes the state increment x(t_b) - x(t_a) too small relative to the observation noise, the physics residual carries almost no gradient, and every entry of W is driven to the lower bound, so all candidates receive an identical score. This is the method failing in that regime, not a missing run.

---

### Comment 6

> The reproducibility of the experiments is somewhat difficult to assess. Repeating the experiments multiple times for each configuration would provide a more robust statistical assessment of the reported results and clarify the extent to which run-to-run variability affects the conclusions. Furthermore, the proposed method achieves AUROC values in the range of 0.6 - 0.8 across the benchmark studies, which, while outperforming the baseline methods, can be considered relatively modest. The comparatively poor performance of the baselines may also indicate that they were not sufficiently optimized or tuned for the considered tasks. Therefore, the authors are encouraged to provide more details on the hyperparameter tuning and configuration procedures used for the baseline methods, including whether the same tuning effort was applied across all approaches.

**Table R2.6a — Run-to-run variability over seeds (mean ± std, 95% CI). Same runs as Table R3.6a.**

| Scene | Method | 2-edges | 3-edges | 4-edges | 5-edges | Macro-AUC | 95% CI | n seeds |
|---|---|---|---|---|---|---|---|---|
| Ecological (n=9) | Ours | 0.457 ± 0.031 | 0.628 ± 0.050 | 0.401 ± 0.111 | 0.489 ± 0.190 | 0.494 ± 0.086 | [0.432, 0.556] | 10 |
| Ecological (n=9) | HyperPINN | 0.619 ± 0.048 | 0.563 ± 0.079 | 0.603 ± 0.045 | 0.717 ± 0.173 | 0.625 ± 0.069 | [0.576, 0.675] | 10 |
| Ecological (n=9) | Kernel-based | 0.438 ± 0.082 | 0.521 ± 0.068 | 0.548 ± 0.067 | 0.548 ± 0.146 | 0.514 ± 0.043 | [0.483, 0.545] | 10 |
| Ecological (n=9) | THIS | 0.541 ± 0.108 | 0.497 ± 0.072 | 0.512 ± 0.130 | 0.611 ± 0.174 | 0.540 ± 0.060 | [0.498, 0.583] | 10 |
| Social (n=8) | Ours | 0.589 ± 0.065 | 0.674 ± 0.085 | 0.631 ± 0.098 | 0.570 ± 0.143 | 0.616 ± 0.064 | [0.570, 0.662] | 10 |
| Social (n=8) | HyperPINN | 0.600 ± 0.094 | 0.679 ± 0.083 | 0.641 ± 0.076 | 0.419 ± 0.280 | 0.585 ± 0.074 | [0.532, 0.638] | 10 |
| Social (n=8) | Kernel-based | 0.471 ± 0.111 | 0.503 ± 0.068 | 0.542 ± 0.108 | 0.475 ± 0.249 | 0.498 ± 0.067 | [0.450, 0.545] | 10 |
| Social (n=8) | THIS | 0.483 ± 0.106 | 0.429 ± 0.113 | 0.427 ± 0.074 | 0.363 ± 0.176 | 0.425 ± 0.075 | [0.372, 0.479] | 10 |
| Neuronal (n=30) | Ours | 0.494 ± 0.022 | 0.504 ± 0.009 | 0.501 ± 0.027 | 0.544 ± 0.101 | 0.511 ± 0.027 | [0.492, 0.530] | 10 |
| Neuronal (n=30) | HyperPINN | 0.482 ± 0.034 | 0.518 ± 0.048 | 0.499 ± 0.045 | 0.516 ± 0.073 | 0.504 ± 0.021 | [0.487, 0.520] | 9 |
| Neuronal (n=30) | Kernel-based | 0.500 ± 0.000 | 0.518 ± 0.053 | 0.490 ± 0.074 | 0.522 ± 0.053 | 0.508 ± 0.032 | [0.485, 0.530] | 10 |
| Neuronal (n=30) | THIS | - | - | - | - | - | - | 0 |

**Table R2.6b — Equal-budget hyperparameter tuning protocol**

| Method | Tuned hyperparameters | Grid | Configurations | Selection seed | Evaluation seeds |
|---|---|---|---|---|---|
| Ours (IntegralPINN) | learning rate x number of sampled intervals | lr {1e-4, 5e-4, 1e-3, 5e-3} x n_intervals {8, 32, 64, 128} | 16 | 11 | 12, 13 |
| HyperPINN | learning rate x epoch budget | lr {1e-4, 5e-4, 1e-3, 5e-3} x epochs {7000, 14000} | 16 | 11 | 12, 13 |
| Kernel-based | not tuned | — | not tuned | - | 11-20 |
| THIS | not tuned | — | not tuned | - | 11-20 |
| Integral SINDy | STLSQ threshold x integral window | threshold {1e-4, 1e-3, 1e-2, 1e-1} x window {5, 20, 50, 100} | 16 | 11 | 12, 13 |

**Table R2.6c — Best configuration found per method, per scene**

| Scene | Method | Best configuration | Macro-AUC (tuning seed) | Macro-AUC (held-out seeds) |
|---|---|---|---|---|
| Ecological (n=9) | Ours | lr=0.005, n_intervals=128 | 0.624 | 0.494 ± 0.086 |
| Ecological (n=9) | HyperPINN | lr=0.0005, epochs=14000 | 0.678 | 0.625 ± 0.069 |
| Ecological (n=9) | Kernel-based | | | |
| Ecological (n=9) | THIS | | | |
| Ecological (n=9) | Integral SINDy | threshold=0.0001, window=20 | 0.504 | 0.479 ± 0.054 |
| Social (n=8) | Ours | lr=0.001, n_intervals=16 | 0.722 | 0.616 ± 0.064 |
| Social (n=8) | HyperPINN | lr=0.005, epochs=7000 | 0.681 | 0.585 ± 0.074 |
| Social (n=8) | Kernel-based | | | |
| Social (n=8) | THIS | | | |
| Social (n=8) | Integral SINDy | threshold=0.0001, window=5 | 0.511 | 0.491 ± 0.012 |
| Neuronal (n=30) | Ours | - | - | - |
| Neuronal (n=30) | HyperPINN | - | - | - |
| Neuronal (n=30) | Kernel-based | | | |
| Neuronal (n=30) | THIS | | | |
| Neuronal (n=30) | Integral SINDy | - | - | - |

---

## Reviewer 3

### Comment 6

> Second, improve the experimental validation. Please report standard deviations over multiple random seeds, include statistical significance or confidence intervals, and evaluate sensitivity to noise level.

**Table R3.6a — Standard deviations, confidence intervals and significance vs. each baseline. Same runs as Table R2.6a.**

| Scene | Method | Macro-AUC (mean ± std) | 95% CI | Δ vs Ours | paired t p | Wilcoxon p | n seeds |
|---|---|---|---|---|---|---|---|
| Ecological (n=9) | Ours | 0.494 ± 0.086 | [0.432, 0.556] | — | — | — | 10 |
| Ecological (n=9) | HyperPINN | 0.625 ± 0.069 | [0.576, 0.675] | +0.131 | 0.0003 | 0.0020 | 10 |
| Ecological (n=9) | Kernel-based | 0.514 ± 0.043 | [0.483, 0.545] | +0.023 | 0.5991 | 0.8203 | 10 |
| Ecological (n=9) | THIS | 0.540 ± 0.060 | [0.498, 0.583] | +0.046 | 0.2398 | 0.2754 | 10 |
| Ecological (n=9) | Integral SINDy | 0.479 ± 0.054 | [0.441, 0.518] | -0.015 | 0.6991 | 0.6953 | 10 |
| Social (n=8) | Ours | 0.616 ± 0.064 | [0.570, 0.662] | — | — | — | 10 |
| Social (n=8) | HyperPINN | 0.585 ± 0.074 | [0.532, 0.638] | -0.031 | 0.3325 | 0.1934 | 10 |
| Social (n=8) | Kernel-based | 0.498 ± 0.067 | [0.450, 0.545] | -0.119 | 0.0009 | 0.0059 | 10 |
| Social (n=8) | THIS | 0.425 ± 0.075 | [0.372, 0.479] | -0.191 | 0.0000 | 0.0020 | 10 |
| Social (n=8) | Integral SINDy | 0.491 ± 0.012 | [0.482, 0.500] | -0.125 | 0.0001 | 0.0020 | 10 |
| Neuronal (n=30) | Ours | 0.511 ± 0.027 | [0.492, 0.530] | — | — | — | 10 |
| Neuronal (n=30) | HyperPINN | 0.504 ± 0.021 | [0.487, 0.520] | -0.007 | 0.5961 | 0.5703 | 9 |
| Neuronal (n=30) | Kernel-based | 0.508 ± 0.032 | [0.485, 0.530] | -0.003 | 0.6488 | 0.9219 | 10 |
| Neuronal (n=30) | THIS | - | - | - | - | - | 0 |
| Neuronal (n=30) | Integral SINDy | 0.510 ± 0.016 | [0.498, 0.521] | -0.001 | 0.8822 | 1.0000 | 10 |

**Table R3.6b — Sensitivity to noise level (Macro-AUC, mean ± std)**

| Scene | Method | sigma=0.00 | sigma=0.01 | sigma=0.02 | sigma=0.05 | sigma=0.10 | sigma=0.20 |
|---|---|---|---|---|---|---|---|
| Ecological (n=9) | Ours | 0.555 ± 0.005 | 0.561 ± 0.015 | 0.540 ± 0.038 | 0.335 ± 0.162 | 0.420 ± 0.113 | 0.552 ± 0.093 |
| Ecological (n=9) | HyperPINN | 0.657 ± 0.023 | 0.656 ± 0.022 | 0.653 ± 0.021 | 0.641 ± 0.018 | 0.605 ± 0.010 | 0.536 ± 0.050 |
| Ecological (n=9) | Kernel-based | 0.410 ± 0.000 | 0.565 ± 0.009 | 0.550 ± 0.017 | 0.449 ± 0.034 | 0.474 ± 0.046 | 0.539 ± 0.046 |
| Ecological (n=9) | THIS | 0.526 ± 0.049 | 0.548 ± 0.030 | 0.529 ± 0.008 | 0.505 ± 0.050 | 0.546 ± 0.061 | 0.514 ± 0.052 |
| Ecological (n=9) | Integral SINDy | 0.518 ± 0.000 | 0.537 ± 0.052 | 0.508 ± 0.030 | 0.469 ± 0.025 | 0.491 ± 0.058 | 0.475 ± 0.057 |
| Social (n=8) | Ours | 0.609 ± 0.122 | 0.626 ± 0.107 | 0.533 ± 0.064 | 0.623 ± 0.071 | 0.597 ± 0.105 | 0.637 ± 0.133 |
| Social (n=8) | HyperPINN | 0.620 ± 0.069 | 0.613 ± 0.050 | 0.621 ± 0.044 | 0.623 ± 0.041 | 0.624 ± 0.063 | 0.618 ± 0.055 |
| Social (n=8) | Kernel-based | 0.407 ± 0.105 | 0.486 ± 0.070 | 0.498 ± 0.073 | 0.534 ± 0.102 | 0.556 ± 0.040 | 0.478 ± 0.029 |
| Social (n=8) | THIS | 0.445 ± 0.129 | 0.478 ± 0.022 | 0.474 ± 0.026 | 0.475 ± 0.016 | 0.435 ± 0.067 | 0.438 ± 0.050 |
| Social (n=8) | Integral SINDy | 0.500 ± 0.045 | 0.503 ± 0.009 | 0.489 ± 0.012 | 0.503 ± 0.001 | 0.507 ± 0.007 | 0.503 ± 0.001 |
| Neuronal (n=30) | Ours | - | - | - | - | - | - |
| Neuronal (n=30) | HyperPINN | - | - | - | - | - | - |
| Neuronal (n=30) | Kernel-based | - | - | - | - | - | - |
| Neuronal (n=30) | THIS | - | - | - | - | - | - |
| Neuronal (n=30) | Integral SINDy | - | - | - | - | - | - |

---

### Comment 7

> Third, include ablations. At minimum, compare: derivative PINN vs integral PINN using the same architecture; with and without staged λ scheduling; with and without the bounded tensor parameterization; different integral window lengths; and different numbers of sampled intervals. These ablations are important to show which component actually drives the improvement.

**Table R3.7a — Component ablations (one switch at a time against the full method)**

| Scene | Variant | 2-edges | 3-edges | 4-edges | 5-edges | Macro-AUC | Δ vs full | paired t p |
|---|---|---|---|---|---|---|---|---|
| Ecological (n=9) | Full method | 0.469 ± 0.032 | 0.609 ± 0.017 | 0.333 ± 0.037 | 0.376 ± 0.037 | 0.447 ± 0.028 | — | — |
| Ecological (n=9) | Derivative PINN (same architecture, cf. Table R2.5a) | 0.653 ± 0.008 | 0.535 ± 0.019 | 0.633 ± 0.022 | 0.865 ± 0.049 | 0.672 ± 0.018 | +0.225 | 0.0042 |
| Ecological (n=9) | Without staged λ scheduling | 0.467 ± 0.092 | 0.614 ± 0.013 | 0.363 ± 0.118 | 0.473 ± 0.255 | 0.479 ± 0.113 | +0.033 | 0.5794 |
| Ecological (n=9) | Without bounded tensor parameterization | 0.626 ± 0.026 | 0.503 ± 0.109 | 0.336 ± 0.126 | 0.358 ± 0.237 | 0.456 ± 0.119 | +0.009 | 0.9217 |
| Social (n=8) | Full method | 0.561 ± 0.090 | 0.665 ± 0.066 | 0.641 ± 0.101 | 0.580 ± 0.107 | 0.612 ± 0.084 | — | — |
| Social (n=8) | Derivative PINN (same architecture, cf. Table R2.5a) | 0.586 ± 0.039 | 0.558 ± 0.056 | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.536 ± 0.007 | -0.076 | 0.2848 |
| Social (n=8) | Without staged λ scheduling | 0.474 ± 0.059 | 0.482 ± 0.063 | 0.455 ± 0.056 | 0.543 ± 0.104 | 0.488 ± 0.043 | -0.123 | 0.2317 |
| Social (n=8) | Without bounded tensor parameterization | 0.582 ± 0.031 | 0.762 ± 0.033 | 0.732 ± 0.100 | 0.722 ± 0.100 | 0.700 ± 0.038 | +0.088 | 0.2267 |
| Neuronal (n=30) | Full method | - | - | - | - | - | — | — |
| Neuronal (n=30) | Derivative PINN (same architecture, cf. Table R2.5a) | - | - | - | - | - | - | - |
| Neuronal (n=30) | Without staged λ scheduling | - | - | - | - | - | - | - |
| Neuronal (n=30) | Without bounded tensor parameterization | - | - | - | - | - | - | - |

**Table R3.7b — Integral window length**

| Scene | Window length h | 2-edges | 3-edges | 4-edges | 5-edges | Macro-AUC |
|---|---|---|---|---|---|---|
| Ecological (n=9) | 0.1 T | 0.666 ± 0.013 | 0.640 ± 0.042 | 0.583 ± 0.076 | 0.708 ± 0.042 | 0.649 ± 0.031 |
| Ecological (n=9) | 0.25 T | 0.477 ± 0.259 | 0.691 ± 0.038 | 0.593 ± 0.160 | 0.801 ± 0.167 | 0.641 ± 0.146 |
| Ecological (n=9) | 0.5 T | 0.469 ± 0.032 | 0.609 ± 0.017 | 0.333 ± 0.037 | 0.376 ± 0.037 | 0.447 ± 0.028 |
| Ecological (n=9) | 1.0 T | 0.565 ± 0.076 | 0.516 ± 0.041 | 0.270 ± 0.128 | 0.243 ± 0.198 | 0.399 ± 0.110 |
| Social (n=8) | 0.1 T | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.500 ± 0.000 |
| Social (n=8) | 0.25 T | 0.514 ± 0.024 | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.494 ± 0.011 | 0.502 ± 0.008 |
| Social (n=8) | 0.5 T | 0.561 ± 0.090 | 0.665 ± 0.066 | 0.641 ± 0.101 | 0.580 ± 0.107 | 0.612 ± 0.084 |
| Social (n=8) | 1.0 T | 0.557 ± 0.055 | 0.644 ± 0.141 | 0.684 ± 0.091 | 0.676 ± 0.088 | 0.640 ± 0.075 |
| Neuronal (n=30) | 0.1 T | - | - | - | - | - |
| Neuronal (n=30) | 0.25 T | - | - | - | - | - |
| Neuronal (n=30) | 0.5 T | - | - | - | - | - |
| Neuronal (n=30) | 1.0 T | - | - | - | - | - |

**Table R3.7c — Number of sampled intervals**

| Scene | Sampled intervals | 2-edges | 3-edges | 4-edges | 5-edges | Macro-AUC |
|---|---|---|---|---|---|---|
| Ecological (n=9) | 8 | 0.564 ± 0.094 | 0.566 ± 0.016 | 0.392 ± 0.064 | 0.515 ± 0.006 | 0.509 ± 0.036 |
| Ecological (n=9) | 32 | 0.521 ± 0.049 | 0.626 ± 0.043 | 0.418 ± 0.009 | 0.554 ± 0.045 | 0.529 ± 0.022 |
| Ecological (n=9) | 64 | 0.469 ± 0.032 | 0.609 ± 0.017 | 0.333 ± 0.037 | 0.376 ± 0.037 | 0.447 ± 0.028 |
| Ecological (n=9) | 128 | 0.454 ± 0.070 | 0.593 ± 0.088 | 0.288 ± 0.190 | 0.281 ± 0.312 | 0.404 ± 0.165 |
| Social (n=8) | 8 | 0.559 ± 0.074 | 0.714 ± 0.033 | 0.626 ± 0.136 | 0.691 ± 0.070 | 0.648 ± 0.076 |
| Social (n=8) | 32 | 0.562 ± 0.097 | 0.685 ± 0.105 | 0.650 ± 0.123 | 0.627 ± 0.193 | 0.631 ± 0.129 |
| Social (n=8) | 64 | 0.561 ± 0.090 | 0.665 ± 0.066 | 0.641 ± 0.101 | 0.580 ± 0.107 | 0.612 ± 0.084 |
| Social (n=8) | 128 | 0.573 ± 0.051 | 0.701 ± 0.016 | 0.623 ± 0.120 | 0.610 ± 0.128 | 0.627 ± 0.044 |
| Neuronal (n=30) | 8 | - | - | - | - | - |
| Neuronal (n=30) | 32 | - | - | - | - | - |
| Neuronal (n=30) | 64 | - | - | - | - | - |
| Neuronal (n=30) | 128 | - | - | - | - | - |

