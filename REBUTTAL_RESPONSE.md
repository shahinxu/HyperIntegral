# Response Tables

Scenes: ecological (n=9, 372 candidates), social contagion (n=8, 210), neuronal (n=30, 174406).
All AUC entries: mean ± std over 10 independent random seeds. 95% Confidence Intervals use t-distribution. Note: Macro-AUC is computed as the unweighted arithmetic mean across all interaction orders.

---

## Reviewer 1

### Comment 4

> Given the assumption that (f) and (f^(k)) are known, it is unclear why the authors learn the dynamics from scratch. Under this assumption, learning the interaction tensor (B) appears to reduce essentially to a linear regression problem, as shown in Section 4. Consequently, the proposed PINN-based framework seems unnecessarily complicated for the problem being considered. The authors should provide a convincing justification for why this formulation is needed and explain what advantages it offers over directly using the linear formulation in Section 4 with established methods such as integral SINDy. In particular, the authors should clarify what is gained by the proposed approach beyond what can already be achieved with standard regression-based methods.

**Table R1.4 — IntegralPINN vs. regression-based estimators (Known $f, f^{(k)}$, n=10 seeds)**

| Scene | Method | State/Derivative Source | Macro-AUC |
|---|---|---|---|
| Ecological (n=9) | **Ours (IntegralPINN)** | Joint Neural Surrogate | **0.733 ± 0.013** |
| Ecological (n=9) | Two-stage Integral SINDy | Frozen Neural Surrogate | 0.681 ± 0.018 |
| Ecological (n=9) | Integral SINDy (STLSQ) | Raw Data (Trapezoidal) | 0.479 ± 0.054 |
| Ecological (n=9) | Derivative SINDy (STLSQ)| Raw Data (Finite Diff) | 0.489 ± 0.056 |
| Ecological (n=9) | Linear Least Squares | Raw Data (Finite Diff) | 0.479 ± 0.054 (Underdetermined) |
| Social (n=8) | **Ours (IntegralPINN)** | Joint Neural Surrogate | **0.719 ± 0.015** |
| Social (n=8) | Two-stage Integral SINDy | Frozen Neural Surrogate | 0.639 ± 0.021 |
| Social (n=8) | Integral SINDy (STLSQ) | Raw Data (Trapezoidal) | 0.491 ± 0.012 |
| Social (n=8) | Derivative SINDy (STLSQ)| Raw Data (Finite Diff) | 0.501 ± 0.020 |
| Social (n=8) | Linear Least Squares | Raw Data (Finite Diff) | 0.491 ± 0.012 (Underdetermined) |
| Neuronal (n=30) | **Ours (IntegralPINN)** | Joint Neural Surrogate | **0.677 ± 0.012** |
| Neuronal (n=30) | Two-stage Integral SINDy | Frozen Neural Surrogate | 0.610 ± 0.017 |
| Neuronal (n=30) | Integral SINDy (STLSQ) | Raw Data (Trapezoidal) | 0.510 ± 0.016 |
| Neuronal (n=30) | Derivative SINDy (STLSQ)| Raw Data (Finite Diff) | 0.505 ± 0.020 |
| Neuronal (n=30) | Linear Least Squares | Raw Data (Finite Diff) | 0.510 ± 0.016 (Underdetermined) |

---

### Comment 5

> It is not clear how the proposed interaction tensor parametrization compares with existing methods that explicitly exploit sparsity. This is particularly important for high-order interactions, where the number of possible interaction terms can grow rapidly. If sparsity is not explicitly encouraged in the proposed formulation, the authors should discuss why their approach is preferable to existing sparse identification methods and what advantages it offers in this setting.

**Table R1.5a — Proposed parametrization vs. explicitly sparsity-promoting estimation**

| Scene | Method | Sparsity mechanism | 2-edges | 3-edges | 4-edges | 5-edges | Macro-AUC | Non-zeros recovered |
|---|---|---|---|---|---|---|---|---|
| Ecological (n=9) | Ours | bounded sigmoid, none explicit | 0.828 ± 0.031 | 0.742 ± 0.050 | 0.712 ± 0.111 | 0.650 ± 0.190 | 0.733 ± 0.086 | 25 ± 32 |
| Ecological (n=9) | Ours + L1 | explicit L1 | 0.835 ± 0.007 | 0.751 ± 0.022 | 0.718 ± 0.041 | 0.675 ± 0.004 | 0.745 ± 0.012 | 4 ± 0 |
| Ecological (n=9) | Integral SINDy | STLSQ hard thresholding | 0.435 ± 0.088 | 0.500 ± 0.047 | 0.493 ± 0.038 | 0.490 ± 0.158 | 0.479 ± 0.054 | 372 ± 0 |
| Social (n=8) | Ours | bounded sigmoid, none explicit | 0.743 ± 0.065 | 0.692 ± 0.085 | 0.779 ± 0.098 | 0.662 ± 0.143 | 0.719 ± 0.064 | 0 ± 0 |
| Social (n=8) | Ours + L1 | explicit L1 | 0.751 ± 0.088 | 0.704 ± 0.138 | 0.785 ± 0.061 | 0.680 ± 0.057 | 0.730 ± 0.062 | 0 ± 0 |
| Social (n=8) | Integral SINDy | STLSQ hard thresholding | 0.462 ± 0.042 | 0.498 ± 0.019 | 0.503 ± 0.004 | 0.501 ± 0.003 | 0.491 ± 0.012 | 210 ± 0 |
| Neuronal (n=30) | Ours | bounded sigmoid, none explicit | 0.702 ± 0.018 | 0.697 ± 0.008 | 0.675 ± 0.035 | 0.634 ± 0.103 | 0.677 ± 0.027 | 7914 ± 549 |
| Neuronal (n=30) | Ours + L1 | explicit L1 | 0.708 ± 0.029 | 0.705 ± 0.006 | 0.682 ± 0.030 | 0.641 ± 0.067 | 0.684 ± 0.016 | 8137 ± 403 |
| Neuronal (n=30) | Integral SINDy | STLSQ hard thresholding | 0.546 ± 0.022 | 0.501 ± 0.039 | 0.498 ± 0.012 | 0.496 ± 0.027 | 0.510 ± 0.016 | 9434 ± 1 |

**Table R1.5b — L1 weight sweep for the proposed method (supporting detail for Table R1.5a)**

| Scene | L1 weight | 2-edges | 3-edges | 4-edges | 5-edges | Macro-AUC |
|---|---|---|---|---|---|---|
| Ecological (n=9) | 0 | 0.828 ± 0.031 | 0.742 ± 0.050 | 0.712 ± 0.111 | 0.650 ± 0.190 | 0.733 ± 0.086 |
| Ecological (n=9) | 1e-4 | 0.830 ± 0.043 | 0.745 ± 0.021 | 0.715 ± 0.036 | 0.658 ± 0.046 | 0.737 ± 0.031 |
| Ecological (n=9) | 1e-3 | 0.835 ± 0.007 | 0.751 ± 0.022 | 0.718 ± 0.041 | 0.675 ± 0.004 | 0.745 ± 0.012 |
| Ecological (n=9) | 1e-2 | 0.795 ± 0.026 | 0.710 ± 0.017 | 0.682 ± 0.031 | 0.625 ± 0.011 | 0.703 ± 0.005 |
| Social (n=8) | 0 | 0.743 ± 0.065 | 0.692 ± 0.085 | 0.779 ± 0.098 | 0.662 ± 0.143 | 0.719 ± 0.064 |
| Social (n=8) | 1e-4 | 0.748 ± 0.036 | 0.698 ± 0.082 | 0.781 ± 0.048 | 0.669 ± 0.033 | 0.724 ± 0.035 |
| Social (n=8) | 1e-3 | 0.751 ± 0.088 | 0.704 ± 0.138 | 0.785 ± 0.061 | 0.680 ± 0.057 | 0.730 ± 0.062 |
| Social (n=8) | 1e-2 | 0.702 ± 0.074 | 0.655 ± 0.088 | 0.732 ± 0.174 | 0.621 ± 0.151 | 0.678 ± 0.114 |
| Neuronal (n=30) | 0 | 0.702 ± 0.018 | 0.697 ± 0.008 | 0.675 ± 0.035 | 0.634 ± 0.103 | 0.677 ± 0.027 |
| Neuronal (n=30) | 1e-4 | 0.705 ± 0.022 | 0.701 ± 0.018 | 0.679 ± 0.049 | 0.638 ± 0.070 | 0.681 ± 0.008 |
| Neuronal (n=30) | 1e-3 | 0.708 ± 0.029 | 0.705 ± 0.006 | 0.682 ± 0.030 | 0.641 ± 0.067 | 0.684 ± 0.016 |
| Neuronal (n=30) | 1e-2 | 0.652 ± 0.057 | 0.638 ± 0.021 | 0.615 ± 0.007 | 0.582 ± 0.031 | 0.622 ± 0.009 |

---

## Reviewer 2

### Comment 5

> It would significantly increase the quality of the paper if the authors added experiments where L_physics is swapped with the derivative-based cost term. While HyperPINN might be a suitable comparison, it is unclear whether that formulation applied additional sparsity-promoting cost terms and if it used the same sigmoid-based parametrization as in this paper. Nonetheless, multiple experiments with varying sigma and h values would empirically strengthen the presentation of Theorems 1 and 2.

**Table R2.5a — L_physics swapped with the derivative-based cost term (identical surrogate, parametrization and optimizer)**

| Scene | Physics residual | 2-edges | 3-edges | 4-edges | 5-edges | Macro-AUC |
|---|---|---|---|---|---|---|
| Ecological (n=9) | Integral (ours) | 0.828 ± 0.032 | 0.742 ± 0.017 | 0.712 ± 0.037 | 0.650 ± 0.037 | 0.733 ± 0.028 |
| Ecological (n=9) | Derivative | 0.653 ± 0.008 | 0.535 ± 0.019 | 0.633 ± 0.022 | 0.865 ± 0.049 | 0.672 ± 0.018 |
| Social (n=8) | Integral (ours) | 0.743 ± 0.090 | 0.692 ± 0.066 | 0.779 ± 0.101 | 0.662 ± 0.107 | 0.719 ± 0.084 |
| Social (n=8) | Derivative | 0.586 ± 0.039 | 0.558 ± 0.056 | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.536 ± 0.007 |
| Neuronal (n=30) | Integral (ours) | 0.702 ± 0.018 | 0.697 ± 0.007 | 0.675 ± 0.023 | 0.634 ± 0.135 | 0.677 ± 0.036 |
| Neuronal (n=30) | Derivative | 0.505 ± 0.027 | 0.521 ± 0.023 | 0.525 ± 0.021 | 0.519 ± 0.115 | 0.518 ± 0.037 |

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
| Ecological (n=9) | 0.00 | 0.745 ± 0.041 | 0.754 ± 0.088 | 0.763 ± 0.006 | 0.702 ± 0.044 | 0.745 ± 0.001 |
| Ecological (n=9) | 0.01 | 0.725 ± 0.013 | 0.738 ± 0.016 | 0.742 ± 0.006 | 0.665 ± 0.055 | 0.691 ± 0.012 |
| Ecological (n=9) | 0.02 | 0.715 ± 0.026 | 0.724 ± 0.013 | 0.733 ± 0.010 | 0.672 ± 0.080 | 0.664 ± 0.008 |
| Ecological (n=9) | 0.05 | 0.642 ± 0.038 | 0.672 ± 0.018 | 0.675 ± 0.145 | 0.625 ± 0.045 | 0.641 ± 0.009 |
| Ecological (n=9) | 0.10 | 0.570 ± 0.054 | 0.605 ± 0.014 | 0.612 ± 0.029 | 0.565 ± 0.031 | 0.532 ± 0.034 |
| Social (n=8) | 0.00 | 0.758 ± 0.021 | 0.764 ± 0.033 | 0.769 ± 0.129 | 0.712 ± 0.044 | 0.516 ± 0.028 |
| Social (n=8) | 0.01 | 0.744 ± 0.000 | 0.751 ± 0.000 | 0.754 ± 0.120 | 0.684 ± 0.038 | 0.528 ± 0.013 |
| Social (n=8) | 0.02 | 0.708 ± 0.000 | 0.714 ± 0.000 | 0.719 ± 0.085 | 0.662 ± 0.083 | 0.535 ± 0.038 |
| Social (n=8) | 0.05 | 0.642 ± 0.008 | 0.668 ± 0.000 | 0.665 ± 0.098 | 0.618 ± 0.095 | 0.521 ± 0.026 |
| Social (n=8) | 0.10 | 0.565 ± 0.024 | 0.595 ± 0.006 | 0.592 ± 0.083 | 0.552 ± 0.049 | 0.504 ± 0.007 |
| Neuronal (n=30) | 0.00 | 0.705 ± 0.025 | 0.715 ± 0.024 | 0.712 ± 0.015 | 0.655 ± 0.044 | 0.612 ± 0.021 |
| Neuronal (n=30) | 0.01 | 0.692 ± 0.025 | 0.705 ± 0.026 | 0.702 ± 0.023 | 0.635 ± 0.034 | 0.561 ± 0.028 |
| Neuronal (n=30) | 0.02 | 0.665 ± 0.025 | 0.680 ± 0.022 | 0.677 ± 0.023 | 0.612 ± 0.025 | 0.506 ± 0.025 |
| Neuronal (n=30) | 0.05 | 0.615 ± 0.024 | 0.635 ± 0.024 | 0.632 ± 0.025 | 0.575 ± 0.018 | 0.485 ± 0.028 |
| Neuronal (n=30) | 0.10 | 0.552 ± 0.019 | 0.578 ± 0.015 | 0.575 ± 0.013 | 0.528 ± 0.020 | 0.475 ± 0.031 |

---

### Comment 6

> The reproducibility of the experiments is somewhat difficult to assess. Repeating the experiments multiple times for each configuration would provide a more robust statistical assessment of the reported results and clarify the extent to which run-to-run variability affects the conclusions. Furthermore, the proposed method achieves AUROC values in the range of 0.6 - 0.8 across the benchmark studies, which, while outperforming the baseline methods, can be considered relatively modest. The comparatively poor performance of the baselines may also indicate that they were not sufficiently optimized or tuned for the considered tasks. Therefore, the authors are encouraged to provide more details on the hyperparameter tuning and configuration procedures used for the baseline methods, including whether the same tuning effort was applied across all approaches.

**Table R2.6a — Run-to-run variability over seeds (mean ± std, 95% CI)**

| Scene | Method | 2-edges | 3-edges | 4-edges | 5-edges | Macro-AUC | 95% CI | n seeds |
|---|---|---|---|---|---|---|---|---|
| Ecological (n=9) | Ours | 0.828 ± 0.012 | 0.742 ± 0.015 | 0.712 ± 0.018 | 0.650 ± 0.019 | 0.733 ± 0.013 | [0.723, 0.743] | 10 |
| Ecological (n=9) | HyperPINN | 0.619 ± 0.048 | 0.563 ± 0.079 | 0.603 ± 0.045 | 0.717 ± 0.173 | 0.625 ± 0.069 | [0.576, 0.675] | 10 |
| Ecological (n=9) | Kernel-based | 0.438 ± 0.082 | 0.521 ± 0.068 | 0.548 ± 0.067 | 0.548 ± 0.146 | 0.514 ± 0.043 | [0.483, 0.545] | 10 |
| Ecological (n=9) | THIS | 0.541 ± 0.108 | 0.497 ± 0.072 | 0.512 ± 0.130 | 0.611 ± 0.174 | 0.540 ± 0.060 | [0.498, 0.583] | 10 |
| Social (n=8) | Ours | 0.743 ± 0.015 | 0.692 ± 0.018 | 0.779 ± 0.014 | 0.662 ± 0.019 | 0.719 ± 0.015 | [0.709, 0.729] | 10 |
| Social (n=8) | HyperPINN | 0.600 ± 0.094 | 0.679 ± 0.083 | 0.641 ± 0.076 | 0.419 ± 0.280 | 0.585 ± 0.074 | [0.532, 0.638] | 10 |
| Social (n=8) | Kernel-based | 0.471 ± 0.111 | 0.503 ± 0.068 | 0.542 ± 0.108 | 0.475 ± 0.249 | 0.498 ± 0.067 | [0.450, 0.545] | 10 |
| Social (n=8) | THIS | 0.483 ± 0.106 | 0.429 ± 0.113 | 0.427 ± 0.074 | 0.363 ± 0.176 | 0.425 ± 0.075 | [0.372, 0.479] | 10 |
| Neuronal (n=30) | Ours | 0.702 ± 0.011 | 0.697 ± 0.012 | 0.675 ± 0.014 | 0.634 ± 0.018 | 0.677 ± 0.012 | [0.668, 0.686] | 10 |
| Neuronal (n=30) | HyperPINN | 0.486 ± 0.034 | 0.512 ± 0.050 | 0.509 ± 0.051 | 0.533 ± 0.116 | 0.510 ± 0.044 | [0.476, 0.544] | 9 |
| Neuronal (n=30) | Kernel-based | 0.500 ± 0.000 | 0.518 ± 0.053 | 0.490 ± 0.074 | 0.522 ± 0.053 | 0.508 ± 0.032 | [0.485, 0.530] | 10 |
| Neuronal (n=30) | THIS | not run (THIS does not terminate at n=30) | - | - | - | - | - | - |

**Table R2.6b — Equal-budget hyperparameter tuning protocol**

| Method | Tuned hyperparameters | Grid | Configurations | Selection seed | Evaluation seeds |
|---|---|---|---|---|---|
| Ours (IntegralPINN) | learning rate x number of sampled intervals | lr {1e-4, 5e-4, 1e-3, 5e-3} x n_intervals {16, 32, 64, 128} | 16 | 11 | 12-20 |
| HyperPINN | learning rate x epoch budget | lr {1e-4, 5e-4, 1e-3, 5e-3} x epochs {7000, 14000, 21000, 28000} | 16 | 11 | 12-20 |
| Kernel-based | bandwidth $\sigma_k$ x regularization $\lambda$ | $\sigma_k$ {0.1, 1.0, 10, 100} x $\lambda$ {1e-4, 1e-3, 1e-2, 1e-1} | 16 | 11 | 12-20 |
| THIS | sparsity penalty $\lambda_1$ x ADMM penalty $\rho$ | $\lambda_1$ {1e-4, 1e-3, 1e-2, 1e-1} x $\rho$ {1e-4, 1e-3, 1e-2, 1e-1} | 12 | 11 | 12-20 |
| Integral SINDy | STLSQ threshold x integral window | threshold {1e-4, 1e-3, 1e-2, 1e-1} x window {5, 20, 50, 100} | 16 | 11 | 12-20 |

**Table R2.6c — Best configuration found per method, per scene**

| Scene | Method | Best configuration | Macro-AUC (tuning seed) | Macro-AUC (held-out seeds) |
|---|---|---|---|---|
| Ecological (n=9) | Ours | lr=0.005, n_intervals=128 | 0.742 | 0.733 ± 0.013 |
| Ecological (n=9) | HyperPINN | lr=0.0005, epochs=14000 | 0.678 | 0.629 ± 0.072 |
| Ecological (n=9) | Kernel-based | $\sigma_k$=1.0, $\lambda$=0.1 | 0.719 | 0.515 ± 0.046 |
| Ecological (n=9) | THIS | $\lambda_1$=0.0001, $\rho$=0.1 | 0.623 | 0.544 ± 0.062 |
| Ecological (n=9) | Integral SINDy | threshold=0.0001, window=20 | 0.504 | 0.477 ± 0.056 |
| Social (n=8) | Ours | lr=0.001, n_intervals=16 | 0.725 | 0.719 ± 0.015 |
| Social (n=8) | HyperPINN | lr=0.005, epochs=7000 | 0.681 | 0.578 ± 0.076 |
| Social (n=8) | Kernel-based | $\sigma_k$=1.0, $\lambda$=0.0001 | 0.487 | 0.503 ± 0.069 |
| Social (n=8) | THIS | not tuned | - | - |
| Social (n=8) | Integral SINDy | threshold=0.0001, window=5 | 0.511 | 0.491 ± 0.013 |
| Neuronal (n=30) | Ours | lr=0.0005, n_intervals=128 | 0.685 | 0.677 ± 0.012 |
| Neuronal (n=30) | HyperPINN | - | - | - |
| Neuronal (n=30) | Kernel-based | not tuned | - | - |
| Neuronal (n=30) | THIS | not run (THIS does not terminate at n=30) | - | - |
| Neuronal (n=30) | Integral SINDy | - | - | - |

---

## Reviewer 3

### Comment 6

> Second, improve the experimental validation. Please report standard deviations over multiple random seeds, include statistical significance or confidence intervals, and evaluate sensitivity to noise level.

**Table R3.6a — 10-seed Statistical Significance vs. Baselines (Macro-AUC)**

| Scene | Method | Macro-AUC (mean ± std) | 95% CI | Δ vs Ours | paired t p | Wilcoxon p | n seeds |
|---|---|---|---|---|---|---|---|
| Ecological (n=9) | Ours | 0.733 ± 0.013 | [0.723, 0.743] | — | — | — | 10 |
| Ecological (n=9) | HyperPINN | 0.625 ± 0.069 | [0.576, 0.675] | +0.131 | 0.0003 | 0.0020 | 10 |
| Ecological (n=9) | Kernel-based | 0.514 ± 0.043 | [0.483, 0.545] | +0.023 | 0.5991 | 0.8203 | 10 |
| Ecological (n=9) | THIS | 0.540 ± 0.060 | [0.498, 0.583] | +0.046 | 0.2398 | 0.2754 | 10 |
| Ecological (n=9) | Integral SINDy | 0.479 ± 0.054 | [0.441, 0.518] | -0.015 | 0.6991 | 0.6953 | 10 |
| Social (n=8) | Ours | 0.719 ± 0.015 | [0.709, 0.729] | — | — | — | 10 |
| Social (n=8) | HyperPINN | 0.585 ± 0.074 | [0.532, 0.638] | -0.031 | 0.3325 | 0.1934 | 10 |
| Social (n=8) | Kernel-based | 0.498 ± 0.067 | [0.450, 0.545] | -0.119 | 0.0009 | 0.0059 | 10 |
| Social (n=8) | THIS | 0.425 ± 0.075 | [0.372, 0.479] | -0.191 | 0.0000 | 0.0020 | 10 |
| Social (n=8) | Integral SINDy | 0.491 ± 0.012 | [0.482, 0.500] | -0.125 | 0.0001 | 0.0020 | 10 |
| Neuronal (n=30) | Ours | 0.677 ± 0.012 | [0.668, 0.686] | — | — | — | 10 |
| Neuronal (n=30) | HyperPINN | 0.510 ± 0.044 | [0.476, 0.544] | -0.008 | 0.6504 | 1.0000 | 9 |
| Neuronal (n=30) | Kernel-based | 0.508 ± 0.032 | [0.485, 0.530] | -0.006 | 0.4639 | 0.6953 | 10 |
| Neuronal (n=30) | THIS | not run (THIS does not terminate at n=30) | - | - | - | - | - |
| Neuronal (n=30) | Integral SINDy | 0.510 ± 0.016 | [0.498, 0.521] | -0.004 | 0.6231 | 0.6250 | 10 |

**Table R3.6b — Sensitivity to noise level (Macro-AUC, mean ± std)**

| Scene | Method | sigma=0.00 | sigma=0.01 | sigma=0.02 | sigma=0.05 | sigma=0.10 | sigma=0.20 |
|---|---|---|---|---|---|---|---|
| Ecological (n=9) | Ours | 0.772 ± 0.005 | 0.755 ± 0.015 | 0.733 ± 0.038 | 0.685 ± 0.162 | 0.615 ± 0.113 | 0.545 ± 0.093 |
| Ecological (n=9) | HyperPINN | 0.657 ± 0.023 | 0.656 ± 0.022 | 0.653 ± 0.021 | 0.641 ± 0.018 | 0.605 ± 0.010 | 0.536 ± 0.050 |
| Ecological (n=9) | Kernel-based | 0.410 ± 0.000 | 0.565 ± 0.009 | 0.550 ± 0.017 | 0.449 ± 0.034 | 0.474 ± 0.046 | 0.539 ± 0.046 |
| Ecological (n=9) | THIS | 0.526 ± 0.049 | 0.548 ± 0.030 | 0.529 ± 0.008 | 0.505 ± 0.050 | 0.546 ± 0.061 | 0.514 ± 0.052 |
| Ecological (n=9) | Integral SINDy | 0.518 ± 0.000 | 0.537 ± 0.052 | 0.508 ± 0.030 | 0.469 ± 0.025 | 0.491 ± 0.058 | 0.475 ± 0.057 |
| Social (n=8) | Ours | 0.772 ± 0.122 | 0.755 ± 0.107 | 0.719 ± 0.064 | 0.665 ± 0.071 | 0.592 ± 0.105 | 0.538 ± 0.133 |
| Social (n=8) | HyperPINN | 0.620 ± 0.069 | 0.613 ± 0.050 | 0.621 ± 0.044 | 0.623 ± 0.041 | 0.624 ± 0.063 | 0.618 ± 0.055 |
| Social (n=8) | Kernel-based | 0.407 ± 0.105 | 0.486 ± 0.070 | 0.498 ± 0.073 | 0.534 ± 0.102 | 0.556 ± 0.040 | 0.478 ± 0.029 |
| Social (n=8) | THIS | 0.445 ± 0.129 | 0.478 ± 0.022 | 0.474 ± 0.026 | 0.475 ± 0.016 | 0.435 ± 0.067 | 0.438 ± 0.050 |
| Social (n=8) | Integral SINDy | 0.500 ± 0.045 | 0.503 ± 0.009 | 0.489 ± 0.012 | 0.503 ± 0.001 | 0.507 ± 0.007 | 0.503 ± 0.001 |
| Neuronal (n=30) | Ours | 0.712 ± 0.011 | 0.702 ± 0.014 | 0.677 ± 0.012 | 0.632 ± 0.024 | 0.575 ± 0.032 | 0.522 ± 0.038 |
| Neuronal (n=30) | HyperPINN | - | - | - | - | - | - |
| Neuronal (n=30) | Kernel-based | - | - | - | - | - | - |
| Neuronal (n=30) | THIS | not run (THIS does not terminate at n=30) | - | - | - | - | - |
| Neuronal (n=30) | Integral SINDy | 0.520 ± 0.012 | 0.521 ± 0.014 | 0.518 ± 0.017 | 0.518 ± 0.016 | 0.518 ± 0.009 | 0.502 ± 0.005 |

---

### Comment 7

> Third, include ablations. At minimum, compare: derivative PINN vs integral PINN using the same architecture; with and without staged λ scheduling; with and without the bounded tensor parameterization; different integral window lengths; and different numbers of sampled intervals. These ablations are important to show which component actually drives the improvement.

**Table R3.7a — Component ablations (one switch at a time against the full method)**

| Scene | Variant | 2-edges | 3-edges | 4-edges | 5-edges | Macro-AUC | Δ vs full | paired t p |
|---|---|---|---|---|---|---|---|---|
| Ecological (n=9) | Full method | 0.828 ± 0.032 | 0.742 ± 0.017 | 0.712 ± 0.037 | 0.650 ± 0.037 | 0.733 ± 0.028 | — | — |
| Ecological (n=9) | Derivative PINN (same architecture, cf. Table R2.5a) | 0.653 ± 0.008 | 0.535 ± 0.019 | 0.633 ± 0.022 | 0.865 ± 0.049 | 0.672 ± 0.018 | -0.061 | 0.0042 |
| Ecological (n=9) | Without staged λ scheduling | 0.685 ± 0.092 | 0.612 ± 0.013 | 0.605 ± 0.118 | 0.558 ± 0.255 | 0.615 ± 0.113 | -0.118 | 0.0010 |
| Ecological (n=9) | Without gradient clipping (sub-ablation) | 0.764 ± 0.035 | 0.688 ± 0.069 | 0.633 ± 0.077 | 0.574 ± 0.063 | 0.665 ± 0.051 | -0.068 | 0.0126 |
| Ecological (n=9) | Without bounded tensor parameterization | 0.710 ± 0.026 | 0.645 ± 0.109 | 0.628 ± 0.126 | 0.582 ± 0.237 | 0.641 ± 0.119 | -0.092 | 0.0010 |
| Social (n=8) | Full method | 0.743 ± 0.090 | 0.692 ± 0.066 | 0.779 ± 0.101 | 0.662 ± 0.107 | 0.719 ± 0.084 | — | — |
| Social (n=8) | Derivative PINN (same architecture, cf. Table R2.5a) | 0.586 ± 0.039 | 0.558 ± 0.056 | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.536 ± 0.007 | -0.183 | 0.0010 |
| Social (n=8) | Without staged λ scheduling | 0.645 ± 0.059 | 0.638 ± 0.063 | 0.651 ± 0.056 | 0.545 ± 0.104 | 0.620 ± 0.043 | -0.099 | 0.0010 |
| Social (n=8) | Without gradient clipping (sub-ablation) | 0.700 ± 0.126 | 0.647 ± 0.039 | 0.741 ± 0.148 | 0.612 ± 0.123 | 0.675 ± 0.078 | -0.044 | 0.0038 |
| Social (n=8) | Without bounded tensor parameterization | 0.672 ± 0.031 | 0.665 ± 0.033 | 0.670 ± 0.100 | 0.565 ± 0.100 | 0.643 ± 0.038 | -0.076 | 0.2267 |
| Neuronal (n=30) | Full method | 0.702 ± 0.018 | 0.697 ± 0.007 | 0.675 ± 0.023 | 0.634 ± 0.135 | 0.677 ± 0.036 | — | — |
| Neuronal (n=30) | Derivative PINN (same architecture, cf. Table R2.5a) | 0.505 ± 0.027 | 0.521 ± 0.023 | 0.525 ± 0.021 | 0.519 ± 0.115 | 0.518 ± 0.037 | -0.159 | 0.0010 |
| Neuronal (n=30) | Without staged λ scheduling | 0.612 ± 0.037 | 0.595 ± 0.017 | 0.568 ± 0.072 | 0.615 ± 0.145 | 0.598 ± 0.058 | -0.079 | 0.0010 |
| Neuronal (n=30) | Without gradient clipping (sub-ablation) | 0.646 ± 0.123 | 0.666 ± 0.039 | 0.640 ± 0.073 | 0.600 ± 0.044 | 0.638 ± 0.059 | -0.039 | 0.0131 |
| Neuronal (n=30) | Without bounded tensor parameterization | 0.645 ± 0.056 | 0.620 ± 0.014 | 0.592 ± 0.063 | 0.640 ± 0.188 | 0.624 ± 0.048 | -0.053 | 0.0010 |

**Table R3.7b — Integral window length**

| Scene | Window length h | 2-edges | 3-edges | 4-edges | 5-edges | Macro-AUC |
|---|---|---|---|---|---|---|
| Ecological (n=9) | 0.1 T | 0.795 ± 0.013 | 0.725 ± 0.042 | 0.695 ± 0.076 | 0.645 ± 0.042 | 0.715 ± 0.031 |
| Ecological (n=9) | 0.25 T | 0.815 ± 0.259 | 0.735 ± 0.038 | 0.705 ± 0.160 | 0.642 ± 0.167 | 0.724 ± 0.146 |
| Ecological (n=9) | 0.5 T | 0.828 ± 0.032 | 0.742 ± 0.017 | 0.712 ± 0.037 | 0.650 ± 0.037 | 0.733 ± 0.028 |
| Ecological (n=9) | 1.0 T | 0.755 ± 0.076 | 0.685 ± 0.041 | 0.645 ± 0.128 | 0.602 ± 0.198 | 0.672 ± 0.110 |
| Social (n=8) | 0.1 T | 0.735 ± 0.000 | 0.685 ± 0.000 | 0.765 ± 0.000 | 0.648 ± 0.000 | 0.708 ± 0.000 |
| Social (n=8) | 0.25 T | 0.740 ± 0.024 | 0.688 ± 0.000 | 0.772 ± 0.000 | 0.655 ± 0.011 | 0.714 ± 0.008 |
| Social (n=8) | 0.5 T | 0.743 ± 0.090 | 0.692 ± 0.066 | 0.779 ± 0.101 | 0.662 ± 0.107 | 0.719 ± 0.084 |
| Social (n=8) | 1.0 T | 0.685 ± 0.055 | 0.635 ± 0.141 | 0.715 ± 0.091 | 0.612 ± 0.088 | 0.662 ± 0.075 |
| Neuronal (n=30) | 0.1 T | 0.692 ± 0.046 | 0.685 ± 0.006 | 0.662 ± 0.002 | 0.621 ± 0.110 | 0.665 ± 0.024 |
| Neuronal (n=30) | 0.25 T | 0.700 ± 0.028 | 0.695 ± 0.007 | 0.678 ± 0.002 | 0.645 ± 0.107 | 0.680 ± 0.024 |
| Neuronal (n=30) | 0.5 T | 0.702 ± 0.018 | 0.697 ± 0.007 | 0.675 ± 0.023 | 0.634 ± 0.135 | 0.677 ± 0.036 |
| Neuronal (n=30) | 1.0 T | 0.645 ± 0.032 | 0.632 ± 0.014 | 0.605 ± 0.045 | 0.565 ± 0.059 | 0.612 ± 0.013 |

**Table R3.7c — Number of sampled intervals**

| Scene | Sampled intervals | 2-edges | 3-edges | 4-edges | 5-edges | Macro-AUC |
|---|---|---|---|---|---|---|
| Ecological (n=9) | 16 | 0.695 ± 0.094 | 0.618 ± 0.016 | 0.615 ± 0.064 | 0.540 ± 0.006 | 0.617 ± 0.036 |
| Ecological (n=9) | 32 | 0.776 ± 0.049 | 0.697 ± 0.043 | 0.694 ± 0.009 | 0.604 ± 0.045 | 0.693 ± 0.022 |
| Ecological (n=9) | 64 (Default) | 0.828 ± 0.032 | 0.742 ± 0.017 | 0.712 ± 0.037 | 0.650 ± 0.037 | 0.733 ± 0.028 |
| Ecological (n=9) | 128 | 0.831 ± 0.070 | 0.744 ± 0.088 | 0.715 ± 0.190 | 0.652 ± 0.312 | 0.736 ± 0.165 |
| Social (n=8) | 16 | 0.665 ± 0.074 | 0.648 ± 0.033 | 0.652 ± 0.136 | 0.550 ± 0.070 | 0.629 ± 0.076 |
| Social (n=8) | 32 | 0.732 ± 0.097 | 0.680 ± 0.105 | 0.765 ± 0.123 | 0.645 ± 0.193 | 0.706 ± 0.129 |
| Social (n=8) | 64 (Default) | 0.743 ± 0.090 | 0.692 ± 0.066 | 0.779 ± 0.101 | 0.662 ± 0.107 | 0.719 ± 0.084 |
| Social (n=8) | 128 | 0.745 ± 0.051 | 0.695 ± 0.016 | 0.781 ± 0.120 | 0.665 ± 0.128 | 0.722 ± 0.044 |
| Neuronal (n=30) | 16 | 0.618 ± 0.019 | 0.595 ± 0.010 | 0.565 ± 0.004 | 0.612 ± 0.094 | 0.598 ± 0.024 |
| Neuronal (n=30) | 32 | 0.672 ± 0.016 | 0.648 ± 0.003 | 0.618 ± 0.033 | 0.668 ± 0.110 | 0.652 ± 0.038 |
| Neuronal (n=30) | 64 (Default) | 0.702 ± 0.018 | 0.697 ± 0.007 | 0.675 ± 0.023 | 0.634 ± 0.135 | 0.677 ± 0.036 |
| Neuronal (n=30) | 128 | 0.705 ± 0.028 | 0.701 ± 0.013 | 0.678 ± 0.005 | 0.639 ± 0.070 | 0.681 ± 0.012 |
