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
| Ecological (n=9) | Ours (IntegralPINN) | | | | | |
| Ecological (n=9) | Integral SINDy (STLSQ) | | | | | |
| Ecological (n=9) | Derivative SINDy (STLSQ) | | | | | |
| Ecological (n=9) | Linear least squares | | | | | |
| Social (n=8) | Ours (IntegralPINN) | | | | | |
| Social (n=8) | Integral SINDy (STLSQ) | | | | | |
| Social (n=8) | Derivative SINDy (STLSQ) | | | | | |
| Social (n=8) | Linear least squares | | | | | |
| Neuronal (n=30) | Ours (IntegralPINN) | | | | | |
| Neuronal (n=30) | Integral SINDy (STLSQ) | | | | | |
| Neuronal (n=30) | Derivative SINDy (STLSQ) | | | | | |
| Neuronal (n=30) | Linear least squares | | | | | |

---

### Comment 5

> It is not clear how the proposed interaction tensor parametrization compares with existing methods that explicitly exploit sparsity. This is particularly important for high-order interactions, where the number of possible interaction terms can grow rapidly. If sparsity is not explicitly encouraged in the proposed formulation, the authors should discuss why their approach is preferable to existing sparse identification methods and what advantages it offers in this setting.

**Table R1.5a — Proposed parametrization vs. explicitly sparsity-promoting estimation**

| Scene | Method | Sparsity mechanism | 2-edges | 3-edges | 4-edges | 5-edges | Macro-AUC | Non-zeros recovered |
|---|---|---|---|---|---|---|---|---|
| Ecological (n=9) | Ours | bounded sigmoid, none explicit | | | | | | |
| Ecological (n=9) | Ours + L1 | explicit L1 | | | | | | |
| Ecological (n=9) | Integral SINDy | STLSQ hard thresholding | | | | | | |
| Social (n=8) | Ours | bounded sigmoid, none explicit | | | | | | |
| Social (n=8) | Ours + L1 | explicit L1 | | | | | | |
| Social (n=8) | Integral SINDy | STLSQ hard thresholding | | | | | | |
| Neuronal (n=30) | Ours | bounded sigmoid, none explicit | | | | | | |
| Neuronal (n=30) | Ours + L1 | explicit L1 | | | | | | |
| Neuronal (n=30) | Integral SINDy | STLSQ hard thresholding | | | | | | |

**Table R1.5b — L1 weight sweep for the proposed method (supporting detail for Table R1.5a)**

| Scene | L1 weight | 2-edges | 3-edges | 4-edges | 5-edges | Macro-AUC |
|---|---|---|---|---|---|---|
| Ecological (n=9) | 0 | | | | | |
| Ecological (n=9) | 1e-4 | | | | | |
| Ecological (n=9) | 1e-3 | | | | | |
| Ecological (n=9) | 1e-2 | | | | | |
| Social (n=8) | 0 | | | | | |
| Social (n=8) | 1e-4 | | | | | |
| Social (n=8) | 1e-3 | | | | | |
| Social (n=8) | 1e-2 | | | | | |
| Neuronal (n=30) | 0 | | | | | |
| Neuronal (n=30) | 1e-4 | | | | | |
| Neuronal (n=30) | 1e-3 | | | | | |
| Neuronal (n=30) | 1e-2 | | | | | |

---

## Reviewer 2

### Comment 5

> It would significantly increase the quality of the paper if the authors added experiments where L_physics is swapped with the derivative-based cost term. While HyperPINN might be a suitable comparison, it is unclear whether that formulation applied additional sparsity-promoting cost terms and if it used the same sigmoid-based parametrization as in this paper. Nonetheless, multiple experiments with varying sigma and h values would empirically strengthen the presentation of Theorems 1 and 2.

**Table R2.5a — L_physics swapped with the derivative-based cost term (identical surrogate, parametrization and optimizer)**

| Scene | Physics residual | 2-edges | 3-edges | 4-edges | 5-edges | Macro-AUC |
|---|---|---|---|---|---|---|
| Ecological (n=9) | Integral (ours) | | | | | |
| Ecological (n=9) | Derivative | | | | | |
| Social (n=8) | Integral (ours) | | | | | |
| Social (n=8) | Derivative | | | | | |
| Neuronal (n=30) | Integral (ours) | | | | | |
| Neuronal (n=30) | Derivative | | | | | |

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
| Ecological (n=9) | 0.00 | | | | | |
| Ecological (n=9) | 0.01 | | | | | |
| Ecological (n=9) | 0.02 | | | | | |
| Ecological (n=9) | 0.05 | | | | | |
| Ecological (n=9) | 0.10 | | | | | |
| Social (n=8) | 0.00 | | | | | |
| Social (n=8) | 0.01 | | | | | |
| Social (n=8) | 0.02 | | | | | |
| Social (n=8) | 0.05 | | | | | |
| Social (n=8) | 0.10 | | | | | |
| Neuronal (n=30) | 0.00 | | | | | |
| Neuronal (n=30) | 0.01 | | | | | |
| Neuronal (n=30) | 0.02 | | | | | |
| Neuronal (n=30) | 0.05 | | | | | |
| Neuronal (n=30) | 0.10 | | | | | |

---

### Comment 6

> The reproducibility of the experiments is somewhat difficult to assess. Repeating the experiments multiple times for each configuration would provide a more robust statistical assessment of the reported results and clarify the extent to which run-to-run variability affects the conclusions. Furthermore, the proposed method achieves AUROC values in the range of 0.6 - 0.8 across the benchmark studies, which, while outperforming the baseline methods, can be considered relatively modest. The comparatively poor performance of the baselines may also indicate that they were not sufficiently optimized or tuned for the considered tasks. Therefore, the authors are encouraged to provide more details on the hyperparameter tuning and configuration procedures used for the baseline methods, including whether the same tuning effort was applied across all approaches.

**Table R2.6a — Run-to-run variability over seeds (mean ± std, 95% CI). Same runs as Table R3.6a.**

| Scene | Method | 2-edges | 3-edges | 4-edges | 5-edges | Macro-AUC | 95% CI | n seeds |
|---|---|---|---|---|---|---|---|---|
| Ecological (n=9) | Ours | | | | | | | |
| Ecological (n=9) | HyperPINN | | | | | | | |
| Ecological (n=9) | Kernel-based | | | | | | | |
| Ecological (n=9) | THIS | | | | | | | |
| Social (n=8) | Ours | | | | | | | |
| Social (n=8) | HyperPINN | | | | | | | |
| Social (n=8) | Kernel-based | | | | | | | |
| Social (n=8) | THIS | | | | | | | |
| Neuronal (n=30) | Ours | | | | | | | |
| Neuronal (n=30) | HyperPINN | | | | | | | |
| Neuronal (n=30) | Kernel-based | | | | | | | |
| Neuronal (n=30) | THIS | | | | | | | |

**Table R2.6b — Equal-budget hyperparameter tuning protocol**

| Method | Tuned hyperparameters | Grid | Configurations | Selection seed | Evaluation seeds |
|---|---|---|---|---|---|
| Ours (IntegralPINN) | learning rate x number of sampled intervals | lr {1e-4, 5e-4, 1e-3, 5e-3} x n_intervals {8, 32, 64, 128} | 16 | | |
| HyperPINN | learning rate x epoch budget | lr {1e-4, 5e-4, 1e-3, 5e-3} x epochs {7000, 14000} | 8 | | |
| Kernel-based | | | | | |
| THIS | | | | | |
| Integral SINDy | STLSQ threshold x integral window | threshold {1e-4, 1e-3, 1e-2, 1e-1} x window {5, 20, 50, 100} | 16 | | |

**Table R2.6c — Best configuration found per method, per scene**

| Scene | Method | Best configuration | Macro-AUC (tuning seed) | Macro-AUC (held-out seeds) |
|---|---|---|---|---|
| Ecological (n=9) | Ours | | | |
| Ecological (n=9) | HyperPINN | | | |
| Ecological (n=9) | Kernel-based | | | |
| Ecological (n=9) | THIS | | | |
| Ecological (n=9) | Integral SINDy | | | |
| Social (n=8) | Ours | | | |
| Social (n=8) | HyperPINN | | | |
| Social (n=8) | Kernel-based | | | |
| Social (n=8) | THIS | | | |
| Social (n=8) | Integral SINDy | | | |
| Neuronal (n=30) | Ours | | | |
| Neuronal (n=30) | HyperPINN | | | |
| Neuronal (n=30) | Kernel-based | | | |
| Neuronal (n=30) | THIS | | | |
| Neuronal (n=30) | Integral SINDy | | | |

---

## Reviewer 3

### Comment 6

> Second, improve the experimental validation. Please report standard deviations over multiple random seeds, include statistical significance or confidence intervals, and evaluate sensitivity to noise level.

**Table R3.6a — Standard deviations, confidence intervals and significance vs. each baseline. Same runs as Table R2.6a.**

| Scene | Method | Macro-AUC (mean ± std) | 95% CI | Δ vs Ours | paired t p | Wilcoxon p | n seeds |
|---|---|---|---|---|---|---|---|
| Ecological (n=9) | Ours | | | — | — | — | |
| Ecological (n=9) | HyperPINN | | | | | | |
| Ecological (n=9) | Kernel-based | | | | | | |
| Ecological (n=9) | THIS | | | | | | |
| Ecological (n=9) | Integral SINDy | | | | | | |
| Social (n=8) | Ours | | | — | — | — | |
| Social (n=8) | HyperPINN | | | | | | |
| Social (n=8) | Kernel-based | | | | | | |
| Social (n=8) | THIS | | | | | | |
| Social (n=8) | Integral SINDy | | | | | | |
| Neuronal (n=30) | Ours | | | — | — | — | |
| Neuronal (n=30) | HyperPINN | | | | | | |
| Neuronal (n=30) | Kernel-based | | | | | | |
| Neuronal (n=30) | THIS | | | | | | |
| Neuronal (n=30) | Integral SINDy | | | | | | |

**Table R3.6b — Sensitivity to noise level (Macro-AUC, mean ± std)**

| Scene | Method | sigma=0.00 | sigma=0.01 | sigma=0.02 | sigma=0.05 | sigma=0.10 | sigma=0.20 |
|---|---|---|---|---|---|---|---|
| Ecological (n=9) | Ours | | | | | | |
| Ecological (n=9) | HyperPINN | | | | | | |
| Ecological (n=9) | Kernel-based | | | | | | |
| Ecological (n=9) | THIS | | | | | | |
| Ecological (n=9) | Integral SINDy | | | | | | |
| Social (n=8) | Ours | | | | | | |
| Social (n=8) | HyperPINN | | | | | | |
| Social (n=8) | Kernel-based | | | | | | |
| Social (n=8) | THIS | | | | | | |
| Social (n=8) | Integral SINDy | | | | | | |
| Neuronal (n=30) | Ours | | | | | | |
| Neuronal (n=30) | HyperPINN | | | | | | |
| Neuronal (n=30) | Kernel-based | | | | | | |
| Neuronal (n=30) | THIS | | | | | | |
| Neuronal (n=30) | Integral SINDy | | | | | | |

---

### Comment 7

> Third, include ablations. At minimum, compare: derivative PINN vs integral PINN using the same architecture; with and without staged λ scheduling; with and without the bounded tensor parameterization; different integral window lengths; and different numbers of sampled intervals. These ablations are important to show which component actually drives the improvement.

**Table R3.7a — Component ablations (one switch at a time against the full method)**

| Scene | Variant | 2-edges | 3-edges | 4-edges | 5-edges | Macro-AUC | Δ vs full | paired t p |
|---|---|---|---|---|---|---|---|---|
| Ecological (n=9) | Full method | | | | | | — | — |
| Ecological (n=9) | Derivative PINN (same architecture, cf. Table R2.5a) | | | | | | | |
| Ecological (n=9) | Without staged λ scheduling | | | | | | | |
| Ecological (n=9) | Without bounded tensor parameterization | | | | | | | |
| Social (n=8) | Full method | | | | | | — | — |
| Social (n=8) | Derivative PINN (same architecture, cf. Table R2.5a) | | | | | | | |
| Social (n=8) | Without staged λ scheduling | | | | | | | |
| Social (n=8) | Without bounded tensor parameterization | | | | | | | |
| Neuronal (n=30) | Full method | | | | | | — | — |
| Neuronal (n=30) | Derivative PINN (same architecture, cf. Table R2.5a) | | | | | | | |
| Neuronal (n=30) | Without staged λ scheduling | | | | | | | |
| Neuronal (n=30) | Without bounded tensor parameterization | | | | | | | |

**Table R3.7b — Integral window length**

| Scene | Window length h | 2-edges | 3-edges | 4-edges | 5-edges | Macro-AUC |
|---|---|---|---|---|---|---|
| Ecological (n=9) | 0.1 T | | | | | |
| Ecological (n=9) | 0.25 T | | | | | |
| Ecological (n=9) | 0.5 T | | | | | |
| Ecological (n=9) | 1.0 T | | | | | |
| Social (n=8) | 0.1 T | | | | | |
| Social (n=8) | 0.25 T | | | | | |
| Social (n=8) | 0.5 T | | | | | |
| Social (n=8) | 1.0 T | | | | | |
| Neuronal (n=30) | 0.1 T | | | | | |
| Neuronal (n=30) | 0.25 T | | | | | |
| Neuronal (n=30) | 0.5 T | | | | | |
| Neuronal (n=30) | 1.0 T | | | | | |

**Table R3.7c — Number of sampled intervals**

| Scene | Sampled intervals | 2-edges | 3-edges | 4-edges | 5-edges | Macro-AUC |
|---|---|---|---|---|---|---|
| Ecological (n=9) | 8 | | | | | |
| Ecological (n=9) | 32 | | | | | |
| Ecological (n=9) | 64 | | | | | |
| Ecological (n=9) | 128 | | | | | |
| Social (n=8) | 8 | | | | | |
| Social (n=8) | 32 | | | | | |
| Social (n=8) | 64 | | | | | |
| Social (n=8) | 128 | | | | | |
| Neuronal (n=30) | 8 | | | | | |
| Neuronal (n=30) | 32 | | | | | |
| Neuronal (n=30) | 64 | | | | | |
| Neuronal (n=30) | 128 | | | | | |

