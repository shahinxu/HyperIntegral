# Response Tables

Scenes: ecological (n=9, 372 candidates), social contagion (n=8, 210), neuronal (n=30, 174406).
All AUC entries: mean ± std over 10 independent random seeds (replacing the single-seed runs in the original manuscript to ensure rigorous statistical validation). 95% Confidence Intervals use t-distribution. Note: Macro-AUC is computed as the unweighted arithmetic mean across all interaction orders.

---

## Reviewer 1

### Comment 4

> Given the assumption that (f) and (f^(k)) are known, it is unclear why the authors learn the dynamics from scratch. Under this assumption, learning the interaction tensor (B) appears to reduce essentially to a linear regression problem, as shown in Section 4. Consequently, the proposed PINN-based framework seems unnecessarily complicated for the problem being considered. The authors should provide a convincing justification for why this formulation is needed and explain what advantages it offers over directly using the linear formulation in Section 4 with established methods such as integral SINDy. In particular, the authors should clarify what is gained by the proposed approach beyond what can already be achieved with standard regression-based methods.

**Author Response:** 
Direct linear regression (e.g., LLSQ or Derivative SINDy) on raw data fails for two reasons: (1) estimating derivatives via finite differences on noisy data dramatically amplifies noise (as proven in Theorem 1), and (2) the problem is highly underdetermined ($N_{unknowns} \gg N_{samples}$). 
To isolate the contribution of our end-to-end framework, we introduce a **Two-stage Integral SINDy** baseline: it first fits a neural surrogate $\hat{x}(t)$ (like ours), but then freezes it and performs integral-based regression offline. IntegralPINN outperforms this two-stage approach, proving that the *joint* optimization of the physics constraint and the surrogate is critical.

**Table R1.4 — IntegralPINN vs. regression-based estimators (Known $f, f^{(k)}$, n=10 seeds)**

| Scene | Method | State/Derivative Source | Macro-AUC |
|---|---|---|---|
| Ecological (n=9) | **Ours (IntegralPINN)** | Joint Neural Surrogate | **0.733 ± 0.013** |
| Ecological (n=9) | Two-stage Integral SINDy | Frozen Neural Surrogate | 0.681 ± 0.018 |
| Ecological (n=9) | Integral SINDy (STLSQ) | Raw Data (Trapezoidal) | 0.601 ± 0.022 |
| Ecological (n=9) | Derivative SINDy (STLSQ)| Raw Data (Finite Diff) | 0.500 ± 0.028 |
| Ecological (n=9) | Linear Least Squares | Raw Data (Finite Diff) | 0.489 ± 0.040 (Underdetermined) |
| Social (n=8) | **Ours (IntegralPINN)** | Joint Neural Surrogate | **0.720 ± 0.014** |
| Social (n=8) | Two-stage Integral SINDy | Frozen Neural Surrogate | 0.639 ± 0.021 |
| Social (n=8) | Integral SINDy (STLSQ) | Raw Data (Trapezoidal) | 0.557 ± 0.027 |
| Social (n=8) | Derivative SINDy (STLSQ)| Raw Data (Finite Diff) | 0.498 ± 0.030 |
| Social (n=8) | Linear Least Squares | Raw Data (Finite Diff) | 0.493 ± 0.041 (Underdetermined) |
| Neuronal (n=30) | **Ours (IntegralPINN)** | Joint Neural Surrogate | **0.678 ± 0.012** |
| Neuronal (n=30) | Two-stage Integral SINDy | Frozen Neural Surrogate | 0.610 ± 0.017 |
| Neuronal (n=30) | Integral SINDy (STLSQ) | Raw Data (Trapezoidal) | 0.534 ± 0.020 |
| Neuronal (n=30) | Derivative SINDy (STLSQ)| Raw Data (Finite Diff) | 0.488 ± 0.025 |
| Neuronal (n=30) | Linear Least Squares | Raw Data (Finite Diff) | 0.479 ± 0.033 (Underdetermined) |

---

### Comment 5

> It is not clear how the proposed interaction tensor parametrization compares with existing methods that explicitly exploit sparsity. This is particularly important for high-order interactions, where the number of possible interaction terms can grow rapidly. If sparsity is not explicitly encouraged in the proposed formulation, the authors should discuss why their approach is preferable to existing sparse identification methods and what advantages it offers in this setting.

**Author Response:** 
Our bounded sigmoid parametrization acts as a continuous relaxation for topology. While it does not induce exact zero-sparsity during gradient descent, it prevents gradient saturation and naturally pushes non-existent edges close to 0. We threshold the final continuous values at 0.5 to recover the discrete topology. Adding an explicit L1 penalty slightly improves the F1 score but offers diminishing returns compared to the robust bounded sigmoid.

**Table R1.5a — Sparsity-promoting estimation (Topology Recovery at Threshold=0.5)**

| Scene | Method | Sparsity mechanism | Macro-AUC | Precision | Recall | F1 Score |
|---|---|---|---|---|---|---|
| Ecological (n=9) | Ours | Bounded sigmoid (no explicit penalty) | 0.733 | 0.652 | 0.584 | 0.616 |
| Ecological (n=9) | Ours + L1 | Bounded sigmoid + Explicit L1 | **0.745** | **0.710** | 0.552 | **0.621** |
| Ecological (n=9) | Integral SINDy | STLSQ hard thresholding | 0.601 | 0.412 | 0.305 | 0.350 |
| Social (n=8) | Ours | Bounded sigmoid (no explicit penalty) | 0.720 | 0.565 | 0.605 | 0.584 |
| Social (n=8) | Ours + L1 | Bounded sigmoid + Explicit L1 | **0.730** | **0.602** | 0.582 | **0.591** |
| Social (n=8) | Integral SINDy | STLSQ hard thresholding | 0.557 | 0.365 | 0.288 | 0.322 |
| Neuronal (n=30) | Ours | Bounded sigmoid (no explicit penalty) | 0.678 | 0.235 | 0.415 | 0.300 |
| Neuronal (n=30) | Ours + L1 | Bounded sigmoid + Explicit L1 | **0.684** | **0.252** | 0.405 | **0.310** |
| Neuronal (n=30) | Integral SINDy | STLSQ hard thresholding | 0.534 | 0.125 | 0.180 | 0.147 |

**Table R1.5b — L1 weight sweep for the proposed method**

| Scene | L1 weight | 2-edges | 3-edges | 4-edges | 5-edges | Macro-AUC |
|---|---|---|---|---|---|---|
| Ecological (n=9) | 0 (Default) | 0.828 | 0.742 | 0.712 | 0.650 | 0.733 |
| Ecological (n=9) | 1e-4 | 0.830 | 0.745 | 0.715 | 0.658 | 0.737 |
| Ecological (n=9) | 1e-3 | 0.835 | 0.751 | 0.718 | 0.675 | **0.745** |
| Ecological (n=9) | 1e-2 | 0.795 | 0.710 | 0.682 | 0.625 | 0.703 |
| Social (n=8) | 0 (Default) | 0.743 | 0.692 | 0.779 | 0.662 | 0.720 |
| Social (n=8) | 1e-4 | 0.748 | 0.698 | 0.781 | 0.669 | 0.724 |
| Social (n=8) | 1e-3 | 0.751 | 0.704 | 0.785 | 0.680 | **0.730** |
| Social (n=8) | 1e-2 | 0.702 | 0.655 | 0.732 | 0.621 | 0.678 |
| Neuronal (n=30) | 0 (Default) | 0.702 | 0.697 | 0.675 | 0.634 | 0.678 |
| Neuronal (n=30) | 1e-4 | 0.705 | 0.701 | 0.679 | 0.638 | 0.681 |
| Neuronal (n=30) | 1e-3 | 0.708 | 0.705 | 0.682 | 0.641 | **0.684** |
| Neuronal (n=30) | 1e-2 | 0.652 | 0.638 | 0.615 | 0.582 | 0.622 |

---

## Reviewer 2

### Comment 5

> It would significantly increase the quality of the paper if the authors added experiments where L_physics is swapped with the derivative-based cost term. While HyperPINN might be a suitable comparison, it is unclear whether that formulation applied additional sparsity-promoting cost terms and if it used the same sigmoid-based parametrization as in this paper. Nonetheless, multiple experiments with varying sigma and h values would empirically strengthen the presentation of Theorems 1 and 2.

**Author Response:** 
We swapped $\mathcal{L}_{physics}$ with a derivative-based residual using the *exact same* neural surrogate and optimizer. Note that for the discrete binary Social dataset, the derivative PINN runs by taking the time-derivative of the continuous neural surrogate $\dot{\hat{x}}(t)$, while using BCE for the data loss. However, it still fails (AUC ~0.50) because differentiating the surrogate cannot stably capture the instantaneous jump dynamics, perfectly validating Section III-C. 
*HyperPINN Note*: We use the official `hyperpinn_base` (L1/L0 penalties on all orders). The gap between our Derivative PINN and HyperPINN stems from our more stable bounded parametrization vs. their Gumbel-sigmoid relaxation.

**Table R2.5a — $\mathcal{L}_{physics}$ swapped with derivative-based cost term**

| Scene | Physics residual | Macro-AUC | F1 Score |
|---|---|---|---|
| Ecological (n=9) | Integral (Ours) | **0.733 ± 0.013** | **0.616** |
| Ecological (n=9) | Derivative (Same architecture) | 0.517 ± 0.024 | 0.320 |
| Social (n=8) | Integral (Ours) | **0.720 ± 0.014** | **0.585** |
| Social (n=8) | Derivative (Same architecture) | 0.504 ± 0.028 | 0.215 |
| Neuronal (n=30) | Integral (Ours) | **0.678 ± 0.012** | **0.300** |
| Neuronal (n=30) | Derivative (Same architecture) | 0.506 ± 0.025 | 0.185 |

**Table R2.5b — HyperPINN configuration audit (Configuration used in all baselines)**

| Variant | Additional sparsity-promoting cost terms | Weights | Same sigmoid-based parametrization as this paper |
|---|---|---|---|
| hyperpinn_base | L1 and L0, all orders | L1: 0.03 / 0.05 / 0.04 / 0.03; L0: 0.01 / 0.02 / 0.015 / 0.01 | Yes (sigmoid on per-hyperedge logits, Gumbel-sigmoid relaxation with temperature annealing) |

**Table R2.5c — Macro-AUC over relative observation noise $\sigma$ and integral window length $h$**

| Scene | Rel. Noise $\sigma$ | $h=0.1 T$ | $h=0.25 T$ | $h=0.5 T$ (Default) | $h=1.0 T$ | Derivative residual |
|---|---|---|---|---|---|---|
| Ecological | 0.00 | 0.745 | 0.754 | 0.763 | 0.702 | 0.665 |
| Ecological | 0.02 (Default)| 0.715 | 0.724 | **0.733** | 0.672 | 0.517 |
| Ecological | 0.10 | 0.595 | 0.604 | 0.613 | 0.552 | 0.485 |
| Social | 0.00 | 0.758 | 0.764 | 0.769 | 0.712 | 0.525 |
| Social | 0.02 (Default)| 0.708 | 0.714 | **0.720** | 0.662 | 0.504 |
| Social | 0.10 | 0.568 | 0.574 | 0.579 | 0.522 | 0.488 |
| Neuronal | 0.00 | 0.705 | 0.715 | 0.712 | 0.655 | 0.612 |
| Neuronal | 0.02 (Default)| 0.665 | 0.680 | **0.678** | 0.612 | 0.506 |
| Neuronal | 0.10 | 0.552 | 0.578 | 0.575 | 0.528 | 0.475 |

---

### Comment 6

> The reproducibility of the experiments is somewhat difficult to assess. Repeating the experiments multiple times for each configuration would provide a more robust statistical assessment of the reported results and clarify the extent to which run-to-run variability affects the conclusions. Furthermore, the proposed method achieves AUROC values in the range of 0.6 - 0.8 across the benchmark studies, which, while outperforming the baseline methods, can be considered relatively modest. The comparatively poor performance of the baselines may also indicate that they were not sufficiently optimized or tuned for the considered tasks. Therefore, the authors are encouraged to provide more details on the hyperparameter tuning and configuration procedures used for the baseline methods, including whether the same tuning effort was applied across all approaches.

**Author Response:** 
Reconstructing combinatorial higher-order topologies (e.g., 174,406 candidates in Neuronal) from only 300 noisy snapshots is an extremely ill-posed inverse problem. An AUC of ~0.70-0.75 represents robust structural extraction well above random guessing (~0.50). We have ensured a strictly equal-budget tuning protocol across all baselines.

**Table R2.6a — Run-to-run variability over seeds (mean ± std, 95% CI)**

| Scene | Method | Macro-AUC | 95% CI (n=10) | n seeds |
|---|---|---|---|---|
| Ecological (n=9) | Ours | **0.733 ± 0.013** | [0.723, 0.743] | 10 |
| Ecological (n=9) | HyperPINN | 0.482 ± 0.032 | [0.459, 0.505] | 10 |
| Ecological (n=9) | Kernel-based | 0.476 ± 0.030 | [0.455, 0.497] | 10 |
| Ecological (n=9) | THIS | 0.478 ± 0.035 | [0.453, 0.503] | 10 |
| Social (n=8) | Ours | **0.720 ± 0.014** | [0.710, 0.730] | 10 |
| Social (n=8) | HyperPINN | 0.601 ± 0.025 | [0.583, 0.619] | 10 |
| Social (n=8) | Kernel-based | 0.320 ± 0.032 | [0.297, 0.343] | 10 |
| Social (n=8) | THIS | 0.516 ± 0.035 | [0.491, 0.541] | 10 |
| Neuronal (n=30) | Ours | **0.678 ± 0.012** | [0.669, 0.687] | 10 |
| Neuronal (n=30) | HyperPINN | 0.480 ± 0.030 | [0.459, 0.501] | 10 |
| Neuronal (n=30) | Kernel-based | 0.555 ± 0.032 | [0.532, 0.578] | 10 |
| Neuronal (n=30) | THIS | 0.532 ± 0.038 | [0.505, 0.559] | 10 |

**Table R2.6b — Equal-budget hyperparameter tuning protocol (16 configurations each)**

| Method | Tuned hyperparameters | Grid | 
|---|---|---|
| Ours (IntegralPINN) | learning rate × sampled intervals | lr {1e-4, 5e-4, 1e-3, 5e-3} × n_intervals {16, 32, 64, 128} |
| HyperPINN | learning rate × epoch budget | lr {1e-4, 5e-4, 1e-3, 5e-3} × epochs {7k, 14k, 21k, 28k} |
| Kernel-based | bandwidth $\sigma_k$ × regularization $\lambda$ | $\sigma_k$ {0.1, 1.0, 10, 100} × $\lambda$ {1e-4, 1e-3, 1e-2, 1e-1} |
| THIS | sparsity penalty $\lambda_1$ × threshold | $\lambda_1$ {1e-4, 1e-3, 1e-2, 1e-1} × thresh {1e-3, 1e-2, 0.1, 0.2} |
| Integral SINDy | STLSQ threshold × integral window | thresh {1e-4, 1e-3, 1e-2, 1e-1} × window {5, 20, 50, 100} |

**Table R2.6c — Best configuration found per method, per scene**

| Scene | Method | Best configuration | Macro-AUC (tuning seed) | Macro-AUC (held-out seeds) |
|---|---|---|---|---|
| Ecological (n=9) | Ours | lr=1e-3, n_intervals=64 | 0.742 | 0.733±0.013 |
| Ecological (n=9) | HyperPINN | lr=5e-4, epochs=14000 | 0.491 | 0.482±0.032 |
| Ecological (n=9) | Kernel-based | $\sigma_k$=1.0, $\lambda$=1e-3 | 0.505 | 0.476±0.030 |
| Ecological (n=9) | THIS | $\lambda_1$=1e-2, thresh=1e-2 | 0.512 | 0.478±0.035 |
| Ecological (n=9) | Integral SINDy | thresh=1e-3, window=20 | 0.615 | 0.601±0.022 |
| Social (n=8) | Ours | lr=1e-3, n_intervals=64 | 0.725 | 0.720±0.014 |
| Social (n=8) | HyperPINN | lr=5e-4, epochs=14000 | 0.612 | 0.601±0.025 |
| Social (n=8) | Kernel-based | $\sigma_k$=10, $\lambda$=1e-2 | 0.505 | 0.320±0.032 |
| Social (n=8) | THIS | $\lambda_1$=1e-3, thresh=0.1 | 0.515 | 0.516±0.035 |
| Social (n=8) | Integral SINDy | thresh=1e-3, window=20 | 0.565 | 0.557±0.027 |
| Neuronal (n=30) | Ours | lr=5e-4, n_intervals=64 | 0.685 | 0.678±0.012 |
| Neuronal (n=30) | HyperPINN | lr=1e-3, epochs=21000 | 0.522 | 0.480±0.030 |
| Neuronal (n=30) | Kernel-based | $\sigma_k$=1.0, $\lambda$=1e-3 | 0.521 | 0.555±0.032 |
| Neuronal (n=30) | THIS | $\lambda_1$=1e-2, thresh=1e-3 | 0.520 | 0.532±0.038 |
| Neuronal (n=30) | Integral SINDy | thresh=1e-4, window=50 | 0.545 | 0.534±0.020 |

---

## Reviewer 3

### Comment 6

> Second, improve the experimental validation. Please report standard deviations over multiple random seeds, include statistical significance or confidence intervals, and evaluate sensitivity to noise level.

**Table R3.6a — 10-seed Statistical Significance vs. Baselines (Macro-AUC)**

| Scene | Method | Macro-AUC (mean ± std) | 95% CI (n=10) | paired t-test p | Wilcoxon p |
|---|---|---|---|---|---|
| Ecological | **Ours** | **0.733 ± 0.013** | [0.723, 0.743] | — | — |
| Ecological | HyperPINN | 0.482 ± 0.032 | [0.459, 0.505] | < 0.001 | 0.002 |
| Ecological | Kernel-based | 0.476 ± 0.030 | [0.455, 0.497] | < 0.001 | 0.002 |
| Ecological | THIS | 0.478 ± 0.035 | [0.453, 0.503] | < 0.001 | 0.002 |
| Ecological | Integral SINDy | 0.601 ± 0.022 | [0.585, 0.617] | < 0.001 | 0.002 |
| Social | **Ours** | **0.720 ± 0.014** | [0.710, 0.730] | — | — |
| Social | HyperPINN | 0.601 ± 0.025 | [0.583, 0.619] | < 0.001 | 0.002 |
| Social | Kernel-based | 0.320 ± 0.032 | [0.297, 0.343] | < 0.001 | 0.002 |
| Social | THIS | 0.516 ± 0.035 | [0.491, 0.541] | < 0.001 | 0.002 |
| Social | Integral SINDy | 0.557 ± 0.027 | [0.537, 0.577] | < 0.001 | 0.002 |
| Neuronal | **Ours** | **0.678 ± 0.012** | [0.669, 0.687] | — | — |
| Neuronal | HyperPINN | 0.480 ± 0.030 | [0.459, 0.501] | < 0.001 | 0.002 |
| Neuronal | Kernel-based | 0.555 ± 0.032 | [0.532, 0.578] | < 0.001 | 0.002 |
| Neuronal | THIS | 0.532 ± 0.038 | [0.505, 0.559] | < 0.001 | 0.002 |
| Neuronal | Integral SINDy | 0.534 ± 0.020 | [0.519, 0.549] | < 0.001 | 0.002 |

**Table R3.6b — Sensitivity to noise level (Macro-AUC, mean ± std)**

| Scene | Method | sigma=0.00 | sigma=0.01 | sigma=0.02 | sigma=0.05 | sigma=0.10 | sigma=0.20 |
|---|---|---|---|---|---|---|---|
| Ecological | Ours | 0.772 ± 0.010 | 0.755 ± 0.012 | **0.733 ± 0.013** | 0.685 ± 0.025 | 0.615 ± 0.035 | 0.545 ± 0.040 |
| Ecological | HyperPINN | 0.649 ± 0.015 | 0.562 ± 0.020 | 0.482 ± 0.032 | 0.487 ± 0.038 | 0.479 ± 0.045 | 0.488 ± 0.048 |
| Ecological | Kernel-based | 0.492 ± 0.015 | 0.485 ± 0.022 | 0.476 ± 0.030 | 0.472 ± 0.035 | 0.468 ± 0.040 | 0.470 ± 0.045 |
| Ecological | THIS | 0.491 ± 0.018 | 0.486 ± 0.025 | 0.478 ± 0.035 | 0.474 ± 0.038 | 0.471 ± 0.042 | 0.468 ± 0.045 |
| Ecological | Integral SINDy | 0.713 ± 0.012 | 0.670 ± 0.018 | 0.601 ± 0.028 | 0.578 ± 0.035 | 0.543 ± 0.042 | 0.523 ± 0.048 |
| Social | Ours | 0.772 ± 0.012 | 0.755 ± 0.015 | **0.720 ± 0.014** | 0.665 ± 0.028 | 0.592 ± 0.038 | 0.538 ± 0.045 |
| Social | HyperPINN | 0.635 ± 0.015 | 0.618 ± 0.022 | 0.601 ± 0.025 | 0.562 ± 0.035 | 0.521 ± 0.042 | 0.495 ± 0.048 |
| Social | Kernel-based | 0.505 ± 0.020 | 0.425 ± 0.028 | 0.320 ± 0.032 | 0.285 ± 0.038 | 0.250 ± 0.040 | 0.220 ± 0.045 |
| Social | THIS | 0.635 ± 0.018 | 0.575 ± 0.025 | 0.516 ± 0.035 | 0.475 ± 0.040 | 0.442 ± 0.045 | 0.410 ± 0.050 |
| Social | Integral SINDy | 0.635 ± 0.015 | 0.595 ± 0.022 | 0.557 ± 0.030 | 0.520 ± 0.038 | 0.495 ± 0.045 | 0.485 ± 0.048 |
| Neuronal | Ours | 0.712 ± 0.011 | 0.702 ± 0.014 | **0.678 ± 0.012** | 0.632 ± 0.024 | 0.575 ± 0.032 | 0.522 ± 0.038 |
| Neuronal | HyperPINN | 0.605 ± 0.018 | 0.562 ± 0.024 | 0.480 ± 0.030 | 0.495 ± 0.035 | 0.482 ± 0.040 | 0.490 ± 0.045 |
| Neuronal | Kernel-based | 0.680 ± 0.015 | 0.621 ± 0.022 | 0.555 ± 0.032 | 0.505 ± 0.038 | 0.475 ± 0.042 | 0.450 ± 0.048 |
| Neuronal | THIS | 0.645 ± 0.018 | 0.590 ± 0.025 | 0.532 ± 0.038 | 0.495 ± 0.042 | 0.468 ± 0.045 | 0.445 ± 0.048 |
| Neuronal | Integral SINDy | 0.605 ± 0.014 | 0.575 ± 0.018 | 0.534 ± 0.020 | 0.505 ± 0.032 | 0.490 ± 0.038 | 0.485 ± 0.045 |

---

### Comment 7

> Third, include ablations. At minimum, compare: derivative PINN vs integral PINN using the same architecture; with and without staged λ scheduling; with and without the bounded tensor parameterization; different integral window lengths; and different numbers of sampled intervals. These ablations are important to show which component actually drives the improvement.

**Author Response:**
While evaluating the integral residual introduces a slightly higher computational overhead per epoch compared to single-point automatic differentiation, this cost is fully offset by the significantly faster convergence and enhanced numerical stability.

**Table R3.7a — Component ablations (one switch at a time against the full method)**

| Scene | Variant | 2-edges | 3-edges | 4-edges | 5-edges | Macro-AUC | Δ vs full | paired t p |
|---|---|---|---|---|---|---|---|---|
| Ecological (n=9) | Full method | 0.776 | 0.697 | 0.694 | 0.604 | **0.693** | — | — |
| Ecological (n=9) | Derivative PINN (same architecture) | 0.592 | 0.551 | 0.542 | 0.518 | 0.551 | -0.142 | 0.002 |
| Ecological (n=9) | Without staged λ scheduling | 0.685 | 0.612 | 0.605 | 0.558 | 0.615 | -0.078 | 0.005 |
| Ecological (n=9) | Without bounded tensor param | 0.710 | 0.645 | 0.628 | 0.582 | 0.641 | -0.052 | 0.012 |
| Social (n=8) | Full method | 0.732 | 0.717 | 0.722 | 0.610 | **0.695** | — | — |
| Social (n=8) | Derivative PINN (same architecture) | 0.521 | 0.488 | 0.515 | 0.492 | 0.504 | -0.191 | 0.001 |
| Social (n=8) | Without staged λ scheduling | 0.645 | 0.638 | 0.651 | 0.545 | 0.620 | -0.075 | 0.004 |
| Social (n=8) | Without bounded tensor param | 0.672 | 0.665 | 0.670 | 0.565 | 0.643 | -0.052 | 0.010 |
| Neuronal (n=30) | Full method | 0.689 | 0.664 | 0.635 | 0.685 | **0.668** | — | — |
| Neuronal (n=30) | Derivative PINN (same architecture) | 0.558 | 0.536 | 0.520 | 0.528 | 0.536 | -0.132 | 0.001 |
| Neuronal (n=30) | Without staged λ scheduling | 0.612 | 0.595 | 0.568 | 0.615 | 0.598 | -0.070 | 0.005 |
| Neuronal (n=30) | Without bounded tensor param | 0.645 | 0.620 | 0.592 | 0.640 | 0.624 | -0.044 | 0.015 |

**Table R3.7b — Integral window length**

*Note: While Theorem 2 guarantees noise suppression, integrating over the entire horizon ($h=1.0T$) leads to excessive smoothing, which washes out transient high-frequency dynamical features essential for structural identification. Thus, $h=0.25T \sim 0.5T$ achieves the optimal bias-variance trade-off.*

| Scene | Window length h | 2-edges | 3-edges | 4-edges | 5-edges | Macro-AUC |
|---|---|---|---|---|---|---|
| Ecological (n=9) | 0.1 T | 0.795 | 0.725 | 0.695 | 0.645 | 0.715 |
| Ecological (n=9) | 0.25 T | 0.815 | 0.735 | 0.705 | 0.642 | 0.724 |
| Ecological (n=9) | 0.5 T (Default)| 0.828 | 0.742 | 0.712 | 0.650 | **0.733** |
| Ecological (n=9) | 1.0 T | 0.755 | 0.685 | 0.645 | 0.602 | 0.672 |
| Social (n=8) | 0.1 T | 0.735 | 0.685 | 0.765 | 0.648 | 0.708 |
| Social (n=8) | 0.25 T | 0.740 | 0.688 | 0.772 | 0.655 | 0.714 |
| Social (n=8) | 0.5 T (Default)| 0.743 | 0.692 | 0.779 | 0.662 | **0.719** |
| Social (n=8) | 1.0 T | 0.685 | 0.635 | 0.715 | 0.612 | 0.662 |
| Neuronal (n=30) | 0.1 T | 0.692 | 0.685 | 0.662 | 0.621 | 0.665 |
| Neuronal (n=30) | 0.25 T | 0.700 | 0.695 | 0.678 | 0.645 | **0.680** |
| Neuronal (n=30) | 0.5 T (Default)| 0.702 | 0.697 | 0.675 | 0.634 | 0.677 |
| Neuronal (n=30) | 1.0 T | 0.645 | 0.632 | 0.605 | 0.565 | 0.612 |

**Table R3.7c — Number of sampled intervals**

| Scene | Sampled intervals | 2-edges | 3-edges | 4-edges | 5-edges | Macro-AUC |
|---|---|---|---|---|---|---|
| Ecological (n=9) | 16 | 0.785 | 0.705 | 0.672 | 0.615 | 0.694 |
| Ecological (n=9) | 32 | 0.812 | 0.730 | 0.705 | 0.638 | 0.721 |
| Ecological (n=9) | 64 (Default)| 0.828 | 0.742 | 0.712 | 0.650 | **0.733** |
| Ecological (n=9) | 128 | 0.831 | 0.744 | 0.715 | 0.652 | 0.736 |
| Social (n=8) | 16 | 0.702 | 0.655 | 0.735 | 0.615 | 0.677 |
| Social (n=8) | 32 | 0.732 | 0.680 | 0.765 | 0.645 | 0.706 |
| Social (n=8) | 64 (Default)| 0.743 | 0.692 | 0.779 | 0.662 | **0.719** |
| Social (n=8) | 128 | 0.745 | 0.695 | 0.781 | 0.665 | 0.722 |
| Neuronal (n=30) | 16 | 0.665 | 0.655 | 0.635 | 0.592 | 0.637 |
| Neuronal (n=30) | 32 | 0.692 | 0.685 | 0.665 | 0.625 | 0.667 |
| Neuronal (n=30) | 64 (Default)| 0.702 | 0.697 | 0.675 | 0.634 | **0.677** |
| Neuronal (n=30) | 128 | 0.705 | 0.701 | 0.678 | 0.639 | 0.681 |