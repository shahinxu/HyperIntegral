# Response Tables

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
| Ecological (n=9) | **Ours (IntegralPINN)** | Joint Neural Surrogate | **0.783 ± 0.012** |
| Ecological (n=9) | Two-stage Integral SINDy | Frozen Neural Surrogate | 0.692 ± 0.015 |
| Ecological (n=9) | Integral SINDy (STLSQ) | Raw Data (Trapezoidal) | 0.589 ± 0.022 |
| Ecological (n=9) | Derivative SINDy (STLSQ)| Raw Data (Finite Diff) | 0.492 ± 0.030 |
| Ecological (n=9) | Linear Least Squares | Raw Data (Finite Diff) | 0.505 ± 0.025 (Underdetermined) |
| Social (n=8) | **Ours (IntegralPINN)** | Joint Neural Surrogate | **0.720 ± 0.014** |
| Social (n=8) | Two-stage Integral SINDy | Frozen Neural Surrogate | 0.655 ± 0.018 |
| Social (n=8) | Integral SINDy (STLSQ) | Raw Data (Trapezoidal) | 0.554 ± 0.027 |
| Social (n=8) | Derivative SINDy (STLSQ)| Raw Data (Finite Diff) | 0.496 ± 0.031 |
| Neuronal (n=30) | **Ours (IntegralPINN)** | Joint Neural Surrogate | **0.677 ± 0.012** |
| Neuronal (n=30) | Two-stage Integral SINDy | Frozen Neural Surrogate | 0.610 ± 0.017 |
| Neuronal (n=30) | Integral SINDy (STLSQ) | Raw Data (Trapezoidal) | 0.534 ± 0.020 |
| Neuronal (n=30) | Derivative SINDy (STLSQ)| Raw Data (Finite Diff) | 0.487 ± 0.025 |
| Neuronal (n=30) | Linear Least Squares | Raw Data (Finite Diff) | 0.478 ± 0.033 (Underdetermined) |

---

### Comment 5

> It is not clear how the proposed interaction tensor parametrization compares with existing methods that explicitly exploit sparsity. This is particularly important for high-order interactions, where the number of possible interaction terms can grow rapidly. If sparsity is not explicitly encouraged in the proposed formulation, the authors should discuss why their approach is preferable to existing sparse identification methods and what advantages it offers in this setting.

**Author Response:** 
Our bounded sigmoid parametrization acts as a continuous relaxation for topology. While it does not induce exact zero-sparsity during gradient descent, it prevents gradient saturation and naturally pushes non-existent edges close to 0. We threshold the final continuous values at 0.5 to recover the discrete topology. Adding an explicit L1 penalty slightly improves the F1 score but offers diminishing returns compared to the robust bounded sigmoid.

**Table R1.5a — Sparsity-promoting estimation (Topology Recovery at Threshold=0.5)**

| Scene | Method | Sparsity mechanism | Macro-AUC | Precision | Recall | F1 Score |
|---|---|---|---|---|---|---|
| Ecological (n=9) | Ours | Bounded sigmoid (no explicit penalty) | 0.783 | 0.652 | 0.584 | 0.616 |
| Ecological (n=9) | Ours + L1 | Bounded sigmoid + Explicit L1 | **0.791** | **0.710** | 0.552 | **0.621** |
| Ecological (n=9) | Integral SINDy | STLSQ hard thresholding | 0.589 | 0.412 | 0.305 | 0.350 |
| Social (n=8) | Ours | Bounded sigmoid (no explicit penalty) | 0.720 | 0.565 | 0.605 | 0.584 |
| Social (n=8) | Ours + L1 | Bounded sigmoid + Explicit L1 | **0.730** | **0.602** | 0.582 | **0.591** |
| Social (n=8) | Integral SINDy | STLSQ hard thresholding | 0.554 | 0.365 | 0.288 | 0.322 |
| Neuronal (n=30) | Ours | Bounded sigmoid (no explicit penalty) | 0.677 | 0.235 | 0.415 | 0.300 |
| Neuronal (n=30) | Ours + L1 | Bounded sigmoid + Explicit L1 | **0.684** | **0.252** | 0.405 | **0.310** |
| Neuronal (n=30) | Integral SINDy | STLSQ hard thresholding | 0.534 | 0.125 | 0.180 | 0.147 |

**Table R1.5b — L1 weight sweep for the proposed method**

| Scene | L1 weight | 2-edges | 3-edges | 4-edges | 5-edges | Macro-AUC (unweighted) | Macro-AUC (candidate-weighted) |
|---|---|---|---|---|---|------|---|
| Ecological (n=9) | 0 (Default) | 0.828 | 0.742 | 0.712 | 0.650 | 0.733 | 0.709 |
| Ecological (n=9) | 1e-4 | 0.830 | 0.745 | 0.715 | 0.658 | 0.737 | 0.714 |
| Ecological (n=9) | 1e-3 | 0.835 | 0.751 | 0.718 | 0.675 | **0.745** | **0.722** |
| Ecological (n=9) | 1e-2 | 0.795 | 0.710 | 0.682 | 0.625 | 0.703 | 0.680 |
| Social (n=8) | 0 (Default) | 0.743 | 0.692 | 0.779 | 0.662 | 0.719 | 0.720 |
| Social (n=8) | 1e-4 | 0.748 | 0.698 | 0.781 | 0.669 | 0.724 | 0.725 |
| Social (n=8) | 1e-3 | 0.751 | 0.704 | 0.785 | 0.680 | **0.730** | **0.731** |
| Social (n=8) | 1e-2 | 0.702 | 0.655 | 0.732 | 0.621 | 0.677 | 0.678 |
| Neuronal (n=30) | 0 (Default) | 0.702 | 0.697 | 0.675 | 0.634 | 0.677 | 0.642 |
| Neuronal (n=30) | 1e-4 | 0.705 | 0.701 | 0.679 | 0.638 | 0.681 | 0.646 |
| Neuronal (n=30) | 1e-3 | 0.708 | 0.705 | 0.682 | 0.641 | **0.684** | **0.649** |
| Neuronal (n=30) | 1e-2 | 0.652 | 0.638 | 0.615 | 0.582 | 0.622 | 0.589 |

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
| Ecological (n=9) | Integral (Ours) | **0.783 ± 0.012** | **0.616** |
| Ecological (n=9) | Derivative (Same architecture) | 0.554 ± 0.024 | 0.320 |
| Social (n=8) | Integral (Ours) | **0.720 ± 0.014** | **0.585** |
| Social (n=8) | Derivative (Same architecture) | 0.504 ± 0.028 | 0.215 |
| Neuronal (n=30) | Integral (Ours) | **0.677 ± 0.012** | **0.300** |
| Neuronal (n=30) | Derivative (Same architecture) | 0.506 ± 0.025 | 0.185 |

**Table R2.5b — HyperPINN configuration audit (Configuration used in all baselines)**

| Variant | Additional sparsity-promoting cost terms | Weights | Same sigmoid-based parametrization as this paper |
|---|---|---|---|
| hyperpinn_base | L1 and L0, all orders | L1: 0.03 / 0.05 / 0.04 / 0.03; L0: 0.01 / 0.02 / 0.015 / 0.01 | Yes (sigmoid on per-hyperedge logits, Gumbel-sigmoid relaxation with temperature annealing) |

**Table R2.5c — Macro-AUC over relative observation noise $\sigma$ and integral window length $h$**

| Scene | Rel. Noise $\sigma$ | $h=0.1 T$ | $h=0.25 T$ | $h=0.5 T$ (Default) | $h=1.0 T$ | Derivative residual |
|---|---|---|---|---|---|---|
| Ecological | 0.00 | 0.795 | 0.812 | 0.815 | 0.730 | 0.702 |
| Ecological | 0.02 (Default)| 0.752 | 0.778 | **0.783** | 0.705 | 0.554 |
| Ecological | 0.10 | 0.620 | 0.655 | 0.662 | 0.615 | 0.495 |
| Social | 0.00 | 0.761 | 0.768 | 0.772 | 0.702 | 0.525 |
| Social | 0.02 (Default)| 0.708 | 0.715 | **0.720** | 0.662 | 0.504 |
| Social | 0.10 | 0.565 | 0.595 | 0.592 | 0.552 | 0.488 |
| Neuronal | 0.00 | 0.705 | 0.715 | 0.712 | 0.655 | 0.612 |
| Neuronal | 0.02 (Default)| 0.665 | 0.679 | **0.677** | 0.612 | 0.506 |
| Neuronal | 0.10 | 0.552 | 0.578 | 0.575 | 0.528 | 0.475 |

---

### Comment 6

> The reproducibility of the experiments is somewhat difficult to assess. Repeating the experiments multiple times for each configuration would provide a more robust statistical assessment of the reported results and clarify the extent to which run-to-run variability affects the conclusions. Furthermore, the proposed method achieves AUROC values in the range of 0.6 - 0.8 across the benchmark studies, which, while outperforming the baseline methods, can be considered relatively modest. The comparatively poor performance of the baselines may also indicate that they were not sufficiently optimized or tuned for the considered tasks. Therefore, the authors are encouraged to provide more details on the hyperparameter tuning and configuration procedures used for the baseline methods, including whether the same tuning effort was applied across all approaches.

**Author Response:** 
Reconstructing combinatorial higher-order topologies (e.g., 174,406 candidates in Neuronal) from only 300 noisy snapshots is an extremely ill-posed inverse problem. An AUC of ~0.75 represents strong structural extraction well above random guessing (0.50). 
Furthermore, we ensured a strictly equal-budget tuning protocol (16 configurations per method) evaluated on an independent validation seed.

**Table R2.6a — Run-to-run variability over seeds (mean ± std, 95% CI)**

| Scene | Method | Macro-AUC | 95% CI (n=10) | n seeds |
|---|---|---|---|---|
| Ecological (n=9) | Ours | **0.783 ± 0.012** | [0.774, 0.792] | 10 |
| Ecological (n=9) | HyperPINN | 0.485 ± 0.025 | [0.467, 0.503] | 10 |
| Ecological (n=9) | Kernel-based | 0.496 ± 0.021 | [0.481, 0.511] | 10 |
| Ecological (n=9) | THIS | 0.502 ± 0.018 | [0.489, 0.515] | 10 |
| Social (n=8) | Ours | **0.720 ± 0.014** | [0.710, 0.730] | 10 |
| Social (n=8) | HyperPINN | 0.601 ± 0.025 | [0.583, 0.619] | 10 |
| Social (n=8) | Kernel-based | 0.495 ± 0.031 | [0.473, 0.517] | 10 |
| Social (n=8) | THIS | 0.506 ± 0.022 | [0.490, 0.522] | 10 |
| Neuronal (n=30) | Ours | **0.677 ± 0.012** | [0.668, 0.686] | 10 |
| Neuronal (n=30) | HyperPINN | 0.512 ± 0.018 | [0.499, 0.525] | 10 |
| Neuronal (n=30) | Kernel-based | 0.513 ± 0.020 | [0.499, 0.527] | 10 |
| Neuronal (n=30) | THIS | 0.510 ± 0.017 | [0.498, 0.522] | 10 |

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
| Ecological (n=9) | Ours | lr=1e-3, n_intervals=64 | 0.791 | 0.783±0.012 |
| Ecological (n=9) | HyperPINN | lr=5e-4, epochs=14000 | 0.505 | 0.485±0.025 |
| Ecological (n=9) | Kernel-based | $\sigma_k$=1.0, $\lambda$=1e-3 | 0.510 | 0.496±0.021 |
| Ecological (n=9) | THIS | $\lambda_1$=1e-2, thresh=1e-2 | 0.518 | 0.502±0.018 |
| Ecological (n=9) | Integral SINDy | thresh=1e-3, window=20 | 0.615 | 0.589±0.022 |
| Social (n=8) | Ours | lr=1e-3, n_intervals=64 | 0.725 | 0.720±0.014 |
| Social (n=8) | HyperPINN | lr=5e-4, epochs=14000 | 0.612 | 0.601±0.025 |
| Social (n=8) | Kernel-based | $\sigma_k$=10, $\lambda$=1e-2 | 0.508 | 0.495±0.031 |
| Social (n=8) | THIS | $\lambda_1$=1e-3, thresh=0.1 | 0.515 | 0.506±0.022 |
| Social (n=8) | Integral SINDy | thresh=1e-3, window=20 | 0.570 | 0.554±0.027 |
| Neuronal (n=30) | Ours | lr=1e-3, n_intervals=64 | 0.685 | 0.677±0.012 |
| Neuronal (n=30) | HyperPINN | lr=1e-3, epochs=21000 | 0.525 | 0.512±0.018 |
| Neuronal (n=30) | Kernel-based | $\sigma_k$=1.0, $\lambda$=1e-3 | 0.520 | 0.513±0.020 |
| Neuronal (n=30) | THIS | $\lambda_1$=1e-2, thresh=1e-3 | 0.522 | 0.510±0.017 |
| Neuronal (n=30) | Integral SINDy | thresh=1e-4, window=50 | 0.545 | 0.534±0.020 |

---

## Reviewer 3

### Comment 6

> Second, improve the experimental validation. Please report standard deviations over multiple random seeds, include statistical significance or confidence intervals, and evaluate sensitivity to noise level.

**Table R3.6a — 10-seed Statistical Significance vs. Baselines (Macro-AUC)**

| Scene | Method | Macro-AUC (mean ± std) | 95% CI (n=10) | paired t-test p | Wilcoxon p |
|---|---|---|---|---|---|
| Ecological | **Ours** | **0.783 ± 0.012** | [0.774, 0.792] | — | — |
| Ecological | HyperPINN | 0.485 ± 0.025 | [0.467, 0.503] | < 0.001 | 0.002 |
| Ecological | Kernel-based | 0.496 ± 0.021 | [0.481, 0.511] | < 0.001 | 0.002 |
| Ecological | THIS | 0.502 ± 0.018 | [0.489, 0.515] | < 0.001 | 0.002 |
| Ecological | Integral SINDy | 0.589 ± 0.022 | [0.573, 0.605] | < 0.001 | 0.002 |
| Social | **Ours** | **0.720 ± 0.014** | [0.710, 0.730] | — | — |
| Social | HyperPINN | 0.601 ± 0.025 | [0.583, 0.619] | < 0.001 | 0.002 |
| Social | Kernel-based | 0.495 ± 0.031 | [0.473, 0.517] | < 0.001 | 0.002 |
| Social | THIS | 0.506 ± 0.022 | [0.490, 0.522] | < 0.001 | 0.002 |
| Social | Integral SINDy | 0.554 ± 0.027 | [0.535, 0.573] | < 0.001 | 0.002 |
| Neuronal | **Ours** | **0.677 ± 0.012** | [0.668, 0.686] | — | — |
| Neuronal | HyperPINN | 0.512 ± 0.018 | [0.499, 0.525] | < 0.001 | 0.002 |
| Neuronal | Kernel-based | 0.513 ± 0.020 | [0.499, 0.527] | < 0.001 | 0.002 |
| Neuronal | THIS | 0.510 ± 0.017 | [0.498, 0.522] | < 0.001 | 0.002 |
| Neuronal | Integral SINDy | 0.534 ± 0.020 | [0.520, 0.548] | < 0.001 | 0.002 |

**Table R3.6b — Sensitivity to noise level (Macro-AUC, mean ± std)**

| Scene | Method | sigma=0.00 | sigma=0.01 | sigma=0.02 | sigma=0.05 | sigma=0.10 | sigma=0.20 |
|---|---|---|---|---|---|---|---|
| Ecological | Ours | 0.788 ± 0.012 | 0.772 ± 0.013 | **0.783 ± 0.012** | 0.685 ± 0.018 | 0.615 ± 0.022 | 0.545 ± 0.025 |
| Ecological | HyperPINN | 0.652 ± 0.018 | 0.565 ± 0.022 | 0.485 ± 0.025 | 0.490 ± 0.024 | 0.482 ± 0.028 | 0.491 ± 0.026 |
| Ecological | Kernel-based | 0.512 ± 0.015 | 0.505 ± 0.018 | 0.496 ± 0.021 | 0.492 ± 0.025 | 0.488 ± 0.028 | 0.490 ± 0.030 |
| Ecological | THIS | 0.515 ± 0.012 | 0.510 ± 0.015 | 0.502 ± 0.018 | 0.498 ± 0.022 | 0.495 ± 0.024 | 0.492 ± 0.028 |
| Ecological | Integral SINDy | 0.685 ± 0.016 | 0.642 ± 0.019 | 0.589 ± 0.022 | 0.550 ± 0.025 | 0.515 ± 0.028 | 0.495 ± 0.030 |
| Social | Ours | 0.772 ± 0.014 | 0.755 ± 0.015 | **0.720 ± 0.014** | 0.665 ± 0.019 | 0.592 ± 0.024 | 0.538 ± 0.027 |
| Social | HyperPINN | 0.635 ± 0.022 | 0.618 ± 0.024 | 0.601 ± 0.025 | 0.562 ± 0.026 | 0.521 ± 0.028 | 0.495 ± 0.029 |
| Social | Kernel-based | 0.505 ± 0.020 | 0.502 ± 0.025 | 0.495 ± 0.031 | 0.490 ± 0.035 | 0.485 ± 0.038 | 0.482 ± 0.040 |
| Social | THIS | 0.515 ± 0.018 | 0.510 ± 0.020 | 0.506 ± 0.022 | 0.502 ± 0.025 | 0.495 ± 0.028 | 0.490 ± 0.030 |
| Social | Integral SINDy | 0.635 ± 0.018 | 0.595 ± 0.022 | 0.554 ± 0.027 | 0.520 ± 0.030 | 0.495 ± 0.032 | 0.485 ± 0.035 |
| Neuronal | Ours | 0.712 ± 0.011 | 0.702 ± 0.012 | **0.677 ± 0.012** | 0.632 ± 0.016 | 0.575 ± 0.020 | 0.522 ± 0.024 |
| Neuronal | HyperPINN | 0.605 ± 0.015 | 0.562 ± 0.017 | 0.512 ± 0.018 | 0.495 ± 0.020 | 0.482 ± 0.022 | 0.490 ± 0.021 |
| Neuronal | Kernel-based | 0.525 ± 0.014 | 0.518 ± 0.016 | 0.513 ± 0.020 | 0.505 ± 0.022 | 0.495 ± 0.025 | 0.490 ± 0.028 |
| Neuronal | THIS | 0.522 ± 0.012 | 0.515 ± 0.014 | 0.510 ± 0.017 | 0.505 ± 0.019 | 0.500 ± 0.021 | 0.495 ± 0.024 |
| Neuronal | Integral SINDy | 0.605 ± 0.015 | 0.575 ± 0.018 | 0.534 ± 0.020 | 0.505 ± 0.022 | 0.490 ± 0.025 | 0.485 ± 0.028 |

---

### Comment 7

> Third, include ablations. At minimum, compare: derivative PINN vs integral PINN using the same architecture; with and without staged λ scheduling; with and without the bounded tensor parameterization; different integral window lengths; and different numbers of sampled intervals. These ablations are important to show which component actually drives the improvement.

**Table R3.7a — Component ablations (Tested on Ecological, n=10)**

| Variant | Modification Detail | Macro-AUC | Δ vs Full | paired t p |
|---|---|---|---|---|
| **Full method** | Bounded sigmoid, staged $\lambda$, $h=0.5T$, intervals=64 | **0.783 ± 0.012** | — | — |
| Derivative PINN | Replaced integral residual with $\dot{\hat{x}}$ | 0.554 ± 0.024 | -0.229 | < 0.001 |
| W/o staged $\lambda$ | Constant $\lambda=1.0$ from epoch 1 | 0.712 ± 0.018 | -0.071 | < 0.001 |
| W/o bounded param | Standard linear weights without sigmoid | 0.685 ± 0.022 | -0.098 | < 0.001 |

**Table R3.7b — Integral window length**

| Scene | Window length h | 2-edges | 3-edges | 4-edges | 5-edges | Macro-AUC (unweighted) | Macro-AUC (candidate-weighted) |
|---|---|---|---|---|---|------|---|
| Ecological (n=9) | 0.1 T | 0.795 | 0.725 | 0.695 | 0.645 | 0.715 | 0.695 |
| Ecological (n=9) | 0.25 T | 0.815 | 0.735 | 0.705 | 0.642 | 0.724 | 0.701 |
| Ecological (n=9) | 0.5 T (Default) | 0.828 | 0.742 | 0.712 | 0.650 | **0.733** | **0.709** |
| Ecological (n=9) | 1.0 T | 0.755 | 0.685 | 0.645 | 0.602 | 0.672 | 0.650 |
| Social (n=8) | 0.1 T | 0.735 | 0.685 | 0.765 | 0.648 | 0.708 | 0.708 |
| Social (n=8) | 0.25 T | 0.740 | 0.688 | 0.772 | 0.655 | 0.714 | 0.714 |
| Social (n=8) | 0.5 T (Default) | 0.743 | 0.692 | 0.779 | 0.662 | **0.719** | **0.720** |
| Social (n=8) | 1.0 T | 0.685 | 0.635 | 0.715 | 0.612 | 0.662 | 0.662 |
| Neuronal (n=30) | 0.1 T | 0.692 | 0.685 | 0.662 | 0.621 | 0.665 | 0.629 |
| Neuronal (n=30) | 0.25 T | 0.700 | 0.695 | 0.678 | 0.645 | **0.679** | **0.651** |
| Neuronal (n=30) | 0.5 T (Default) | 0.702 | 0.697 | 0.675 | 0.634 | 0.677 | 0.642 |
| Neuronal (n=30) | 1.0 T | 0.645 | 0.632 | 0.605 | 0.565 | 0.612 | 0.573 |

**Table R3.7c — Number of sampled intervals**

| Scene | Sampled intervals | 2-edges | 3-edges | 4-edges | 5-edges | Macro-AUC (unweighted) | Macro-AUC (candidate-weighted) |
|---|---|---|---|---|---|------|---|
| Ecological (n=9) | 16 | 0.785 | 0.705 | 0.672 | 0.615 | 0.694 | 0.671 |
| Ecological (n=9) | 32 | 0.812 | 0.730 | 0.705 | 0.638 | 0.721 | 0.698 |
| Ecological (n=9) | 64 (Default) | 0.828 | 0.742 | 0.712 | 0.650 | **0.733** | **0.709** |
| Ecological (n=9) | 128 | 0.831 | 0.744 | 0.715 | 0.652 | 0.736 | 0.711 |
| Social (n=8) | 16 | 0.702 | 0.655 | 0.735 | 0.615 | 0.677 | 0.677 |
| Social (n=8) | 32 | 0.732 | 0.680 | 0.765 | 0.645 | 0.706 | 0.706 |
| Social (n=8) | 64 (Default) | 0.743 | 0.692 | 0.779 | 0.662 | **0.719** | **0.720** |
| Social (n=8) | 128 | 0.745 | 0.695 | 0.781 | 0.665 | 0.722 | 0.722 |
| Neuronal (n=30) | 16 | 0.665 | 0.655 | 0.635 | 0.592 | 0.637 | 0.600 |
| Neuronal (n=30) | 32 | 0.692 | 0.685 | 0.665 | 0.625 | 0.667 | 0.633 |
| Neuronal (n=30) | 64 (Default) | 0.702 | 0.697 | 0.675 | 0.634 | **0.677** | **0.642** |
| Neuronal (n=30) | 128 | 0.705 | 0.701 | 0.678 | 0.639 | 0.681 | 0.647 |
