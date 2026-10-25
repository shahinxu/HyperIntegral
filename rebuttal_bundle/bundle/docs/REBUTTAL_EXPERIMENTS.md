# Reviewer comments → experiments mapping

Status legend: **Done** = code written and run; **Running** = sweep in flight;
**Code ready** = switch implemented, sweep not yet launched.

All runs use the identical data pipeline, candidate hyperedge library and
evaluation (per-order AUC + macro-AUC), so every comparison is attributable
to the one thing that changes.

---

## 1. Comment-by-comment plan

| # | Reviewer comment (abridged) | What is required | How we answer it | Code | Status |
|---|---|---|---|---|---|
| R?-4 | Given `f` and `f^(k)` are known, learning `B` reduces to linear regression; why a PINN at all? What is gained over integral SINDy? | A direct, equal-footing comparison against (integral) SINDy and against plain least squares | New baseline that builds the *same* integral design matrix `(∫Φdt)A = Δx − ∫f dt` and solves it with STLSQ (integral SINDy), with classical collocation SINDy, and with dense ridge least squares. Identical library, identical `f`, identical scoring. | `models/sindy/train.py` | Running (`suite=main`) |
| R?-5 | How does the tensor parametrization compare with methods that explicitly exploit sparsity? Important for high-order interactions. | Comparison against an explicitly sparsity-promoting estimator; justification for not encouraging sparsity | STLSQ is exactly such an estimator and is reported per order (2/3/4/5-edges), so the high-order regime is visible. Additionally an optional L1 penalty on the interaction tensor (`--l1`) lets us report our method with and without explicit sparsity. | `models/sindy/train.py` (STLSQ), `--l1` in `models/integral/train.py` | Running / Code ready |
| R?-5 (2nd set) | Add experiments where `L_physics` is swapped with the derivative-based cost term; vary sigma and h to support Theorems 1–2. | (a) derivative-cost ablation; (b) a σ × h grid | (a) `--loss_mode derivative` keeps the architecture, the parametrization and the optimizer fixed and only replaces the integral residual with a finite-difference collocation residual. (b) `suite=sigma_h` sweeps observation noise σ ∈ {0, .01, .05, .1} × integration step h (controlled via `--n_samples` ∈ {100, 200, 300, 600}). | `--loss_mode`, `experiments/sweep.py:suite_sigma_h` | Done (ablation) / Code ready (σ×h) |
| R?-6 (2nd set) | Reproducibility: repeat each configuration several times; AUROC 0.6–0.8 is modest; baselines may be under-tuned — report their tuning protocol. | Multi-seed statistics; an explicit, equal-budget tuning protocol for baselines | `--seed` now seeds the ODE initial condition, the noise realization, the parameter init and the batch sampling, so seeds are genuinely independent replicates. `suite=tuning` gives the baseline a 4×4 grid (sparsity threshold × integral window) and our method an equally sized 4×4 grid (learning rate × number of intervals), reported in full. | `--seed`, `experiments/sweep.py:suite_tuning`, `experiments/aggregate.py` | Running / Code ready |
| R?-6 (3rd set) | Report standard deviations over multiple random seeds, statistical significance or confidence intervals, and sensitivity to noise level. | mean ± std, 95% CI, significance tests, noise sweep | The aggregator reports mean ± std, a Student-t 95% CI, and paired t-test + Wilcoxon signed-rank tests computed per seed against a reference variant. The noise axis is a first-class sweep dimension (`--noises`). | `experiments/aggregate.py` | Running |
| R?-7 (3rd set) | Ablations: derivative vs integral PINN with the same architecture; with/without staged λ scheduling; with/without bounded tensor parametrization; different integral window lengths; different numbers of sampled intervals. | Five ablation axes | One flag each, all one-switch-at-a-time against the full method: `--loss_mode {integral,derivative}`, `--lambda_schedule {none,staged}`, `--bounded {1,0}`, `--window_frac`, `--n_intervals`. | `models/integral/train.py`, `experiments/sweep.py:suite_ablation` | Running |

---

## 2. What changed in the code

### `models/integral/train.py`
| Flag | Default | Answers | Meaning |
|---|---|---|---|
| `--seed` | 42 | R?-6, R?-6 | Seeds the ODE initial condition (via `HG_DATA_SEED`), the noise draw, the tensor init and the window sampling. Previously the trajectory seed was hard-coded to 42, so repeated runs were not independent replicates. |
| `--loss_mode` | `integral` | R?-5, R?-7 | `derivative` replaces the integral residual with `dx/dt − f − ΦA` using central differences, keeping everything else fixed. |
| `--bounded` | 1 | R?-7 | `0` removes the box constraint on the interaction tensor. |
| `--lambda_schedule` | `none` | R?-7 | `staged` fades in the interaction orders one stage at a time (2-edges first, then 3, 4, 5), each ramping linearly within its stage. |
| `--window_frac` | 1.0 | R?-7 | Caps the integral window length as a fraction of the observation horizon. |
| `--n_intervals` | 32 | R?-7 | Number of integral windows (or collocation points) sampled per step. |
| `--l1` | 0.0 | R?-5 | Optional L1 sparsity penalty on the interaction tensor. |
| `--run_tag` | timestamp | — | Deterministic output directory, needed for parallel sweeps. |

The integral residual is now evaluated with cumulative integrals, so a whole
batch of windows costs one pass instead of a Python loop over 32 windows.
`summary.json` additionally records `macro_auc` and the full configuration.

### New files
- `models/sindy/train.py` — integral SINDy (STLSQ on the weak-form system), classical SINDy, and dense ridge least squares.
- `experiments/sweep.py` — suites `main` / `ablation` / `sigma_h` / `tuning`, multi-GPU scheduling, collects every run into `results.jsonl`.
- `experiments/aggregate.py` — mean ± std, 95% CI, paired t-test and Wilcoxon, Markdown tables + tidy CSV.

### Scene libraries
`lib_rossler_oscillator/hypergraph.py` and `lib_neuronal_synchronization/hypergraph.py`
read `HG_DATA_SEED` for the initial condition / parameter draws (default 42, so
existing behaviour is unchanged).

---

## 3. Reproducing the tables

```bash
# Ablations (R?-7): one switch at a time, 3 seeds, 2 noise levels
python -m experiments.sweep --suite ablation --seeds 1 2 3 \
    --scenes rossler:8:5 --noises 0.0 0.05 --n_epochs 5000 --lr 0.01 \
    --gpus 3 5 --jobs_per_gpu 3 --out results/rebuttal/ablation

# Ours vs SINDy baselines (R?-4, R?-5), noise sensitivity (R?-6)
python -m experiments.sweep --suite main --seeds 1 2 3 \
    --scenes rossler:8:5 --noises 0.0 0.01 0.05 0.1 --n_epochs 5000 --lr 0.01 \
    --gpus 4 6 --jobs_per_gpu 3 --out results/rebuttal/main

# sigma x h grid for Theorems 1-2 (R?-5)
python -m experiments.sweep --suite sigma_h --seeds 1 2 3 \
    --scenes rossler:8:5 --n_epochs 5000 --lr 0.01 \
    --gpus 3 5 --jobs_per_gpu 3 --out results/rebuttal/sigma_h

# Equal-budget baseline tuning protocol (R?-6)
python -m experiments.sweep --suite tuning --seeds 1 \
    --scenes rossler:8:5 --noises 0.0 0.05 --n_epochs 5000 \
    --gpus 4 6 --jobs_per_gpu 3 --out results/rebuttal/tuning

# Tables
python -m experiments.aggregate results/rebuttal/ablation/results.jsonl \
    --reference full --title "Ablation study" \
    --out_md results/rebuttal/ablation.md --out_csv results/rebuttal/ablation.csv
python -m experiments.aggregate results/rebuttal/main/results.jsonl \
    --reference ours --title "Comparison against sparse-regression baselines" \
    --out_md results/rebuttal/main.md --out_csv results/rebuttal/main.csv
```

---

## 4. Results

Setting for every table: Rossler oscillators, N = 8, max order 5, 300 simulated
samples resampled onto a 500-point grid, 5000 epochs, lr = 0.01. The reported
quantity is macro-AUC, the mean of the per-order AUCs for 2/3/4/5-edges.
`--seed` controls the ODE initial condition, the noise realization, the tensor
initialization and the window sampling, so seeds are independent replicates.
Significance is a paired test over seeds (each seed contributes one matched
pair), with the Wilcoxon signed-rank test as the distribution-free check.

**All 572 runs across the seven sweeps completed without error (0 failures).**

### 4.1 Against tuned sparse-regression baselines (R?-4, R?-5) + noise sensitivity (R?-6)

Baselines use the same candidate library, the same known `f` and the same
scoring, so the gap is attributable to the estimator alone.
`sindy_integral_stlsq` is integral SINDy, `sindy_integral_lstsq` is the plain
linear-regression solution the reviewer describes, `sindy_derivative_stlsq` is
classical collocation SINDy. Baseline hyperparameters are the per-noise winners
of the equal-budget grid in Section 4.3, selected on seed 1; the table below is
evaluated on eight **held-out** seeds (11-18).

## scene=rossler, n_nodes=8, noise=0.0

| Variant | n seeds | Macro-AUC (mean ± std) | 95% CI | Δ vs ref | paired t p | Wilcoxon p | sig |
|---|---|---|---|---|---|---|---|
| ours **(ref)** | 8 | 0.7785 ± 0.0584 | [0.7297, 0.8273] | — | — | — | — |
| sindy_derivative_stlsq | 8 | 0.6804 ± 0.0776 | [0.6155, 0.7453] | -0.0981 | 0.0228 | 0.0156 | * |
| sindy_integral_stlsq | 8 | 0.5296 ± 0.0863 | [0.4575, 0.6018] | -0.2489 | 0.0010 | 0.0078 | *** |
| sindy_integral_lstsq | 8 | 0.5296 ± 0.0863 | [0.4575, 0.6018] | -0.2489 | 0.0010 | 0.0078 | *** |

## scene=rossler, n_nodes=8, noise=0.01

| Variant | n seeds | Macro-AUC (mean ± std) | 95% CI | Δ vs ref | paired t p | Wilcoxon p | sig |
|---|---|---|---|---|---|---|---|
| ours **(ref)** | 8 | 0.7664 ± 0.0576 | [0.7183, 0.8146] | — | — | — | — |
| sindy_integral_stlsq | 8 | 0.5606 ± 0.0936 | [0.4824, 0.6389] | -0.2058 | 0.0015 | 0.0078 | ** |
| sindy_integral_lstsq | 8 | 0.5389 ± 0.0676 | [0.4824, 0.5954] | -0.2275 | 0.0001 | 0.0078 | *** |
| sindy_derivative_stlsq | 8 | 0.5124 ± 0.0700 | [0.4538, 0.5709] | -0.2541 | 0.0002 | 0.0078 | *** |

## scene=rossler, n_nodes=8, noise=0.05

| Variant | n seeds | Macro-AUC (mean ± std) | 95% CI | Δ vs ref | paired t p | Wilcoxon p | sig |
|---|---|---|---|---|---|---|---|
| ours **(ref)** | 8 | 0.7022 ± 0.0608 | [0.6513, 0.7531] | — | — | — | — |
| sindy_derivative_stlsq | 8 | 0.4726 ± 0.0895 | [0.3977, 0.5474] | -0.2296 | 0.0024 | 0.0078 | ** |
| sindy_integral_lstsq | 8 | 0.4613 ± 0.0881 | [0.3876, 0.5349] | -0.2409 | 0.0019 | 0.0078 | ** |
| sindy_integral_stlsq | 8 | 0.4233 ± 0.0473 | [0.3838, 0.4629] | -0.2789 | 0.0000 | 0.0078 | *** |

## scene=rossler, n_nodes=8, noise=0.1

| Variant | n seeds | Macro-AUC (mean ± std) | 95% CI | Δ vs ref | paired t p | Wilcoxon p | sig |
|---|---|---|---|---|---|---|---|
| ours **(ref)** | 8 | 0.6249 ± 0.0573 | [0.5770, 0.6728] | — | — | — | — |
| sindy_integral_stlsq | 8 | 0.5155 ± 0.0810 | [0.4478, 0.5833] | -0.1094 | 0.0131 | 0.0156 | * |
| sindy_derivative_stlsq | 8 | 0.4893 ± 0.0853 | [0.4179, 0.5606] | -0.1356 | 0.0050 | 0.0391 | ** |
| sindy_integral_lstsq | 8 | 0.4872 ± 0.0879 | [0.4137, 0.5607] | -0.1377 | 0.0056 | 0.0156 | ** |


**Reading.** The proposed method is ahead at every noise level, and with eight
seeds every gap is significant. The margin is largest in the low-noise regime
(sigma = 0 to 0.05: +0.10 to +0.28 macro-AUC) and narrows at sigma = 0.1
(+0.11 to +0.14) where every method degrades. Note that the tuned baselines are
*weaker* than the untuned ones at sigma = 0 (0.530 vs 0.610 earlier): tuning on
seed 1 and evaluating on held-out seeds removes a selection advantage the
earlier table had, which is the more honest protocol.

This is the substantive answer to "what is gained beyond standard
regression-based methods". It is **not** that the problem is non-linear: with
`f` known the system genuinely is linear in `B`, exactly as the reviewer says.
It is that the constrained, iteratively-optimized estimator conditions the
problem far better than a one-shot (thresholded) least-squares solve on a
design matrix whose high-order columns are strongly collinear.

### 4.2 Ablations (R?-7)

## scene=rossler, n_nodes=8, noise=0.0

| Variant | n seeds | Macro-AUC (mean ± std) | 95% CI | Δ vs ref | paired t p | Wilcoxon p | sig |
|---|---|---|---|---|---|---|---|
| staged_lambda | 8 | 0.8626 ± 0.0297 | [0.8378, 0.8874] | 0.0841 | 0.0028 | 0.0156 | ** |
| derivative_loss | 8 | 0.8317 ± 0.0528 | [0.7876, 0.8759] | 0.0533 | 0.0201 | 0.0234 | * |
| window_0.1 | 8 | 0.8035 ± 0.0664 | [0.7480, 0.8590] | 0.0251 | 0.0285 | 0.0547 | * |
| window_0.25 | 8 | 0.7897 ± 0.0738 | [0.7280, 0.8514] | 0.0112 | 0.1869 | 0.2500 | n.s. |
| window_0.5 | 8 | 0.7832 ± 0.0666 | [0.7274, 0.8389] | 0.0047 | 0.3785 | 0.3828 | n.s. |
| full **(ref)** | 8 | 0.7785 ± 0.0584 | [0.7297, 0.8273] | — | — | — | — |
| intervals_64 | 8 | 0.7781 ± 0.0614 | [0.7268, 0.8295] | -0.0003 | 0.8421 | 0.9453 | n.s. |
| intervals_128 | 8 | 0.7780 ± 0.0629 | [0.7254, 0.8305] | -0.0005 | 0.8279 | 0.9453 | n.s. |
| intervals_8 | 8 | 0.7682 ± 0.0592 | [0.7186, 0.8177] | -0.0103 | 0.0812 | 0.1484 | n.s. |
| no_bounded | 8 | 0.5726 ± 0.0997 | [0.4893, 0.6559] | -0.2059 | 0.0016 | 0.0078 | ** |

## scene=rossler, n_nodes=8, noise=0.05

| Variant | n seeds | Macro-AUC (mean ± std) | 95% CI | Δ vs ref | paired t p | Wilcoxon p | sig |
|---|---|---|---|---|---|---|---|
| window_0.1 | 8 | 0.7728 ± 0.0608 | [0.7220, 0.8237] | 0.0548 | 0.2045 | 0.2500 | n.s. |
| window_0.25 | 8 | 0.7452 ± 0.0671 | [0.6891, 0.8014] | 0.0272 | 0.3973 | 0.5469 | n.s. |
| window_0.5 | 8 | 0.7421 ± 0.0413 | [0.7075, 0.7766] | 0.0241 | 0.5403 | 0.6406 | n.s. |
| intervals_64 | 8 | 0.7416 ± 0.0774 | [0.6768, 0.8063] | 0.0236 | 0.4588 | 0.5469 | n.s. |
| intervals_8 | 8 | 0.7213 ± 0.0543 | [0.6759, 0.7667] | 0.0033 | 0.9022 | 0.8438 | n.s. |
| full **(ref)** | 8 | 0.7180 ± 0.1022 | [0.6326, 0.8034] | — | — | — | — |
| intervals_128 | 8 | 0.7153 ± 0.0209 | [0.6979, 0.7328] | -0.0027 | 0.9443 | 0.9453 | n.s. |
| derivative_loss | 8 | 0.6861 ± 0.0430 | [0.6501, 0.7220] | -0.0320 | 0.4342 | 0.4609 | n.s. |
| staged_lambda | 8 | 0.6363 ± 0.0911 | [0.5601, 0.7124] | -0.0817 | 0.0505 | 0.0391 | n.s. |
| no_bounded | 8 | 0.5466 ± 0.0897 | [0.4717, 0.6216] | -0.1714 | 0.0061 | 0.0156 | ** |


**Reading — and one result that does not support the paper's current claim.**

- *Bounded tensor parametrization* is the component that carries the method.
  Removing it costs 0.206 macro-AUC on clean data (p = 0.0016) and 0.171 at
  sigma = 0.05 (p = 0.0061). This is the only axis that is significant and in
  the expected direction at both noise levels.
- *Integral vs derivative residual* **does not reproduce the expected
  ordering.** At eight seeds the derivative-cost variant is *better* on clean
  data (+0.053, p = 0.020) and statistically indistinguishable at sigma = 0.05
  (-0.032, p = 0.43). The three-seed pilot suggested the opposite; the
  three-seed result did not survive more seeds. Taken together with 4.1, the
  advantage over SINDy comes from the bounded parametrization and the iterative
  optimizer, **not** from the integral form of the residual. The paper's
  motivation for the integral formulation is not supported by this experiment
  and needs either a setting where it does hold (coarser sampling, larger
  noise, irregular grids) or a weaker claim.
- *Staged lambda scheduling* is conditional: +0.084 on clean data (p = 0.0028)
  but -0.082 at sigma = 0.05 (Wilcoxon p = 0.039). It should not be presented
  as an unconditional improvement.
- *Integral window length* and *number of sampled intervals* have small effects
  (|delta| <= 0.055, none significant except window_0.1 on clean data), i.e.
  the method is insensitive to these two knobs over the ranges tested. Shorter
  windows are never worse, which is itself mild evidence against the long-window
  integral formulation.

### 4.3 sigma x h grid (R?-5, Theorems 1-2)

Macro-AUC, mean ± std over 3 seeds. The integration step h is inversely
proportional to `n_samples`: the trajectory is simulated on `n_samples`
points over a fixed horizon, so a larger T means a finer quadrature grid.

| sigma \\ n_samples (finer h →) | 100 | 200 | 300 | 600 |
|---|---|---|---|---|
| **0.0** | 0.8040 ± 0.0436 | 0.8025 ± 0.0412 | 0.8031 ± 0.0421 | 0.8032 ± 0.0435 |
| **0.01** | 0.8024 ± 0.1000 | 0.8254 ± 0.0782 | 0.7707 ± 0.0868 | 0.7767 ± 0.0668 |
| **0.05** | 0.7132 ± 0.0605 | 0.6945 ± 0.0719 | 0.7313 ± 0.0485 | 0.7750 ± 0.0259 |
| **0.1** | 0.6230 ± 0.0909 | 0.5961 ± 0.0578 | 0.7043 ± 0.0574 | 0.7330 ± 0.0674 |

## Marginal trends

| axis | value | macro-AUC (mean ± std over the other axis and seeds) |
|---|---|---|
| sigma | 0.0 | 0.8032 ± 0.0364 |
| sigma | 0.01 | 0.7938 ± 0.0750 |
| sigma | 0.05 | 0.7285 ± 0.0559 |
| sigma | 0.1 | 0.6641 ± 0.0836 |
| n_samples | 100 | 0.7356 ± 0.1021 |
| n_samples | 200 | 0.7296 ± 0.1101 |
| n_samples | 300 | 0.7523 ± 0.0653 |
| n_samples | 600 | 0.7720 ± 0.0529 |

**Reading.** Degradation in sigma is monotone and clean (0.803 -> 0.794 ->
0.729 -> 0.664). The h axis interacts with sigma as the theory would predict:
at sigma = 0 the grid resolution is irrelevant (0.8040 to 0.8032, flat), while
at sigma = 0.1 a finer quadrature grid recovers 0.11 macro-AUC
(0.623 at T=100 -> 0.733 at T=600). This is usable support for Theorems 1-2.

### 4.4 Baseline tuning protocol (R?-6)

Both arms received a grid of exactly 16 configurations per noise level, tuned
on seed 1 and evaluated on held-out seeds. Baseline grid: STLSQ threshold in
{1e-4, 1e-3, 1e-2, 1e-1} x integral window in {5, 20, 50, 100}. Our grid:
learning rate in {1e-3, 3e-3, 1e-2, 3e-2} x number of intervals in
{8, 32, 64, 128}.

| sigma | best baseline (config) | best ours (config) |
|---|---|---|
| 0.0 | 0.6612 (thr = 1e-4, window = 50) | 0.7804 (lr = 0.03, n_intervals = 128) |
| 0.01 | 0.6152 (thr = 0.1, window = 5) | 0.8384 (lr = 0.03, n_intervals = 8) |
| 0.05 | 0.5587 (thr = 1e-4, window = 20) | 0.7347 (lr = 0.03, n_intervals = 32) |
| 0.1 | 0.5292 (thr = 1e-3, window = 50) | 0.7783 (lr = 0.003, n_intervals = 32) |

Full per-configuration numbers are in `results/rebuttal/tuning*/results.jsonl`.
The baseline's spread across its grid (0.57 to 0.66 at sigma = 0) is smaller
than ours (0.48 to 0.78), so the baseline is not merely under-tuned; it is
tuning-insensitive in this setting.

---

### 4.5 Integral vs derivative residual: resolved (R?-5, R?-7)

Section 4.2 found the integral residual no better than a derivative residual,
contradicting the paper's motivation. A dedicated 960-run study
(`suite=diagnostic`, full analysis in `results/rebuttal/crossover.md`) resolves
this: **the comparison is governed by the noise level, with a crossover at
σ ≈ 0.05.**

| σ | mean Δ (integral − derivative) | cells with Δ>0 | p | verdict |
|---|---|---|---|---|
| 0.01 | −0.0536 | 0/8 | 0.0013 | derivative clearly better |
| 0.05 | −0.0037 | 4/8 | 0.64 | tie (crossover) |
| 0.1 | +0.0436 | 7/8 | 0.022 | integral better |
| 0.15 | +0.0263 | 7/8 | 0.017 | integral better |
| 0.2 | +0.0205 | 5/8 | 0.12 | integral mildly better, n.s. |
| 0.3 | +0.0273 | 6/8 | 0.13 | integral mildly better |

Each cell is one (control × grid) configuration averaged over 8 seeds; the
controls vary whether `Phi`/`f` are built from the noise-free trajectory
(`oracle`) and whether the dense resampling is skipped (`coarse`). The sign
flip reproduces under every control, so it is not a pipeline artifact.

This is a bias–variance crossover: integration averages observation noise down
but introduces quadrature bias, so below the threshold the integral form is
actively harmful and above it the variance reduction pays. The benefit does not
keep growing — past σ = 0.1 it plateaus near +0.02 to +0.03 as both estimators
decay toward chance.

A secondary finding: errors-in-variables in `Phi` is a large but separate
effect. The `oracle` control lifts *both* estimators at high noise (σ = 0.1,
T = 300: 0.681/0.602 → 0.719/0.703) and *shrinks* the integral advantage
(+0.078 → +0.016), so part of what integration buys is averaging noise in the
regressor, not only in the matched increment.

**Consequence.** The unconditional claim should be replaced by two supported
claims: (i) constrained estimation beats one-shot sparse regression at every
noise level (Section 4.1, the main result, independent of the integral form);
(ii) the integral residual buys robustness above a characterized noise
threshold, peak +0.044 at σ = 0.1. Stating the threshold is stronger than
asserting the claim unconditionally, and it is what Theorems 1–2 should predict.

---

### 4.6 Scene coverage, and a hard identifiability ceiling (R?-6)

With negative candidates subsampled per order (`--max_candidates_per_order`,
needed because the full support is ~6e6 hyperedges at N=60), the three
remaining scenes now run. Full tables in `results/rebuttal/scenes.md`.

| scene | N | σ | ours | best baseline |
|---|---|---|---|---|
| social | 8 | 0.0 | 0.4935 ± 0.045 | 0.5192 ± 0.077 |
| social | 8 | 0.05 | **0.5852 ± 0.058** | 0.5023 ± 0.007 |
| neuronal | 60 | 0.0 | **0.5326 ± 0.016** | 0.5024 ± 0.012 |
| neuronal | 60 | 0.05 | **0.5375 ± 0.015** | 0.5039 ± 0.008 |
| ecological | 60 | 0.0 | 0.4414 ± 0.009 | 0.5056 ± 0.012 |
| ecological | 60 | 0.05 | 0.4649 ± 0.019 | 0.5351 ± 0.013 |

**Everything is at or near chance.** The Rossler N=8 result (0.78) does not
transfer. This is not under-training: on neuronal, raising the epoch budget 6x
(5000 -> 30000) and the learning rate 10x (0.01 -> 0.1) moves macro-AUC by
0.0003 (0.5387 -> 0.5384). The optimizer has converged; there is nothing to
find.

The reason is a property of the benchmark, quantified in
`results/rebuttal/identifiability.md`. The dynamics scale each order's coupling
by `1/n^(k-1)`, so at N=60 the measured contributions to `dθ/dt` are:

| order | contribution RMS | fraction of total signal | vs σ=0.05 noise floor | observed AUC |
|---|---|---|---|---|
| 2-edge | ~2.5e-2 coupling | — | above | 0.609 |
| 3-edge | 1.66e-03 | 1.0e-03 | 0.02x | 0.505 |
| 4-edge | 1.11e-05 | 6.6e-06 | 1.3e-04x | 0.526 |
| 5-edge | 6.68e-08 | 4.0e-08 | 8.1e-07x | 0.514 |

The 4- and 5-edge terms are 1e4 to 1e6 times *below* the noise floor and near
float32 resolution. **No method can identify them**, and the measured AUCs
confirm it: ours and every baseline sit at 0.50 for orders 3-5, with only the
pairwise term carrying recoverable signal. This also explains the decay already
recorded in `Mcro-AUC.txt` (ITC: 0.905 at n=8 down to 0.622 at n=100) — that
curve measures the benchmark's normalization, not any method.

**This is the answer to "AUROC 0.6-0.8 is modest."** The ceiling is set by how
much of the dynamics the higher-order terms actually drive. Reporting per-order
signal-to-noise next to per-order AUC makes the ceiling explicit and converts
the complaint into a characterization. The ecological scene scoring *below*
chance (0.44, consistently) is a separate open issue — a consistent sub-chance
AUC indicates inverted signal, not absence of signal, and needs a sign check.

### 4.7 Explicit sparsity (R?-5, second half)

Adding an L1 penalty to our objective changes nothing material on Rossler
(5 seeds, σ=0.05): L1=1e-2 gives +0.019, L1=1e-4 gives +0.011, L1=1e-3 gives
-0.038, none significant, against 0.712 for our method unchanged. Explicitly
sparsity-promoting STLSQ on the same system reaches 0.516 (-0.196, p=0.020).
So the bounded parametrization already supplies whatever regularization this
problem needs; explicit sparsity neither helps nor is required.

---

## 5. Still open

- **Ecological scores consistently below chance** (0.441/0.465, significantly
  worse than the baselines' 0.49-0.54). A stable sub-0.5 AUC means the score is
  anti-correlated with the truth, which points at a sign or index-convention
  mismatch in that scene's `dynamic_phi` rather than at an estimation failure.
  This should be checked before anything about the ecological scene is reported.
- Large-N claims cannot be made on the current benchmark (Section 4.6). Either
  rescale the coupling constants so each order stays O(1) in N and disclose the
  change, or restrict reported results to the regime where the higher-order
  terms are above the noise floor.
- Seed counts: 8 for the Rossler tables and the crossover study, 5 for the
  scene-coverage and sparsity tables. Raise the latter before submission.
- The σ x h grid (Section 4.3) was run only for the integral loss. Re-running it
  with `--loss_mode derivative` would let Theorems 1-2 be checked against both
  residual operators directly.
