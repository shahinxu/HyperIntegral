# Defects found and fixed while producing the rebuttal experiments

Every result in `REBUTTAL_RESPONSE.md` was produced after all of these were
fixed. Items 1-6 changed measured numbers; 7-11 made runs possible or
comparable in the first place.

## Affecting correctness of results

**1. Surrogate time tensor had the wrong rank — `models/integral/train_nn.py`**
`TimeResNet.forward` already does `t.unsqueeze(-1)`, but callers passed `[T,1]`,
so the prediction came back as `[T,1,out]`. Subtracting the `[T,out]`
observations broadcast into an all-pairs `[T,T,out]` tensor: the data loss
compared every predicted time against every observed time (810000 elements
instead of 2700). The surrogate was therefore never fit to the trajectory, and
the integral residual was computed on a meaningless `x̂(t)`.
*Effect:* the proposed method sat at chance on every scene. Also crashed the
derivative ablation with `shape '[64,1,9]' is invalid for input of size 64`.

**2. `clamp` inside the autograd graph — `models/integral/train_nn.py`**
`A_eff = sigmoid(clamp(W, -c, c))` has zero gradient outside `[-c, c]`. With
`w_init = -3.0` and `c = 2.0`, 100% of entries started outside, so `W` received
no gradient at all and never moved (verified: 200 Adam steps, max |ΔW| = 0).
Even with a valid init, any entry that later drifted past ±c froze permanently,
making the solution depend on which entries happened to saturate first.
*Fix:* keep `sigmoid(W)` in the graph and project `W` back into `[-c, c]` after
each optimizer step. Also added a startup check that rejects `|w_init| >= c`.

**3. Binary data loss clamped an unbounded output — `models/integral/train_nn.py`**
On the social-contagion scene the Bernoulli loss used
`p = clamp(net_output, 1e-4, 1-1e-4)`. `TimeResNet` is unbounded, so once a
prediction left that interval the gradient vanished and the surrogate stopped
learning entirely (`L_data` and `L_physics` were constant for the whole run).
*Fix:* squash with a sigmoid instead of clamping.
*Effect:* social macro-AUC went from 0.481 ± 0.066 to 0.609 ± 0.069.

**4. THIS ignored the seed — `models/this/run.py`**
`social-run.jl` reads `SC_GRAPH_SEED` / `SC_TRAIN_SEED` / `SC_TEST_SEED`; the
launcher only set `SC_SEED`. All ten seeds ran an identical configuration
(std = 0.000 across seeds).

**5. Candidate pools contained no negatives — all three `lib_*/hypergraph.py`**
`generate_all_possible_hyperedges()` returned `get_hyperedge_config()`, i.e. the
ground truth. Any consumer scoring against it had `y_true` all ones, so every
AUC was undefined. The same bug appeared again inside the HyperPINN scene
scripts, where `all_*edges` were set from the ground-truth config and
`true_5edges` was built from `all_5edges`.
*After the fix the candidate sets match the paper:* 372 / 210 / 174406.

**6. Baselines scored against different candidate sets on neuronal**
The kernel baseline cannot solve the full 174406-column system (~121 GB), so
every method uses a subsampled pool (all true hyperedges + 3000 negatives per
order = 9435). HyperPINN and THIS bypassed this and enumerated all 174406,
giving them 18x more negatives and making their AUC incomparable.
*Fix:* `--max_candidates_per_order` for HyperPINN; for THIS the Python exporter
writes `candidates.json` and the Julia driver scores against it.

## Preventing runs or losing results

**7. `evaluate_edges_triangles` had no `return`** in `HyperPINN_Ecosystem.py`, so
the ecological HyperPINN run crashed before reporting any AUC.

**8. Order-5 AUC computed but never printed**, and `--max_order` was overwritten
by the scene default, so HyperPINN reported only 3 of 4 orders.

**9. Baseline launchers parsed `auc_scores.txt`**, which the inner scripts never
write; every baseline `summary.json` had an empty `auc` field.

**10. `--l1` was missing from the trainer**, so all 18 explicit-sparsity runs
failed with `unrecognized arguments`.

**11. THIS serialized on a shared export directory.** All THIS jobs wrote to
`models/this/<scene>-data`, so they could not run concurrently. Each run now
gets its own directory and they execute in parallel.

## Found in the reporting pipeline

**12. The tuning grid never reached the kernel and THIS baselines — `experiments/paper_sweep.py`**
`suite_tuning` built 16 `tune_kernel` jobs and 16 `tune_this` jobs, but the grid
point only went into the record's `meta`; the command line was identical every
time. All 16 kernel runs re-ran the model's own internal sigma/gamma search and
all 16 THIS runs used the Julia default lambda/rho, so the "equal budget"
claim in Table R2.6b was not backed by anything.
*Fix:* `models/kernel/train.py` gained `--sigma_k` / `--lam` (given, they pin
the internal grid to that single point), `models/this/run.py` gained
`--lambda1` / `--rho` (exported as `{PREFIX}_LAMBDA` / `{PREFIX}_RHO`), and the
sweep appends them to the job's CLI. The stale records were deleted and rerun.
THIS is tuned on lambda x rho, which are its real knobs; the table text said
"threshold" and was corrected.

**13. Killed runs were never retried as far as the tables were concerned — `experiments/paper_sweep.py`**
`record()` took `macro_auc` only from `summary.json`'s `extra_metrics`, which the
kernel, THIS and SINDy baselines do not write. Their records were stored with
`macro_auc: null`, which `fill_tables.py` skips, so finished runs looked
missing and had to be rescued by hand with `repair_results.py` every time.
*Fix:* `record()` now falls back to the mean of the per-order AUCs, the same
definition `repair_results.py` uses.

**14. The integral SINDy system ran the GPU out of memory at n=30 — `models/sindy/train.py`**
`build_integral_system` formed `cumsum(Phi[:-1] * dt)` over the whole
`[T, N, D, E]` tensor. At T=500, E=9435 each temporary is ~570 MB and three of
them are live at once, which OOMed whenever the GPU was shared.
*Fix:* take the cumulative integral a slice of candidates at a time and write
each slice straight into the output array. Verified bit-identical to the
previous expression (max |diff| = 0 over several window lengths and chunk
sizes).

**15. The held-out column included the selection seed — `experiments/fill_tables.py`**
Table R2.6c's "Macro-AUC (held-out seeds)" averaged all ten main-suite seeds,
including seed 11, which is the seed the configuration was selected on. The
held-out set is seeds 12-20 (nine seeds); the table said so while reporting
something else.
*Fix:* `SELECTION_SEED = 11` is excluded from that column, and the "Evaluation
seeds" column of R2.6b is now derived from the seeds actually present.

**16. "not tuned" leaked into tables that have nothing to do with tuning — `experiments/fill_tables.py`**
`annotate_unavailable` matched the method name anywhere in the document, so the
empty neuronal Kernel-based row of the *noise* table R3.6b was labelled "not
tuned" rather than left blank.
*Fix:* the annotation is confined to the span of table R2.6c.

**17. Table R2.6c reported a result for THIS on the neuronal scene**
The row carried `lambda1=1e-2, thresh=1e-3 | 0.525 | 0.511 +- 0.017` while the
other three tables said "not run (THIS does not terminate at n=30)". There are
three THIS neuronal records and all three exited non-zero; no such measurement
exists. The row now matches the others.

**18. `fill_tables.py` degrades silently without scipy**
`ci95` falls back from `t.ppf(0.975, n-1)` to 1.96 and the paired tests return
nothing. Run under an interpreter that has scipy, or every confidence interval
in the document silently narrows (for n=10, 2.262 -> 1.96).

**19. The noise sweep queued THIS on the neuronal scene — `experiments/paper_sweep.py`**
`suite_main` already skipped it (THIS never terminates at n=30), but
`suite_noise` did not, so 18 jobs that can never finish each held a worker slot
until the three-hour timeout. On a two-GPU run that is the entire sweep's
capacity, and the neuronal noise rows stayed empty for hours with nothing
apparently wrong.
*Fix:* the same guard in `suite_noise`, plus a `--skip_variants` flag so the
hand-maintained `ours` / `tune_ours` jobs need not be rerun either.

**20. `experiments/coverage.py` mis-scored the tuning-protocol table**
Coverage counts a row as filled when any cell past the second matches
`\d\.\d`. In table R2.6b the result column is an integer count sitting beside
prose columns whose grid text contains decimals, so rows with no result were
counted as filled and rows with one were not. The table read 1/5 while it was
in fact complete.
*Fix:* an explicit rule for that table — column 3 must be a positive integer.

**21. The HyperPINN epoch grid collapsed to a single point on neuronal — `experiments/paper_sweep.py`**
`hyperpinn()` clamps the neuronal budget with `min(epochs, bl_epochs_neuronal)`
because the full budget cannot finish inside the per-job timeout at n=30. In
`suite_tuning` that clamp flattened the entire epoch axis: all four budgets
became 3000, so the 16-point grid was really 4 learning rates run four times
each, at roughly nine GPU-hours for four distinct measurements. Same class of
defect as item 12 — the grid existed only in the record's `meta`.
*Fix:* scale the grid into the feasible budget instead of clamping it, giving
{750, 1500, 2250, 3000} on neuronal, and record the budget that actually ran.
The Grid cell of table R2.6b states the scaled values.

**22. Table R2.6c scored the default configuration, not the one it names**
The table is titled "Best configuration found per method" and its last column
is "Macro-AUC (held-out seeds)", but that column was read off the *main*
suite, which runs each baseline's default hyperparameters. For Integral SINDy
the table named `threshold=1e-4` beside a number measured at `threshold=1e-2`;
the kernel baseline's main runs use its own internal sigma/gamma search rather
than the selected grid point, and THIS's use the Julia defaults
(lambda=0.05, rho=0.001) rather than the selected pair. The column therefore
reported untuned performance under a tuned configuration's label.
*Fix:* a new `holdout` suite reruns each baseline's selected configuration on
the held-out seeds (12-20), and `fill_tables.py` reads that column from those
runs, falling back to the main suite only while they do not exist. The
proposed method is excluded, as its rows are maintained by hand.

**23. The main tables mixed in runs from other noise levels — `experiments/fill_tables.py`**
`load()` indexes每 (scene, variant) by seed alone and applied no noise filter.
The sweeps reuse seeds 11-13 at all six noise levels, so a sigma=0.2 run
silently overwrote the sigma=0.02 run of the same seed. 78 of 380
(scene, variant, seed) combinations were affected, each spanning five noise
levels. Every table filled through `load()` was wrong: R1.4, R1.5a, R2.6a,
R3.6a, R3.7a and the held-out column of R2.6c.
It showed up as inflated spread. Ecological HyperPINN read 0.625 +- 0.069 in
R2.6a while the main suite's ten seeds give 0.661 +- 0.022, and the same
method read 0.653 +- 0.021 in R3.6b, which was right because `fill_noise`
groups by noise. The three numbers disagreeing with each other is what made
the defect visible.
*Fix:* `load(..., noise=MAIN_NOISE)` keeps only runs at the main noise level.

**24. Concurrent jobs exhausted the sandbox's pid limit — `experiments/paper_sweep.py`**
Workers inherited a 256-core machine with no thread-pool limits, so OpenMP and
the BLAS libraries each sized their pools to the core count: a single job was
measured at 127 threads. The sandbox cgroup allows 8192 pids in total, and
nineteen concurrent jobs put the machine at 7993 - 97.5% of the limit. Jobs
then died about 70 seconds in, before their first epoch, with

    libgomp: Thread creation failed: Resource temporarily unavailable

and the sweep recorded them as ordinary failures, so the neuronal noise rows
kept not filling with nothing in the log to explain it.

The oversubscription was also the reason every surviving job was slow: a
750-epoch HyperPINN run took 3635 s where the main suite's 3000-epoch runs
took 2121 s, a 7x per-epoch slowdown.
*Fix:* `env_for()` pins `OMP_NUM_THREADS` and the MKL / OpenBLAS / NumExpr /
veclib / Julia equivalents to `SWEEP_THREADS_PER_JOB` (4 by default).
Measured after the change: 7 threads per job instead of 127, 3377 total
instead of 7993, at the same 19-way concurrency.

**25. Tables mixed runs made under different training budgets — `experiments/fill_tables.py`**
`load()` was handed every stage directory at once and indexes by seed, so the
stage read last won each seat. The suites do not share a budget: the sparsity
suite trains for 20000 epochs where the main suite trains for 8000. For
ecological `ours` at seed 11 the three stages hold

    main      0.5104   n_epochs=8000   stage1=1500  ramp=1000
    noise     0.4960   n_epochs=8000   stage1=1500  ramp=1000
    sparsity  0.4658   n_epochs=20000  stage1=2500  ramp=2000

and `sparsity` sorts last, so seeds 11-13 of R1.4, R2.6a and R3.6a came from
20000-epoch runs while seeds 14-20 came from 8000-epoch runs. 39 of the 481
(scene, variant, seed) combinations at the main noise level were affected.
*Fix:* each table is read from the suite that produced it - `main` and
`main_neuronal` for those three tables, plus `holdout` for the tuning table's
held-out column.

**26. Recovered records lost their seed, collapsing four runs into one — `experiments/repair_results.py`**
Records rebuilt from a bare `summary.json` carried `meta.seed = None`. Since
`load()` keys by seed, every such record for a given (scene, variant) landed
in the same slot and all but one were discarded. Four ecological kernel runs
shared one seat, so that row reported n=7 while ten runs existed, and the
sweep would not rerun them either, because `load_done` matches on job name
and the job names were present.
*Fix:* `repair_results` takes the seed and the noise level from the run tag
when the metadata lacks them. All 25 rows that carry an n column now read 10.

## THIS on the neuronal scene: it was never a hang

For most of this work THIS was recorded as `not run (THIS does not terminate
at n=30)`. A probe had run 40 minutes with no output and every attempt exited
non-zero, and the four affected cells were treated as unfillable.

That was wrong. The Julia output is block-buffered, so an empty log says
nothing about progress, and the run was not hanging - it was working through
three separate quadratic accumulation patterns and one Gram matrix that does
not fit in any machine. All four are fixed and a neuronal run now completes.

**27. `inv(θθ' + ρI)` formed an m×m Gram matrix — `models/this/this.jl`**
`mySINDy` solved the ridge system as `Y*θ'*inv(θ*θ' + ρI_m)`. m is the number
of monomials of degree at most dmax, C(n+dmax, dmax): 2002 at n=9 and 1287 at
n=8, but **324632 at n=30**, where `θ*θ'` alone is 785 GB and its inverse
about 3.4e16 flops.
*Fix:* the push-through identity `θ'*inv(θθ' + ρI_m) == inv(θ'θ + ρI_T)*θ'`
gives the same matrix from a T×T solve (T=300), independent of m. `ridge_solve`
picks whichever side is smaller. Verified against the direct form: exact where
m <= T, relative error 1e-9 to 7.6e-8 where the identity is used, which is the
conditioning of the ridge problem itself.

**28. `get_θd` grew the dictionary with `vcat` — `models/this/this-tools.jl`**
Each append recopied the whole accumulated block, so building θ was O(m²) in
copying rather than O(m). At n=30 this alone ran over three and a half hours
without finishing.
*Fix:* size each level from its children and write it once. Verified
bit-identical (`==`, not approximate) against the original for both θ and d at
(n,dmax) = (4,3), (5,4), (6,5) and (9,5).

**29. The adjacency reconstruction grew `Ainf` with `vcat` — `models/this/this.jl`**
Same pattern in the loop that rebuilds the tensors. Without hard thresholding
`coeff` is dense, so the loop runs over every squarefree monomial and `Ainf`
passes five million rows; the recopying took more than seven hours.
*Fix:* collect the blocks and `reduce(vcat, ...)` once per order.

Regression check on the ecological scene, which was already recorded: the new
code reproduces the stored run exactly.

    recorded   2-edges 0.5864  3-edges 0.4468  4-edges 0.4681  5-edges 0.5827
    new code   2-edges 0.5864  3-edges 0.4468  4-edges 0.4681  5-edges 0.5827

A neuronal run also completes at n=60 (m = C(65,5) = 8.26 million, θ is
19.8 GB), which the original could not approach.

## THIS on the neuronal scene: three quadratic accumulations, not a hang

THIS was recorded throughout this work as not terminating at n=30, and four
cells were annotated `not run` on that basis. That was wrong. The Julia
implementation grows a matrix with `vcat` inside a loop in three places, so
each append recopies everything accumulated so far - O(rows²) instead of
O(rows). At n=9 the dictionary has 2002 monomials and none of it shows; at
n=30 it has C(35,5) = 324632 and the three costs compound. Julia block-buffers
its output, so the log stayed empty and the run looked hung.

**27. `mySINDy` inverted the dictionary's Gram matrix — `models/this/this.jl`**
`Ξ = Y*θ'*inv(θ*θ' + ρ*Id(m))` forms an m×m matrix: 785 GB at n=30, and about
3.4e16 flops to invert. The push-through identity
`θ'*inv(θθ' + ρI_m) == inv(θ'θ + ρI_T)*θ'` gives the same matrix from a T×T
solve, T being 300. Added as `ridge_solve`, which picks whichever side is
smaller.

**28. `get_θd` built the dictionary by repeated `vcat` — `models/this/this-tools.jl`**
Each level appended n children one at a time, recopying the block each time.
Rewritten to size each level from its children and write once. Verified
element-for-element against the original at (n,dmax) = (4,3), (5,4), (6,5)
and (9,5): `θ` and `d` compare `==`.

**29. The adjacency reconstruction appended row blocks the same way — `models/this/this.jl`**
`Ainf[o] = vcat(Ainf[o], ...)` inside the loop over nonzero monomials.
Without hard thresholding `coeff` is dense, so at n=30 the loop runs over
174436 squarefree monomials and `Ainf` grows past five million rows. Now the
blocks are collected per order and concatenated once.

Checked end to end against the version in git at HEAD, same inputs: `coeff`
identical element-for-element, `relerr` difference 0.0, and every order of
`Ainf` identical. The ecological and social numbers are unchanged.

| | n=30 | n=60 |
|---|---|---|
| before | 7.4 h, unfinished | unfinished |
| after | **29.2 s** | **8 m 52 s** |

`suite_main`, `suite_noise` and `suite_tuning` no longer skip the scene, and
the four cells that carried `not run` now carry measured values. Coverage is
164/164.

## Open finding, not fixed: the proposed method's means are not in the records

### Where those means come from: nowhere in the repository

Searched both result trees - the current `results/paper` and
`results/rebuttal`, the superseded first round - for a group of runs that
reproduces each printed row. The test is whole-row: all four per-order means
must come from the same (tree, scene, variant, noise) group, because matching
one cell at a time proves nothing when three-decimal values collide and there
are 431 candidate groups.

| Row set | Rows reproduced |
|---|---|
| R2.5a Derivative (control, not an Ours row) | **3/3** |
| R2.5a Integral (ours) | 0/3 |
| R3.7c (every row is ours) | 0/12 |

The control reproduces exactly, and names its source
(`paper/ablation`, `derivative_loss`, sigma=0.02), so the search works. The
Ours rows match nothing, in either tree. The same values do appear in
`REBUTTAL_RESPONSE-v3.md`, `-v4.md` and `REBUTTAL_EXPERIMENTS.md`, which are
hand-written documents, not run output.

This is as far as it can be taken without editing the rows. What the tables
should say is the author's decision. Reproduce with
`python -m experiments.check_ours_rows`.

### The lost n=30 neuronal preset is not recoverable from the runs either

`results/rebuttal` holds neuronal runs only at n=60, so no first-round record
carries the n=30 hyperedge set. The preset in the tree was regenerated from
the n60 design rule and checked for within:cross = 2.00, equal per-module
counts and full node coverage, but it is not the original, and neuronal
results are therefore not comparable to the paper's Table III.

Checked cell by cell against the ablation suite at the main noise level. The
scope is wider than it first appeared - it is every ablation table, not just
two:

| Table / rows | means match | sd match |
|---|---|---|
| R2.5a, `Integral (ours)` | 0/4 per row | 4/4 per row |
| R2.5a, `Derivative` | **4/4 per row** | 4/4 per row |
| R3.7a, `Full method`, `Without staged λ`, `Without bounded` | 0/4 per row | 4/4 per row |
| R3.7a, `Without gradient clipping` | 0/4 per row | **0/4 per row** |
| R3.7a, `Derivative PINN` | **4/4 per row** | 4/4 per row |
| R3.7b, all 12 rows | 0/4 per row | 4/4 per row |
| R3.7c, all 12 rows | 0/4 per row | 4/4 per row |

The agreement on the standard deviations is exact to the last digit, so those
came from the runs; the means paired with them did not. The rows that do match
on both are exactly the derivative baselines - the arm the tables compare
against. `Without gradient clipping` matches on neither.

Side by side, ecological at sigma=0.02:

    Integral (ours)  printed   0.828  0.742  0.712  0.650
                     measured  0.469  0.609  0.333  0.376
    Derivative       printed   0.653  0.535  0.633  0.865
                     measured  0.653  0.535  0.633  0.865

This changes the direction of the comparison the tables are making. Measured
macro-AUC at sigma=0.02, three seeds:

| Scene | Integral (`full`) | Derivative (`derivative_loss`) | measured | as printed |
|---|---|---|---|---|
| ecological | 0.447 +- 0.028 | 0.672 +- 0.018 | -0.225 | +0.061 |
| social | 0.612 +- 0.084 | 0.536 +- 0.007 | +0.076 | +0.183 |
| neuronal | 0.495 +- 0.036 | 0.518 +- 0.037 | -0.023 | +0.159 |

Left exactly as it is. These are Ours rows, which are maintained by hand and
which this work is under instruction not to touch, and the discrepancy is not
a code defect: the pipeline reproduces the recorded runs correctly. Deciding
what the tables should say is the author's call, not something to patch.

One hypothesis for why the measured integral runs come out below their own
derivative ablation was checked and ruled out. The integral residual
`(Δx − ∫f dt)²` carries a factor of the window length where the derivative
residual `(dx/dt − f)²` does not, so under a shared `lambda_physics` the
integral physics term could have been systematically under-weighted. Measured
on ecological, seed 11, at equal budget, it is not:

    integral    L_phys = 1.15e-03
    derivative  L_phys = 2.35e-04

The integral term is the larger of the two, so the weighting is not the
explanation and this is not a scaling defect. Noted so the check is not
repeated.

One related observation, not acted on because it would change the method:
window lengths are drawn as `max_window * rand`, so a share of the sampled
windows are near-degenerate and carry almost no signal.

Reproduce with `python -m experiments.check_ours_rows`, or
`--side-by-side` to see both numbers for every cell. The script is read-only.

## Infrastructure added

- `experiments/paper_sweep.py` — the six suites, resumable: each finished run is
  appended to `results.jsonl` immediately and a restart skips completed jobs.
- `experiments/repair_results.py` — re-attaches `summary.json` files to records
  whose run succeeded but whose result was never linked (recovered 113 runs).
- `experiments/fill_tables.py` — writes measured values into the response tables
  without touching their structure.
- `experiments/coverage.py` — reports how much of each table is filled.
- `lib_neuronal_synchronization/presets/neuronal_synchronization_n30_hypergraph.json`
  — regenerated from the n60 design rule; the original was never under version
  control. Neuronal results are therefore not comparable to the paper's Table III.
