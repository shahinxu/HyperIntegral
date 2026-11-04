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
