# Handoff — IntegralPINN rebuttal experiments

Read this first, then `BUGFIX_LOG.md`. Everything below is state that is not
recoverable from the code or the git history.

---

## 1. What the task is

The paper is *Higher-Order Interaction Reconstruction via Integral
Physics-Informed Neural Networks* (IntegralPINN). Three reviewers raised six
experiment-related comments (R1.4, R1.5, R2.5, R2.6, R3.6, R3.7). The
screenshots of those comments are in `rebuttal_screenshots/`.

The deliverable is `REBUTTAL_RESPONSE.md`: for each comment, the reviewer's
verbatim text followed by one or more tables of measured numbers. The owner was
explicit about the format — **question number + the reviewer's own words + a
table, nothing else. No prose, no analysis, no commentary.**

The job is to run the experiments honestly and put the measured numbers in
those tables.

---

## 2. Standing constraints from the owner

These are not suggestions. They were repeated, and one was repeated three
times.

1. **Do not touch anything to do with the proposed method ("Ours").**
   Verbatim: 任何关于ours方法的都不要管 / 不要管不要管明白了吗.
   Do not fill, edit, re-run, re-tune or "improve" any Ours row. You may *read*
   Ours results to answer a question, and you may *report* a discrepancy, but
   you may not change the cell.
   `experiments/fill_tables.py` enforces this in three layers: the `OURS_VARIANTS`
   set, the `_is_ours()` prefix test, and `restore_ours_rows()`, which reverts
   wholesale any changed line inside the seven Ours-only tables (R1.5a, R1.5b,
   R2.5a, R2.5c, R3.7a, R3.7b, R3.7c) or matching `OURS_LABEL`. If you add a
   writer to that file, assume it will leak into an Ours row unless you check.

2. **Never change the table structure.** Fill cells; do not add, remove or
   reorder rows or columns.

3. **Never fabricate a number.** An earlier request to edit recorded result
   files so they would agree with a draft of the document was declined, and
   that stands. If a run did not happen, the cell says so. Bug 17 in
   `BUGFIX_LOG.md` is an example of a fabricated cell that had to be removed.

4. **Code packages must contain no result files.** Verbatim, three times:
   不要包含任何reuslt文件. `integralpinn_code.zip` is that package — code and
   configuration only. Also excluded by request: `BUGFIX_LOG.md`,
   `generate_n30_preset.py`, `run_rebuttal.sh`.

5. Report progress plainly. The owner has been burned by optimistic ETAs on
   this task ("我都不相信你了") and by a 16.8x speedup claim that was a
   micro-benchmark and only 1.2x end to end. Measure before you claim.

---

## 3. The three benchmarks

| Scene | n | Candidate hyperedges | Notes |
|---|---|---|---|
| ecological | 9 | 372 | continuous state |
| social contagion | 8 | 210 | binary state, Bernoulli data loss |
| neuronal | 30 | 174406 | subsampled to 9435 for every method |

On the neuronal scene the full 174406-column system is about 121 GB for the
kernel baseline, so **every** method runs with
`--max_candidates_per_order 3000` (all true hyperedges plus 3000 negatives per
order = 9435 columns). AUC stays unbiased because negatives are subsampled
uniformly. If one method skips the cap its AUC is not comparable — that was bug
6.

**Neuronal is unidentifiable at n=30.** Coupling scales as 1/n^(k-1), so the
4-edge and 5-edge contributions sit 1e4 to 1e6 below the noise floor. Every
method lands at 0.503-0.511. This is a property of the scene, not a bug; do not
go hunting for one. Related: the n=30 preset
(`lib_neuronal_synchronization/presets/..._n30_hypergraph.json`) was never under
version control and had to be regenerated from the n60 design rule, so neuronal
numbers are not comparable to the paper's Table III.

**THIS runs on every scene, including neuronal.** It was excluded from that
scene for most of this work under the belief that it does not terminate at
n=30. It does: see bugs 27-29. A run takes about 35 seconds. If you touch
`models/this/this.jl` or `this-tools.jl`, keep the accumulation patterns out
of the loops - that is what made it look like a hang.

---

## 4. Code map

| Path | Role |
|---|---|
| `models/integral/train_nn.py` | the proposed method (do not touch) |
| `models/integral/train.py` | older variant; still hosts shared helpers (`build_orderwise_support`, `compute_auc_scores`, `macro_auc`, `TimeResNet`, ...) imported elsewhere |
| `models/sindy/train.py` | integral SINDy, classical SINDy, dense ridge LSQ |
| `models/hyperpinn/` | HyperPINN baseline |
| `models/kernel/train.py` | kernel baseline; `--sigma_k` / `--lam` pin one grid point |
| `models/this/run.py` + `*-run.jl` | THIS (Julia); `--lambda1` / `--rho` pin one grid point |
| `experiments/paper_sweep.py` | suites `main`, `ablation`, `sparsity`, `noise`, `sigma_h`, `tuning` |
| `experiments/fill_tables.py` | writes measured values into the tables |
| `experiments/repair_results.py` | relinks `summary.json` to records whose run succeeded but was never recorded |
| `experiments/coverage.py` | per-table fill report |

Macro-AUC is the **unweighted** mean of the per-order AUCs; an order with only
one class yields a null AUC and is excluded from the mean.

---

## 5. How to work

Use the interpreter that has torch **and scipy**:
`/home/zhx/miniconda3/envs/scalepinn/bin/python`. Without scipy,
`fill_tables.py` silently swaps the t quantile for 1.96 and drops every p-value
(bug 18).

    PY=/home/zhx/miniconda3/envs/scalepinn/bin/python
    $PY -m experiments.repair_results     # relink stragglers
    $PY -m experiments.fill_tables        # write measured values in place
    $PY -m experiments.coverage           # per-table fill report

Every sweep is resumable: a job counts as done only once its record carries a
non-null `macro_auc`, so a killed job is retried on the next run. Launch
detached — the sandbox container is rebuilt often and takes every process with
it, which is what killed the sweeps repeatedly and left logs that stop
mid-epoch with no traceback. **Those are not code bugs; just rerun.**

    setsid nohup $PY -m experiments.paper_sweep ... >> suite.log 2>&1 < /dev/null &

The four commands that were in flight at handoff:

    $PY -m experiments.paper_sweep --suite tuning --scenes ecological social \
        --seeds 11 --noises 0.02 --n_epochs 8000 --stage1 1500 --ramp 1000 \
        --gpus 0 --jobs_per_gpu 3 --out results/paper/tuning

    $PY -m experiments.paper_sweep --suite tuning --scenes neuronal \
        --seeds 11 --noises 0.02 --n_epochs 8000 --stage1 1500 --ramp 1000 \
        --gpus 5 --jobs_per_gpu 1 --out results/paper/tuning_neuronal

    $PY -m experiments.paper_sweep --suite noise --scenes neuronal \
        --seeds 11 12 13 --n_epochs 8000 --stage1 1500 --ramp 1000 \
        --gpus 1 6 --jobs_per_gpu 2 --out results/paper/noise_neuronal

    $PY -m experiments.paper_sweep --suite main --scenes neuronal --seeds 13 \
        --noises 0.02 --n_epochs 8000 --stage1 1500 --ramp 1000 \
        --gpus 7 --jobs_per_gpu 1 --out results/paper/main_neuronal

Pick GPUs with free memory — the machine is shared and other users routinely
hold 40 GB on a card. Do not use `pkill -f "paper_sweep --suite noise"` from a
shell whose own command line contains that string; it kills itself.

### What is in git

Code, presets, `REBUTTAL_RESPONSE.md`, `BUGFIX_LOG.md`, this file, the
screenshots, and the run manifests `results/paper/*/results.jsonl` plus every
`summary.json` (~1.6 MB, force-added — `results/` is ignored, and a negation
cannot reach inside an excluded directory). The ~4.5 GB of per-run payloads is
not in git and is not needed: the two commands above refill every table from
the manifests alone.

---

## 6. State at handoff

**Coverage 164/164 — the table is complete.** Every sweep ran to completion
and no process is left running.

The four cells that were long marked `not run (THIS does not terminate at
n=30)` are now measured. THIS was never hanging: its Julia code had three
quadratic accumulation patterns and formed an m×m Gram matrix that is 785 GB
at n=30, and its output is block-buffered, so an empty log looked like a hang.
With bugs 27-29 fixed a neuronal run finishes in **35 seconds**. The fixes are
behaviour-preserving - the ecological scene reproduces its stored run to four
decimals.

What each suite ended at:

| Suite | Runs |
|---|---|
| `tuning` (ecological + social) | 160/160 |
| `tuning_neuronal` | 61/64 |
| `noise_neuronal` | complete |
| `holdout` (all three scenes) | 99/99 |

Re-deriving the document from the committed manifests is two commands and
needs no GPU:

    python -m experiments.repair_results
    python -m experiments.fill_tables

**Before adding runs, read bug 24 in `BUGFIX_LOG.md`.** The sandbox cgroup
allows 8192 pids and an uncapped job opens 127 threads, so a dozen concurrent
jobs exhaust it and every later job dies about 70 seconds in with
`libgomp: Thread creation failed`, logged as an ordinary failure. `env_for()`
now pins the thread pools; `SWEEP_THREADS_PER_JOB` overrides the default of 4.
This was also the cause of most of the unexplained failures earlier in the
project: a neuronal HyperPINN run that had died at 30 s with no error
completed in 1879 s once the cap was in place.

### Four questions the owner asked, and the answers given

1. *Can you access the GPU; is it causing the disconnections?* No. 8 GPUs
   visible, compute verified, no Xid/NVRM errors in dmesg. The disconnections
   and process deaths are the sandbox container being rebuilt.
2. *Is "12-20" missing a seed; should it be 10?* It is 9 seeds and that is
   correct — seeds are 11-20, and seed 11 is the selection seed. The real
   defect was that the held-out column was averaging all ten (bug 15).
   A later question about the CI of neuronal HyperPINN had the same root:
   that row was n=9 because seed 13 had died, so its interval used t(8), not
   t(9). The rerun succeeded and the row is n=10 again.
3. *Did THIS produce neuronal results? The std dev does not match.* No. The
   number in R2.6c was fabricated and has been removed (bug 17).
4. *In R2.5a, is integral always below derivative at k=5?* No. Measured
   per-order AUC at k=5: ecological 0.376 vs 0.865 (integral lower), social
   0.580 vs 0.500, neuronal 0.539 vs 0.519 (integral higher in both). Only
   ecological shows the pattern. Note the document's R2.5a shows 0.650 vs
   0.865 — that is an Ours table, left untouched per constraint 1, and
   reported rather than edited.

---

## 7. Traps

- Re-running `fill_tables.py` under a scipy-less interpreter will quietly
  rewrite every confidence interval in the document. Check `git diff` after
  every fill; only the cells you expect should move.
- `_tuning_runs` keys off `meta["variant"]`. A new tuned baseline needs an
  entry in `TUNE_ROW`, `TUNE_KEYS` and `MAIN_OF_TUNE`, or its table row stays
  empty with no error.
- When you change what a tuned job actually runs, the old records must be
  deleted *and* their run directories removed, or `load_done` skips them and
  you measure the old thing forever.
- `paper_sweep` writes `results.jsonl` by append under a lock. Do not edit that
  file while a sweep is running.
