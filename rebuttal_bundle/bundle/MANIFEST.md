# Rebuttal experiment bundle

## Environment
- Python: /home/zhx/miniconda3/envs/scalepinn/bin/python  (torch 2.11.0+cu128, CUDA available)
- Julia:  ~/.local/julia-1.10.5/bin/julia  (required by the THIS baseline; `Pkg.instantiate()` in models/this)
- GPUs:   8x NVIDIA RTX A6000 (shared with other users; set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True)

## Benchmarks (match the paper)
| Scene | n | max_order | candidate hyperedges | preset / env var |
|---|---|---|---|---|
| ecological | 9 | 5 | 372 | ECOLOGY_HYPERGRAPH_PRESET=n9 |
| social | 8 | 5 | 210 | (default preset) |
| neuronal | 30 | 5 | 174406 | NEURONAL_HYPERGRAPH_FILE=<n30 json> |

The n30 neuronal preset was NOT under version control and could not be recovered
(`.gitignore` had a blanket `*.json`). `config/generate_n30_preset.py` regenerates it
from the n60 design rule (3 modules, within:cross = 2:1 per order, within-module edges
split equally, counts decreasing by order -> 90/45/18/9). It is a regenerated topology,
not the original one, so neuronal numbers are not comparable to the paper's Table III.

## Layout
- `code/` — the proposed method, baselines and sweep drivers (see `all_changes.diff` for every edit)
- `config/` — scene presets used, and the n30 generator
- `results/paper/` — current runs on the paper's benchmarks (results.jsonl + summary.json + logs)
- `results/rebuttal_superseded/` — earlier runs on Rossler / N=60, superseded, kept for reference only
- `docs/REBUTTAL_RESPONSE.md` — comment-by-comment response tables (the deliverable)
- `docs/REBUTTAL_EXPERIMENTS.md` — earlier working document; its results section is superseded

## Metric
Per-order AUROC over that order's full candidate set (true hyperedges labelled 1).
`macro_auc` = unweighted mean of the per-order AUROCs that are defined; an order with
only one class present is recorded as null and excluded, so the denominator can vary
between runs.

## Commands
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/miniconda3/envs/scalepinn/bin/python

# main comparison: ours vs HyperPINN / Kernel / THIS / SINDy variants
$PY -m experiments.paper_sweep --suite main \
    --scenes ecological social --seeds 11 12 13 --noises 0.02 \
    --gpus 2 7 0 1 --jobs_per_gpu 2 --out results/paper/main

# ablations: full / derivative_loss / no_staged / no_bounded / window_* / intervals_*
$PY -m experiments.paper_sweep --suite ablation \
    --scenes ecological social --seeds 11 12 13 --noises 0.02 \
    --gpus 2 7 0 1 --jobs_per_gpu 2 --out results/paper/ablation

# single run of the proposed method
ECOLOGY_HYPERGRAPH_PRESET=n9 $PY -m models.integral.train_nn \
    --scene ecological --n_samples 300 --noise 0.02 --seed 11 \
    --n_epochs 20000 --stage1_epochs 2500 --physics_ramp_epochs 2000 \
    --lr 0.001 --gpu_id 2 --results_root results/paper/single --run_tag demo

# tables
$PY -m experiments.aggregate results/paper/main/results.jsonl \
    --reference ours --out_md results/paper/main.md --out_csv results/paper/main.csv
```

## Status at time of packaging
- Main sweep on ecological + social: 42 jobs, 27 produced a summary.
  The 6 THIS jobs failed inside the sweep (THIS runs correctly standalone); under investigation.
- Ablation sweep: not yet run.
- Neuronal (n=30, 174406 candidates): not yet run.
- Response tables in `docs/REBUTTAL_RESPONSE.md` are therefore still mostly empty.

## Known issues fixed in this bundle (they affected correctness)
1. `generate_all_possible_hyperedges()` returned the ground-truth configuration in all
   three scenes, so any candidate pool built from it had no negatives and every AUC was
   undefined.
2. `HyperPINN_Ecosystem.py: evaluate_edges_triangles()` had no `return`, so the ecological
   HyperPINN run crashed before reporting any AUC.
3. HyperPINN scene scripts ignored `--n_epochs` / `--lr` (never forwarded) and had no
   `--seed`; `np.random.seed()` was called with no argument, so runs were not reproducible.
4. HyperPINN `all_*edges` were set to the ground-truth configuration (same class of bug as 1).
5. Baseline launchers parsed AUCs from an `auc_scores.txt` that the inner scripts never
   write, so every baseline `summary.json` had an empty `auc` field.
6. `models/hyperpinn/run.py` never set `ROSSLER_HYPERGRAPH_PRESET`, so `--n_nodes 8`
   still loaded the 100-node hypergraph.
7. Kernel passed `seed` into `generate_training_data` bypassing its signature check.
8. THIS: Julia shelled out to a bare `python` without PYTHONPATH; the export scripts
   rejected the `--n_nodes` / `--max_order` the Julia driver passes; output went to a
   shared directory so parallel runs overwrote each other; AUC keys used `order=k`.
9. `ECOLOGY_HYPERGRAPH_PRESET` was ignored by `get_default_params()`, silently selecting
   the 60-node scene (5,985,137 candidates) instead of the paper's 9-node one.
10. Output directories were keyed only by timestamp, so parallel sweep runs collided.
