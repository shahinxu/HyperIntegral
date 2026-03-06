## Project Layout (Unified)

Top-level now follows a unified structure:

- `models/`: all model entrypoints and unified launchers
- `lib_*/`: scene/data-generation libraries
- `results*/`: experiment outputs
- `run_*.sh`: convenience shell runners
- `*.md`: docs

### Model Folder Overview

- `models/run_unified.py`: single unified CLI across methods/scenes
- `models/integral/`: Integral-based methods
- `models/kernel/`: Kernel-Integral methods
- `models/baseline/`: baseline unified launcher
- `models/baseline/HyperPINN/`: HyperPINN model scripts
- `models/this/`: THIS unified launcher

Each method folder now exposes a consistent entrypoint:

- `models/<method>/run.py`

with a common argument interface (unused args are ignored by methods that do not need them):

- `--scene --n_samples --noise --gpu_id --n_nodes --max_order --n_epochs --lr --n_trajectories --results_root --python --bin_thresh`

### Strict Slim Version

- Top-level compatibility wrappers have been removed.
- HyperPINN model files are under `models/baseline/HyperPINN/`.
- HyperPINN outputs are in top-level `results_hyperpinn*` directories.
- Use `python models/run_unified.py ...` or scripts under `models/*` directly.
