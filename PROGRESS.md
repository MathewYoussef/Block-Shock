# Progress

## Milestone 0 - Repo skeleton + rules of the game

### 0.1 Create the repo layout

- `configs/` (base + phase + method + mask + workload)
- `src/` (main, orchestrator, methods, sparsity, metrics, logging, distributed)
- `analysis/`
- `results/` (raw/tables/plots)

**Definition of Done**

- You can run `python -m src.main --help` (even if it prints "not implemented yet")
- Git has a clean first commit

**Status**

- Done (layout created; `python -m src.main --help` verified)
- First git commit pending user action

### 0.2 Write your "experiment contract" into README

Include:

- What counts as a "run"
- Where outputs go
- How configs are composed
- What baselines exist
- What "Block-Shock" is in one paragraph

**Definition of Done**

- A stranger could reproduce a run by copy/pasting two commands

**Status**

- Done (experiment contract documented in `README.md`)

## Milestone 1 - Config system (YAML composition) + run registry

### 1.1 Config loader + merger

Create:

- `src/config.py` (or `src/utils_config.py`)
- `configs/base.yaml`
- `configs/phases/phase0_correctness.yaml`
- `configs/methods/dense_single.yaml`
- `configs/workloads/gaussian.yaml`

What it must do:

- Load multiple YAMLs and merge them in order
- Support a unique `run_id`
- Dump the resolved config into the run folder

**Definition of Done**

- Running main prints the resolved config and writes it to `results/raw/<run_id>/config.yaml`

**Status**

- Done (resolved config printed; `config.yaml` written to `results/raw/<run_id>/`)
- Verification command and sample output:

```bash
python3 -m src.main --config configs/base.yaml --phase configs/phases/phase0_correctness.yaml --method configs/methods/dense_single.yaml --workload configs/workloads/gaussian.yaml --hardware configs/hardware/local_2gpu.yaml --print-config
```

```yaml
experiment:
  name: block_shock
  seed: 1234
model:
  dtype: bf16
  N: 4096
  B: 64
phase:
  name: phase0_correctness
  forward_only: true
  check_correctness: true
  check_grad_input: false
  train_step: false
  record_timings: false
method:
  name: dense_single
  mode: dense
  tensor_parallel: none
mask:
  name: complement_1100_0011
workload:
  name: gaussian
  type: random_normal
  mean: 0.0
  std: 1.0
hardware:
  name: local_2gpu
  world_size: 2
  backend: nccl
logging:
  out_dir: results/raw
  table_dir: results/tables
  plot_dir: results/plots
run_id: 20260117_004122_56ac8c1d
config_paths:
- configs/base.yaml
- configs/phases/phase0_correctness.yaml
- configs/methods/dense_single.yaml
- configs/workloads/gaussian.yaml
- configs/hardware/local_2gpu.yaml
```

### 1.2 Run folder + metadata

Create:

- `src/logging_utils.py`

What it must do:

- Create a run directory
- Save config + environment metadata (torch version, GPU name, capability)
- Save a seed used for reproducibility

**Definition of Done**

- Every run creates a folder with:
- config
- env info
- seed
- an empty `metrics.jsonl` ready for writing

**Status**

- Done (env.json, seed.txt, and metrics.jsonl created in run folder)
