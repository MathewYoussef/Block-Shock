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

## Milestone 2 - Timing + metrics discipline (before any methods)

### 2.1 Timer regions

Create:

- `src/metrics.py`

What it must do:

- Named timing regions: `build`, `forward`, `backward`, `opt_step`, `compress`, `allreduce`, `total_step`
- Handle CUDA sync correctly (explicitly decide when to sync)

**Definition of Done**

- A toy "sleep + CUDA op" test yields believable timings

**Status**

- Done (timing regions implemented; smoke test shows ~50 ms sleep and GPU GEMM timing)
- Verified with `python3 -m src.timing_smoke` inside container
- Verification command and sample output:

```bash
python3 -m src.timing_smoke
```

```text
{'build': {'count': 0.0, 'sum_ms': 0.0, 'avg_ms': 0.0}, 'forward': {'count': 0.0, 'sum_ms': 0.0, 'avg_ms': 0.0}, 'backward': {'count': 0.0, 'sum_ms': 0.0, 'avg_ms': 0.0}, 'opt_step': {'count': 0.0, 'sum_ms': 0.0, 'avg_ms': 0.0}, 'compress': {'count': 0.0, 'sum_ms': 0.0, 'avg_ms': 0.0}, 'allreduce': {'count': 0.0, 'sum_ms': 0.0, 'avg_ms': 0.0}, 'total_step': {'count': 0.0, 'sum_ms': 0.0, 'avg_ms': 0.0}, 'sleep': {'count': 1.0, 'sum_ms': 50.14619417488575, 'avg_ms': 50.14619417488575}, 'cuda_gemm': {'count': 1.0, 'sum_ms': 8.188785053789616, 'avg_ms': 8.188785053789616}}
```

### 2.2 Metrics logging

Update:

- `src/logging_utils.py` to write JSONL lines

What it must do:

- Write one JSON record per measurement block
- Include: method name, phase, N, batch, dtype, world_size, timings, memory peak

**Definition of Done**

- `metrics.jsonl` has valid records you can read as a dataset

**Status**

- Done (metrics.jsonl written by `src/timing_smoke.py`)
- Verification record:

```json
{"timestamp_unix": 1768612523.4315023, "run_id": "timing_smoke", "phase": "smoke", "method": "smoke", "N": 0, "B": 0, "dtype": "n/a", "world_size": 1, "timings_ms": {"build": {"count": 0.0, "sum_ms": 0.0, "avg_ms": 0.0}, "forward": {"count": 0.0, "sum_ms": 0.0, "avg_ms": 0.0}, "backward": {"count": 0.0, "sum_ms": 0.0, "avg_ms": 0.0}, "opt_step": {"count": 0.0, "sum_ms": 0.0, "avg_ms": 0.0}, "compress": {"count": 0.0, "sum_ms": 0.0, "avg_ms": 0.0}, "allreduce": {"count": 0.0, "sum_ms": 0.0, "avg_ms": 0.0}, "total_step": {"count": 0.0, "sum_ms": 0.0, "avg_ms": 0.0}, "sleep": {"count": 1.0, "sum_ms": 50.17054406926036, "avg_ms": 50.17054406926036}, "cuda_gemm": {"count": 1.0, "sum_ms": 8.224955992773175, "avg_ms": 8.224955992773175}}, "memory_peak_bytes": 46270464}
```

## Milestone 3 - Distributed plumbing (even before TP)

### 3.1 Distributed init

Create:

- `src/distributed.py`

What it must do:

- Initialize torch distributed when `world_size > 1`
- Assign device by local rank
- Provide:
  - `is_distributed()`
  - `rank()`, `world_size()`
  - `barrier()`
  - `allreduce_sum(tensor)`

**Definition of Done**

- A tiny smoke test: rank0 prints "hello", rank1 prints "hello", then both allreduce a scalar and get the same answer

**Status**

- Done (allreduce smoke test passed with `torchrun --standalone --nproc_per_node=2 -m src.allreduce_smoke`)
- Verification output:

```text
rank=0 world_size=2 allreduce_sum=1.0rank=1 world_size=2 allreduce_sum=1.0
```

## Milestone 4 - Workloads (data generation + simple losses)

### 4.1 Workload generator

Create:

- `src/workloads.py`

What it must do:

- Generate input `X` with configurable distribution:
  - gaussian
  - uniform
  - fixed seed deterministic
- Optionally generate target `T` or define a simple loss (e.g., MSE vs zeros)

**Definition of Done**

- For fixed seed, two runs produce identical `X` on the same device/dtype

**Status**

- Done (workloads implemented; repetition smoke test passed)
- Verification command and sample output:

```bash
python3 -m src.workload_gen_repetition_smoke
```

```text
random_normal: same=True shape=(64, 4096) dtype=torch.float32 device=cuda:0
uniform: same=True shape=(64, 4096) dtype=torch.float32 device=cuda:0
activation_like: same=True shape=(64, 4096) dtype=torch.float32 device=cuda:0
transformer_mlp: same=True shape=(64, 4096) dtype=torch.float32 device=cuda:0
attention_like: same=True shape=(64, 4096) dtype=torch.float32 device=cuda:0
vision_conv: same=True shape=(64, 4096) dtype=torch.float32 device=cuda:0
```
- New files:
  - `configs/workloads/transformer_mlp_like.yaml`
  - `configs/workloads/attention_like.yaml`
  - `configs/workloads/vision_conv.yaml`
  - `src/workload_gen_repetition_smoke.py`

## Milestone 5 - Orchestrator (phases as pipelines)

### 5.1 Orchestrator

Create:

- `src/orchestrator.py`

What it must do:

- Execute Phase 0 and Phase 1 pipelines:
  - Phase 0: build -> forward -> compare to reference -> log correctness
  - Phase 1: warmup -> timed forward loops -> log speed

**Definition of Done**

- You can run Phase 0 + Phase 1 using a placeholder method that just calls dense matmul

**Status**

- Done (orchestrator smoke test passed with sync + cuda_events timing)
- Verification command and sample output:

```bash
python3 -m src.orchestrator_smoke
```

```text
phase0_correctness results: {'max_abs_error': 127.12849426269531, 'rel_error': 1.5325625906519214, 'phase': 'phase0_correctness', 'timings_ms': {'build': {'count': 0.0, 'sum_ms': 0.0, 'avg_ms': 0.0, 'p50_ms': 0.0, 'p95_ms': 0.0}, 'forward': {'count': 1.0, 'sum_ms': 44.575911946594715, 'avg_ms': 44.575911946594715, 'p50_ms': 44.575911946594715, 'p95_ms': 44.575911946594715}, 'backward': {'count': 0.0, 'sum_ms': 0.0, 'avg_ms': 0.0, 'p50_ms': 0.0, 'p95_ms': 0.0}, 'opt_step': {'count': 0.0, 'sum_ms': 0.0, 'avg_ms': 0.0, 'p50_ms': 0.0, 'p95_ms': 0.0}, 'compress': {'count': 0.0, 'sum_ms': 0.0, 'avg_ms': 0.0, 'p50_ms': 0.0, 'p95_ms': 0.0}, 'allreduce': {'count': 0.0, 'sum_ms': 0.0, 'avg_ms': 0.0, 'p50_ms': 0.0, 'p95_ms': 0.0}, 'total_step': {'count': 0.0, 'sum_ms': 0.0, 'avg_ms': 0.0, 'p50_ms': 0.0, 'p95_ms': 0.0}}}
phase1_forward results: {'phase': 'phase1_forward', 'timings_ms': {'build': {'count': 0.0, 'sum_ms': 0.0, 'avg_ms': 0.0, 'p50_ms': 0.0, 'p95_ms': 0.0}, 'forward': {'count': 3.0, 'sum_ms': 0.04828799981623888, 'avg_ms': 0.016095999938746292, 'p50_ms': 0.014976000413298607, 'p95_ms': 0.020390399731695652}, 'backward': {'count': 0.0, 'sum_ms': 0.0, 'avg_ms': 0.0, 'p50_ms': 0.0, 'p95_ms': 0.0}, 'opt_step': {'count': 0.0, 'sum_ms': 0.0, 'avg_ms': 0.0, 'p50_ms': 0.0, 'p95_ms': 0.0}, 'compress': {'count': 0.0, 'sum_ms': 0.0, 'avg_ms': 0.0, 'p50_ms': 0.0, 'p95_ms': 0.0}, 'allreduce': {'count': 0.0, 'sum_ms': 0.0, 'avg_ms': 0.0, 'p50_ms': 0.0, 'p95_ms': 0.0}, 'total_step': {'count': 0.0, 'sum_ms': 0.0, 'avg_ms': 0.0, 'p50_ms': 0.0, 'p95_ms': 0.0}}, 'iterations': 3, 'warmup_iters': 2}
```
