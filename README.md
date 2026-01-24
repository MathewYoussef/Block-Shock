# Block-Shock

A multi-GPU "dense-equivalent" training method using semi-structured 2:4 sparse kernels.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Elevator Pitch](#elevator-pitch)
- [Repo Architecture](#repo-architecture)
- [Usage Examples](#usage-examples)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Requirements

- Python 3.8 or higher
- PyTorch 2.0+ with CUDA support
- NVIDIA GPU with Tensor Core support (Ampere or newer recommended for optimal 2:4 sparse performance)
- 2 GPUs for multi-GPU experiments
- CUDA 11.8 or higher

### Setup

1. Clone the repository:
```bash
git clone https://github.com/MathewYoussef/Block-Shock.git
cd Block-Shock
```

2. Install dependencies:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

3. Verify installation:
```bash
python verify_setup.py
```

This will check that all dependencies are installed and your environment is properly configured.

## Quick Start

### Try the Examples

The easiest way to get started is with the example scripts:

```bash
# Run a simple correctness check (requires 2 GPUs)
torchrun --standalone --nproc_per_node=2 examples/simple_correctness_check.py

# Benchmark single GPU dense baseline
python examples/benchmark_comparison.py --method dense_single --N 4096

# Benchmark Block-Shock (requires 2 GPUs)
torchrun --standalone --nproc_per_node=2 examples/benchmark_comparison.py --method block_shock --N 8192
```

See [examples/README.md](examples/README.md) for more details.

### Using the Main Runner

Run a simple correctness check with dense single-GPU baseline:
```bash
python -m src.main \
  --config configs/base.yaml \
  --phase configs/phases/phase0_correctness.yaml \
  --method configs/methods/dense_single.yaml \
  --workload configs/workloads/gaussian.yaml \
  --hardware configs/hardware/local_2gpu.yaml
```

Run a 2-GPU Block-Shock forward benchmark:
```bash
torchrun --standalone --nproc_per_node=2 -m src.main \
  --config configs/base.yaml \
  --phase configs/phases/phase1_forward.yaml \
  --method configs/methods/block_shock_2gpu.yaml \
  --workload configs/workloads/gaussian.yaml \
  --hardware configs/hardware/local_2gpu.yaml
```

## Elevator pitch

You want dense model capacity (all weights exist and get updated), but you want to ride NVIDIA's 2:4 sparse Tensor Core / cuSPARSELt path.

Write the dense weight as a sum of multiple 2:4-sparse matrices:

```text
W = W(0) + W(1)
```

Each `W(g)` is 2:4 sparse (50% zeros in each 4-wide group) and is placed on a different GPU. Then:

```text
y = x W^T
  = x (W(0) + W(1))^T
  = x W(0)^T + x W(1)^T
```

Compute each term on its GPU with semi-structured sparse matmul, and sum the outputs with an all-reduce. This is the core Block-Shock "dense by superposition" trick.

This is structurally aligned with how semi-structured kernels work: sparse weights stored in a compressed form plus a metadata mask.

## Why this might work (and why it might not)

### Why it might work

- Semi-structured 2:4 is fixed at 50% sparsity, with a theoretical 2x improvement in the ideal case.
- PyTorch's semi-structured format explicitly targets compressed weights and sparse GEMM dispatch (cuSPARSELt) rather than "masked dense."
- For inference, the workflow is explicitly: prune/mask -> compress -> run sparse kernel.

### Why it might not

- You do two sparse matmuls (one per GPU) plus a reduction, instead of one dense matmul. Communication and overhead can eat your win.
- Training adds another pain point: weights change each step, so "compress once" stops being true.
- Semi-structured ops support is limited to a specific set (mm/addmm/linear and transposes).
- Speedups vary a lot by architecture and workload; even PyTorch's own tutorial warns speedups may differ.

## Repo architecture

Phases 0-3 share a single experiment engine. Each phase is just a different pipeline of toggles:

- Phase 0: forward correctness (small N) + equivalence checks
- Phase 1: forward throughput (timing only)
- Phase 2: forward + backward wrt input (weights frozen)
- Phase 3: full training step (forward + backward + optimizer + optional recompress)

Key directories:

- `configs/`: YAML stack (base + sweeps + phases + methods + masks + workloads + hardware)
- `src/`: core runner and shared modules used by all phases
- `results/`: raw runs, tables, and plots
- `analysis/`: aggregation and plotting scripts

## Experiment contract (rules of the game)

### What counts as a "run"

A run is one invocation of the experiment runner with a fully resolved config. Each run produces a unique `run_id`, a config snapshot, environment metadata, and one or more metrics records.

### Where outputs go

Each run writes to `results/raw/<phase>/<method>/<run_id>/` with:

- `config.yaml` (resolved config)
- `env.json` (hardware + software metadata)
- `seed.txt` (seed used for reproducibility)
- `metrics.jsonl` (one JSON record per measurement block)

Aggregated tables and plots are written to `results/tables/` and `results/plots/`.

Official runs

To keep an official record under version control, run with `configs/official.yaml`. Each run writes to:

- `results/official/runs/<run_group>/<phase>/<method>/<run_id>/`

`run_group` is auto-generated (UTC timestamp) so new runs never collide.

Sweeps write to:

- `results/official/sweeps/<tag>/<phase>/<method>/<run_id>/`

### How configs are composed

Configs are merged in order (later files override earlier keys):

1) `configs/base.yaml`
2) one phase config from `configs/phases/`
3) one method config from `configs/methods/`
4) one workload config from `configs/workloads/`
5) one hardware config from `configs/hardware/`
6) optional sweep config from `configs/sweeps/`

### What baselines exist

- Dense single GPU (`configs/methods/dense_single.yaml`)
- Dense TP 2-GPU (`configs/methods/dense_tp.yaml`)
- Masked split dense (ablation) (`configs/methods/masked_split_dense.yaml`)
- Block-Shock 2-GPU (`configs/methods/block_shock_2gpu.yaml`)

### Block-Shock in one paragraph

Block-Shock writes a dense weight as the sum of multiple 2:4-sparse matrices placed on different GPUs, computes each sparse matmul with semi-structured kernels, then all-reduces the partial outputs. This preserves dense capacity while attempting to exploit the 2:4 sparse Tensor Core path.

Note: The `mask` section appears in merged configs because it is set in `configs/base.yaml`. It is ignored by non-sparse baselines (e.g., `dense_single`, `dense_tp`) and only used by sparse/masked methods.

Note: For TP baselines, inputs are currently broadcast from rank 0 to ensure correctness. A future improvement is to generate `X` once on rank 0 and scatter feature shards to each GPU.

## Phase 0 reference experiment (correctness)

Phase 0 compares each method against a single dense reference:

- **Reference**: dense single-GPU `F.linear` using weight `W` (and optional bias `b`).
- **Test**: each method computes its output on the same input `X` and same `W`.

Determinism rules:

- `X` is generated with a fixed seed.
- `W` (and optional `b`) are generated with the same seed for both reference and test.
- Exact zeros in `W` are nudged to `eps` to avoid accidental 1-of-4 blocks in 2:4 validation; this is applied consistently across methods.
- This ensures exact reproducibility across runs.

Memory metrics:

- Runs log per-method weight storage estimates (e.g., `weight_bytes_total`, `weight_bytes_sparse_est`) to help attribute memory overhead alongside timing.
- For large-N Phase 1 sweeps, you can enable `method.drop_full_weight: true` to release the dense `W_full` after shards/compressed weights are created (kept when correctness checks are enabled).

Comparison metrics:

- `max_abs_error`
- `mean_abs_error`
- `max_rel_error` (with configurable `phase.rel_eps`)
- `passed` (boolean, based on `phase.tol_max_abs` and `phase.tol_max_rel`)

Phase 0 also runs configurable warmup and timed iterations:

- `phase.warmup_iters` (default 10)
- `phase.timed_iters` (default 100)

### Phase 0 vs Phase 1 (what changes)

- Phase 0: correctness against dense reference, logs error metrics, uses `sync` timing.
- Phase 1: forward-only throughput, no reference comparisons, uses `cuda_events` timing.
- If a method records communication timing (e.g., dense TP), it populates `timings_ms.allreduce` with p50/p95 stats.
- Methods that call collectives run `collective_prep` first; its timing is logged under `timings_ms.layout_fix`, along with `layout_fix_trigger_rate` and `layout_fix_bytes_per_copy`.

Bias handling:

- If a method uses bias, the reference includes the same bias.
- If a method does not use bias, the reference bias is disabled.

## Dtype standard (bf16)

The project defaults to `bf16` because NVIDIA 2:4 semi-structured kernels require `bf16`/`fp16` and dimensions multiple of 64. Keep `model.dtype: bf16` unless you are running a specific fp32 debug check.

### Reproduce a run (two commands)

```bash
python -m src.main --help
python -m src.main --config configs/base.yaml --phase configs/phases/phase0_correctness.yaml --method configs/methods/dense_single.yaml --workload configs/workloads/gaussian.yaml --hardware configs/hardware/local_2gpu.yaml
```

## Timing modes

- `sync`: CPU wall time with `torch.cuda.synchronize()` before/after each region. Use for correctness and debugging.
- `cuda_events`: GPU event timing on the current stream, sync once at summary. Use for Phase 1 benchmarking.
- `none`: No sync. Not recommended for benchmarking.

## Usage Examples

### Running Different Phases

**Phase 0 - Correctness Check:**
```bash
# Single GPU dense baseline
torchrun --standalone --nproc_per_node=1 -m src.main \
  --config configs/base.yaml \
  --phase configs/phases/phase0_correctness.yaml \
  --method configs/methods/dense_single.yaml \
  --workload configs/workloads/gaussian.yaml \
  --hardware configs/hardware/local_2gpu.yaml

# Block-Shock 2-GPU
torchrun --standalone --nproc_per_node=2 -m src.main \
  --config configs/base.yaml \
  --phase configs/phases/phase0_correctness.yaml \
  --method configs/methods/block_shock_2gpu.yaml \
  --workload configs/workloads/gaussian.yaml \
  --hardware configs/hardware/local_2gpu.yaml
```

**Phase 1 - Forward Throughput Benchmark:**
```bash
# Dense tensor parallel baseline
torchrun --standalone --nproc_per_node=2 -m src.main \
  --config configs/base.yaml \
  --phase configs/phases/phase1_forward.yaml \
  --method configs/methods/dense_tp.yaml \
  --workload configs/workloads/gaussian.yaml \
  --hardware configs/hardware/local_2gpu.yaml

# Masked split dense (ablation)
torchrun --standalone --nproc_per_node=2 -m src.main \
  --config configs/base.yaml \
  --phase configs/phases/phase1_forward.yaml \
  --method configs/methods/masked_split_dense.yaml \
  --workload configs/workloads/gaussian.yaml \
  --hardware configs/hardware/local_2gpu.yaml
```

**Phase 2 - Backward Input Gradients:**
```bash
torchrun --standalone --nproc_per_node=2 -m src.main \
  --config configs/base.yaml \
  --phase configs/phases/phase2_backward_input.yaml \
  --method configs/methods/dense_tp.yaml \
  --workload configs/workloads/gaussian.yaml \
  --hardware configs/hardware/local_2gpu.yaml
```

### Running Sweeps

Run an N-dimension sweep and generate plots:
```bash
# Execute sweep
python scripts/run_sweep.py --sweep configs/sweeps/N_sweep.yaml

# Aggregate results
python analysis/aggregate.py \
  --input results/official/sweeps/<tag> \
  --output results/tables/runs.csv

# Generate plots
python analysis/plot_speedups.py \
  --input results/tables/runs.csv \
  --out-dir results/plots
```

### Customizing Experiments

Create your own workload configuration in `configs/workloads/`:
```yaml
# configs/workloads/my_workload.yaml
workload:
  name: my_custom_workload
  type: random_normal
  mean: 0.0
  std: 0.5
```

Override model size in command line:
```bash
python -m src.main \
  --config configs/base.yaml \
  --phase configs/phases/phase1_forward.yaml \
  --method configs/methods/block_shock_2gpu.yaml \
  --workload configs/workloads/gaussian.yaml \
  --hardware configs/hardware/local_2gpu.yaml
```

## Project Structure

```
Block-Shock/
├── .github/              # GitHub configuration
│   └── workflows/        # CI/CD workflows
│       └── tests.yml     # Automated testing workflow
├── analysis/             # Data aggregation and visualization scripts
│   ├── aggregate.py      # Aggregate JSONL metrics to CSV
│   ├── plot_speedups.py  # Generate performance plots
│   └── report.md         # Analysis reports
├── configs/              # YAML configuration files
│   ├── base.yaml         # Base configuration
│   ├── phases/           # Phase configurations (0, 1, 2)
│   ├── methods/          # Method implementations config
│   ├── workloads/        # Input data generation patterns
│   ├── hardware/         # Hardware setup (GPU counts, backend)
│   ├── masks/            # 2:4 sparsity mask patterns
│   └── sweeps/           # Parameter sweep configurations
├── examples/             # Example scripts and tutorials
│   ├── README.md         # Examples documentation
│   ├── simple_correctness_check.py  # Basic correctness example
│   └── benchmark_comparison.py      # Performance comparison example
├── results/              # Experimental outputs
│   ├── raw/              # Raw run data
│   ├── official/         # Versioned official runs
│   ├── tables/           # Aggregated CSV data
│   └── plots/            # Generated visualizations
├── scripts/              # Utility scripts
│   └── run_sweep.py      # Sweep execution script
├── src/                  # Source code
│   ├── main.py           # Entry point
│   ├── orchestrator.py   # Phase pipeline orchestration
│   ├── config.py         # Config loading and merging
│   ├── distributed.py    # Distributed training utilities
│   ├── logging_utils.py  # Logging and metrics I/O
│   ├── metrics.py        # Timing and metric tracking
│   ├── workloads.py      # Input data generation
│   ├── validation.py     # Correctness checks
│   ├── methods/          # Implementation of all methods
│   │   ├── dense_single.py      # Single GPU dense baseline
│   │   ├── dense_tp.py          # Dense tensor parallel
│   │   ├── masked_split_dense.py # Dense compute with masks
│   │   └── block_shock.py       # Block-Shock sparse method
│   └── sparsity/         # Sparsity utilities
│       ├── masks.py      # Mask generation and validation
│       └── semistructured.py # Semi-structured sparse ops
├── tests/                # Unit tests
│   ├── __init__.py
│   ├── test_config.py    # Config system tests
│   ├── test_masks.py     # Mask generation tests
│   └── test_workloads.py # Workload generation tests
├── .gitignore
├── CONTRIBUTING.md       # Contribution guidelines
├── LICENSE
├── README.md
├── PROGRESS.md           # Development progress tracking
└── requirements.txt      # Python dependencies
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Project TODO list (milestones, top-to-bottom)

### Milestone X - Forward-only drift test (Phase X)

**X.1 Drift test pipeline**

Create:

- `src/orchestrator.py` (new pipeline)
- `configs/phases/phaseX_drift.yaml`

What it must do:

- Repeatedly apply a forward-only block for T steps
- Compare trajectories vs dense reference (error vs step)
- Log drift metrics (max/mean/rel error per step)

**Definition of Done**

- A run produces per-step error curves for dense vs Block-Shock forward

### Milestone 13 (deferred) - Phase 2/3 backward + optimizer

Phase 2 and Phase 3 are deferred for now. Semi-structured sparse backward is not supported in the current PyTorch build, so the focus remains on forward-only experiments.

### Milestone 14 - Sweeps + analysis

**14.1 Sweeps**

Create:

- `configs/sweeps/N_sweep.yaml`
- `configs/sweeps/batch_sweep.yaml`

**Definition of Done**

- You can launch a sweep and get one run folder per configuration

**14.2 Aggregation and plots**

Create:

- `analysis/aggregate.py`
- `analysis/plot_speedups.py`

**Definition of Done**

- One chart: throughput vs N (for each method)
- One table: step time breakdown (forward/backward/comm/compress)

Sweep usage (forward-only):

```bash
python scripts/run_sweep.py --sweep configs/sweeps/N_sweep.yaml
python analysis/aggregate.py --input results/official/sweeps/<tag> --output results/tables/runs.csv
python analysis/plot_speedups.py --input results/tables/runs.csv --out-dir results/plots
```

Plots produced:

- Phase 1 forward/allreduce/layout-fix timing vs N (avg + p50/p95; one line per method)
- Phase 1 forward minus layout-fix timing vs N (avg + p50/p95; optimistic upper bound)
- Phase 1 memory plots vs N (peak allocated memory + weight storage estimates + best-effort actual bytes, GiB + bytes)
- Phase 1 forward avg normalized by weight bytes (ns/byte)
- Phase 0 error metrics vs N (`max_abs_error`, `mean_abs_error`, `max_rel_error`)
- Quality-adjusted speed: `(1/forward_avg_ms) / (1 + mean_abs_error)`

Note: If you run a Phase 1-only sweep, the Phase 0 error plots and quality-adjusted plot will be empty.

## One rule to stay sane

At every milestone, do a vertical slice:

- Implement just enough to run one method through one phase
- Lock correctness
- Then add the next method

## Experimental design

### Target layer

A single giant linear layer:

```text
W in R^(N x N)  (e.g., 4096 or 40960)
X in R^(B x N)  (B = batch/tokens)
Y in R^(B x N)
```

Use bf16 (bf16 + Blackwell). Semi-structured bf16 requires 2D and both dims multiples of 64.

### Block-Shock method (2 GPUs)

Choose complementary masks per 4-wide group (example shown; any complementary pair works):

```text
GPU0 mask per block: 1100
GPU1 mask per block: 0011
```

Construct:

```text
W(0) = W .* M(0)
W(1) = W .* M(1)
```

Each is valid 2:4.

#### Forward

On each GPU `g`:

- Replicate `X` on both GPUs
- Compute partial output: `Y(g) = X W(g)^T` using semi-structured sparse ops
- All-reduce sum: `Y = Y(0) + Y(1)`

PyTorch's semi-structured sparse ops include `torch.mm(dense, sparse)` and `aten.linear.default(dense, sparse, bias)` among others.

#### Backward (conceptual)

Because `Y = sum_g Y(g)`:

```text
grad_X = sum_g grad_Y W(g)
```

Each GPU computes its contribution, then all-reduce sum.

Gradient wrt weights:

```text
grad_W(g) = (grad_Y)^T X
```

Apply the mask so each GPU updates only its half of weights.

### Weight update policy (two variants)

This is where the experiment becomes interesting.

#### Variant T1: dense master shard + compress each step (honest training)

- Each GPU stores `W(g)` as a dense masked tensor (bf16).
- After optimizer update, re-apply the mask (keep 2:4 pattern).
- Re-compress to semi-structured (`to_sparse_semi_structured`) for the next forward.

This explicitly measures whether "compress + sparse GEMM" can beat dense, step after step.

#### Variant T2: frozen weights / forward-only (kernel ceiling)

- No optimizer updates.
- Compress once.
- Pure forward benchmark (and optionally grad wrt input only).

This gives an upper bound on the sparse kernel win without the "compress cost" dominating.

## Baselines

### Baseline A: single-GPU dense

Standard `nn.Linear` (or `X @ W.T`) on one GPU. Purpose: absolute reference for correctness and speed.

### Baseline B: 2-GPU dense tensor parallel (standard)

Pick a standard TP split (two good choices):

**B1) Row-parallel (sum outputs)**
- Shard input features and weight columns
- Each GPU computes partial `Y(g)`
- All-reduce sum outputs

This baseline matches Block-Shock's "sum outputs" comms pattern.

**B2) Column-parallel (gather outputs)**
- Shard output features and weight rows
- All-gather output shards

Include B2 if you want a broader view; B1 is the fairest apples-to-apples.

### Baseline C (optional, recommended): masked split but dense compute

Same split `W = W(0) + W(1)` using the same masks, but do dense GEMM on each GPU (so zeros do not get skipped). This isolates whether the win is really from cuSPARSELt semi-structured kernels.

## Metrics (what you report)

### Correctness

- Forward: max abs error and relative error vs dense baseline
- Backward (if training): compare gradients vs dense (spot checks)

### Performance

For each method:

- Forward time (ms)
- Backward time (ms)
- Optimizer step time (ms) (for training variants)
- End-to-end step time (ms)
- Achieved throughput (samples/s or tokens/s)
- Communication overhead (time spent in all-reduce)

### Memory

- Peak allocated memory per GPU
- Weight storage size (dense vs semi-structured)

### Profiling artifacts

- `torch.profiler` trace (kernel names + NCCL ops)
- Optional: per-kernel time breakdown

## Sizing question: 4096 vs 40960

40960 x 40960 is doable and a great stress test. It is divisible by 64 (good for bf16 semi-structured).

### Dense bf16 weight size

```text
40960^2 = 1,677,721,600 elements
bf16 = 2 bytes -> 3.125 GiB weight tensor
```

### Semi-structured bf16 compression factor

PyTorch documents 2:4 bf16 as compression factor 9/16.

```text
one 2:4 compressed copy ~= 3.125 * 9/16 ~= 1.758 GiB
two complementary copies ~= 3.516 GiB total
```

That is about 12.5% overhead vs dense weight storage.

Training memory is dominated by gradients and optimizer states. Start with:

1) forward-only
2) SGD
3) Adam (if you want to feel pain in 4D)

## Hypotheses (make it falsifiable)

- H1 (kernel ceiling): In forward-only mode, Block-Shock's two-GPU semi-structured method achieves higher throughput than two-GPU dense TP for sufficiently large N and B.
- H2 (training reality): In full training, Block-Shock's advantage shrinks or disappears unless the compress/repack overhead is small relative to GEMM time.
- H3 (ablation): Masked-split dense (Baseline C) performs worse than Block-Shock, proving the gain (if any) is from semi-structured kernels, not just splitting.

## Implementation plan (clean and incremental)

### Phase 0 - correctness microcheck (small N = 4096)

Verify Block-Shock output matches dense exactly (it should if masks are complementary and you preserve all weights).

### Phase 1 - forward-only benchmark

- N sweep: 4096 -> 8192 -> 16384 -> 40960
- Batch sweep: B = 16, 64, 256, 1024 (as memory allows)
- Measure forward time and comm time

### Phase 2 - backward wrt input only

- Freeze weights, enable grad on X, backprop
- Verify grad correctness vs dense
- Time forward + backward

### Phase 3 - full training loop

- SGD first (minimize state)
- Then Adam (realistic)
- Include compress cost explicitly in timing
