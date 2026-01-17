# Block-Shock

A multi-GPU "dense-equivalent" training method using semi-structured 2:4 sparse kernels.

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

Each run writes to `results/raw/<run_id>/` with:

- `config.yaml` (resolved config)
- `env.json` (hardware + software metadata)
- `seed.txt` (seed used for reproducibility)
- `metrics.jsonl` (one JSON record per measurement block)

Aggregated tables and plots are written to `results/tables/` and `results/plots/`.

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

### Reproduce a run (two commands)

```bash
python -m src.main --help
python -m src.main --config configs/base.yaml --phase configs/phases/phase0_correctness.yaml --method configs/methods/dense_single.yaml --workload configs/workloads/gaussian.yaml --hardware configs/hardware/local_2gpu.yaml
```

## Project TODO list (milestones, top-to-bottom)

### Milestone 2 - Timing + metrics discipline (before any methods)

### Milestone 3 - Distributed plumbing (even before TP)

### Milestone 4 - Workloads (data generation + simple losses)

**4.1 Workload generator**

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

### Milestone 5 - Orchestrator (phases as pipelines)

**5.1 Orchestrator**

Create:

- `src/orchestrator.py`

What it must do:

- Execute Phase 0 and Phase 1 pipelines:
  - Phase 0: build -> forward -> compare to reference -> log correctness
  - Phase 1: warmup -> timed forward loops -> log speed

**Definition of Done**

- You can run Phase 0 + Phase 1 using a placeholder method that just calls dense matmul

### Milestone 6 - Baseline A: single-GPU dense (the reference truth)

**6.1 Dense single method**

Create:

- `src/methods/dense_single.py`

What it must do:

- Build weight `W` (NxN)
- Run forward `Y = X @ W.T` (or `F.linear(X, W)`)
- Provide hooks used by orchestrator: `build`, `forward`

**Definition of Done**

- Phase 0 passes (self-reference)
- Phase 1 produces stable timing numbers

### Milestone 7 - Baseline B: 2-GPU dense tensor parallel (row-parallel sum baseline)

**7.1 Dense TP method**

Create:

- `src/methods/dense_tp.py`

What it must do:

- Row-parallel or column-parallel (start row-parallel since Block-Shock sums outputs)
- Forward produces correct result vs dense single
- Uses `allreduce_sum` or gather as required

**Definition of Done**

- Phase 0: outputs match dense single within tolerance for N=4096
- Phase 1: timings logged with comm time separated

### Milestone 8 - Mask system (2:4 generators + validators)

**8.1 Mask generation + validation**

Create:

- `src/sparsity/masks.py`
- `configs/masks/complement_1100_0011.yaml`

What it must do:

- Generate complementary masks for 2-of-4 along the last dimension
- Validate 2:4 compliance for a dense masked matrix

**Definition of Done**

- You can generate `(M0, M1)` where `M0 + M1 == 1` (boolean-wise) for the masked positions

### Milestone 9 - Baseline C (ablation): masked split but dense compute

**9.1 Masked split dense**

Create:

- `src/methods/masked_split_dense.py`

What it must do:

- Two GPUs each hold their masked shard `W0`, `W1`
- Each computes `Yg = X @ Wg.T` using dense GEMM
- Allreduce sum `Y = Y0 + Y1`

**Definition of Done**

- Phase 0: exact match to dense single
- Phase 1: compare it to dense TP and see overhead

### Milestone 10 - Semi-structured compression module (2:4)

**10.1 Semi-structured wrapper**

Create:

- `src/sparsity/semistructured.py`

What it must do:

- Convert masked dense `W_24` -> `W_sparse = to_sparse_semi_structured(W_24)`
- Validate constraints (2D, bf16/fp16, N multiple of 64, CUDA)
- Optional debug: decompress to dense for checking

**Definition of Done**

- You can compress a 4096x4096 masked matrix and confirm sparse matmul output matches dense masked output

### Milestone 11 - Block-Shock forward (Phase 1 focus)

**11.1 Block-Shock method (forward)**

Create:

- `src/methods/block_shock.py`
- `configs/methods/block_shock_2gpu.yaml`

What it must do:

- GPU0: holds `W0_24` -> compressed `W0_sparse`
- GPU1: holds `W1_24` -> compressed `W1_sparse`
- Compute partial outputs with supported sparse ops
- Allreduce sum outputs

**Definition of Done**

- Phase 0: output matches dense single for N=4096
- Phase 1: produces timings + comm breakdown

### Milestone 12 - Phase 2: backward wrt input (weights frozen)

**12.1 Extend orchestrator for Phase 2**

Update:

- `src/orchestrator.py`

What it must do:

- Enable grad on X
- Define a simple loss on Y (sum or MSE)
- Compute dX
- Compare dX against dense single reference

**Definition of Done**

- dX allclose within tolerance for all methods that claim to support backward

### Milestone 13 - Phase 3: full training step (SGD first)

**13.1 Add optimizer module**

Create:

- `src/optim.py` (or integrate into methods cleanly)

What it must do:

- Implement SGD first (lightweight state)
- Optionally Adam later

**13.2 Decide compress cadence**

Config switch options:

- compress once (inference-style)
- compress every step (true dynamic training)
- compress every K steps (hybrid)

**Definition of Done**

- You can run 10 training steps and log:
  - forward time
  - backward time
  - opt step time
  - compress time (explicit)

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
