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
