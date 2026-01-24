# Examples

This directory contains example scripts demonstrating how to use Block-Shock.

## Available Examples

### 1. Simple Correctness Check (`simple_correctness_check.py`)

Demonstrates how to run a Phase 0 correctness check comparing Block-Shock against a dense baseline.

**Usage:**
```bash
torchrun --standalone --nproc_per_node=2 examples/simple_correctness_check.py
```

**What it does:**
- Loads and merges configuration files
- Initializes 2-GPU distributed training
- Generates synthetic Gaussian workload
- Runs Block-Shock forward pass
- Compares output against dense reference
- Reports pass/fail with error metrics

**Output:**
- Console output with pass/fail status
- Error metrics (max absolute, max relative)
- Metrics written to `results/raw/<run_id>/metrics.jsonl`

---

### 2. Benchmark Comparison (`benchmark_comparison.py`)

Compare performance of different methods with configurable matrix sizes.

**Usage:**

Single GPU (dense baseline):
```bash
python examples/benchmark_comparison.py --method dense_single --N 4096 --batch-size 64
```

Multi-GPU methods:
```bash
# Dense tensor parallel
torchrun --standalone --nproc_per_node=2 examples/benchmark_comparison.py --method dense_tp --N 8192

# Masked split dense (ablation)
torchrun --standalone --nproc_per_node=2 examples/benchmark_comparison.py --method masked_split_dense

# Block-Shock
torchrun --standalone --nproc_per_node=2 examples/benchmark_comparison.py --method block_shock --N 16384
```

**What it does:**
- Runs Phase 1 forward-only throughput benchmark
- Measures forward pass timing (avg, p50, p95)
- Tracks communication overhead for multi-GPU methods
- Calculates samples/second throughput

**Output:**
- Forward pass timing statistics
- All-reduce communication timing (if applicable)
- Throughput (samples/second)

---

## Command-Line Options

### `benchmark_comparison.py`

- `--method`: Method to benchmark
  - Choices: `dense_single`, `dense_tp`, `masked_split_dense`, `block_shock`
  - Default: `block_shock`
  
- `--N`: Matrix dimension (square matrix NxN)
  - Default: 4096
  - Must be multiple of 64 for bf16/fp16
  
- `--batch-size`: Batch size
  - Default: 64

---

## Requirements

- Python 3.8+
- PyTorch 2.0+ with CUDA support
- For single-GPU examples: 1 NVIDIA GPU
- For multi-GPU examples: 2 NVIDIA GPUs with Tensor Core support
- See main [README.md](../README.md) for full installation instructions

---

## Tips

1. **Start small**: Begin with smaller matrix sizes (N=4096) to verify setup
2. **Monitor memory**: Use `nvidia-smi` to track GPU memory usage
3. **Timing modes**: Examples use default timing from configs; modify config files for different timing modes
4. **Custom configs**: Create your own config files in `configs/` and reference them in examples

---

## Next Steps

After running these examples:

1. Explore configuration options in `configs/` directory
2. Run parameter sweeps with `scripts/run_sweep.py`
3. Analyze results with `analysis/aggregate.py` and `analysis/plot_speedups.py`
4. Contribute your own examples or improvements!

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines on contributing.
