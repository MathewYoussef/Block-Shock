#!/usr/bin/env python3
"""
Benchmark example: Compare performance of different methods.

This example demonstrates how to:
1. Run multiple methods (dense single, dense TP, Block-Shock)
2. Collect timing metrics
3. Compare throughput and communication overhead

Requirements:
- For single GPU methods: 1 GPU with CUDA support
- For multi-GPU methods: 2 GPUs with CUDA support
- PyTorch 2.0+ with CUDA

Usage:
    # Single GPU (dense baseline)
    python examples/benchmark_comparison.py --method dense_single
    
    # Multi-GPU methods
    torchrun --standalone --nproc_per_node=2 examples/benchmark_comparison.py --method dense_tp
    torchrun --standalone --nproc_per_node=2 examples/benchmark_comparison.py --method block_shock
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config as config_utils
from src import distributed as dist_utils
from src import orchestrator
from src import workloads
from src.methods import block_shock, dense_single, dense_tp, masked_split_dense


METHOD_MODULES = {
    "dense_single": dense_single,
    "dense_tp": dense_tp,
    "masked_split_dense": masked_split_dense,
    "block_shock": block_shock,
}


def main():
    """Run benchmark comparison."""
    
    parser = argparse.ArgumentParser(description="Benchmark Block-Shock methods")
    parser.add_argument(
        "--method",
        choices=list(METHOD_MODULES.keys()),
        default="block_shock",
        help="Method to benchmark"
    )
    parser.add_argument(
        "--N",
        type=int,
        default=4096,
        help="Matrix dimension (default: 4096)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (default: 64)"
    )
    args = parser.parse_args()
    
    # Define config paths
    method_file = f"configs/methods/{args.method}.yaml"
    config_paths = [
        Path("configs/base.yaml"),
        Path("configs/phases/phase1_forward.yaml"),
        Path(method_file),
        Path("configs/workloads/gaussian.yaml"),
        Path("configs/hardware/local_2gpu.yaml"),
    ]
    
    # Load and merge configs
    cfg = config_utils.resolve_config(config_paths)
    
    # Override model size if specified
    cfg["model"]["N"] = args.N
    cfg["model"]["B"] = args.batch_size
    
    # Initialize distributed
    dist_utils.init_distributed(cfg)
    
    rank = dist_utils.rank()
    world_size = dist_utils.world_size()
    
    if rank == 0:
        print("=" * 70)
        print(f"BENCHMARK: {args.method}")
        print("=" * 70)
        print(f"GPUs: {world_size}")
        print(f"Matrix size: {args.N}x{args.N}")
        print(f"Batch size: {args.batch_size}")
        print(f"Dtype: {cfg['model']['dtype']}")
        print("=" * 70)
    
    # Generate workload
    inputs = workloads.build_inputs(cfg)
    
    # Build method
    method_module = METHOD_MODULES[args.method]
    method_state = method_module.build(cfg)
    
    # Run Phase 1 forward benchmark
    if rank == 0:
        print("\nRunning forward throughput benchmark...")
        print(f"Warmup iterations: {cfg['phase']['warmup_iters']}")
        print(f"Timed iterations: {cfg['phase']['timed_iters']}")
    
    results = orchestrator.run_phase1(
        cfg=cfg,
        method_state=method_state,
        method_module=method_module,
        inputs=inputs,
    )
    
    # Report results
    if rank == 0:
        timings = results.get("timings_ms", {})
        forward = timings.get("forward", {})
        allreduce = timings.get("allreduce", {})
        
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        
        if forward:
            print(f"Forward avg:  {forward.get('avg_ms', 0):.4f} ms")
            print(f"Forward p50:  {forward.get('p50_ms', 0):.4f} ms")
            print(f"Forward p95:  {forward.get('p95_ms', 0):.4f} ms")
        
        if allreduce and allreduce.get('count', 0) > 0:
            print(f"\nAll-reduce avg: {allreduce.get('avg_ms', 0):.4f} ms")
            print(f"All-reduce p50: {allreduce.get('p50_ms', 0):.4f} ms")
            print(f"All-reduce p95: {allreduce.get('p95_ms', 0):.4f} ms")
        
        # Calculate throughput
        if forward and forward.get('avg_ms', 0) > 0:
            samples_per_sec = 1000 * args.batch_size / forward['avg_ms']
            print(f"\nThroughput: {samples_per_sec:.2f} samples/sec")
        
        print("=" * 70)
    
    # Cleanup
    dist_utils.destroy_process_group()


if __name__ == "__main__":
    main()
