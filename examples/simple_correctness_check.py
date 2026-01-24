#!/usr/bin/env python3
"""
Simple example: Run Phase 0 correctness check with Block-Shock method.

This example demonstrates how to:
1. Load and merge configuration files
2. Initialize distributed training
3. Generate synthetic workload
4. Run a correctness check comparing Block-Shock to dense baseline

Requirements:
- 2 GPUs with CUDA support
- PyTorch 2.0+ with CUDA
- Run with: torchrun --standalone --nproc_per_node=2 examples/simple_correctness_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config as config_utils
from src import distributed as dist_utils
from src import logging_utils
from src import orchestrator
from src import workloads
from src.methods import block_shock


def main():
    """Run a simple Block-Shock correctness check."""
    
    # Define config paths
    config_paths = [
        Path("configs/base.yaml"),
        Path("configs/phases/phase0_correctness.yaml"),
        Path("configs/methods/block_shock_2gpu.yaml"),
        Path("configs/workloads/gaussian.yaml"),
        Path("configs/hardware/local_2gpu.yaml"),
    ]
    
    # Load and merge configs
    print("Loading configuration...")
    cfg = config_utils.resolve_config(config_paths)
    
    # Initialize distributed
    print("Initializing distributed...")
    dist_utils.init_distributed(cfg)
    
    rank = dist_utils.rank()
    world_size = dist_utils.world_size()
    
    if rank == 0:
        print(f"Running on {world_size} GPU(s)")
        print(f"Model size: {cfg['model']['N']}x{cfg['model']['N']}")
        print(f"Batch size: {cfg['model']['B']}")
        print(f"Dtype: {cfg['model']['dtype']}")
    
    # Setup logging
    run_dir = logging_utils.get_run_dir(cfg)
    if rank == 0:
        logging_utils.setup_run_dir(cfg, run_dir)
        print(f"Run directory: {run_dir}")
    
    dist_utils.barrier()
    
    # Generate workload
    if rank == 0:
        print("\nGenerating workload...")
    inputs = workloads.build_inputs(cfg)
    
    # Build method
    if rank == 0:
        print("Building Block-Shock method...")
    method_state = block_shock.build(cfg)
    
    # Run Phase 0 correctness check
    if rank == 0:
        print("\nRunning Phase 0 correctness check...")
    
    results = orchestrator.run_phase0(
        cfg=cfg,
        method_state=method_state,
        method_module=block_shock,
        inputs=inputs,
    )
    
    # Report results
    if rank == 0:
        passed = results.get("passed", False)
        max_abs_error = results.get("max_abs_error", float('inf'))
        max_rel_error = results.get("max_rel_error", float('inf'))
        
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Passed: {passed}")
        print(f"Max absolute error: {max_abs_error:.6e}")
        print(f"Max relative error: {max_rel_error:.6e}")
        
        if passed:
            print("\n✓ Block-Shock correctness check PASSED!")
        else:
            print("\n✗ Block-Shock correctness check FAILED!")
        print("=" * 60)
        
        # Write metrics
        logging_utils.write_metrics(results, run_dir)
        print(f"\nMetrics written to: {run_dir / 'metrics.jsonl'}")
    
    # Cleanup
    dist_utils.destroy_process_group()


if __name__ == "__main__":
    main()
