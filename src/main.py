### block_shock/src/main.py
## Entrypoint for the experiment runner.
# Loads YAML config stack, initializes distributed, iterates methods, runs phases, logs results.

import argparse
import os
from pathlib import Path

from . import config as config_utils
from . import distributed as dist_utils
from . import logging_utils
from . import orchestrator
from . import workloads
from .methods import block_shock, dense_single, dense_tp, masked_split_dense

#TODO: initialize distributed if requested
#TODO: build workload inputs
#TODO: iterate methods and run the phase pipeline
#TODO: write JSONL results per run


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="block-shock",
        description="Block-Shock experiment runner (scaffold).",
    )
    parser.add_argument("--config", action="append", default=[], help="Base config YAML path.")
    parser.add_argument("--phase", help="Phase config YAML path.")
    parser.add_argument("--method", help="Method config YAML path.")
    parser.add_argument("--workload", help="Workload config YAML path.")
    parser.add_argument("--hardware", help="Hardware config YAML path.")
    parser.add_argument("--sweep", help="Sweep config YAML path (optional).")
    parser.add_argument("--run-id", help="Override run_id (optional).")
    parser.add_argument("--print-config", action="store_true", help="Print resolved config and exit.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config_paths = [Path(p) for p in (args.config or ["configs/base.yaml"])]
    if args.phase:
        config_paths.append(Path(args.phase))
    if args.method:
        config_paths.append(Path(args.method))
    if args.workload:
        config_paths.append(Path(args.workload))
    if args.hardware:
        config_paths.append(Path(args.hardware))
    if args.sweep:
        config_paths.append(Path(args.sweep))

    resolved = config_utils.resolve_config(config_paths, run_id=args.run_id)

    env_world_size = os.environ.get("WORLD_SIZE")
    if env_world_size is not None or int(resolved.get("hardware", {}).get("world_size", 1)) <= 1:
        dist_utils.init_distributed(resolved)
    else:
        print("warning: WORLD_SIZE not set; running in single-process mode", flush=True)

    if dist_utils.is_distributed():
        run_id = resolved["run_id"] if dist_utils.rank() == 0 else ""
        run_id = dist_utils.broadcast_object(run_id, src=0)
        resolved["run_id"] = run_id

    out_dir = Path(resolved.get("logging", {}).get("out_dir", "results/raw"))
    phase_name = str(resolved.get("phase", {}).get("name", "phase"))
    method_name = str(resolved.get("method", {}).get("name", "method"))
    run_dir = out_dir / phase_name / method_name / resolved["run_id"]

    logger = None
    if dist_utils.rank() == 0:
        config_utils.write_config(resolved, run_dir)
        logger = logging_utils.init_logger(resolved, run_dir)
        print(config_utils.config_to_yaml(resolved), end="")

    if args.print_config:
        if dist_utils.is_distributed():
            dist_utils.barrier()
        return

    inputs = workloads.build_inputs(resolved)
    if dist_utils.is_distributed():
        dist_utils.broadcast_tensor(inputs["X"], src=0)
        if inputs.get("T") is not None:
            dist_utils.broadcast_tensor(inputs["T"], src=0)

    method_name = str(resolved.get("method", {}).get("name", "dense_single"))
    method_map = {
        "dense_single": dense_single,
        "dense_tp": dense_tp,
        "block_shock_2gpu": block_shock,
        "masked_split_dense": masked_split_dense,
    }
    method = method_map.get(method_name)
    if method is None:
        raise ValueError(f"Unknown method: {method_name}")

    orchestrator.run_phase(resolved, method, inputs, logger=logger)
    dist_utils.destroy_process_group()


if __name__ == "__main__":
    main()
