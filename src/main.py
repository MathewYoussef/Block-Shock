### block_shock/src/main.py
## Entrypoint for the experiment runner.
# Loads YAML config stack, initializes distributed, iterates methods, runs phases, logs results.

import argparse
from pathlib import Path

from . import config as config_utils
from . import logging_utils

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
    out_dir = Path(resolved.get("logging", {}).get("out_dir", "results/raw"))
    run_dir = out_dir / resolved["run_id"]
    config_utils.write_config(resolved, run_dir)
    logging_utils.init_logger(resolved, run_dir)
    print(config_utils.config_to_yaml(resolved), end="")

    if args.print_config:
        return

    #TODO: run the experiment once orchestration is implemented


if __name__ == "__main__":
    main()
