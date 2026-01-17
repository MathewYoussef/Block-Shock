### block_shock/src/main.py
## Entrypoint for the experiment runner.
# Loads YAML config stack, initializes distributed, iterates methods, runs phases, logs results.

import argparse

#TODO: implement YAML merge stack (base + phase + method + workload + hardware + sweeps)
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
    #TODO: parse args, load config, run experiment
    _ = build_arg_parser().parse_args()
    raise NotImplementedError("Scaffold only: implement main entrypoint.")


if __name__ == "__main__":
    main()
