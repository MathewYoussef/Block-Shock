#!/usr/bin/env python3
### block_shock/scripts/run_sweep.py
## Launch Phase 0/1 sweeps across N and methods using torchrun.

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
import datetime as dt
from typing import Iterable

import yaml


DEFAULT_METHODS = [
    "configs/methods/dense_single.yaml",
    "configs/methods/dense_tp.yaml",
    "configs/methods/masked_split_dense.yaml",
    "configs/methods/block_shock_2gpu.yaml",
]


def _load_sweep_values(path: Path) -> list[int]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    sweep = data.get("sweep", {})
    param = sweep.get("param")
    if param != "model.N":
        raise ValueError("Only sweep.param == model.N is supported for now")
    values = sweep.get("values")
    if not isinstance(values, list) or not values:
        raise ValueError("sweep.values must be a non-empty list")
    return [int(v) for v in values]


def _write_override_n(n: int) -> Path:
    fd, path = tempfile.mkstemp(prefix="block_shock_n_", suffix=".yaml")
    override = {"model": {"N": int(n)}}
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(override, handle, sort_keys=False)
    return Path(path)


def _write_logging_override(tag: str) -> Path:
    fd, path = tempfile.mkstemp(prefix="block_shock_log_", suffix=".yaml")
    override = {"logging": {"out_dir": f"results/official/sweeps/{tag}", "auto_group": False}}
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(override, handle, sort_keys=False)
    return Path(path)


def _torchrun_cmd() -> list[str]:
    torchrun = shutil.which("torchrun")
    if torchrun:
        return [torchrun]
    return [sys.executable, "-m", "torch.distributed.run"]


def _run_command(cmd: list[str], dry_run: bool) -> int:
    print("run_sweep:", " ".join(cmd), flush=True)
    if dry_run:
        return 0
    return subprocess.call(cmd)


def _iter_phases(phases: Iterable[str]) -> list[str]:
    return [p for p in phases if p]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 0/1 sweeps across N and methods.")
    parser.add_argument("--base", default="configs/base.yaml", help="Base config path.")
    parser.add_argument("--official", default="configs/official.yaml", help="Official config path.")
    parser.add_argument("--phase0", default="configs/phases/phase0_correctness.yaml")
    parser.add_argument("--phase1", default="configs/phases/phase1_forward.yaml")
    parser.add_argument("--workload", default="configs/workloads/gaussian.yaml")
    parser.add_argument("--hardware", default="configs/hardware/local_2gpu.yaml")
    parser.add_argument("--sweep", default="configs/sweeps/N_sweep.yaml")
    parser.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    parser.add_argument("--phases", nargs="*", default=None, help="Override phase list.")
    parser.add_argument("--tag", help="Optional output tag (defaults to UTC timestamp).")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()

    sweep_path = Path(args.sweep)
    n_values = _load_sweep_values(sweep_path)
    phase_list = _iter_phases(args.phases) if args.phases else [args.phase0, args.phase1]

    torchrun_cmd = _torchrun_cmd()
    base = Path(args.base)
    official = Path(args.official)
    workload = Path(args.workload)
    hardware = Path(args.hardware)

    tag = args.tag or dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    logging_override = _write_logging_override(tag)

    exit_code = 0
    try:
        for n in n_values:
            override_path = _write_override_n(n)
            try:
                for method_path in args.methods:
                    method = Path(method_path)
                    nproc = 1 if method.name == "dense_single.yaml" else 2
                    for phase_path in phase_list:
                        phase = Path(phase_path)
                        cmd = (
                            torchrun_cmd
                            + ["--standalone", f"--nproc_per_node={nproc}", "-m", "src.main"]
                            + ["--config", str(base), "--config", str(official)]
                            + ["--config", str(logging_override)]
                            + ["--config", str(override_path)]
                            + ["--phase", str(phase)]
                            + ["--method", str(method)]
                            + ["--workload", str(workload)]
                            + ["--hardware", str(hardware)]
                        )
                        code = _run_command(cmd, args.dry_run)
                        if code != 0:
                            exit_code = code
                            if not args.keep_going:
                                return exit_code
            finally:
                if override_path.exists():
                    override_path.unlink()
    finally:
        if logging_override.exists():
            logging_override.unlink()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
