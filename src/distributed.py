### block_shock/src/distributed.py
## Distributed helpers: process group init, device assignment, all-reduce.

from __future__ import annotations

import os
from typing import Any, Mapping

try:  # Optional dependency during scaffolding
    import torch  # type: ignore
    import torch.distributed as dist  # type: ignore
except Exception:  # pragma: no cover - allow import without torch
    torch = None
    dist = None

#TODO: add timing hooks for collectives


def _cfg_value(cfg: Mapping[str, Any], path: list[str], default: Any = None) -> Any:
    cur: Any = cfg
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def init_distributed(cfg: Mapping[str, Any]) -> None:
    world_size = _env_int("WORLD_SIZE", _cfg_value(cfg, ["hardware", "world_size"], 1))
    if world_size <= 1:
        return
    if torch is None or dist is None:
        raise RuntimeError("torch.distributed unavailable but world_size > 1")
    if dist.is_initialized():
        return

    backend = _cfg_value(cfg, ["hardware", "backend"], "nccl")
    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", rank)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def is_distributed() -> bool:
    return dist is not None and dist.is_initialized()


def rank() -> int:
    if not is_distributed():
        return 0
    return dist.get_rank()


def world_size() -> int:
    if not is_distributed():
        return 1
    return dist.get_world_size()


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def allreduce_sum(tensor):
    if not is_distributed():
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def destroy_process_group() -> None:
    if is_distributed():
        dist.destroy_process_group()


def broadcast_tensor(tensor, src: int = 0):
    if not is_distributed():
        return tensor
    dist.broadcast(tensor, src=src)
    return tensor


def broadcast_object(obj, src: int = 0):
    if not is_distributed():
        return obj
    obj_list = [obj]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]
