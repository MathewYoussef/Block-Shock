### block_shock/src/distributed.py
## Distributed helpers: process group init, device assignment, all-reduce.

from __future__ import annotations

import os
import time
from typing import Any, Mapping

try:  # Optional dependency during scaffolding
    import torch  # type: ignore
    import torch.distributed as dist  # type: ignore
except Exception:  # pragma: no cover - allow import without torch
    torch = None
    dist = None

#TODO: add timing hooks for collectives

_AUTOGRAD_ALL_REDUCE = None
if torch is not None:
    try:  # Prefer autograd-aware functional collectives
        import torch.distributed.nn.functional as dist_nn  # type: ignore

        _AUTOGRAD_ALL_REDUCE = dist_nn.all_reduce
    except Exception:
        _AUTOGRAD_ALL_REDUCE = None


def _cuda_sync_if_needed(sync: bool) -> None:
    """Synchronize CUDA if requested and available.
    
    Args:
        sync: Whether to perform synchronization
    """
    if not sync:
        return
    if torch is None:
        return
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def collective_prep(tensor, timing_mode: str = "none"):
    """Prepare tensor for collective operations by ensuring contiguity.
    
    Non-contiguous tensors must be copied before collective operations.
    This function handles the copy and optional timing of the operation.
    
    Args:
        tensor: Input tensor
        timing_mode: Timing mode ('none', 'sync', or 'cuda_events')
        
    Returns:
        Tuple of (ready_tensor, metadata, events, duration_ms):
            - ready_tensor: Contiguous tensor ready for collective
            - metadata: Dict with contiguity info and copy stats
            - events: CUDA events tuple for cuda_events mode, else None
            - duration_ms: Duration in ms for sync mode, else 0.0
            
    Raises:
        RuntimeError: If torch is not available
    """
    if torch is None:
        raise RuntimeError("torch is required for collective_prep")
    was_contig = tensor.is_contiguous()
    meta = {
        "was_contig": bool(was_contig),
        "did_copy": False,
        "bytes": 0,
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "stride": tuple(tensor.stride()),
        "storage_offset": int(tensor.storage_offset()),
    }

    if was_contig:
        return tensor, meta, None, 0.0

    meta["did_copy"] = True
    meta["bytes"] = int(tensor.numel() * tensor.element_size())

    if timing_mode == "cuda_events" and tensor.is_cuda:
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record(torch.cuda.current_stream())
        tensor_ready = tensor.contiguous()
        end_evt.record(torch.cuda.current_stream())
        return tensor_ready, meta, (start_evt, end_evt), None

    sync = timing_mode == "sync" and tensor.is_cuda
    _cuda_sync_if_needed(sync)
    t0 = time.perf_counter()
    tensor_ready = tensor.contiguous()
    _cuda_sync_if_needed(sync)
    duration_ms = (time.perf_counter() - t0) * 1e3
    return tensor_ready, meta, None, duration_ms


def _cfg_value(cfg: Mapping[str, Any], path: list[str], default: Any = None) -> Any:
    """Navigate nested config dictionary by path.
    
    Args:
        cfg: Configuration dictionary
        path: List of keys to traverse
        default: Default value if path not found
        
    Returns:
        Value at path or default
    """
    cur: Any = cfg
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _env_int(name: str, default: int) -> int:
    """Get integer from environment variable with fallback.
    
    Args:
        name: Environment variable name
        default: Default value if not set or invalid
        
    Returns:
        Integer value from environment or default
    """
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def init_distributed(cfg: Mapping[str, Any]) -> None:
    """Initialize distributed process group if world_size > 1.
    
    Reads configuration from:
    - Environment variables: WORLD_SIZE, RANK, LOCAL_RANK
    - Config: hardware.world_size, hardware.backend
    
    Sets CUDA device based on LOCAL_RANK if CUDA is available.
    
    Args:
        cfg: Configuration dictionary
        
    Raises:
        RuntimeError: If torch.distributed unavailable but world_size > 1
    """
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
    """Check if distributed mode is active.
    
    Returns:
        True if torch.distributed is initialized, False otherwise
    """
    return dist is not None and dist.is_initialized()


def rank() -> int:
    """Get current process rank.
    
    Returns:
        Process rank (0 if not distributed)
    """
    if not is_distributed():
        return 0
    return dist.get_rank()


def world_size() -> int:
    """Get total number of processes.
    
    Returns:
        World size (1 if not distributed)
    """
    if not is_distributed():
        return 1
    return dist.get_world_size()


def barrier() -> None:
    """Synchronize all processes at a barrier.
    
    No-op if not in distributed mode.
    """
    if is_distributed():
        dist.barrier()


def allreduce_sum(tensor, allow_autograd: bool = True):
    """All-reduce sum across all processes.
    
    Uses autograd-aware functional collective if available and tensor
    requires gradients.
    
    Args:
        tensor: Tensor to reduce
        allow_autograd: Whether to use autograd-aware collective
        
    Returns:
        Reduced tensor (in-place modification)
    """
    if not is_distributed():
        return tensor
    if allow_autograd and getattr(tensor, "requires_grad", False) and _AUTOGRAD_ALL_REDUCE is not None:
        return _AUTOGRAD_ALL_REDUCE(tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def destroy_process_group() -> None:
    """Destroy distributed process group.
    
    No-op if not in distributed mode.
    """
    if is_distributed():
        dist.destroy_process_group()


def broadcast_tensor(tensor, src: int = 0):
    """Broadcast tensor from source rank to all ranks.
    
    Args:
        tensor: Tensor to broadcast (modified in-place on all ranks)
        src: Source rank (default: 0)
        
    Returns:
        Broadcasted tensor (same object, modified in-place)
    """
    if not is_distributed():
        return tensor
    dist.broadcast(tensor, src=src)
    return tensor


def broadcast_object(obj, src: int = 0):
    """Broadcast Python object from source rank to all ranks.
    
    Args:
        obj: Object to broadcast
        src: Source rank (default: 0)
        
    Returns:
        Broadcasted object
    """
    if not is_distributed():
        return obj
    obj_list = [obj]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]
