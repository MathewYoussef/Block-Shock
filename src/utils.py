### block_shock/src/utils.py
## Misc utilities.

try:  # Optional dependency during scaffolding
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None

#TODO: set seeds for reproducibility


def set_seed(_seed: int) -> None:
    #TODO: seed python, numpy, torch
    raise NotImplementedError("Scaffold only: implement seed setting.")


def nudge_zeros(tensor, chunk_elems: int = 16_777_216):
    if torch is None:
        raise RuntimeError("torch is required for nudge_zeros")
    if not torch.is_floating_point(tensor):
        return tensor
    eps = torch.finfo(tensor.dtype).eps
    flat = tensor.view(-1)
    total = flat.numel()
    if chunk_elems <= 0:
        chunk_elems = total
    with torch.no_grad():
        for start in range(0, total, chunk_elems):
            chunk = flat[start : start + chunk_elems]
            mask = chunk == 0
            if torch.any(mask):
                chunk[mask] = eps
    return tensor


def _storage_nbytes(tensor) -> int:
    if torch is None:
        return 0
    try:
        return int(tensor.untyped_storage().nbytes())
    except Exception:
        pass
    try:
        return int(tensor.storage().nbytes())
    except Exception:
        pass
    try:
        return int(tensor.numel() * tensor.element_size())
    except Exception:
        return 0


def tensor_storage_nbytes(obj) -> int:
    if obj is None or torch is None:
        return 0
    if torch.is_tensor(obj):
        size = _storage_nbytes(obj)
        if size:
            return size
    for attr in ("packed", "_packed", "values", "_values"):
        inner = getattr(obj, attr, None)
        if torch.is_tensor(inner):
            size = _storage_nbytes(inner)
            if size:
                return size
    return 0
