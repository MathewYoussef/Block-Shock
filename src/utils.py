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


def nudge_zeros(tensor):
    if torch is None:
        raise RuntimeError("torch is required for nudge_zeros")
    if not torch.is_floating_point(tensor):
        return tensor
    zero_mask = tensor == 0
    if torch.any(zero_mask):
        eps = torch.finfo(tensor.dtype).eps
        tensor = tensor + zero_mask.to(dtype=tensor.dtype) * eps
    return tensor
