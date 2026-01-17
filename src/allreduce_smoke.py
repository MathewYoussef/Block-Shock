### block_shock/src/allreduce_smoke.py
## Tiny distributed allreduce smoke test.

from __future__ import annotations

import os

try:
    import torch  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency during scaffolding
    torch = None
    _torch_error = exc
else:
    _torch_error = None

from .distributed import allreduce_sum, destroy_process_group, init_distributed, rank, world_size


def main() -> None:
    if torch is None:
        print(f"torch import failed: {_torch_error}")
        return

    cfg = {
        "hardware": {
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "backend": os.environ.get("BACKEND", "nccl"),
        }
    }
    init_distributed(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    value = torch.tensor([rank()], dtype=torch.float32, device=device)
    allreduce_sum(value)

    print(f"rank={rank()} world_size={world_size()} allreduce_sum={value.item()}")
    destroy_process_group()


if __name__ == "__main__":
    main()
