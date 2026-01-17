# Failure note

This Phase 0 dense_tp run failed correctness.

Cause:
- Each rank generated its own input X (no broadcast).
- The all-reduce summed outputs from different X tensors.
- This broke comparison vs the dense reference.

Fix applied:
- Broadcast X (and T if present) from rank 0 in `src/main.py` using `distributed.broadcast_tensor`.
- Rerun Phase 0 after the broadcast fix.
