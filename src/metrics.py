### block_shock/src/metrics.py
## Timing, memory stats, correctness metrics.

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import math
import time
from typing import Dict, Iterable

try:  # Best-effort CUDA sync support
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional dependency during scaffolding
    torch = None

#TODO: add sync_mode support:
#TODO: - "cuda_events" (preferred for perf benchmarking)
#TODO: - "sync" (debug reliable, current default)
#TODO: - "none" (not for benchmarking)

DEFAULT_REGIONS = (
    "build",
    "forward",
    "backward",
    "opt_step",
    "compress",
    "allreduce",
    "total_step",
)


def _cuda_sync_if_needed(sync: bool) -> None:
    if not sync:
        return
    if torch is None:
        return
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * q
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


@dataclass
class TimerRegion:
    name: str
    durations_s: list[float] = field(default_factory=list)
    _start_s: float | None = None

    def start(self, sync: bool = True) -> None:
        if self._start_s is not None:
            raise RuntimeError(f"Timer region '{self.name}' already started.")
        _cuda_sync_if_needed(sync)
        self._start_s = time.perf_counter()

    def stop(self, sync: bool = True) -> None:
        if self._start_s is None:
            raise RuntimeError(f"Timer region '{self.name}' was not started.")
        _cuda_sync_if_needed(sync)
        end_s = time.perf_counter()
        self.durations_s.append(end_s - self._start_s)
        self._start_s = None

    def summary_ms(self) -> Dict[str, float]:
        if not self.durations_s:
            return {
                "count": 0.0,
                "sum_ms": 0.0,
                "avg_ms": 0.0,
                "p50_ms": 0.0,
                "p95_ms": 0.0,
            }
        total_s = sum(self.durations_s)
        return {
            "count": float(len(self.durations_s)),
            "sum_ms": total_s * 1e3,
            "avg_ms": (total_s / len(self.durations_s)) * 1e3,
            "p50_ms": _percentile(self.durations_s, 0.50) * 1e3,
            "p95_ms": _percentile(self.durations_s, 0.95) * 1e3,
        }


class TimerRegistry:
    def __init__(self, regions: Iterable[str] = DEFAULT_REGIONS, sync: bool = True) -> None:
        self._sync = sync
        self._regions: Dict[str, TimerRegion] = {name: TimerRegion(name) for name in regions}

    def region(self, name: str) -> TimerRegion:
        if name not in self._regions:
            self._regions[name] = TimerRegion(name)
        return self._regions[name]

    @contextmanager
    def time(self, name: str):
        region = self.region(name)
        region.start(sync=self._sync)
        try:
            yield
        finally:
            region.stop(sync=self._sync)

    def summary(self) -> Dict[str, Dict[str, float]]:
        return {name: region.summary_ms() for name, region in self._regions.items()}


#TODO: correctness metrics (max abs, relative error)
#TODO: peak memory capture


def record_metrics(results: Dict[str, object], timers: TimerRegistry) -> Dict[str, object]:
    results = dict(results)
    results["timings_ms"] = timers.summary()
    return results
