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


def _cuda_events_available() -> bool:
    return torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available()


@dataclass
class TimerRegion:
    name: str
    mode: str = "sync"
    durations_s: list[float] = field(default_factory=list)
    event_pairs: list[tuple[object, object]] = field(default_factory=list)
    _start_s: float | None = None
    _start_event: object | None = None

    def start(self) -> None:
        if self._start_s is not None or self._start_event is not None:
            raise RuntimeError(f"Timer region '{self.name}' already started.")

        if self.mode == "cuda_events" and _cuda_events_available():
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record(torch.cuda.current_stream())
            return

        sync = self.mode == "sync"
        _cuda_sync_if_needed(sync)
        self._start_s = time.perf_counter()

    def stop(self) -> None:
        if self._start_s is None and self._start_event is None:
            raise RuntimeError(f"Timer region '{self.name}' was not started.")

        if self.mode == "cuda_events" and _cuda_events_available() and self._start_event is not None:
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record(torch.cuda.current_stream())
            self.event_pairs.append((self._start_event, end_event))
            self._start_event = None
            return

        sync = self.mode == "sync"
        _cuda_sync_if_needed(sync)
        end_s = time.perf_counter()
        self.durations_s.append(end_s - self._start_s)
        self._start_s = None

    def _durations_s(self) -> list[float]:
        if self.event_pairs:
            durations = []
            for start_event, end_event in self.event_pairs:
                ms = start_event.elapsed_time(end_event)
                durations.append(ms / 1e3)
            return durations
        return self.durations_s

    def summary_ms(self) -> Dict[str, float]:
        durations = self._durations_s()
        if not durations:
            return {
                "count": 0.0,
                "sum_ms": 0.0,
                "avg_ms": 0.0,
                "p50_ms": 0.0,
                "p95_ms": 0.0,
            }
        total_s = sum(durations)
        return {
            "count": float(len(durations)),
            "sum_ms": total_s * 1e3,
            "avg_ms": (total_s / len(durations)) * 1e3,
            "p50_ms": _percentile(durations, 0.50) * 1e3,
            "p95_ms": _percentile(durations, 0.95) * 1e3,
        }


class TimerRegistry:
    def __init__(
        self,
        regions: Iterable[str] = DEFAULT_REGIONS,
        sync: bool | None = True,
        sync_mode: str | None = None,
    ) -> None:
        if sync_mode is None:
            self._sync_mode = "sync" if sync else "none"
        else:
            self._sync_mode = sync_mode
        self._regions: Dict[str, TimerRegion] = {
            name: TimerRegion(name, mode=self._sync_mode) for name in regions
        }

    def region(self, name: str) -> TimerRegion:
        if name not in self._regions:
            self._regions[name] = TimerRegion(name, mode=self._sync_mode)
        return self._regions[name]

    @contextmanager
    def time(self, name: str):
        region = self.region(name)
        region.start()
        try:
            yield
        finally:
            region.stop()

    def summary(self) -> Dict[str, Dict[str, float]]:
        if self._sync_mode == "cuda_events" and _cuda_events_available():
            torch.cuda.synchronize()
        return {name: region.summary_ms() for name, region in self._regions.items()}


#TODO: correctness metrics (max abs, relative error)
#TODO: peak memory capture


def record_metrics(results: Dict[str, object], timers: TimerRegistry) -> Dict[str, object]:
    results = dict(results)
    results["timings_ms"] = timers.summary()
    return results


def summarize_samples_ms(samples_ms: list[float]) -> Dict[str, float]:
    if not samples_ms:
        return {
            "count": 0.0,
            "sum_ms": 0.0,
            "avg_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
        }
    total = sum(samples_ms)
    return {
        "count": float(len(samples_ms)),
        "sum_ms": total,
        "avg_ms": total / len(samples_ms),
        "p50_ms": _percentile(samples_ms, 0.50),
        "p95_ms": _percentile(samples_ms, 0.95),
    }
