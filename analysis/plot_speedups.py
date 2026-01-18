### block_shock/analysis/plot_speedups.py
## Plot throughput, step time, and comm overhead.

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    return rows


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _plot_lines(rows: list[dict[str, Any]], y_key: str, out_path: Path, title: str) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        print(f"plot_speedups: matplotlib not available ({exc}); skipping {out_path.name}")
        return

    by_method: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        if row.get("phase") != "phase1_forward":
            continue
        method = row.get("method") or "unknown"
        n = _to_float(row.get("N"))
        y_val = _to_float(row.get(y_key))
        if n is None or y_val is None:
            continue
        by_method.setdefault(method, []).append((n, y_val))

    if not by_method:
        print(f"plot_speedups: no data for {y_key}")
        return

    plt.figure()
    for method, points in by_method.items():
        points = sorted(points, key=lambda p: p[0])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        plt.plot(xs, ys, marker="o", label=method)

    plt.xlabel("N")
    plt.ylabel(y_key)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def _plot_lines_scaled(
    rows: list[dict[str, Any]],
    y_key: str,
    out_path: Path,
    title: str,
    scale: float,
    y_label: str,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        print(f"plot_speedups: matplotlib not available ({exc}); skipping {out_path.name}")
        return

    by_method: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        if row.get("phase") != "phase1_forward":
            continue
        method = row.get("method") or "unknown"
        n = _to_float(row.get("N"))
        y_val = _to_float(row.get(y_key))
        if n is None or y_val is None:
            continue
        by_method.setdefault(method, []).append((n, y_val / scale))

    if not by_method:
        print(f"plot_speedups: no data for {y_key}")
        return

    plt.figure()
    for method, points in by_method.items():
        points = sorted(points, key=lambda p: p[0])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        plt.plot(xs, ys, marker="o", label=method)

    plt.xlabel("N")
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def _plot_lines_bytes(
    rows: list[dict[str, Any]],
    y_key: str,
    out_path: Path,
    title: str,
) -> None:
    _plot_lines(rows, y_key, out_path, title)


def _get_weight_bytes(row: dict[str, Any]) -> float | None:
    for key in ("weight_bytes_total_actual", "weight_bytes_total"):
        val = _to_float(row.get(key))
        if val is not None and val > 0:
            return val
    return None


def _plot_time_per_byte(
    rows: list[dict[str, Any]],
    time_key: str,
    out_path: Path,
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        print(f"plot_speedups: matplotlib not available ({exc}); skipping {out_path.name}")
        return

    by_method: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        if row.get("phase") != "phase1_forward":
            continue
        method = row.get("method") or "unknown"
        n = _to_float(row.get("N"))
        time_ms = _to_float(row.get(time_key))
        weight_bytes = _get_weight_bytes(row)
        if n is None or time_ms is None or weight_bytes is None:
            continue
        ns_per_byte = (time_ms * 1e6) / weight_bytes
        by_method.setdefault(method, []).append((n, ns_per_byte))

    if not by_method:
        print(f"plot_speedups: no data for {time_key} per byte")
        return

    plt.figure()
    for method, points in by_method.items():
        points = sorted(points, key=lambda p: p[0])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        plt.plot(xs, ys, marker="o", label=method)

    plt.xlabel("N")
    plt.ylabel("ns per byte")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def _plot_lines_derived(
    rows: list[dict[str, Any]],
    y_key_a: str,
    y_key_b: str,
    out_path: Path,
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        print(f"plot_speedups: matplotlib not available ({exc}); skipping {out_path.name}")
        return

    by_method: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        if row.get("phase") != "phase1_forward":
            continue
        method = row.get("method") or "unknown"
        n = _to_float(row.get("N"))
        base = _to_float(row.get(y_key_a))
        subtract = _to_float(row.get(y_key_b))
        if n is None or base is None or subtract is None:
            continue
        adjusted = max(base - subtract, 0.0)
        by_method.setdefault(method, []).append((n, adjusted))

    if not by_method:
        print(f"plot_speedups: no data for derived {y_key_a}-{y_key_b}")
        return

    plt.figure()
    for method, points in by_method.items():
        points = sorted(points, key=lambda p: p[0])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        plt.plot(xs, ys, marker="o", label=method)

    plt.xlabel("N")
    plt.ylabel(f"{y_key_a} - {y_key_b}")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def _plot_phase0_errors(rows: list[dict[str, Any]], y_key: str, out_path: Path, title: str) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        print(f"plot_speedups: matplotlib not available ({exc}); skipping {out_path.name}")
        return

    by_method: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        if row.get("phase") != "phase0_correctness":
            continue
        method = row.get("method") or "unknown"
        n = _to_float(row.get("N"))
        y_val = _to_float(row.get(y_key))
        if n is None or y_val is None:
            continue
        by_method.setdefault(method, []).append((n, y_val))

    if not by_method:
        print(f"plot_speedups: no data for {y_key}")
        return

    plt.figure()
    for method, points in by_method.items():
        points = sorted(points, key=lambda p: p[0])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        plt.plot(xs, ys, marker="o", label=method)

    plt.xlabel("N")
    plt.ylabel(y_key)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def _plot_quality_adjusted(rows: list[dict[str, Any]], out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        print(f"plot_speedups: matplotlib not available ({exc}); skipping {out_path.name}")
        return

    phase0_errors: dict[tuple[str, float], float] = {}
    for row in rows:
        if row.get("phase") != "phase0_correctness":
            continue
        method = row.get("method") or "unknown"
        n = _to_float(row.get("N"))
        err = _to_float(row.get("mean_abs_error"))
        if n is None or err is None:
            continue
        phase0_errors[(method, n)] = err

    by_method: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        if row.get("phase") != "phase1_forward":
            continue
        method = row.get("method") or "unknown"
        n = _to_float(row.get("N"))
        forward_ms = _to_float(row.get("forward_avg_ms"))
        if n is None or forward_ms is None:
            continue
        err = phase0_errors.get((method, n))
        if err is None:
            continue
        score = (1.0 / forward_ms) / (1.0 + err)
        by_method.setdefault(method, []).append((n, score))

    if not by_method:
        print("plot_speedups: no data for quality_adjusted_speed")
        return

    plt.figure()
    for method, points in by_method.items():
        points = sorted(points, key=lambda p: p[0])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        plt.plot(xs, ys, marker="o", label=method)

    plt.xlabel("N")
    plt.ylabel("quality_adjusted_speed (1/ms)/(1+mean_abs_error)")
    plt.title("Phase 1 speed normalized by Phase 0 mean_abs_error")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot phase1 forward timing curves.")
    parser.add_argument("--input", default="results/tables/runs.csv", help="Input CSV path.")
    parser.add_argument("--out-dir", default="results/plots", help="Output directory.")
    args = parser.parse_args()

    rows = _load_rows(Path(args.input))
    out_dir = Path(args.out_dir)

    _plot_lines(
        rows,
        "forward_avg_ms",
        out_dir / "phase1_forward_avg_ms.png",
        "Phase 1 forward avg (ms) vs N",
    )
    _plot_lines(
        rows,
        "forward_p50_ms",
        out_dir / "phase1_forward_p50_ms.png",
        "Phase 1 forward p50 (ms) vs N",
    )
    _plot_lines(
        rows,
        "forward_p95_ms",
        out_dir / "phase1_forward_p95_ms.png",
        "Phase 1 forward p95 (ms) vs N",
    )
    _plot_lines(
        rows,
        "allreduce_avg_ms",
        out_dir / "phase1_allreduce_avg_ms.png",
        "Phase 1 allreduce avg (ms) vs N",
    )
    _plot_lines(
        rows,
        "allreduce_p50_ms",
        out_dir / "phase1_allreduce_p50_ms.png",
        "Phase 1 allreduce p50 (ms) vs N",
    )
    _plot_lines(
        rows,
        "allreduce_p95_ms",
        out_dir / "phase1_allreduce_p95_ms.png",
        "Phase 1 allreduce p95 (ms) vs N",
    )
    _plot_lines(
        rows,
        "layout_fix_avg_ms",
        out_dir / "phase1_layout_fix_avg_ms.png",
        "Phase 1 layout-fix avg (ms) vs N",
    )
    _plot_lines(
        rows,
        "layout_fix_p50_ms",
        out_dir / "phase1_layout_fix_p50_ms.png",
        "Phase 1 layout-fix p50 (ms) vs N",
    )
    _plot_lines(
        rows,
        "layout_fix_p95_ms",
        out_dir / "phase1_layout_fix_p95_ms.png",
        "Phase 1 layout-fix p95 (ms) vs N",
    )
    _plot_lines_derived(
        rows,
        "forward_avg_ms",
        "layout_fix_avg_ms",
        out_dir / "phase1_forward_minus_layout_fix_avg_ms.png",
        "Phase 1 forward avg minus layout-fix avg (ms) vs N",
    )
    _plot_lines_derived(
        rows,
        "forward_p50_ms",
        "layout_fix_p50_ms",
        out_dir / "phase1_forward_minus_layout_fix_p50_ms.png",
        "Phase 1 forward p50 minus layout-fix p50 (ms) vs N",
    )
    _plot_lines_derived(
        rows,
        "forward_p95_ms",
        "layout_fix_p95_ms",
        out_dir / "phase1_forward_minus_layout_fix_p95_ms.png",
        "Phase 1 forward p95 minus layout-fix p95 (ms) vs N",
    )
    gib = 1024.0 ** 3
    _plot_lines_scaled(
        rows,
        "memory_peak_bytes",
        out_dir / "phase1_memory_peak_gib.png",
        "Phase 1 peak allocated memory (GiB) vs N",
        gib,
        "GiB",
    )
    _plot_lines_scaled(
        rows,
        "weight_bytes_total",
        out_dir / "phase1_weight_bytes_total_gib.png",
        "Phase 1 total weight bytes (GiB) vs N",
        gib,
        "GiB",
    )
    _plot_lines_scaled(
        rows,
        "weight_bytes_dense",
        out_dir / "phase1_weight_bytes_dense_gib.png",
        "Phase 1 dense weight bytes (GiB) vs N",
        gib,
        "GiB",
    )
    _plot_lines_scaled(
        rows,
        "weight_bytes_sparse_est",
        out_dir / "phase1_weight_bytes_sparse_est_gib.png",
        "Phase 1 sparse weight estimate (GiB) vs N",
        gib,
        "GiB",
    )
    _plot_lines_bytes(
        rows,
        "weight_bytes_total_actual",
        out_dir / "phase1_weight_bytes_total_actual_bytes.png",
        "Phase 1 total weight bytes (actual) vs N",
    )
    _plot_lines_bytes(
        rows,
        "weight_bytes_total",
        out_dir / "phase1_weight_bytes_total_bytes.png",
        "Phase 1 total weight bytes (estimate) vs N",
    )
    _plot_lines_scaled(
        rows,
        "weight_bytes_total_actual",
        out_dir / "phase1_weight_bytes_total_actual_gib.png",
        "Phase 1 total weight bytes (actual GiB) vs N",
        gib,
        "GiB",
    )
    _plot_lines_scaled(
        rows,
        "weight_bytes_dense_actual",
        out_dir / "phase1_weight_bytes_dense_actual_gib.png",
        "Phase 1 dense weight bytes (actual GiB) vs N",
        gib,
        "GiB",
    )
    _plot_lines_scaled(
        rows,
        "weight_bytes_masked_actual",
        out_dir / "phase1_weight_bytes_masked_actual_gib.png",
        "Phase 1 masked weight bytes (actual GiB) vs N",
        gib,
        "GiB",
    )
    _plot_lines_scaled(
        rows,
        "weight_bytes_shard_actual",
        out_dir / "phase1_weight_bytes_shard_actual_gib.png",
        "Phase 1 shard weight bytes (actual GiB) vs N",
        gib,
        "GiB",
    )
    _plot_lines_scaled(
        rows,
        "weight_bytes_sparse_actual",
        out_dir / "phase1_weight_bytes_sparse_actual_gib.png",
        "Phase 1 sparse weight bytes (actual GiB) vs N",
        gib,
        "GiB",
    )
    _plot_time_per_byte(
        rows,
        "forward_avg_ms",
        out_dir / "phase1_forward_avg_ns_per_byte.png",
        "Phase 1 forward avg normalized by weight bytes (ns/byte) vs N",
    )
    _plot_phase0_errors(
        rows,
        "max_abs_error",
        out_dir / "phase0_max_abs_error.png",
        "Phase 0 max_abs_error vs N",
    )
    _plot_phase0_errors(
        rows,
        "mean_abs_error",
        out_dir / "phase0_mean_abs_error.png",
        "Phase 0 mean_abs_error vs N",
    )
    _plot_phase0_errors(
        rows,
        "max_rel_error",
        out_dir / "phase0_max_rel_error.png",
        "Phase 0 max_rel_error vs N",
    )
    _plot_quality_adjusted(rows, out_dir / "phase1_quality_adjusted_speed.png")


if __name__ == "__main__":
    main()
