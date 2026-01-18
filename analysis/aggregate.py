### block_shock/analysis/aggregate.py
## Aggregate JSONL runs into summary CSVs.

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _flatten_timings(timings: dict[str, Any], row: dict[str, Any]) -> None:
    for region, stats in timings.items():
        if not isinstance(stats, dict):
            continue
        for stat, value in stats.items():
            row[f"{region}_{stat}"] = value


def _load_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for metrics_path in root.rglob("metrics.jsonl"):
        with metrics_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                row: dict[str, Any] = dict(record)
                timings = record.get("timings_ms")
                if isinstance(timings, dict):
                    _flatten_timings(timings, row)
                records.append(row)
    return records


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate JSONL runs into CSV tables.")
    parser.add_argument("--input", default="results/official", help="Root results directory.")
    parser.add_argument("--output", default="results/tables/runs.csv", help="Output CSV path.")
    args = parser.parse_args()

    root = Path(args.input)
    rows = _load_records(root)
    _write_csv(Path(args.output), rows)


if __name__ == "__main__":
    main()
