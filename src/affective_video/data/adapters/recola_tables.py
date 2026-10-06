"""Small dense numeric CSV/ARFF readers and timestamp diagnostics, without alignment."""

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from affective_video.data.adapters.recola_schema import AnnotationSpec, TableSpec


@dataclass(frozen=True)
class ParsedTrace:
    timestamps: tuple[float, ...]
    values: tuple[float | None, ...]
    timing: dict
    statistics: dict


def timestamp_audit(timestamps: tuple[float, ...], *, gap_factor: float = 5.0) -> dict:
    times = np.asarray(timestamps, dtype=np.float64)
    finite = np.isfinite(times)
    deltas = np.diff(times)
    positive = deltas[np.isfinite(deltas) & (deltas > 0)]
    median = float(np.median(positive)) if len(positive) else None
    return {
        "count": len(times),
        "negative_count": int(np.sum(times < 0)),
        "nonfinite_count": int(np.sum(~finite)),
        "duplicate_count": len(times) - len(set(timestamps)),
        "nonincreasing_intervals": int(np.sum(deltas <= 0)),
        "monotonic": bool(np.all(finite) and np.all(deltas > 0)),
        "start": float(times[0]) if len(times) and finite[0] else None,
        "end": float(times[-1]) if len(times) and finite[-1] else None,
        "median_interval": median,
        "frequency_hz": 1.0 / median if median else None,
        "irregular_intervals": int(np.sum(~np.isclose(deltas, median, rtol=0.01, atol=1e-8)))
        if median
        else 0,
        "large_gap_count": int(np.sum(deltas > median * gap_factor)) if median else 0,
    }


def value_statistics(values: tuple[float | None, ...]) -> dict:
    valid = np.asarray([value for value in values if value is not None], dtype=np.float64)
    result: dict = {
        "trace_length": len(values),
        "valid_count": len(valid),
        "missing_count": len(values) - len(valid),
        "missing_fraction": 1 - len(valid) / len(values) if values else None,
    }
    for name in ("min", "max", "mean", "median", "std"):
        result[name] = float(getattr(np, name)(valid)) if len(valid) else None
    result["quantiles"] = (
        {str(q): float(np.quantile(valid, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}
        if len(valid)
        else {}
    )
    return result


def _read_table(path: Path, spec: TableSpec) -> tuple[list[str], list[list[str]]]:
    with path.open(encoding=spec.encoding, newline="") as handle:
        if spec.format == "csv":
            reader = csv.reader(handle, delimiter=spec.delimiter)
            try:
                header = next(reader)
            except StopIteration as error:
                raise ValueError("empty table") from error
            rows = list(reader)
        else:
            header, rows = [], []
            in_data = False
            for line in handle:
                line = line.strip()
                if not line or line.startswith("%"):
                    continue
                if in_data:
                    if line.startswith("{"):
                        raise ValueError("sparse ARFF is unsupported; review this release format")
                    rows.append(next(csv.reader([line], delimiter=",")))
                elif line.lower().startswith("@attribute"):
                    match = re.fullmatch(
                        r"@attribute\s+(?:'([^']+)'|\"([^\"]+)\"|(\S+))\s+(\S+)",
                        line,
                        flags=re.IGNORECASE,
                    )
                    if not match or match.group(4).lower() not in ("numeric", "real", "integer"):
                        raise ValueError("only dense numeric ARFF attributes are supported")
                    header.append(next(group for group in match.groups()[:3] if group is not None))
                elif line.lower() == "@data":
                    in_data = True
                elif not line.lower().startswith("@relation"):
                    raise ValueError(f"unknown ARFF header: {line}")
            if not in_data:
                raise ValueError("ARFF @data section missing")
    header = [item.strip() for item in header]
    if len(set(header)) != len(header):
        raise ValueError("duplicate table columns")
    if not header or any(len(row) != len(header) for row in rows):
        raise ValueError("inconsistent column count")
    return header, rows


def parse_trace(path: Path, spec: TableSpec) -> ParsedTrace:
    header, rows = _read_table(path, spec)
    if spec.timestamp_column not in header:
        raise ValueError(f"timestamp column {spec.timestamp_column!r} not found")
    time_index = header.index(spec.timestamp_column)
    value_index = None
    if isinstance(spec, AnnotationSpec):
        if spec.value_column not in header:
            raise ValueError(f"value column {spec.value_column!r} not found")
        value_index = header.index(spec.value_column)
    times, values = [], []
    for row_number, row in enumerate(rows, start=2):
        try:
            timestamp = float(row[time_index])
            if not math.isfinite(timestamp):
                raise ValueError("nonfinite timestamp")
            times.append(timestamp / 1000 if spec.timestamp_unit == "milliseconds" else timestamp)
            if value_index is not None:
                token = row[value_index].strip()
                value = None if token in spec.missing_tokens else float(token)
                if value is not None and not math.isfinite(value):
                    raise ValueError("nonfinite value not declared as a missing token")
                values.append(value)
        except ValueError as error:
            raise ValueError(f"row {row_number}: {error}") from error
    return ParsedTrace(
        tuple(times), tuple(values), timestamp_audit(tuple(times)), value_statistics(tuple(values))
    )
