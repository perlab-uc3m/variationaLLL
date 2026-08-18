#!/usr/bin/env python3
"""Validate canonical benchmark output before it is used in the paper."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


MAIN_ALGORITHMS = ["LLL", "Deep-Var", "SS-GG", "Thermal-Adaptive", "G-DLLL"]
MAIN_FAMILIES = ["gaussian", "qary", "goldstein-mayer"]
MAIN_DIMENSIONS = [40, 80, 120, 160]
METRICS = [
    "mean_ops",
    "std_ops",
    "mean_equiv_swaps",
    "std_equiv_swaps",
    "mean_time",
    "std_time",
    "mean_delta0",
    "std_delta0",
    "mean_final_var",
    "std_final_var",
]


def validate_main_results(data: dict, expected_runs: int = 30) -> None:
    meta = data.get("meta", {})
    if meta.get("canonical_gdlll_scan") != "exhaustive":
        raise ValueError("canonical G-DLLL must use an exhaustive scan")
    if meta.get("main_only") != 1:
        raise ValueError("paper results must be produced with --main-only")

    results = data.get("results")
    if not isinstance(results, dict):
        raise ValueError("missing results object")

    for family in MAIN_FAMILIES:
        family_results = results.get(family)
        if not isinstance(family_results, dict):
            raise ValueError(f"missing family {family}")
        for dimension in MAIN_DIMENSIONS:
            cell = family_results.get(str(dimension))
            if not isinstance(cell, dict):
                raise ValueError(f"missing {family} d={dimension}")
            for algorithm in MAIN_ALGORITHMS:
                record = cell.get(algorithm)
                if not isinstance(record, dict):
                    raise ValueError(
                        f"missing {family} d={dimension} {algorithm}"
                    )
                if record.get("completed_runs") != expected_runs:
                    raise ValueError(
                        f"{family} d={dimension} {algorithm}: expected "
                        f"{expected_runs} completed runs, got "
                        f"{record.get('completed_runs')}"
                    )
                if record.get("limit_hits") != 0:
                    raise ValueError(
                        f"{family} d={dimension} {algorithm}: operation limit hit"
                    )
                if record.get("non_lll_outputs") != 0:
                    raise ValueError(
                        f"{family} d={dimension} {algorithm}: non-LLL output"
                    )
                for metric in METRICS:
                    value = record.get(metric)
                    if not isinstance(value, (int, float)) or not math.isfinite(value):
                        raise ValueError(
                            f"{family} d={dimension} {algorithm}: invalid {metric}"
                        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--expected-runs", type=int, default=30)
    args = parser.parse_args()

    with args.path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    validate_main_results(data, args.expected_runs)
    print(f"Validated {args.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
