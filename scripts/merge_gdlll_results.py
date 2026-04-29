#!/usr/bin/env python3
"""
merge_gdlll_results.py
======================
Merge multiple partial G-DLLL benchmark JSON files into one.

Usage:
    python3 merge_gdlll_results.py part1.json part2.json ... -o merged.json

Each input has the structure:
    {"meta": {...}, "results": {"family": {"dim": {"alg": {...}}}}}

The merge combines all family/dim entries.  If the same family+dim
appears in multiple inputs, the later file wins.
"""

import json
import sys
import re


def load_json_tolerant(path):
    """Load JSON, auto-closing any truncated trailing braces."""
    with open(path) as f:
        raw = f.read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try closing open braces
        depth = raw.count("{") - raw.count("}")
        if depth > 0:
            raw = raw.rstrip().rstrip(",") + "}" * depth
            return json.loads(raw)
        raise


def merge(files, out_path):
    meta = None
    results = {}

    for path in files:
        data = load_json_tolerant(path)
        if meta is None:
            meta = data.get("meta", {})

        for family, dims in data.get("results", {}).items():
            if family not in results:
                results[family] = {}
            for dim, algos in dims.items():
                results[family][dim] = algos

    # Sort families and dims
    sorted_results = {}
    for fam in sorted(results.keys()):
        sorted_results[fam] = {}
        for dim in sorted(results[fam].keys(), key=lambda x: int(x)):
            sorted_results[fam][dim] = results[fam][dim]

    merged = {"meta": meta, "results": sorted_results}

    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Merged {len(files)} files -> {out_path}")
    for fam, dims in sorted_results.items():
        print(f"  {fam}: dims {', '.join(dims.keys())}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "-o" in args:
        idx = args.index("-o")
        out_path = args[idx + 1]
        args = args[:idx] + args[idx + 2 :]
    else:
        out_path = "gdlll_merged.json"

    if not args:
        print("Usage: merge_gdlll_results.py part1.json part2.json ... -o out.json")
        sys.exit(1)

    merge(args, out_path)
