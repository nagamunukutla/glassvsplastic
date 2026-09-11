#!/usr/bin/env python3
"""Union every deep-result file under an out dir into one ``deep_results.csv``.

Why this exists: training runs write a per-run accumulator into ``<out>/deep/`` and a curated copy
to ``<out>/deep_results.csv``. If a later run ever truncates the curated copy, the per-run
accumulators are still on disk — this tool rebuilds the union from those, keyed on
(model, variant, img_size, epochs, split) so a re-run replaces its own row and nothing else.

It adds no numbers: every field comes from a file on disk. ``variant`` is recovered from the
``notes`` field when a row predates that column.

    python scripts/merge_deep_results.py --out results/real
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from typing import Dict, List, Tuple


def key(r: dict) -> Tuple:
    return (r.get("model", ""), r.get("variant", ""), str(r.get("img_size", "")),
            str(r.get("epochs", "")), r.get("split", ""))


def variant_from_notes(notes: str) -> str:
    n = (notes or "")
    for v in ("object_crop", "full_frame"):
        if v in n:
            return v
    return ""


def merge(out_dir: str) -> str:
    sources = sorted(set(
        glob.glob(os.path.join(out_dir, "deep", "**", "deep_results.csv"), recursive=True) +
        glob.glob(os.path.join(out_dir, "deep_*", "**", "deep_results.csv"), recursive=True) +
        ([os.path.join(out_dir, "deep_results.csv")]
         if os.path.exists(os.path.join(out_dir, "deep_results.csv")) else [])
    ))
    merged: Dict[Tuple, dict] = {}
    for src in sources:
        with open(src, newline="") as fh:
            for r in csv.DictReader(fh):
                r = {k: v for k, v in r.items() if v not in (None, "")}
                if not r.get("variant"):
                    r["variant"] = variant_from_notes(r.get("notes", ""))
                k = key(r)
                prev = merged.get(k)
                if prev is None or len(r) > len(prev):   # prefer the richer row
                    merged[k] = r
    # Fill wall_seconds from the durable per-run records when the CSV row lost it.
    for run_file in glob.glob(os.path.join(out_dir, "deep", "run_*.json")):
        try:
            with open(run_file) as fh:
                run = json.load(fh)
        except Exception:
            continue
        k = (run.get("model", ""), run.get("variant", ""), str(run.get("img_size", "")),
             str(run.get("epochs", "")), "test")
        if k in merged and not merged[k].get("wall_seconds"):
            merged[k]["wall_seconds"] = str(run["wall_seconds"])

    rows = [merged[k] for k in sorted(merged, key=lambda k: (k[0], k[1], k[2], k[3], k[4]))]
    if not rows:
        raise SystemExit(f"no deep results found under {out_dir}")
    cols: List[str] = []
    for r in rows:
        for c in r:
            if c not in cols:
                cols.append(c)
    dst = os.path.join(out_dir, "deep_results.csv")
    with open(dst, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"merged {len(sources)} source file(s) -> {dst} ({len(rows)} rows)")
    for r in rows:
        print(f"  {r.get('model',''):24s} {r.get('variant',''):12s} "
              f"{float(r['balanced_accuracy']):.4f}")
    return dst


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/real")
    a = ap.parse_args()
    merge(a.out)
