#!/usr/bin/env python3
"""Convert public waste-classification datasets into this repository's manifest schema.

Two layouts are supported:

1. ``--layout splitfiles``  (TrashNet)
   The dataset ships official split files in the form ``<filename> <class_index>``. Two variants
   exist and they use *different* index bases -- this is a real trap:

       one-indexed-files-notrash_*.txt   1=glass 2=paper 3=cardboard 4=plastic 5=metal 6=trash
       zero-indexed-files*.txt           0=glass 1=paper 2=cardboard 3=plastic 4=metal 5=trash

   Both are handled via ``--index-base``. Getting this wrong silently relabels plastic as metal,
   so the script prints the class histogram it derived and refuses to proceed if the requested
   class names are absent.

2. ``--layout folders``  (RealWaste, TACO-style dumps, your own data)
   One directory per class; the script makes a stratified, object-level split.

It also estimates the **background type** of every image from its border ring (white studio \
backdrop vs brown cardboard vs other). That single derived column enables the most important
real-data experiment in ``results/REAL_DATA_RESULTS.md``: how much of a benchmark's glass-vs-plastic
accuracy is carried by the scene rather than the object.

Usage
-----
    python scripts/prepare_real_dataset.py --layout splitfiles \
        --root data/raw/trashnet --images dataset-resized --index-base 1 \
        --splits one-indexed-files-notrash_{train,val,test} \
        --positive glass --negative plastic --out data/raw/trashnet

    python scripts/prepare_real_dataset.py --layout folders \
        --root data/raw/realwaste --images RealWaste \
        --positive Glass --negative Plastic --out data/raw/realwaste
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from collections import Counter
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# The class order used by TrashNet's official split files (see module docstring).
TRASHNET_CLASSES = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# --------------------------------------------------------------------------------------
# derived metadata
# --------------------------------------------------------------------------------------


def background_type(img: np.ndarray) -> Tuple[str, Dict[str, float]]:
    """Classify the scene backdrop from the image border ring (no object required).

    Returns one of ``white`` / ``cardboard`` / ``grey`` / ``colour`` plus the statistics used,
    so the decision is auditable rather than a black box.
    """
    h, w = img.shape[:2]
    m = max(3, int(min(h, w) * 0.10))
    ring = np.concatenate([
        img[:m].reshape(-1, 3), img[-m:].reshape(-1, 3),
        img[:, :m].reshape(-1, 3), img[:, -m:].reshape(-1, 3)]).astype(np.float32)
    b, g, r = ring[:, 0].mean(), ring[:, 1].mean(), ring[:, 2].mean()
    brightness = float(ring.mean())
    hsv = cv2.cvtColor(ring.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    sat = float(hsv[:, 1].mean())
    std = float(ring.std())
    stats = {"bg_brightness": round(brightness, 1), "bg_saturation": round(sat, 1),
             "bg_std": round(std, 1), "bg_b": round(float(b), 1), "bg_g": round(float(g), 1),
             "bg_r": round(float(r), 1)}
    if brightness > 232 and sat < 28:
        kind = "white"
    elif sat > 28 and r > g > b and brightness < 232:
        kind = "cardboard"
    elif sat < 45 and 120 < brightness <= 232:
        kind = "grey"
    else:
        kind = "colour"
    stats["bg_kind"] = kind
    return kind, stats


def object_stats(img: np.ndarray) -> Dict[str, float]:
    """Crude object extent from the same estimator the classical pipeline uses."""
    h, w = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    m = max(2, int(min(h, w) * 0.06))
    border = np.concatenate([lab[:m].reshape(-1, 3), lab[-m:].reshape(-1, 3),
                             lab[:, :m].reshape(-1, 3), lab[:, -m:].reshape(-1, 3)])
    mu, sd = border.mean(0), border.std(0) + 1e-3
    dist = np.sqrt((((lab - mu) / sd) ** 2).sum(-1))
    dist = cv2.GaussianBlur(dist.astype(np.float32), (0, 0), 1.5)
    dn = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(dn, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    n, _lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = 1 + int(np.argmax(areas))
        mask = (_lab == keep).astype(np.uint8) * 255
    frac = float((mask > 0).mean())
    if frac < 0.03 or frac > 0.95:
        frac = float("nan")  # estimator failed; keep it explicit rather than plausible
    return {"obj_area_frac": round(frac, 4) if frac == frac else -1.0}


# --------------------------------------------------------------------------------------
# writers
# --------------------------------------------------------------------------------------


def _write_dataset(root: str, images_subdir: str, rows: List[dict], out_dir: str,
                   positive: str, negative: str, desc: str) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    out: Dict[str, str] = {}
    for split in sorted({r["split"] for r in rows}):
        sub = [r for r in rows if r["split"] == split]
        path = os.path.join(out_dir, f"manifest_{split}.csv")
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(sub[0].keys()))
            w.writeheader()
            w.writerows(sub)
        n_pos = sum(int(r["label"]) for r in sub)
        out[split] = path
        print(f"  {split:>6}: {len(sub):5d} images  "
              f"({positive}={n_pos}, {negative}={len(sub) - n_pos})  -> {path}")
    print(f"\n{desc}: {len(rows)} images total, "
          f"backgrounds={dict(Counter(r['background'] for r in rows))}")
    return out


def prepare_splitfiles(root: str, images_subdir: str, split_names: List[str], index_base: int,
                       positive: str, negative: str, out_dir: str,
                       max_per_class: Optional[int] = None, seed: int = 0) -> Dict[str, str]:
    classes = TRASHNET_CLASSES
    if positive not in classes or negative not in classes:
        raise SystemExit(f"{positive!r}/{negative!r} not in {classes}")
    pos_idx = classes.index(positive) + index_base
    neg_idx = classes.index(negative) + index_base
    class_of = {classes.index(c) + index_base: c for c in classes}

    rows: List[dict] = []
    for split, fname in zip(("train", "val", "test"), split_names):
        fpath = os.path.join(root, fname if fname.endswith(".txt") else fname + ".txt")
        if not os.path.exists(fpath):
            raise SystemExit(f"missing split file {fpath}")
        kept = 0
        with open(fpath) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                name, idx = line.rsplit(" ", 1)
                idx = int(idx)
                if idx not in (pos_idx, neg_idx):
                    continue
                cls = class_of[idx]
                rel = os.path.join(images_subdir, cls, name)
                if not os.path.exists(os.path.join(root, rel)):
                    continue
                img = cv2.imread(os.path.join(root, rel), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                bg, stats = background_type(img)
                rows.append({
                    "sample_id": f"{split}_{cls}_{os.path.splitext(name)[0]}",
                    "path": rel, "label": 1 if idx == pos_idx else 0, "material": cls,
                    "object": "unknown", "background": bg, "split": split, "shift": "none",
                    "ambiguous": 0, "width": img.shape[1], "height": img.shape[0],
                    **stats, **object_stats(img),
                })
                kept += 1
        print(f"  {split:>6}: kept {kept} {positive}/{negative} images")

    if not rows:
        raise SystemExit(
            "no images of the requested classes were kept from these split files.\n"
            "The usual cause is a wrong --index-base:\n"
            "  one-indexed-files-notrash_*.txt -> 1=glass 2=paper 3=cardboard 4=plastic 5=metal 6=trash\n"
            "  zero-indexed-files*.txt         -> 0=glass 1=paper 2=cardboard 3=plastic 4=metal 5=trash\n"
            "Re-run with the matching --index-base (or check that the images directory is correct).\n"
            "A silent class loss here would quietly void every downstream comparison, so this "
            "fails loudly instead.")
    empty = [s for s in ("train", "val", "test") if not any(r["split"] == s for r in rows)]
    if empty:
        print(f"  WARNING: no {positive}/{negative} images found for split(s): {', '.join(empty)}")

    if max_per_class:
        rng = random.Random(seed)
        for split in {r["split"] for r in rows}:
            idx_pos = [i for i, r in enumerate(rows) if r["split"] == split and r["label"] == 1]
            idx_neg = [i for i, r in enumerate(rows) if r["split"] == split and r["label"] == 0]
            drop = set(rng.sample(idx_pos, max(0, len(idx_pos) - max_per_class))) | \
                set(rng.sample(idx_neg, max(0, len(idx_neg) - max_per_class)))
            rows = [r for i, r in enumerate(rows) if i not in drop]
    return _write_dataset(root, images_subdir, rows, out_dir, positive, negative, "splitfiles")


def prepare_folders(root: str, images_subdir: str, positive: str, negative: str, out_dir: str,
                    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15), seed: int = 0,
                    max_per_class: Optional[int] = None) -> Dict[str, str]:
    rng = random.Random(seed)
    rows: List[dict] = []
    for cls, lab in ((positive, 1), (negative, 0)):
        cdir = os.path.join(root, images_subdir, cls)
        if not os.path.isdir(cdir):
            # tolerate case differences (RealWaste uses 'Glass'/'Plastic')
            cands = [d for d in os.listdir(os.path.join(root, images_subdir))
                     if d.lower() == cls.lower()]
            if not cands:
                raise SystemExit(f"no class directory for {cls!r} under {cdir}")
            cdir = os.path.join(root, images_subdir, cands[0])
        files = sorted(f for f in os.listdir(cdir) if f.lower().endswith(IMG_EXT))
        rng.shuffle(files)
        if max_per_class:
            files = files[:max_per_class]
        n = len(files)
        n_tr, n_va = int(n * ratios[0]), int(n * ratios[1])
        for i, f in enumerate(files):
            split = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
            rel = os.path.join(images_subdir, os.path.basename(cdir), f)
            img = cv2.imread(os.path.join(root, rel), cv2.IMREAD_COLOR)
            if img is None:
                continue
            bg, stats = background_type(img)
            rows.append({
                "sample_id": f"{split}_{cls}_{os.path.splitext(f)[0]}",
                "path": rel, "label": lab, "material": cls.lower(), "object": "unknown",
                "background": bg, "split": split, "shift": "none", "ambiguous": 0,
                "width": img.shape[1], "height": img.shape[0], **stats, **object_stats(img),
            })
    return _write_dataset(root, images_subdir, rows, out_dir, positive, negative, "folders")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layout", choices=("splitfiles", "folders"), required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--images", default="dataset-resized", help="subdirectory holding the images")
    ap.add_argument("--out", required=True, help="where to write manifest_*.csv")
    ap.add_argument("--positive", default="glass")
    ap.add_argument("--negative", default="plastic")
    ap.add_argument("--index-base", type=int, default=1, choices=(0, 1),
                    help="TrashNet split files: 1 for one-indexed-*, 0 for zero-indexed-*")
    ap.add_argument("--splits", nargs=3,
                    default=["one-indexed-files-notrash_train", "one-indexed-files-notrash_val",
                             "one-indexed-files-notrash_test"],
                    help="train/val/test split filenames (with or without .txt)")
    ap.add_argument("--max-per-class", type=int, default=None,
                    help="optionally cap images per class per split (for fast runs)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    print(f"preparing {a.positive} vs {a.negative} from {a.root} (layout={a.layout})")
    os.makedirs(a.out, exist_ok=True)
    if a.layout == "splitfiles":
        paths = prepare_splitfiles(a.root, a.images, a.splits, a.index_base,
                                   a.positive, a.negative, a.out, a.max_per_class, a.seed)
    else:
        paths = prepare_folders(a.root, a.images, a.positive, a.negative, a.out,
                                seed=a.seed, max_per_class=a.max_per_class)
    print(f"\nwrote {len(paths)} manifests to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
