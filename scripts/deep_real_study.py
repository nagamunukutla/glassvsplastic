#!/usr/bin/env python3
"""Deep-learning tier on real data (TrashNet glass vs plastic), including the leakage fix.

Runs the backbones from the registry on the *same* official splits, and compares:

    full frame   vs   object-cropped

The crop variant is the deployment-honest version: it removes the studio backdrop, which the
classical controls show carries ~0.80 balanced accuracy on its own. If a backbone keeps its
accuracy after cropping, it is using the object; if it collapses, the benchmark number was
scene-driven.

    PYTHONPATH=src python scripts/deep_real_study.py --models tv:resnet18,tv:mobilenet_v3_large

Resolution is a memory-budget decision: this sandbox has ~1.4 GB free, and
ResNet-18 at 192 px / batch 12 fits while MobileNetV3-Large and EfficientNet-B0 need
160 px / batch 8 (the 192 px attempt was OOM-killed). The full-frame vs crop pair is
always trained at one resolution, so the comparison is unaffected; absolute numbers
across backbones are not perfectly matched and the report says so.

Writes ``results/real/deep_results.csv`` (+ per-run training histories) and prints a summary with
parameters, MACs and measured CPU latency.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from gvp.features import estimate_foreground  # noqa: E402


# --------------------------------------------------------------------------------------
# build a cropped variant of the dataset (fixes scene leakage by construction)
# --------------------------------------------------------------------------------------


def build_cropped_copy(src_dir: str, dst_dir: str, margin: float = 0.12) -> Dict[str, str]:
    """Crop every image to its estimated object bbox (plus margin) and rewrite manifests."""
    os.makedirs(os.path.join(dst_dir, "images"), exist_ok=True)
    out_paths: Dict[str, str] = {}
    for split in ("train", "val", "test"):
        src_manifest = os.path.join(src_dir, f"manifest_{split}.csv")
        if not os.path.exists(src_manifest):
            continue
        with open(src_manifest, newline="") as fh:
            rows = list(csv.DictReader(fh))
        out_rows = []
        n_fallback = 0
        for r in rows:
            img = cv2.imread(os.path.join(src_dir, r["path"]), cv2.IMREAD_COLOR)
            if img is None:
                continue
            mask = estimate_foreground(img)
            ys, xs = np.nonzero(mask)
            if len(xs) < 32:
                n_fallback += 1
                y0, y1, x0, x1 = 0, img.shape[0], 0, img.shape[1]
            else:
                h, w = img.shape[:2]
                bw, bh = xs.max() - xs.min(), ys.max() - ys.min()
                mx, my = int(bw * margin), int(bh * margin)
                x0, x1 = max(0, xs.min() - mx), min(w, xs.max() + mx)
                y0, y1 = max(0, ys.min() - my), min(h, ys.max() + my)
                if (x1 - x0) < 24 or (y1 - y0) < 24:  # degenerate crop -> keep the frame
                    n_fallback += 1
                    x0, y0, x1, y1 = 0, 0, w, h
            crop = img[y0:y1, x0:x1]
            rel = os.path.join("images", f"{r['sample_id']}.jpg")
            cv2.imwrite(os.path.join(dst_dir, rel), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            out_rows.append({**r, "path": rel, "crop_x0": x0, "crop_y0": y0,
                             "crop_w": int(x1 - x0), "crop_h": int(y1 - y0)})
        dst_manifest = os.path.join(dst_dir, f"manifest_{split}.csv")
        with open(dst_manifest, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)
        out_paths[split] = dst_manifest
        print(f"[crop] {split}: {len(out_rows)} images ({n_fallback} fell back to full frame)")
    return out_paths


# --------------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/trashnet")
    ap.add_argument("--cropped-dir", default="data/raw/trashnet_cropped")
    ap.add_argument("--out", default="results/real")
    ap.add_argument("--models", default="tv:resnet18")
    ap.add_argument("--crop-models", default="tv:resnet18",
                    help="models also trained on the object-cropped variant")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--build-crops", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    if a.build_crops or not os.path.exists(os.path.join(a.cropped_dir, "manifest_train.csv")):
        print("[crop] building object-cropped dataset variant…")
        build_cropped_copy(a.data, a.cropped_dir)

    from gvp.deep import TrainConfig, train_and_eval, count_macs, count_params, build_backbone

    runs: List[Dict[str, object]] = []
    models = [m for m in a.models.split(",") if m]
    crop_models = [m for m in a.crop_models.split(",") if m]

    for model in models:
        runs.append({"model": model, "variant": "full_frame", "data_dir": a.data})
    for model in crop_models:
        runs.append({"model": model, "variant": "object_crop", "data_dir": a.cropped_dir})

    rows: List[dict] = []
    for run in runs:
        tag = f"{run['model'].split(':')[-1]}_{run['variant']}"
        print(f"\n=== {tag} ({a.epochs} epochs, {a.img_size}px, batch {a.batch_size}) ===")
        t0 = time.perf_counter()
        cfg = TrainConfig(model=run["model"], data_dir=run["data_dir"],
                          out_dir=os.path.join(a.out, "deep"), epochs=a.epochs,
                          batch_size=a.batch_size, lr=a.lr, img_size=a.img_size, seed=a.seed,
                          num_workers=2, test_splits=("test",),
                          notes=f"real TrashNet glass-vs-plastic, {run['variant']}")
        res = train_and_eval(cfg)
        # Durable per-run record: the curated CSV is a union that later runs rewrite, so anything
        # only in it can be lost. wall_seconds is therefore also written next to the history.
        with open(os.path.join(a.out, "deep", f"run_{tag}.json"), "w") as fh:
            json.dump({"model": run["model"], "variant": run["variant"],
                       "img_size": a.img_size, "epochs": a.epochs, "batch_size": a.batch_size,
                       "seed": a.seed, "wall_seconds": round(time.perf_counter() - t0, 1),
                       "notes": f"real TrashNet glass-vs-plastic, {run['variant']}"}, fh, indent=1)
        for r in res["rows"]:
            r["variant"] = run["variant"]
            r["wall_seconds"] = round(time.perf_counter() - t0, 1)
            rows.append(r)
            print(f"  -> BA={r['balanced_accuracy']:.4f} acc={r['accuracy']:.4f} "
                  f"glass_recall={r['glass_recall']:.3f} plastic_recall={r['plastic_recall']:.3f} "
                  f"({r['params_m']}M params, {r['macs_g']}G MACs, "
                  f"{r['latency_ms_per_image']:.1f} ms/img, {r['wall_seconds']}s total)")

    def _key(r: dict) -> tuple:
        """Identity of a measurement: re-running the same config replaces it, others accumulate.

        (The previous filter dropped every row whose notes mentioned this dataset, so each
        backbone silently erased the previous backbone's measurements. See docs/errors in the
        study log: an overwrite like this is invisible unless you count rows.)
        """
        return (r.get("model"), r.get("variant"), str(r.get("img_size")),
                str(r.get("epochs")), r.get("split"))

    path = os.path.join(a.out, "deep_results.csv")
    existing: List[dict] = []
    if os.path.exists(path):
        with open(path, newline="") as fh:
            existing = list(csv.DictReader(fh))
    new_keys = {_key(r) for r in rows}
    kept = [r for r in existing if _key(r) not in new_keys]
    allrows = kept + rows
    print(f"[deep] {len(kept)} existing rows kept, {len(rows)} written")
    cols: List[str] = []
    for r in allrows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(allrows)
    print(f"\n[deep] wrote {path} ({len(rows)} new rows)")

    print("\n=== summary: full frame vs object crop ===")
    print(f"{'model':28s} {'variant':12s} {'bal acc':>8s} {'glass rec':>10s} {'plast rec':>10s}"
          f" {'params M':>9s} {'ms/img':>8s}")
    for r in sorted(rows, key=lambda r: (r["model"], r["variant"])):
        print(f"{r['model']:28s} {r['variant']:12s} {r['balanced_accuracy']:8.4f} "
              f"{r['glass_recall']:10.3f} {r['plastic_recall']:10.3f} "
              f"{r['params_m']:9.2f} {r['latency_ms_per_image']:8.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
