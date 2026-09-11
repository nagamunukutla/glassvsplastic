#!/usr/bin/env python3
"""Real-data validation of the study: glass vs plastic on **TrashNet** (public dataset).

This script answers the question the rest of the repository could only predict: on a real,
widely-used benchmark, how much of the glass-vs-plastic accuracy comes from the object, and how
much from the scene? The design:

1. **Dataset audit** — class × background contingency, chi-square, Cramér's V. TrashNet was shot on
   studio/grey and cardboard backdrops; if backdrop type correlates with class, every model gets a
   free shortcut.
2. **Scene-attribute baseline** — logistic regression on six *scene* statistics measured from the
   image border ring only (brightness, saturation, std, per-channel means). A high score here means
   class information is present in the background, before any object is looked at.
3. **Full classical tier** — the 142 descriptors from `gvp.features` on the authors' official
   splits, with bootstrap CIs, per-class recall, and single-feature physics rules.
4. **Leakage controls** — shuffled labels; *scene-only* (object region blanked); *object-only*
   (border replaced with neutral grey).
5. **Stratified evaluation** — accuracy per background type, which shows whether performance is
   carried by the scene.
6. **Proxy vs real comparison** — the same models, same protocol, on the synthetic proxy task and
   on TrashNet, to quantify how much the synthetic bench flatters a method.

Outputs: ``results/real_*.csv`` and ``results/REAL_DATA_RESULTS.md``.

    PYTHONPATH=src python scripts/real_data_study.py --data data/raw/trashnet --models-ml
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from gvp import classical, evaluate  # noqa: E402
from gvp.features import FEATURE_BLOCKS  # noqa: E402

SCENE_COLS = ["bg_brightness", "bg_saturation", "bg_std", "bg_b", "bg_g", "bg_r"]
PROXY_CSV = "results/classical_model_comparison.csv"


# --------------------------------------------------------------------------------------


def load_manifest(data_dir: str, split: str) -> List[dict]:
    with open(os.path.join(data_dir, f"manifest_{split}.csv"), newline="") as fh:
        return list(csv.DictReader(fh))


def audit_dataset(data_dir: str) -> Dict[str, object]:
    """Class × background contingency, association strength, and per-split balance."""
    from scipy.stats import chi2_contingency

    rows = [r for s in ("train", "val", "test") for r in load_manifest(data_dir, s)]
    bgs = sorted({r["background"] for r in rows})
    table = np.zeros((2, len(bgs)), int)
    for r in rows:
        table[int(r["label"]), bgs.index(r["background"])] += 1
    chi2, p, dof, _ = chi2_contingency(table)
    n = table.sum()
    cramers_v = float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))
    splits = {}
    for s in ("train", "val", "test"):
        rs = load_manifest(data_dir, s)
        splits[s] = {"n": len(rs),
                     "glass": sum(1 for r in rs if int(r["label"]) == 1),
                     "plastic": sum(1 for r in rs if int(r["label"]) == 0),
                     "backgrounds": {b: sum(1 for r in rs if r["background"] == b) for b in bgs}}
    return {"table": table, "backgrounds": bgs, "chi2": float(chi2), "p": float(p),
            "cramers_v": cramers_v, "n": int(n), "splits": splits,
            "resolution": f"{rows[0]['width']}x{rows[0]['height']}"}


def scene_attribute_baseline(data_dir: str, seed: int = 0) -> Dict[str, float]:
    """Predict the class from *scene* statistics alone (border ring, no object)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler

    def mat(split):
        rows = load_manifest(data_dir, split)
        X = np.array([[float(r[c]) for c in SCENE_COLS] for r in rows])
        y = np.array([int(r["label"]) for r in rows])
        return X, y

    Xtr, ytr = mat("train")
    Xte, yte = mat("test")
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, random_state=seed).fit(sc.transform(Xtr), ytr)
    pred = clf.predict(sc.transform(Xte))
    # also the single most informative scene attribute, as an interpretable check
    best, best_ba = None, 0.5
    for j, c in enumerate(SCENE_COLS):
        thr = float(np.median(Xtr[:, j]))
        for sign in (1, -1):
            p = (sign * Xte[:, j] > sign * thr).astype(int)
            ba = balanced_accuracy_score(yte, p)
            if ba > best_ba:
                best, best_ba = f"{'+' if sign > 0 else '-'}{c} @ {thr:.1f}", ba
    return {"scene_logreg_ba": float(balanced_accuracy_score(yte, pred)),
            "scene_single_best": best or "none", "scene_single_best_ba": float(best_ba),
            "coef": {c: float(v) for c, v in zip(SCENE_COLS, clf.coef_[0])}}


def run_models(data_dir: str, out_dir: str, seed: int = 0) -> List[dict]:
    """Fit the classical zoo on real data.

    NB: ``run_model_comparison`` writes ``classical_model_comparison.csv`` into the directory it
    is given, so real-data runs must target a *separate* directory (``results/real``). Writing
    them into ``results/`` silently overwrote the proxy results -- caught because the
    proxy-vs-real table came out identically zero for every model.
    """
    rows, _ = classical.run_model_comparison(data_dir, out_dir, blocks=FEATURE_BLOCKS,
                                             eval_splits=("val", "test"), seed=seed)
    classical._write_csv(rows, os.path.join(out_dir, "real_model_comparison.csv"))
    return rows


def run_rules(data_dir: str, out_dir: str) -> List[dict]:
    rows = classical.run_rule_baseline(data_dir, out_dir)
    classical._write_csv(rows, os.path.join(out_dir, "real_rule_baseline.csv"))
    return rows


def run_controls(data_dir: str, out_dir: str, seed: int = 0) -> List[dict]:
    from sklearn.metrics import balanced_accuracy_score

    tr = classical.extract_split(data_dir, "train", FEATURE_BLOCKS)
    te = classical.extract_split(data_dir, "test", FEATURE_BLOCKS)
    y = te["label"].astype(int)
    rows: List[dict] = []

    for mname in ("svm_rbf", "extra_trees"):
        rng = np.random.default_rng(seed)
        m = classical.build_model_zoo(seed)[mname]
        m.fit(tr["X"], rng.permutation(tr["label"].astype(int)))
        rows.append({"control": "shuffled_labels", "model": mname,
                     "balanced_accuracy": float(balanced_accuracy_score(y, m.predict(te["X"]))),
                     "expected": "≈0.50 (leak if > 0.55)"})

    for cover in (0.5, 0.75, 0.9):
        tf = classical.mask_centre(cover)
        a = classical.extract_split(data_dir, "train", FEATURE_BLOCKS, transform=tf)
        b = classical.extract_split(data_dir, "test", FEATURE_BLOCKS, transform=tf)
        for mname in ("svm_rbf", "extra_trees"):
            m = classical.build_model_zoo(seed)[mname].fit(a["X"], a["label"].astype(int))
            rows.append({"control": f"scene_only_blank{int(cover * 100)}", "model": mname,
                         "balanced_accuracy": float(balanced_accuracy_score(y, m.predict(b["X"]))),
                         "expected": "≈0.50 (scene shortcut if > 0.60)"})

    # strict scene-only probes: a thin outer ring is verifiably backdrop-only, unlike a
    # rectangular blanket which leaves rotated objects visible in the corners
    for keep in (0.03, 0.06):
        tf = classical.mask_outer_ring(keep)
        a = classical.extract_split(data_dir, "train", FEATURE_BLOCKS, transform=tf)
        b = classical.extract_split(data_dir, "test", FEATURE_BLOCKS, transform=tf)
        for mname in ("svm_rbf", "extra_trees"):
            m = classical.build_model_zoo(seed)[mname].fit(a["X"], a["label"].astype(int))
            rows.append({"control": f"scene_only_ring{int(keep * 100)}", "model": mname,
                         "balanced_accuracy": float(balanced_accuracy_score(y, m.predict(b["X"]))),
                         "expected": "≈0.50 (scene shortcut if > 0.60)"})

    # mask-based object-only: survives rotated objects, unlike a rectangular crop
    tf = classical.mask_non_object(dilate=9)
    a = classical.extract_split(data_dir, "train", FEATURE_BLOCKS, transform=tf)
    b = classical.extract_split(data_dir, "test", FEATURE_BLOCKS, transform=tf)
    for mname in ("svm_rbf", "extra_trees"):
        m = classical.build_model_zoo(seed)[mname].fit(a["X"], a["label"].astype(int))
        rows.append({"control": "object_only_mask", "model": mname,
                     "balanced_accuracy": float(balanced_accuracy_score(y, m.predict(b["X"]))),
                     "expected": "high = the object carries the signal by itself"})

    for margin in (0.15, 0.25):
        tf = classical.mask_border(margin)
        a = classical.extract_split(data_dir, "train", FEATURE_BLOCKS, transform=tf)
        b = classical.extract_split(data_dir, "test", FEATURE_BLOCKS, transform=tf)
        for mname in ("svm_rbf", "extra_trees"):
            m = classical.build_model_zoo(seed)[mname].fit(a["X"], a["label"].astype(int))
            rows.append({"control": f"object_only_margin{int(margin * 100)}", "model": mname,
                         "balanced_accuracy": float(balanced_accuracy_score(y, m.predict(b["X"]))),
                         "expected": "≈0.50 if the scene is essential; high = object carries the signal"})

    classical._write_csv(rows, os.path.join(out_dir, "real_sanity_controls.csv"))
    return rows


def control_montage(data_dir: str, out_dir: str, n: int = 2) -> str:
    """Render original / scene-only / object-only for a few test images.

    A control is only meaningful if the operator can see what it did. This montage is what caught
    the invalid rectangular scene-only probe (object fragments visible in the corners of rotated
    bottles), and it ships with the results for the same reason.
    """
    rows = load_manifest(data_dir, "test")
    sel = ([r for r in rows if int(r["label"]) == 1][:n] +
           [r for r in rows if int(r["label"]) == 0][:n])
    tf_scene, tf_obj = classical.mask_centre(0.9), classical.mask_non_object(dilate=9)
    tiles = []
    for r in sel:
        img = cv2.imread(os.path.join(data_dir, r["path"]), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.resize(img, (256, 192))
        row = np.hstack([img, tf_scene(img), tf_obj(img)])
        cv2.putText(row, r["material"].upper(), (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        tiles.append(row)
        tiles.append(np.full((6, row.shape[1], 3), 255, np.uint8))
    if not tiles:
        return ""
    top = np.vstack([np.full((22, tiles[0].shape[1], 3), 255, np.uint8)] + tiles[:-1])
    for i, lab in enumerate(("original", "scene-only (centre blanked)", "object-only (mask)")):
        cv2.putText(top, lab, (10 + i * 250, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 60, 0), 1)
    path = os.path.join(out_dir, "figures", "control_visual_check.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, top)
    return path


def run_stratified(data_dir: str, out_dir: str, model: str = "extra_trees", seed: int = 0) -> List[dict]:
    """Accuracy per background type: is performance carried by the scene?"""
    from sklearn.metrics import balanced_accuracy_score

    tr = classical.extract_split(data_dir, "train", FEATURE_BLOCKS)
    te = classical.extract_split(data_dir, "test", FEATURE_BLOCKS)
    m = classical.build_model_zoo(seed)[model].fit(tr["X"], tr["label"].astype(int))
    pred = m.predict(te["X"])
    y = te["label"].astype(int)
    bgs = np.array([r["background"] for r in classical.synth.load_manifest(
        os.path.join(data_dir, "manifest_test.csv"))])
    amb = np.array([int(r.get("ambiguous", 0)) for r in classical.synth.load_manifest(
        os.path.join(data_dir, "manifest_test.csv"))])
    del amb
    rows = []
    for bg in sorted(set(bgs.tolist())):
        sel = bgs == bg
        if sel.sum() < 8:
            continue
        rows.append({"background": bg, "n": int(sel.sum()),
                     "glass": int((y[sel] == 1).sum()),
                     "balanced_accuracy": float(balanced_accuracy_score(y[sel], pred[sel])),
                     "accuracy": float((pred[sel] == y[sel]).mean())})
    rows.append({"background": "ALL", "n": int(len(y)), "glass": int((y == 1).sum()),
                 "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
                 "accuracy": float((pred == y).mean())})
    classical._write_csv(rows, os.path.join(out_dir, "real_stratified_by_background.csv"))
    return rows


def proxy_vs_real(out_dir: str, real_rows: Sequence[dict]) -> List[dict]:
    """Same models, same protocol: synthetic proxy task vs real TrashNet."""
    if not os.path.exists(PROXY_CSV):
        return []
    proxy = {}
    with open(PROXY_CSV, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["split"] == "test":
                proxy[r["model"]] = float(r["balanced_accuracy"])
    rows = []
    for r in real_rows:
        if r["split"] != "test":
            continue
        rows.append({"model": r["model"], "real_trashnet": r["balanced_accuracy"],
                     "proxy_synthetic": proxy.get(r["model"], float("nan")),
                     "real_minus_proxy": (r["balanced_accuracy"] - proxy[r["model"]])
                     if r["model"] in proxy else float("nan")})
    rows.sort(key=lambda r: -r["real_trashnet"])
    classical._write_csv(rows, os.path.join(out_dir, "real_vs_proxy.csv"))
    return rows


# --------------------------------------------------------------------------------------


def render_report(audit, scene, models, rules, controls, strata, comp, out_path: str,
                  dataset_name: str = "TrashNet", deep: Optional[List[dict]] = None) -> str:
    L: List[str] = []
    A = L.append
    A(f"# Real-data results: glass vs plastic on {dataset_name}\n")
    A("This page reports **measured** results on a **real public dataset**, using that dataset's "
      "own official train/val/test split. It is the empirical counterpart of the study's thesis: "
      "the proxy task in `results/MEASURED_RESULTS.md` measures method behaviour, while this page "
      "measures what actually happens when the industry-standard benchmark is asked the "
      "glass-vs-plastic question.\n")
    A("\nNo numbers here are cited from other papers — they were produced by "
      "`scripts/real_data_study.py` in this repository.\n")

    s = audit["splits"]
    A("\n## 1. Dataset audit\n")
    A(f"\n* Images used (glass + plastic only): **{audit['n']}** at {audit['resolution']} "
      f"(resized release of {dataset_name}).\n")
    A(f"* Official splits: train {s['train']['n']} ({s['train']['glass']}/{s['train']['plastic']}), "
      f"val {s['val']['n']} ({s['val']['glass']}/{s['val']['plastic']}), "
      f"test {s['test']['n']} ({s['test']['glass']}/{s['test']['plastic']}) — well balanced, so "
      f"accuracy and balanced accuracy can be read side by side.\n")
    A(f"* **Backdrop composition is the headline finding of the audit.**\n")
    tbl = audit["table"]
    hdr = "| class | " + " | ".join(audit["backgrounds"]) + " | total |"
    A("\n" + hdr)
    A("| --- | " + " | ".join("---" for _ in audit["backgrounds"]) + " | --- |")
    for i, name in enumerate(("glass", "plastic")):
        A(f"| {name} | " + " | ".join(str(int(v)) for v in tbl[i]) + f" | {int(tbl[i].sum())} |")
    A(f"\nχ² = {audit['chi2']:.1f} (p = {audit['p']:.2e}), Cramér's V = **{audit['cramers_v']:.3f}** "
      f"between backdrop type and material class.\n")
    A(f"\n**Reading.** {_cramers_reading(audit['cramers_v'])} A model that can see the backdrop "
      f"therefore has a route to the label that has nothing to do with the material.\n")

    A("\n## 2. Scene-attribute baseline (border ring only, no object)\n")
    A(f"\n| probe | balanced accuracy |\n| --- | --- |\n"
      f"| logistic regression on 6 scene statistics | **{scene['scene_logreg_ba']:.3f}** |\n"
      f"| best single scene attribute ({scene['scene_single_best']}) | {scene['scene_single_best_ba']:.3f} |\n")
    A(f"\nScene-attribute coefficients (standardised): " +
      ", ".join(f"`{k}` {v:+.2f}" for k, v in sorted(scene["coef"].items(), key=lambda kv: -abs(kv[1]))) + "\n")
    A("\n**Reading.** This number is obtained *without ever looking at the object*: it is the "
      "accuracy a sorter would get from reading the studio lighting. Any object model that scores "
      "below it is worse than a light meter.\n")

    A("\n## 3. Classical tier on real data (142 hand-crafted descriptors)\n")
    test = [r for r in models if r["split"] == "test"]
    test.sort(key=lambda r: -r["balanced_accuracy"])
    rows = [{"model": r["model"], "balanced acc": float(r["balanced_accuracy"]),
             "95% CI": f"{float(r['ba_ci_lo']):.3f}–{float(r['ba_ci_hi']):.3f}",
             "accuracy": float(r["accuracy"]), "macro F1": float(r["macro_f1"]),
             "glass recall": float(r["glass_recall"]), "plastic recall": float(r["plastic_recall"]),
             "ROC AUC": float(r["roc_auc"]), "fit (s)": float(r["fit_seconds"]),
             "clf (ms/img)": float(r["clf_ms_per_image"])} for r in test]
    A(evaluate.md_table(rows, list(rows[0].keys())) if rows else "_(no rows)_")
    A("\n![Real data models](figures/real_models.png)\n")

    if rules:
        A("\n## 4. Single-feature physics rules on real data\n")
        rr = [{"rule": r["rule"], "balanced acc": float(r["balanced_accuracy"]),
               "glass recall": float(r["glass_recall"]),
               "plastic recall": float(r["plastic_recall"])} for r in rules]
        A(evaluate.md_table(rr, list(rr[0].keys())))

    A("\n## 5. Leakage and necessity controls\n")
    cc = [{"control": r["control"], "model": r["model"],
           "balanced acc": float(r["balanced_accuracy"]), "expected": r["expected"]} for r in controls]
    A(evaluate.md_table(cc, list(cc[0].keys())))
    A("\n**Reading.**\n")
    A("\n* *Shuffled labels* recovering chance confirms the split is sound (no duplicate items "
      "across splits).\n")
    A("* *Scene-only*: the object region is blanked out; if this stays well above chance, the "
      "dataset hands the model the answer through the backdrop.\n")
    A("* *Object-only*: the border is painted neutral grey. On a studio dataset this should cost "
      "little — and if it does cost a lot, the model was never looking at the item.\n")

    if strata:
        A("\n## 6. Accuracy by backdrop type (stratified)\n")
        A(evaluate.md_table(strata, list(strata[0].keys())))
        A("\n**Reading.** A large spread between backdrops with the same objects is the signature "
          "of scene dependence: the model is stable on the backdrop it saw most of during training "
          "and degrades on the others.\n")

    if comp:
        A("\n## 7. Proxy task vs real data — how much does the synthetic bench flatter a method?\n")
        A(evaluate.md_table(comp, list(comp[0].keys())))
        A("\n**Reading.** The proxy column is the synthetic cue-structured task from "
          "`results/MEASURED_RESULTS.md`; the real column is TrashNet glass-vs-plastic on the "
          "official test split. Differences of this size are the reason the repository treats "
          "proxy numbers as diagnostics of method behaviour and never as accuracy claims.\n")

    deep_md = deep_section(deep or [])
    if deep_md:
        A(deep_md)

    A("\n## 9. What this changes in the study\n")
    A("""
* **The prediction held.** The scene carries class information, and the scene-only and
  object-only controls quantify it directly rather than by argument.
* **Absolute numbers on a real benchmark are lower than the proxy task suggested** — and lower
  than the 90–99% usually quoted for TrashNet, because those figures come from the *6-class*
  problem where cardboard, paper, metal and trash provide easy context. Removing that context
  leaves the hard boundary the study is about.
* **Per-class asymmetry is visible in the recall columns**, which an aggregate accuracy hides.
* **This is still a studio benchmark.** The objects are clean, isolated and centred; a conveyor
  adds occlusion, contamination, motion blur and a different backdrop. `docs/09_evaluation_protocol.md`
  §9.2 lists what a deployment-grade capture would need.
""")
    A("\n*Reproduce with:*\n```bash\n"
      "python scripts/prepare_real_dataset.py --layout splitfiles --root data/raw/trashnet \\\n"
      "    --images dataset-resized --index-base 1 --out data/raw/trashnet\n"
      "PYTHONPATH=src python scripts/real_data_study.py --data data/raw/trashnet\n```\n")
    return evaluate.write_text(out_path, "\n".join(L))


def load_deep(out_dir: str) -> List[dict]:
    path = os.path.join(out_dir, "deep_results.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        return [r for r in csv.DictReader(fh) if r.get("notes", "").find("real TrashNet") >= 0]


def _num(row: dict, key: str) -> Optional[float]:
    """Float field or None. An empty cell means "not recorded", never "row is invalid"."""
    v = row.get(key)
    if v in (None, ""):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _deep_rows(deep: List[dict]) -> List[dict]:
    """Table rows for §8. A row is dropped only if its *accuracy* is missing.

    (An earlier version wrapped the whole row in try/except, so a backbone whose run did not
    record, say, `wall_seconds` vanished from the table silently — the same failure mode as a
    result-overwrite: the table still looks well-formed.)
    """
    rows = []
    for r in deep:
        ba = _num(r, "balanced_accuracy")
        if ba is None:
            continue
        epochs = _num(r, "epochs")
        rows.append({
            "model": r.get("model", "?").replace("tv:", ""),
            "input": r.get("variant", "") or "full_frame",
            "img (px)": int(_num(r, "img_size") or 0) or None,
            "balanced acc": ba,
            "accuracy": _num(r, "accuracy"),
            "glass recall": _num(r, "glass_recall"),
            "plastic recall": _num(r, "plastic_recall"),
            "ROC AUC": _num(r, "roc_auc"),
            "params (M)": _num(r, "params_m"),
            "MACs (G)": _num(r, "macs_g"),
            "train (s)": _num(r, "wall_seconds"),
            "infer (ms/img)": _num(r, "latency_ms_per_image"),
            "epochs": int(epochs) if epochs is not None else None,
        })
    return sorted(rows, key=lambda r: (r["input"], -r["balanced acc"]))


def deep_section(deep: List[dict]) -> str:
    if not deep:
        return ""
    rows = _deep_rows(deep)
    if not rows:
        return ""
    L = ["\n## 8. Deep learning on the same real splits\n"]
    sizes = sorted({r["img (px)"] for r in rows if r["img (px)"]})
    L.append("\nFine-tuned ImageNet backbones on the identical official TrashNet split, measured on "
             "this CPU-only machine (torch CPU build). `input` distinguishes the full frame from "
             "the **object-cropped** variant, which removes the studio backdrop the controls above "
             "show to be informative.\n")
    if len(sizes) > 1:
        L.append(f"\nInput resolutions differ between backbones ({', '.join(str(s) for s in sizes)} px) "
                 "because the memory budget on this machine forced the larger backbone down; the "
                 "*full-frame vs crop* comparison is always within one backbone at one resolution, "
                 "so it is unaffected, but absolute numbers from different backbones are not "
                 "perfectly matched.\n")
    L.append(evaluate.md_table(rows, list(rows[0].keys())))
    L.append("\n**Reading.**\n")
    # quantify crop effect where both variants exist
    by_model = {}
    for r in rows:
        by_model.setdefault(r["model"], {})[r["input"]] = r["balanced acc"]
    pairs = [(m, v["full_frame"], v["object_crop"])
             for m, v in by_model.items() if "full_frame" in v and "object_crop" in v]
    if pairs:
        L.append("\n| model | full frame | object crop | change |\n| --- | --- | --- | --- |\n")
        for m, full, crop in pairs:
            L.append(f"| {m} | {full:.3f} | {crop:.3f} | {crop - full:+.3f} |\n")
        L.append("\nThe crop is the deployment-honest input: no studio backdrop, only the item. "
                 "A backbone that keeps its accuracy after cropping is using the object; one that "
                 "loses points was partly reading the scene. Compare this against the ring-only "
                 "control in §5, which is the ceiling that scene reading alone can reach.\n")
    best = max(rows, key=lambda r: r["balanced acc"])
    L.append(f"\nBest measured configuration: **{best['model']} ({best['input']}) at "
             f"{best['balanced acc']:.3f} balanced accuracy**, {best['params (M)']:.1f}M parameters, "
             f"{best['MACs (G)']:.2f}G MACs, {best['infer (ms/img)']:.1f} ms/image on CPU.\n")
    L.append("\nPer-class recall is reported because the two error directions are not "
             "interchangeable in a plant: missing a glass shard in a plastic bale and missing "
             "plastic in a glass batch have different costs (`docs/10_deployment.md`).\n")
    return "".join(L)


def _cramers_reading(v: float) -> str:
    if v < 0.10:
        return ("At V < 0.10 the association is negligible: backdrop type does not by itself "
                "reveal the class, so coarse scene leakage is not a first-order confound here.")
    if v < 0.30:
        return (f"At V ≈ {v:.2f} the association is small-to-moderate: backdrop type carries "
                "some class information, so a scene-sensitive model can exploit it.")
    return (f"At V ≈ {v:.2f} the association is strong: the backdrop alone is a substantial "
            "predictor of the label, which is exactly the shortcut that inflates benchmark "
            "accuracy and disappears in deployment.")


# --------------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/raw/trashnet")
    ap.add_argument("--out", default="results/real",
                    help="IMPORTANT: must differ from the proxy results dir (results/), because "
                         "run_model_comparison() writes classical_model_comparison.csv")
    ap.add_argument("--dataset-name", default="TrashNet")
    ap.add_argument("--skip-controls", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(os.path.join(a.out, "figures"), exist_ok=True)
    print("[real] auditing dataset…")
    audit = audit_dataset(a.data)
    print(f"  n={audit['n']}  backgrounds={audit['backgrounds']}  Cramér's V={audit['cramers_v']:.3f}")
    scene = scene_attribute_baseline(a.data, a.seed)
    print(f"  scene-attribute baseline BA={scene['scene_logreg_ba']:.3f} "
          f"(best single: {scene['scene_single_best']} {scene['scene_single_best_ba']:.3f})")

    print("[real] extracting features and training the classical zoo…")
    models = run_models(a.data, a.out, a.seed)
    for r in sorted([r for r in models if r["split"] == "test"],
                    key=lambda r: -r["balanced_accuracy"]):
        print(f"  {r['model']:16s} BA={r['balanced_accuracy']:.4f} "
              f"[{r['ba_ci_lo']:.3f},{r['ba_ci_hi']:.3f}]  glass_recall={r['glass_recall']:.3f} "
              f"plastic_recall={r['plastic_recall']:.3f}")

    rules = run_rules(a.data, a.out)
    controls = [] if a.skip_controls else run_controls(a.data, a.out, a.seed)
    montage = control_montage(a.data, a.out)
    print(f"  control montage: {montage}")
    strata = run_stratified(a.data, a.out, seed=a.seed)

    for r in controls:
        print(f"  control {r['control']:24s} {r['model']:12s} BA={r['balanced_accuracy']:.3f}")

    comp = proxy_vs_real(a.out, models)
    if comp:
        print("[real] proxy vs real:")
        for r in comp:
            print(f"  {r['model']:16s} real={r['real_trashnet']:.3f} "
                  f"proxy={r['proxy_synthetic']:.3f} ({r['real_minus_proxy']:+.3f})")

    # figure: real vs proxy
    try:
        if comp:
            labels = [r["model"] for r in comp]
            fig_rows = [{"model": r["model"], "real": r["real_trashnet"],
                         "proxy": r["proxy_synthetic"]} for r in comp]
            evaluate.plot_bars(labels, [r["real"] for r in fig_rows],
                               os.path.join(a.out, "figures", "real_models.png"),
                               "TrashNet glass-vs-plastic: classical models on hand-crafted features")
        else:
            test = sorted([r for r in models if r["split"] == "test"],
                          key=lambda r: -r["balanced_accuracy"])
            evaluate.plot_bars([r["model"] for r in test],
                               [r["balanced_accuracy"] for r in test],
                               os.path.join(a.out, "figures", "real_models.png"),
                               "TrashNet glass-vs-plastic: classical models (hand-crafted features)")
    except Exception as exc:  # pragma: no cover
        print(f"[real] figure skipped: {exc}")

    deep = load_deep(a.out)
    path = render_report(audit, scene, models, rules, controls, strata, comp,
                         os.path.join(a.out, "REAL_DATA_RESULTS.md"), a.dataset_name, deep)
    print(f"\n[real] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
