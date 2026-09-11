"""Classical-ML study driver: model zoo, feature ablations, cue ablations, robustness sweeps.

Everything here operates on *features* extracted by :mod:`gvp.features`, so training is
CPU-only and finishes in seconds/minutes -- that is exactly the point of the classical tier
in the comparison (``docs/06_model_comparison.md``).

The GPU/deep-learning tier lives in :mod:`gvp.deep`.
"""

from __future__ import annotations

import csv
import hashlib
import os
import pickle
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from . import evaluate, synth
from .features import FEATURE_BLOCKS, extract_features

CACHE_DIRNAME = ".cache"


# --------------------------------------------------------------------------------------
# feature extraction with caching
# --------------------------------------------------------------------------------------


def _cache_path(data_dir: str, manifest: str, blocks: Sequence[str], tag: str = "") -> str:
    """Cache key = blocks + manifest *content* + feature-code fingerprint.

    Hashing the manifest and the generator/feature source means the cache can never serve
    stale features after the data or the code changes -- a class of silent methodological bug
    worth the extra few milliseconds.
    """
    from . import features as features_mod
    fp: List[str] = []
    for mod in (synth, features_mod):  # anything that changes the *content* of X
        try:
            with open(mod.__file__, "rb") as fh:
                fp.append(hashlib.md5(fh.read()).hexdigest())
        except Exception:
            fp.append("")
    fp.append(f"{os.path.getsize(__file__)}:{os.path.getmtime(__file__)}")  # extraction loop
    mpath = os.path.join(data_dir, f"manifest_{manifest}.csv")
    if os.path.exists(mpath):
        with open(mpath, "rb") as fh:
            fp.append(hashlib.md5(fh.read()).hexdigest())
    payload = "|".join(sorted(blocks)) + manifest + tag + "|".join(fp)
    key = hashlib.md5(payload.encode()).hexdigest()[:12]
    return os.path.join(data_dir, CACHE_DIRNAME, f"feat_{key}.npz")


def mask_centre(cover: float = 0.8):
    """Factory: blank the central ``cover`` fraction of the frame with the border colour.

    Used by the *background-only control*. If a classifier trained on these images still
    separates glass from plastic, the dataset leaks class information through the *scene*
    (lighting, background choice, camera rig) rather than through the material -- the most
    common way published waste-classification numbers are inflated.

    ``cover`` must be large enough to remove the object. Run at two window sizes: a small
    window leaves peripherally visible object parts (bottle necks, caps, shard tips) and the
    control scores *above* chance, which is a property of the control, not a leak. Seeing the
    score fall to ~0.50 as the window grows is the diagnostic that the dataset is clean.
    """
    def _fn(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        out = img.copy()
        b = max(2, int(min(h, w) * 0.06))
        border = np.concatenate([img[:b].reshape(-1, 3), img[-b:].reshape(-1, 3),
                                 img[:, :b].reshape(-1, 3), img[:, -b:].reshape(-1, 3)])
        fill = np.median(border, axis=0)
        lo, hi = 0.5 - cover / 2, 0.5 + cover / 2
        out[int(h * lo):int(h * hi), int(w * lo):int(w * hi)] = fill.astype(np.uint8)
        return out

    _fn.__name__ = f"blankcentre{int(cover * 100)}"
    return _fn


def mask_border(margin: float = 0.18, fill: int = 127):
    """Factory: replace the image border ring with a constant, keeping the centre.

    The *object-only* control, complementing :func:`mask_centre` (the scene-only control).
    Together they separate two failure modes:

    * high score on ``mask_centre``  -> the model is reading the scene, not the item;
    * big drop        on ``mask_border`` -> the model *needs* the scene, i.e. it is not deciding
      from the object at all.

    ``fill`` is a neutral grey by default (not the border colour), so the replaced region carries
    no information about the original scene.
    """
    def _fn(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        out = np.full_like(img, fill)
        y0, y1 = int(h * margin), int(h * (1 - margin))
        x0, x1 = int(w * margin), int(w * (1 - margin))
        out[y0:y1, x0:x1] = img[y0:y1, x0:x1]
        return out

    _fn.__name__ = f"border{margin}"
    return _fn


def mask_outer_ring(keep: float = 0.03, fill: int = 127):
    """Factory: keep only a thin outer ring, replacing everything else with ``fill``.

    The *strict* scene-only control. Rectangular blanking is leaky on datasets whose objects are
    rotated and reach into the corners (verified visually on TrashNet: at 90% blanking, object
    fragments remain visible on a diagonally-placed bottle). A thin outer ring is the region that
    is genuinely backdrop, so a score here is attributable to the scene alone.
    """
    def _fn(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        out = np.full_like(img, fill)
        m = max(2, int(min(h, w) * keep))
        out[:m], out[-m:], out[:, :m], out[:, -m:] = img[:m], img[-m:], img[:, :m], img[:, -m:]
        return out

    _fn.__name__ = f"ring{int(keep * 100)}"
    return _fn


def mask_non_object(dilate: int = 9, fill: int = 127):
    """Factory: keep only the estimated object, replace the rest with neutral grey.

    A mask-based *object-only* control, which unlike a rectangular crop survives rotated objects.
    Uses the same foreground estimator as the feature pipeline (``gvp.features.estimate_foreground``)
    so it inherits, and exposes, the estimator's own failure modes.
    """
    def _fn(img: np.ndarray) -> np.ndarray:
        from .features import estimate_foreground
        mask = estimate_foreground(img)
        if dilate:
            mask = cv2.dilate(mask, np.ones((dilate, dilate), np.uint8), iterations=1)
        out = np.full_like(img, fill)
        sel = mask > 0
        out[sel] = img[sel]
        return out

    _fn.__name__ = f"objonly{dilate}"
    return _fn


def extract_split(data_dir: str, manifest: str, blocks: Sequence[str] = FEATURE_BLOCKS,
                  tag: str = "", use_cache: bool = True,
                  transform=None) -> Dict[str, np.ndarray]:
    """Feature-matrix for one manifest (train/val/test/shift_*). Cached on disk."""
    cpath = _cache_path(data_dir, manifest, blocks, tag + (getattr(transform, "__name__", "") if transform else ""))
    meta_keys = ("label", "material", "object", "background", "sample_id", "shift", "ambiguous")
    if use_cache and os.path.exists(cpath):
        z = np.load(cpath, allow_pickle=True)
        return {"X": z["X"], "names": list(z["names"]), **{k: z[k] for k in meta_keys}}

    rows = synth.load_manifest(os.path.join(data_dir, f"manifest_{manifest}.csv"))
    feats: List[Dict[str, float]] = []
    t0 = time.perf_counter()
    for r in rows:
        img = cv2.imread(os.path.join(data_dir, r["path"]), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(os.path.join(data_dir, r["path"]))
        if transform is not None:
            img = transform(img)
        feats.append(extract_features(img, blocks=blocks))
    dt = (time.perf_counter() - t0) / max(len(rows), 1) * 1000.0
    names = sorted(feats[0].keys())
    X = np.array([[f.get(n, 0.0) for n in names] for f in feats], dtype=np.float64)
    out = {"X": X, "names": names}
    out.update({k: np.array([r[k] for r in rows]) for k in meta_keys})
    out["feature_ms_per_image"] = np.array([dt])
    if use_cache:
        os.makedirs(os.path.dirname(cpath), exist_ok=True)
        np.savez_compressed(cpath, X=X, names=np.array(names),
                            **{k: out[k] for k in meta_keys})
    return out


# --------------------------------------------------------------------------------------
# model zoo (classical / shallow)
# --------------------------------------------------------------------------------------


def build_model_zoo(seed: int = 0) -> Dict[str, object]:
    """Shallow models used in the study. Kept deliberately small and bibliographic:
    linear -> kernel -> probabilistic -> tree ensembles -> boosted -> shallow NN."""
    z: Dict[str, object] = {
        "logreg_l2": Pipeline([("sc", StandardScaler()),
                               ("clf", LogisticRegression(max_iter=3000, C=1.0, random_state=seed))]),
        "svm_linear": Pipeline([("sc", StandardScaler()), ("clf", LinearSVC(C=0.5, random_state=seed))]),
        "svm_rbf": Pipeline([("sc", StandardScaler()),
                             ("clf", SVC(C=4.0, gamma="scale", random_state=seed))]),
        "knn_5": Pipeline([("sc", StandardScaler()),
                           ("clf", KNeighborsClassifier(n_neighbors=5, weights="distance"))]),
        "gaussian_nb": Pipeline([("sc", StandardScaler()), ("clf", GaussianNB())]),
        "lda_shrinkage": Pipeline([("sc", StandardScaler()),
                                   ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))]),
        "random_forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                                n_jobs=-1, random_state=seed),
        "extra_trees": ExtraTreesClassifier(n_estimators=600, min_samples_leaf=1,
                                            n_jobs=-1, random_state=seed),
        "hist_gbdt": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                                    random_state=seed),
        "mlp_64_32": Pipeline([("sc", StandardScaler()),
                               ("clf", MLPClassifier((64, 32), alpha=1e-3, max_iter=800,
                                                     early_stopping=True, random_state=seed))]),
    }
    return z


def _scores(model: object, X: np.ndarray) -> Optional[np.ndarray]:
    if hasattr(model, "predict_proba"):
        try:
            return np.asarray(model.predict_proba(X))[:, 1]
        except Exception:
            pass
    if hasattr(model, "decision_function"):
        try:
            return np.asarray(model.decision_function(X)).ravel()
        except Exception:
            pass
    return None


def _latency_ms(model: object, X: np.ndarray, repeats: int = 5) -> float:
    model.predict(X[: min(8, len(X))])  # warm-up
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model.predict(X)
        times.append((time.perf_counter() - t0) / len(X) * 1000.0)
    return float(np.median(times))


# --------------------------------------------------------------------------------------
# 1. model comparison
# --------------------------------------------------------------------------------------


def run_model_comparison(data_dir: str, out_dir: str,
                         blocks: Sequence[str] = FEATURE_BLOCKS,
                         models: Optional[Sequence[str]] = None,
                         train_split: str = "train", eval_splits: Sequence[str] = ("val", "test"),
                         seed: int = 0) -> Tuple[List[dict], Dict[str, object]]:
    """Fit every shallow model on hand-crafted features; measure accuracy, latency and size."""
    os.makedirs(out_dir, exist_ok=True)
    tr = extract_split(data_dir, train_split, blocks)
    Xtr, ytr = tr["X"], tr["label"].astype(int)
    evals = {s: extract_split(data_dir, s, blocks) for s in eval_splits}

    zoo = build_model_zoo(seed)
    names = list(models or zoo.keys())
    rows: List[dict] = []
    fitted: Dict[str, object] = {}
    for name in names:
        model = zoo[name]
        t0 = time.perf_counter()
        model.fit(Xtr, ytr)
        train_time = time.perf_counter() - t0
        size_kb = len(pickle.dumps(model)) / 1024.0
        for split, ev in evals.items():
            X, y = ev["X"], ev["label"].astype(int)
            pred = model.predict(X)
            sc = _scores(model, X)
            m = evaluate.binary_metrics(y, pred, sc)
            lo, hi = evaluate.bootstrap_ci(y, pred, n_boot=300, seed=seed)
            m.update({
                "model": name, "split": split, "family": "classical (hand-crafted features)",
                "n_features": int(X.shape[1]), "feature_blocks": "+".join(blocks),
                "fit_seconds": round(train_time, 2),
                "model_size_kb": round(size_kb, 1),
                "clf_ms_per_image": round(_latency_ms(model, X), 4),
                "ba_ci_lo": lo, "ba_ci_hi": hi,
            })
            rows.append(m)
        fitted[name] = model

    rows.sort(key=lambda r: (r["split"], -r["balanced_accuracy"]))
    _write_csv(rows, os.path.join(out_dir, "classical_model_comparison.csv"))
    return rows, fitted


# --------------------------------------------------------------------------------------
# 2. feature-block ablation
# --------------------------------------------------------------------------------------


def run_feature_ablation(data_dir: str, out_dir: str,
                         blocks: Sequence[str] = FEATURE_BLOCKS,
                         models: Sequence[str] = ("logreg_l2", "svm_rbf", "extra_trees"),
                         eval_split: str = "test", seed: int = 0) -> List[dict]:
    """Single-block, all-blocks and leave-one-out feature ablations."""
    combos: List[Tuple[str, Tuple[str, ...]]] = [("all", tuple(blocks))]
    combos += [(b, (b,)) for b in blocks]
    combos += [(f"all_minus_{b}", tuple(x for x in blocks if x != b)) for b in blocks]

    tr_blocks_cache: Dict[Tuple[str, ...], Dict[str, np.ndarray]] = {}
    rows: List[dict] = []
    zoo = build_model_zoo(seed)
    for label, combo in combos:
        if combo not in tr_blocks_cache:
            tr_blocks_cache[combo] = extract_split(data_dir, "train", combo)
        tr = tr_blocks_cache[combo]
        ev = extract_split(data_dir, eval_split, combo)
        y = ev["label"].astype(int)
        for mname in models:
            model = build_model_zoo(seed)[mname]
            model.fit(tr["X"], tr["label"].astype(int))
            pred = model.predict(ev["X"])
            rows.append({
                "combo": label, "model": mname, "n_features": int(tr["X"].shape[1]),
                "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
                "accuracy": float((pred == y).mean()),
            })
    _write_csv(rows, os.path.join(out_dir, "feature_block_ablation.csv"))
    return rows


# --------------------------------------------------------------------------------------
# 3. cue ablation ("what is the model actually looking at?")
# --------------------------------------------------------------------------------------


def run_cue_ablation(out_dir: str,
                     root_dir: str = "data/processed/cue_ablation",
                     n_per_class: int = 400,
                     size: int = 128,
                     models: Sequence[str] = ("svm_rbf", "extra_trees", "logreg_l2"),
                     seed: int = 0,
                     cues: Sequence[str] = synth.CUE_KEYS) -> List[dict]:
    """Regenerate the proxy dataset with one physical cue removed at a time.

    A large drop when cue *c* is disabled means the models rely on *c* -- i.e. the cue is
    load-bearing for the decision. A small drop means the remaining cues/features are
    redundant (which is what you want for robustness, and what you must verify before
    trusting any accuracy figure from an RGB-only pipeline).
    """
    rows: List[dict] = []
    configs: List[Tuple[str, synth.CueStrength]] = [("none_removed", synth.CueStrength())]
    configs += [(f"without_{c}", synth.CueStrength().without(c)) for c in cues]
    configs += [("all_cues_weakened", synth.CueStrength.uniform(0.15))]

    for label, cue_strength in configs:
        ddir = os.path.join(root_dir, label)
        if not os.path.exists(os.path.join(ddir, "manifest_test.csv")):
            synth.build_proxy_dataset(ddir, n_per_class=n_per_class, size=size, seed=seed,
                                      cues=cue_strength, ambiguous_fraction=0.10)
        tr = extract_split(ddir, "train", FEATURE_BLOCKS, tag="cue")
        ev = extract_split(ddir, "test", FEATURE_BLOCKS, tag="cue")
        y = ev["label"].astype(int)
        for mname in models:
            model = build_model_zoo(seed)[mname]
            model.fit(tr["X"], tr["label"].astype(int))
            pred = model.predict(ev["X"])
            m = evaluate.binary_metrics(y, pred, _scores(model, ev["X"]))
            lo, hi = evaluate.bootstrap_ci(y, pred, n_boot=400, seed=seed)
            m.update({"cue_removed": label, "model": mname,
                      "ba_ci_lo": lo, "ba_ci_hi": hi,
                      "ba_ci_width": hi - lo,
                      "n_test": int(len(y))})
            rows.append(m)
    _write_csv(rows, os.path.join(out_dir, "cue_ablation.csv"))
    return rows


# --------------------------------------------------------------------------------------
# 4. domain-shift robustness
# --------------------------------------------------------------------------------------


def run_domain_shift(data_dir: str, out_dir: str,
                     blocks: Sequence[str] = FEATURE_BLOCKS,
                     models: Sequence[str] = ("svm_rbf", "extra_trees", "hist_gbdt"),
                     shifts: Sequence[str] = synth.SHIFT_KINDS,
                     seed: int = 0) -> List[dict]:
    """Train on clean in-distribution data, test on degraded / shifted acquisitions."""
    tr = extract_split(data_dir, "train", blocks)
    zoo = build_model_zoo(seed)
    rows: List[dict] = []
    for mname in models:
        model = zoo[mname]
        model.fit(tr["X"], tr["label"].astype(int))
        for split in ("test", *(f"shift_{s}" for s in shifts)):
            if not os.path.exists(os.path.join(data_dir, f"manifest_{split}.csv")):
                continue
            ev = extract_split(data_dir, split, blocks)
            y = ev["label"].astype(int)
            pred = model.predict(ev["X"])
            m = evaluate.binary_metrics(y, pred, _scores(model, ev["X"]))
            m.update({"model": mname, "split": split, "shift": split.replace("shift_", ""),
                      "drop_vs_clean": None})
            rows.append(m)
    for mname in models:  # compute drop relative to the clean test split
        base = next((r["balanced_accuracy"] for r in rows if r["model"] == mname and r["split"] == "test"), None)
        for r in rows:
            if r["model"] == mname and base is not None:
                r["drop_vs_clean"] = float(r["balanced_accuracy"] - base)
    _write_csv(rows, os.path.join(out_dir, "domain_shift_robustness.csv"))
    return rows


# --------------------------------------------------------------------------------------
# 5. interpretable one-feature physics baseline
# --------------------------------------------------------------------------------------


RULE_FEATURES = ("spec_peak_spikiness", "spec_saturated_frac", "tr_haze_index",
                 "tex_lap_var_inside")


def run_rule_baseline(data_dir: str, out_dir: str,
                      blocks: Sequence[str] = FEATURE_BLOCKS,
                      eval_split: str = "test") -> List[dict]:
    """Single-threshold rules: the 'physics prior' floor that a learned model must beat.

    Thresholds are calibrated on the *train* split only (median split), then frozen.
    """
    tr = extract_split(data_dir, "train", blocks)
    ev = extract_split(data_dir, eval_split, blocks)
    y = ev["label"].astype(int)
    names = tr["names"]
    rows: List[dict] = []
    for fname in RULE_FEATURES:
        if fname not in names:
            continue
        j = names.index(fname)
        thr = float(np.median(tr["X"][:, j]))
        direction = 1.0
        if balanced_accuracy_score(tr["label"].astype(int), (tr["X"][:, j] > thr).astype(int)) < 0.5:
            direction = -1.0
        pred = (direction * ev["X"][:, j] > direction * thr).astype(int)
        m = evaluate.binary_metrics(y, pred, direction * ev["X"][:, j])
        m.update({"rule": f"{'high' if direction > 0 else 'low'}_{fname}", "threshold": thr,
                  "split": eval_split, "n_features": 1})
        rows.append(m)
    _write_csv(rows, os.path.join(out_dir, "rule_baseline.csv"))
    return rows


# --------------------------------------------------------------------------------------
# 6. shortcut checks (must be run before believing anything)
# --------------------------------------------------------------------------------------


def run_sanity_checks(data_dir: str, out_dir: str, seed: int = 0,
                      blocks: Sequence[str] = FEATURE_BLOCKS) -> List[dict]:
    """(a) shuffled-label control, (b) background-only control.

    * Shuffled labels: any score above ~0.55 means a leak (e.g. split contamination).
    * Background-only: features computed on the *image border* only. If the model still
      scores well, the dataset carries a scene/background shortcut -- the single most common
      failure mode of published waste-classification results.
    """
    rows: List[dict] = []
    tr = extract_split(data_dir, "train", blocks)
    ev = extract_split(data_dir, "test", blocks)
    y = ev["label"].astype(int)
    rng = np.random.default_rng(seed)

    for mname in ("svm_rbf", "extra_trees"):
        model = build_model_zoo(seed)[mname]
        y_shuf = rng.permutation(tr["label"].astype(int))
        model.fit(tr["X"], y_shuf)
        pred = model.predict(ev["X"])
        rows.append({"check": "shuffled_labels", "model": mname,
                     "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
                     "expected": "≈0.50 (leak if > 0.55)",
                     "interpretation": "control: no label information should be learnable"})

    # background-only (scene) control, at two window sizes
    for cover in (0.5, 0.8):
        tf = mask_centre(cover)
        tr_bg = extract_split(data_dir, "train", blocks, transform=tf)
        ev_bg = extract_split(data_dir, "test", blocks, transform=tf)
        for mname in ("svm_rbf", "extra_trees"):
            model = build_model_zoo(seed)[mname]
            model.fit(tr_bg["X"], tr_bg["label"].astype(int))
            pred = model.predict(ev_bg["X"])
            rows.append({"check": f"background_only_cover{int(cover * 100)}", "model": mname,
                         "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
                         "expected": "≈0.50 (scene shortcut if > 0.60)",
                         "interpretation": f"central {int(cover * 100)}% blanked; "
                                           f"small windows leave peripheral object pixels visible"})
    _write_csv(rows, os.path.join(out_dir, "sanity_checks.csv"))
    return rows


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _write_csv(rows: List[dict], path: str) -> str:
    if not rows:
        return path
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cols: List[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return path


def top_discriminative_features(fitted: object, names: Sequence[str], top_k: int = 20) -> List[Tuple[str, float]]:
    """Feature importances / coefficients from a fitted tree ensemble or linear model."""
    est = fitted
    if isinstance(est, Pipeline):
        est = est[-1]
    imp = getattr(est, "feature_importances_", None)
    if imp is None:
        coef = getattr(est, "coef_", None)
        imp = None if coef is None else np.abs(np.ravel(coef))
    if imp is None:
        return []
    order = np.argsort(np.abs(imp))[::-1][:top_k]
    return [(names[i], float(imp[i])) for i in order]
