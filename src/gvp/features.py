"""Hand-crafted, interpretable descriptors for glass-vs-plastic discrimination.

Five feature blocks are implemented, each mapped to a physical cue the literature
identifies as separable (see ``docs/02_physics_and_cues.md`` and
``docs/04_classical_pipelines.md``):

=============  =====================================================================
block          what it measures (and the physics behind it)
=============  =====================================================================
``color``      HSV/Lab statistics, saturation, tint, colour entropy.  Weak but cheap;
               glass is achromatic/cool-cast, plastic often dyed or white.
``specular``   highlight morphology: bright-pixel fraction, blob count/size, peak
               spikiness, elongation.  Glass = few small near-saturated highlights;
               plastic = broad, softer sheen.
``texture``    LBP + GLCM + high-frequency energy + local entropy.  Plastic carries
               mould lines, ribs, grain, film wrinkles; glass is smooth.
``edge``       Canny density, gradient orientation histogram, contour geometry,
               corner/line counts, largest-contour solidity and aspect ratio.
``transparency``  does the background *continue through* the object? Compares gradient
               energy and background-contrast statistics inside an estimated object
               region against an annulus outside it.  Proxy for transmission vs haze.
=============  =====================================================================

Important design constraint: **every feature must be computable from the raw image
alone** -- no ground-truth mask.  The object region is *estimated* with
:func:`estimate_foreground`, so the transparency block inherits the segmentation error
that a real deployment would have.  That is deliberate: it keeps the offline study
honest about the fact that a coarse segmenter is part of the classical pipeline.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np

try:  # scikit-image is optional for the texture block
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

    _HAVE_SKIMAGE = True
except Exception:  # pragma: no cover
    _HAVE_SKIMAGE = False

FEATURE_BLOCKS: Tuple[str, ...] = ("color", "specular", "texture", "edge", "transparency")


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    p = p / p.sum()
    return float(-(p * np.log2(p)).sum())


def estimate_foreground(img_bgr: np.ndarray) -> np.ndarray:
    """Cheap segmentation: distance from the border colour statistics + Otsu + cleanup.

    Returns a uint8 mask (0/255). Errors here propagate into the transparency block.
    """
    h, w = img_bgr.shape[:2]
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    b = max(2, int(min(h, w) * 0.06))
    border = np.concatenate([
        img_lab[:b].reshape(-1, 3), img_lab[-b:].reshape(-1, 3),
        img_lab[:, :b].reshape(-1, 3), img_lab[:, -b:].reshape(-1, 3),
    ])
    mu = border.mean(0)
    sd = border.std(0) + 1e-3
    dist = np.sqrt((((img_lab - mu) / sd) ** 2).sum(-1))
    dist = cv2.GaussianBlur(dist.astype(np.float32), (0, 0), 1.5)
    dn = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(dn, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n > 1:
        keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = np.where(lab == keep, 255, 0).astype(np.uint8)
    frac = float((mask > 0).mean())
    if frac < 0.04 or frac > 0.92:  # fall back to a centred box
        mask = np.zeros((h, w), np.uint8)
        mask[int(h * 0.15):int(h * 0.85), int(w * 0.15):int(w * 0.85)] = 255
    return mask


# --------------------------------------------------------------------------------------
# blocks
# --------------------------------------------------------------------------------------


def color_features(img: np.ndarray) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hh = cv2.calcHist([hsv], [0], None, [16], [0, 180]).ravel()
    ss = cv2.calcHist([hsv], [1], None, [8], [0, 256]).ravel()
    vv = cv2.calcHist([hsv], [2], None, [8], [0, 256]).ravel()
    feats.update({f"col_h{i}": float(v) for i, v in enumerate(hh / max(hh.sum(), 1))})
    feats.update({f"col_s{i}": float(v) for i, v in enumerate(ss / max(ss.sum(), 1))})
    feats.update({f"col_v{i}": float(v) for i, v in enumerate(vv / max(vv.sum(), 1))})

    feats["col_hue_entropy"] = _entropy(hh / max(hh.sum(), 1))
    feats["col_sat_mean"] = float(hsv[..., 1].mean() / 255.0)
    feats["col_sat_std"] = float(hsv[..., 1].std() / 255.0)
    feats["col_sat_p95"] = float(np.percentile(hsv[..., 1], 95) / 255.0)
    feats["col_val_mean"] = float(hsv[..., 2].mean() / 255.0)
    feats["col_val_std"] = float(hsv[..., 2].std() / 255.0)
    feats["col_dark_frac"] = float((g < 60).mean())
    feats["col_bright_frac"] = float((g > 200).mean())
    feats["col_lab_a_mean"] = float(lab[..., 1].mean() / 255.0)
    feats["col_lab_b_mean"] = float(lab[..., 2].mean() / 255.0)
    # cool/warm cast proxies (glass tends slightly cool/green, PET often blue)
    b, gr, r = img[..., 0].astype(np.float32), img[..., 1].astype(np.float32), img[..., 2].astype(np.float32)
    feats["col_tint_b_minus_r"] = float((b - r).mean() / 255.0)
    feats["col_tint_g_minus_rb"] = float((gr - 0.5 * (b + r)).mean() / 255.0)
    feats["col_chroma_max"] = float(np.maximum.reduce([b, gr, r]).mean() / 255.0)
    return feats


def specular_features(img: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Highlight morphology. The strongest single cue family for glass."""
    feats: Dict[str, float] = {}
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    inside = g[mask > 0]
    if inside.size < 32:
        inside = g.ravel()

    p99 = float(np.percentile(inside, 99))
    p95 = float(np.percentile(inside, 95))
    p50 = float(np.percentile(inside, 50))
    mx = float(inside.max())
    feats["spec_p99"] = p99 / 255.0
    feats["spec_p50"] = p50 / 255.0
    feats["spec_peak_spikiness"] = float((mx - p99) / max(p99 - p50, 1e-3))
    feats["spec_upper_tail"] = float((p99 - p95) / max(p95 - p50, 1e-3))
    feats["spec_saturated_frac"] = float((inside >= 245).mean())
    feats["spec_bright_frac"] = float((inside >= np.percentile(inside, 99)).mean())

    thr = max(200.0, p99)
    bright = ((g >= thr) & (mask > 0)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bright, 8)
    areas = stats[1:, cv2.CC_STAT_AREA] if n > 1 else np.array([0])
    areas = np.sort(areas)[::-1]
    feats["spec_blob_count"] = float(max(0, n - 1))
    feats["spec_blob_area_max"] = float(areas[0] / inside.size)
    feats["spec_blob_area_mean"] = float(areas[:5].mean() / inside.size)
    # highlight compactness / elongation from the largest bright blob
    elong = 0.0
    feats["spec_highlight_area_frac"] = 0.0  # always emitted: keeps the feature vector fixed
    if (bright.sum() > 8):
        cnts, _ = cv2.findContours(bright * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            if len(c) >= 5:
                try:
                    (_, _), (w0, h0), _ = cv2.fitEllipse(c)
                    if np.isfinite(w0) and np.isfinite(h0) and min(w0, h0) > 1e-6:
                        elong = float(max(w0, h0) / min(w0, h0))
                except cv2.error:
                    elong = 0.0
                feats["spec_highlight_area_frac"] = float(cv2.contourArea(c) / inside.size)
    feats["spec_elongation"] = float(min(elong, 25.0))
    feats["spec_highlight_frac"] = float(bright.mean())
    return feats


def texture_features(img: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sel = mask > 0
    if sel.sum() < 64:
        sel = np.ones_like(g, bool)

    if _HAVE_SKIMAGE:
        for P, R, nb in ((8, 1, 10), (16, 2, 18)):
            lbp = local_binary_pattern(g, P, R, "uniform")
            hist, _ = np.histogram(lbp[sel], bins=nb, range=(0, nb), density=False)
            hist = hist / max(hist.sum(), 1)
            feats.update({f"tex_lbp{P}_{i}": float(v) for i, v in enumerate(hist)})
            feats[f"tex_lbp{P}_entropy"] = _entropy(hist)
        q = (g // 32).astype(np.uint8)  # 8 levels
        glcm = graycomatrix(q, distances=[1, 3], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                            levels=8, symmetric=True, normed=True)
        for prop in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation"):
            vals = graycoprops(glcm, prop)
            feats[f"tex_glcm_{prop}_mean"] = float(np.mean(vals))
            feats[f"tex_glcm_{prop}_d1"] = float(np.mean(vals[0]))  # fine scale
            feats[f"tex_glcm_{prop}_d3"] = float(np.mean(vals[1]))  # coarse scale
        feats["tex_glcm_anisotropy"] = float(
            abs(np.mean(graycoprops(glcm, "contrast")[0, 0]) -
                np.mean(graycoprops(glcm, "contrast")[1, 2])) /
            max(np.mean(graycoprops(glcm, "contrast")) + 1e-6, 1e-6))
    else:  # pragma: no cover - fallback wording
        feats["tex_lbp_missing"] = 1.0

    lap = cv2.Laplacian(g, cv2.CV_32F)
    feats["tex_lap_var_inside"] = float(lap[sel].var() / 1000.0)
    blur = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 1.0)
    hf = (g.astype(np.float32) - blur)[sel]
    feats["tex_hf_energy"] = float(np.sqrt((hf ** 2).mean()) / 10.0)
    feats["tex_hf_kurtosis"] = float(((hf ** 4).mean() / max((hf ** 2).mean() ** 2, 1e-6)))
    mean = cv2.blur(g.astype(np.float32), (9, 9))
    sq = cv2.blur((g.astype(np.float32)) ** 2, (9, 9))
    local_std = np.sqrt(np.clip(sq - mean ** 2, 0, None))
    feats["tex_local_std_mean"] = float(local_std[sel].mean() / 50.0)
    feats["tex_local_std_p90"] = float(np.percentile(local_std[sel], 90) / 50.0)
    # stripes / mould lines: energy in a middle frequency band
    f = np.abs(np.fft.fftshift(np.fft.fft2(cv2.resize(g, (128, 128)).astype(np.float32))))
    yy, xx = np.mgrid[0:128, 0:128] - 64
    rad = np.sqrt(xx ** 2 + yy ** 2)
    low = f[(rad > 2) & (rad <= 6)].mean()
    mid = f[(rad > 8) & (rad <= 20)].mean()
    feats["tex_band_mid_over_low"] = float(mid / max(low, 1e-3))
    return feats


def edge_features(img: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (0, 0), 1.0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    feats["edge_mag_mean"] = float(mag.mean() / 255.0)
    feats["edge_mag_p95"] = float(np.percentile(mag, 95) / 255.0)
    feats["edge_mag_std"] = float(mag.std() / 255.0)

    ang = (np.arctan2(gy, gx) + np.pi) % np.pi
    hist, _ = np.histogram(ang[mask > 0], bins=8, range=(0, np.pi), weights=mag[mask > 0])
    hist = hist / max(hist.sum(), 1e-6)
    feats.update({f"edge_orient{i}": float(v) for i, v in enumerate(hist)})
    feats["edge_orient_entropy"] = _entropy(hist)

    canny = cv2.Canny(g, 60, 160)
    feats["edge_canny_density"] = float(canny.mean() / 255.0)

    cnts, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    feats["edge_n_contours"] = float(min(len(cnts), 200))
    if cnts and cv2.contourArea(cnts[0]) > 20:
        c = cnts[0]
        a = float(cv2.contourArea(c))
        p = float(cv2.arcLength(c, True))
        feats["edge_main_circularity"] = float(4 * np.pi * a / max(p ** 2, 1e-6))
        x, y, w0, h0 = cv2.boundingRect(c)
        feats["edge_main_aspect"] = float(max(w0, h0) / max(min(w0, h0), 1))
        feats["edge_main_extent"] = float(a / max(w0 * h0, 1))
        hull = cv2.convexHull(c)
        feats["edge_main_solidity"] = float(a / max(cv2.contourArea(hull), 1e-6))
        feats["edge_main_rect_fill"] = float(a / (g.shape[0] * g.shape[1]))
        peri = hasattr(cv2, "isContourConvex") and True
        del peri
    else:
        for k in ("edge_main_circularity", "edge_main_aspect", "edge_main_extent",
                  "edge_main_solidity", "edge_main_rect_fill"):
            feats[k] = 0.0

    lines = cv2.HoughLinesP(canny, 1, np.pi / 180, threshold=28,
                            minLineLength=max(12, g.shape[0] // 10), maxLineGap=4)
    if lines is not None:
        L = lines[:, 0, :]
        lengths = np.hypot(L[:, 2] - L[:, 0], L[:, 3] - L[:, 1])
        feats["edge_n_lines"] = float(min(len(L), 100))
        feats["edge_line_len_mean"] = float(lengths.mean() / g.shape[0])
        feats["edge_line_len_max"] = float(lengths.max() / g.shape[0])
    else:
        feats["edge_n_lines"] = 0.0
        feats["edge_line_len_mean"] = 0.0
        feats["edge_line_len_max"] = 0.0

    corners = cv2.goodFeaturesToTrack(g, maxCorners=90, qualityLevel=0.05, minDistance=5)
    feats["edge_n_corners"] = float(0 if corners is None else len(corners))
    return feats


def transparency_features(img: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Does the scene continue *through* the object?

    Compares the gradient/contrast statistics of the estimated object interior against a
    surrounding annulus, plus how much of the interior is explained by the blurred
    background statistics (transmission) rather than an opaque albedo (diffuse).
    """
    feats: Dict[str, float] = {}
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sel = mask > 0
    ann = (cv2.dilate(mask, np.ones((15, 15), np.uint8)) > 0) & (~sel)
    if sel.sum() < 64 or ann.sum() < 64:
        for k in ("tr_in_grad_mean", "tr_out_grad_mean", "tr_grad_ratio", "tr_in_contrast",
                  "tr_out_contrast", "tr_contrast_ratio", "tr_haze_index", "tr_bg_continuity",
                  "tr_area_frac", "tr_inside_gradient_orient_coh"):
            feats[k] = 0.0
        return feats

    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)

    feats["tr_in_grad_mean"] = float(mag[sel].mean() / 255.0)
    feats["tr_out_grad_mean"] = float(mag[ann].mean() / 255.0)
    feats["tr_grad_ratio"] = float(feats["tr_in_grad_mean"] / max(feats["tr_out_grad_mean"], 1e-4))
    feats["tr_in_contrast"] = float(g[sel].std() / 255.0)
    feats["tr_out_contrast"] = float(g[ann].std() / 255.0)
    feats["tr_contrast_ratio"] = float(feats["tr_in_contrast"] / max(feats["tr_out_contrast"], 1e-4))

    # haze index: interior variance vs a blurred version -> opaque diffuse surfaces lose fine detail
    inside = g[sel]
    blurred = cv2.GaussianBlur(g, (0, 0), 2.5)[sel]
    feats["tr_haze_index"] = float((inside - blurred).std() / max(inside.std(), 1e-4))

    # background continuity: compare interior colour to the local background estimate
    bg_est = cv2.medianBlur(cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2LAB), (32, 32)).astype(np.uint8), 5)
    bg_est = cv2.resize(bg_est, (g.shape[1], g.shape[0]))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    feats["tr_bg_continuity"] = float(
        (1.0 - np.abs(lab[sel].mean(0) - bg_est[sel].mean(0)).mean() / 128.0))

    # orientational coherence of interior gradients (transmitted scene keeps global structure)
    try:
        e, v = np.linalg.eigh(np.cov(np.stack([gx[sel], gy[sel]])))
        feats["tr_inside_gradient_orient_coh"] = float(e[-1] / max(e.sum(), 1e-6))
    except Exception:
        feats["tr_inside_gradient_orient_coh"] = 0.0
    feats["tr_area_frac"] = float(sel.mean())
    del v
    return feats


# --------------------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------------------

_BLOCK_FUNCS = {
    "color": color_features,
    "specular": specular_features,
    "texture": texture_features,
    "edge": edge_features,
    "transparency": transparency_features,
}


def extract_features(img_bgr: np.ndarray,
                     blocks: Sequence[str] = FEATURE_BLOCKS,
                     mask: np.ndarray | None = None) -> Dict[str, float]:
    """Extract a flat dict of features for one image."""
    unknown = set(blocks) - set(FEATURE_BLOCKS)
    if unknown:
        raise ValueError(f"unknown feature blocks: {sorted(unknown)}")
    if mask is None:
        mask = estimate_foreground(img_bgr)
    feats: Dict[str, float] = {}
    for name in blocks:
        fn = _BLOCK_FUNCS[name]
        out = fn(img_bgr, mask) if name != "color" else fn(img_bgr)
        feats.update(out)
    # Safety net: degenerate images (uniform frames, zero-variance patches, collinear
    # contours) can make a descriptor undefined. Map those to 0.0 *explicitly* rather than
    # letting NaN propagate into training -- a NaN silently voiding a model fit is exactly
    # the kind of bug that makes a benchmark unreproducible.
    for k, v in list(feats.items()):
        if v is None or not np.isfinite(v):
            feats[k] = 0.0
    return feats


def feature_names(blocks: Iterable[str] = FEATURE_BLOCKS) -> List[str]:
    """Enumerate feature names by running the extractor on a tiny probe image."""
    probe = np.zeros((64, 64, 3), np.uint8)
    probe[16:48, 16:48] = 200
    h, w = 64, 64
    cv2.circle(probe, (w // 2, h // 2), 14, (240, 235, 230), -1)
    feats = extract_features(probe, blocks=tuple(blocks))
    return sorted(feats.keys())
