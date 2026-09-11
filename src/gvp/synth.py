"""Synthetic glass / plastic proxy image generator.

WHY THIS EXISTS
---------------
Real glass-vs-plastic benchmarks require large labelled datasets (TrashNet, RealWaste,
ZeroWaste, TACO, ...) that cannot be redistributed inside a code repository. This module
generates a *physically-motivated proxy dataset* so that every stage of the study
(feature extraction -> classical models -> deep models -> robustness sweeps -> reporting)
is executable and unit-testable on a laptop with no downloads and no GPU.

WHAT IT MODELS (and what it does not)
-------------------------------------
The renderer deliberately encodes the *cues* that the literature says separate glass from
plastic in RGB imagery (see ``docs/02_physics_and_cues.md``):

    cue            glass                        plastic
    -------------  ---------------------------  -----------------------------------
    transparency   background transmitted,      background attenuated + blurred (haze),
                   sharp, mildly refracted      strong diffuse albedo
    specular       few, small, very bright,     fewer, large, softer, lower peak
                   elongated, near silhouettes
    chromatic      visible fringing on edges    negligible
    texture        smooth, near-uniform        ribs, mould marks, grain, film wrinkles
    label/marking  occasional (jar labels)      common (stickers, resin codes)
    silhouette     thin, high-contrast rim      thicker, softer, often thicker wall

Every cue has a strength knob (``CueStrength``). Setting a knob to 0 removes that cue from
*both* classes, which is how ``gvp.classical.run_cue_ablation`` measures how much
classification performance actually depends on each cue -- i.e. a shortcut-learning probe.

CAVEAT, STATED PLAINLY: this is a renderer, not a camera. Numbers obtained on it are
diagnostics of *method behaviour* (which models, which features, which hyper-parameters,
how robust to degradation), NOT measurements of real-world glass/plastic accuracy. The
reporting layer prints this caveat next to every number it emits.
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

CUE_KEYS: Tuple[str, ...] = (
    "transparency",   # background visibility through the object
    "specular",       # highlight sharpness / size separation
    "haze",           # plastic diffuse attenuation of the background
    "texture",        # ribs, grain, film wrinkles
    "label",          # stickers / printed markings
    "distortion",     # refraction (glass) vs flat transmission (plastic)
    "chromatic",      # chromatic fringing on strong edges
    "edge_contrast",  # rim gradient strength (thin bright rim for glass)
)

SCENE_BACKGROUNDS: Tuple[str, ...] = (
    "conveyor_dark",
    "conveyor_blue",
    "table_wood",
    "white_lab",
    "cluttered",
    "gradient_grey",
)

#: Special background used only by the ``whitebg`` domain shift: a featureless studio
#: backdrop. The object is kept intact, the *scene context* is removed -- which is precisely
#: the transferable-cue experiment: no transmitted background to look at, only the object.
FLAT_WHITE_BG = "flat_white"

OBJECT_KINDS: Tuple[str, ...] = ("bottle", "jar", "cup", "shard", "sheet")

SHIFT_KINDS: Tuple[str, ...] = (
    "blur",           # defocus / motion blur
    "dark",           # low illumination
    "noise",          # sensor noise
    "jpeg",           # compression artefacts
    "lowcontrast",    # washed-out illumination
    "clutter",        # debris, other items and glare on the belt
    "whitebg",        # featureless studio backdrop: removes the scene-context cue
                       # (implemented by re-rendering on a flat background, so the object
                       #  itself is preserved -- only the transmitted context is removed)
)


@dataclass
class CueStrength:
    """Per-cue strength multipliers in [0, 1+]. 0 disables the cue for both classes."""

    transparency: float = 1.0
    specular: float = 1.0
    haze: float = 1.0
    texture: float = 1.0
    label: float = 1.0
    distortion: float = 1.0
    chromatic: float = 1.0
    edge_contrast: float = 1.0

    def as_dict(self) -> Dict[str, float]:
        return {k: float(getattr(self, k)) for k in CUE_KEYS}

    @classmethod
    def uniform(cls, value: float) -> "CueStrength":
        return cls(**{k: float(value) for k in CUE_KEYS})

    def without(self, key: str) -> "CueStrength":
        d = self.as_dict()
        d[key] = 0.0
        return CueStrength(**d)


# --------------------------------------------------------------------------------------
# Low level rendering helpers
# --------------------------------------------------------------------------------------


def _rng(seed: Optional[int] = None) -> np.random.Generator:
    return np.random.default_rng(seed)


def _background(rng: np.random.Generator, size: int, kind: str) -> np.ndarray:
    """Render a textured *scene* background. Texture matters: it is what transmission reveals."""
    img = np.zeros((size, size, 3), np.float32)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)

    if kind == "conveyor_dark":
        base = rng.uniform(38, 60)
        img[:] = base
        period = rng.integers(9, 22)
        stripes = (np.sin(xx * 2 * math.pi / period) > 0).astype(np.float32)
        img += stripes[..., None] * rng.uniform(4, 12)
        img += rng.normal(0, 2.5, img.shape)
    elif kind == "conveyor_blue":
        img[:] = np.array([rng.uniform(70, 95), rng.uniform(45, 65), rng.uniform(25, 40)], np.float32)
        img += (np.sin(xx * 2 * math.pi / rng.integers(14, 30)) > 0)[..., None] * 8.0
        img += rng.normal(0, 3.0, img.shape)
    elif kind == "table_wood":
        base = rng.uniform(90, 130)
        grain = np.sin(xx * 0.35 + 3.0 * np.sin(yy * 0.06)) * 12.0
        img[:] = base + grain[..., None] * np.array([0.6, 0.9, 0.4], np.float32)
        img += rng.normal(0, 4.0, img.shape)
    elif kind == "white_lab":
        img[:] = rng.uniform(215, 245)
        img += rng.normal(0, 1.6, img.shape)
    elif kind == FLAT_WHITE_BG:
        img[:] = rng.uniform(238, 252)  # featureless: nothing to transmit
        img += rng.normal(0, 0.8, img.shape)
    elif kind == "cluttered":
        img[:] = rng.uniform(120, 190)
        for _ in range(int(rng.integers(5, 14))):
            x0, y0 = rng.integers(0, size, 2)
            w, h = rng.integers(size // 12, size // 3, 2)
            colour = rng.integers(30, 235, 3).astype(np.float32)
            shape = int(rng.integers(0, 2))
            if shape == 0:
                cv2.rectangle(img, (int(x0), int(y0)), (int(x0 + w), int(y0 + h)), colour.tolist(), -1)
            else:
                cv2.circle(img, (int(x0), int(y0)), int(min(w, h) // 2), colour.tolist(), -1)
        img = cv2.GaussianBlur(img, (0, 0), 1.5)
        img += rng.normal(0, 4.0, img.shape)
    else:  # gradient_grey
        lo, hi = sorted(rng.uniform(45, 230, 2))
        ramp = (xx / max(size - 1, 1))
        img[:] = (lo + (hi - lo) * ramp)[..., None]
        img += rng.normal(0, 2.0, img.shape)

    # global illumination
    gain = rng.uniform(0.85, 1.15)
    img *= gain
    # vignette
    r = np.sqrt((xx - size / 2) ** 2 + (yy - size / 2) ** 2) / (size * 0.75)
    img *= (1.0 - 0.25 * r)[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)


def _object_mask(rng: np.random.Generator, size: int, kind: str) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Return a hard binary mask (0/255) of a plausible recyclable silhouette."""
    mask = np.zeros((size, size), np.uint8)
    cx = int(size * rng.uniform(0.34, 0.66))
    cy = int(size * rng.uniform(0.40, 0.62))
    scale = size * rng.uniform(0.42, 0.68)

    if kind == "bottle":
        bw = int(scale * rng.uniform(0.42, 0.58) * 1.6)
        bh = int(scale * 1.25)
        top, bottom = cy - bh // 2, cy + bh // 2
        body_top = top + int(bh * 0.24)
        cv2.rectangle(mask, (cx - bw // 2, body_top), (cx + bw // 2, bottom), 255, -1)
        neck_w = max(4, int(bw * rng.uniform(0.16, 0.26)))
        cv2.rectangle(mask, (cx - neck_w // 2, top), (cx + neck_w // 2, body_top), 255, -1)
        cv2.ellipse(mask, (cx, bottom - 2), (bw // 2, int(bw * 0.10)), 0, 0, 180, 255, -1)
        cap_h = int(bh * 0.07)
        cv2.rectangle(mask, (cx - int(neck_w * 0.75), top), (cx + int(neck_w * 0.75), top + cap_h), 255, -1)
    elif kind == "jar":
        bw = int(scale * rng.uniform(0.7, 0.95))
        bh = int(bw * rng.uniform(0.85, 1.25))
        cv2.rectangle(mask, (cx - bw // 2, cy - bh // 2), (cx + bw // 2, cy + bh // 2), 255, -1)
        k = int(bw * 0.14)
        for ys in (cy - bh // 2, cy + bh // 2):
            for xs in (cx - bw // 2, cx + bw // 2):
                cv2.circle(mask, (xs, ys), k, 255, -1)
        cv2.rectangle(mask, (cx - int(bw * 0.3), cy - bh // 2 - int(bh * 0.12)),
                      (cx + int(bw * 0.3), cy - bh // 2 + 2), 255, -1)  # lid
    elif kind == "cup":
        bw_top = int(scale * rng.uniform(0.6, 0.8))
        bw_bot = int(bw_top * rng.uniform(0.58, 0.76))
        bh = int(scale * rng.uniform(0.8, 1.05))
        pts = np.array([
            [cx - bw_top // 2, cy - bh // 2],
            [cx + bw_top // 2, cy - bh // 2],
            [cx + bw_bot // 2, cy + bh // 2],
            [cx - bw_bot // 2, cy + bh // 2],
        ], np.int32)
        cv2.fillPoly(mask, [pts], 255)
        cv2.ellipse(mask, (cx, cy - bh // 2), (bw_top // 2, int(bw_top * 0.12)), 0, 0, 360, 255, -1)
    elif kind == "shard":
        n = int(rng.integers(4, 8))
        angles = np.sort(rng.uniform(0, 2 * math.pi, n))
        radii = scale * 0.5 * rng.uniform(0.55, 1.0, n)
        pts = np.stack([cx + radii * np.cos(angles), cy + radii * np.sin(angles)], 1).astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)
    else:  # sheet / film
        w = int(scale * rng.uniform(0.9, 1.3))
        h = int(scale * rng.uniform(0.45, 0.8))
        rect = ((cx, cy), (w, h), float(rng.uniform(0, 180)))
        box = cv2.boxPoints(rect).astype(np.int32)
        cv2.fillPoly(mask, [box], 255)
        if rng.random() < 0.6:  # wrinkled film
            for _ in range(int(rng.integers(2, 6))):
                p0 = (int(rng.integers(0, size)), int(rng.integers(0, size)))
                p1 = (int(rng.integers(0, size)), int(rng.integers(0, size)))
                cv2.line(mask, p0, p1, 255, int(rng.integers(1, 4)))

    if mask.max() == 0:  # degenerate draw
        cv2.circle(mask, (cx, cy), int(scale * 0.35), 255, -1)

    ys, xs = np.nonzero(mask)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    return mask, bbox


def _elastic(a_map: np.ndarray, rng: np.random.Generator, magnitude: float) -> np.ndarray:
    """Smooth random displacement field applied to an image (refraction / transmission proxy)."""
    h, w = a_map.shape[:2]
    low = max(4, int(min(h, w) / 24))
    dx = rng.normal(0, 1, (low, low)).astype(np.float32)
    dy = rng.normal(0, 1, (low, low)).astype(np.float32)
    dx = cv2.resize(dx, (w, h), interpolation=cv2.INTER_CUBIC) * magnitude
    dy = cv2.resize(dy, (w, h), interpolation=cv2.INTER_CUBIC) * magnitude
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(a_map, gx + dx, gy + dy, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)


def _anisotropic_blob(shape: Tuple[int, int], rng: np.random.Generator,
                      sigma_lo: float, sigma_hi: float, peak: float,
                      size_scale: float = 1.0) -> np.ndarray:
    h, w = shape
    sigma = rng.uniform(sigma_lo, sigma_hi) * size_scale
    aspect = rng.uniform(1.6, 6.0) if rng.random() < 0.7 else rng.uniform(1.0, 1.6)
    theta = rng.uniform(0, math.pi)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = rng.uniform(0, w), rng.uniform(0, h)
    xr = (xx - cx) * math.cos(theta) + (yy - cy) * math.sin(theta)
    yr = -(xx - cx) * math.sin(theta) + (yy - cy) * math.cos(theta)
    blob = np.exp(-(xr ** 2 / (2 * (sigma * aspect) ** 2) + yr ** 2 / (2 * sigma ** 2)))
    return (blob * peak).astype(np.float32)


def _feather(mask: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    a = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), sigma)
    return np.clip(a, 0.0, 1.0)


# --------------------------------------------------------------------------------------
# Material rendering
# --------------------------------------------------------------------------------------


def _render_glass(bg: np.ndarray, mask: np.ndarray, rng: np.random.Generator, cues: CueStrength) -> np.ndarray:
    h, w = mask.shape
    inner = bg.copy()

    # 1. transmission with refraction
    if cues.distortion > 0:
        inner = _elastic(inner, rng, magnitude=float(rng.uniform(2.0, 7.0)) * cues.distortion)
    if cues.transparency > 0:
        # clear glass: background almost fully visible, slight cool cast, sharpness kept
        cast = np.array([rng.uniform(0.97, 1.03), rng.uniform(0.99, 1.06), rng.uniform(1.00, 1.08)], np.float32)
        inner = inner.astype(np.float32) * cast
        inner = inner * (1.0 - 0.06 * cues.transparency) + bg * (0.06 * cues.transparency)
        # slight local contrast boost (glass reads as "crisp")
        blur = cv2.GaussianBlur(inner, (0, 0), 1.2)
        inner = np.clip(inner + (inner - blur) * 0.35 * cues.transparency, 0, 255)
    else:
        # CUE REMOVED. Both materials must now be rendered with the *same* transmission
        # model, otherwise disabling the knob would sharpen the contrast between classes
        # instead of erasing the cue. Glass falls back to the attenuated-diffuse model.
        grey = float(rng.uniform(150, 215))
        inner = 0.5 * cv2.GaussianBlur(bg.astype(np.float32), (0, 0), 1.0) + 0.5 * grey

    # 2. rim: thin bright band just inside the silhouette
    if cues.edge_contrast > 0:
        er = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
        band = cv2.bitwise_and(mask, cv2.bitwise_not(er))
        inner[band > 0] = np.clip(inner[band > 0] * (1.06 + 0.16 * cues.edge_contrast) + 12.0, 0, 255)

    # 3. specular: small, bright, elongated, edge-hugging
    if cues.specular > 0:
        acc = np.zeros((h, w), np.float32)
        for _ in range(int(rng.integers(1, 4))):
            acc += _anisotropic_blob((h, w), rng, 1.4, 3.2, float(rng.uniform(215, 255)))
        acc = cv2.GaussianBlur(acc, (0, 0), 0.7) * cues.specular
        ys, xs = np.nonzero(cv2.dilate(mask, np.ones((5, 5), np.uint8)) > 0)
        keep = np.zeros((h, w), np.float32)
        if len(ys):
            idx = rng.choice(len(ys), size=min(len(ys), max(40, len(ys) // 3)), replace=False)
            keep[ys[idx], xs[idx]] = 1.0
        acc *= cv2.GaussianBlur(keep, (0, 0), 3.0)
        inner = inner + acc[..., None]
    return inner


def _render_plastic(bg: np.ndarray, mask: np.ndarray, rng: np.random.Generator, cues: CueStrength) -> np.ndarray:
    h, w = mask.shape
    palette = np.array([
        [235, 235, 238],   # white HDPE
        [175, 150, 210],   # light blue PET
        [120, 175, 120],   # green
        [90, 120, 195],    # amber/brown
        [200, 205, 215],   # grey
        [140, 140, 150],
        [60, 70, 90],      # dark
        [230, 200, 120],   # cream
    ], np.float32)
    albedo = palette[int(rng.integers(0, len(palette)))] * rng.uniform(0.85, 1.05)

    # 1. attenuation + haze: background is blurred and dimmed behind the wall
    if cues.haze > 0:
        sigma = float(rng.uniform(2.0, 5.0)) * cues.haze
        transmitted = cv2.GaussianBlur(bg, (0, 0), sigma) * (1.0 - 0.45 * cues.haze)
    else:
        transmitted = bg.astype(np.float32)
    inner = transmitted.copy()

    # 2. diffuse albedo, opaque-ish (lower transparency than glass). The coupling to the
    # transparency knob is only active while the cue exists, so that removing the cue
    # erases the *difference* between the materials rather than amplifying it.
    opacity = float(rng.uniform(0.45, 0.85))
    if cues.transparency > 0:
        opacity *= (1.0 - 0.35 * cues.transparency)
    yy = np.mgrid[0:h, 0:w][0].astype(np.float32)
    shade = 1.0 + 0.18 * (yy / max(h - 1, 1)) - 0.09
    inner = inner * (1.0 - opacity) + (albedo[None, None, :] * shade[..., None]) * opacity

    # 3. surface texture: mould ribs / grain / wrinkles
    if cues.texture > 0:
        period = rng.uniform(5.0, 16.0)
        ribs = (np.sin(yy * 2 * math.pi / period + rng.uniform(0, 3)) * 9.0) * cues.texture
        inner = inner + ribs[..., None]
        grain = cv2.GaussianBlur(rng.normal(0, 1.0, (h, w)).astype(np.float32), (0, 0), 0.6)
        inner = inner + (grain * 11.0 * cues.texture)[..., None]

    # 4. specular: large, soft, low peak
    if cues.specular > 0:
        acc = np.zeros((h, w), np.float32)
        for _ in range(int(rng.integers(1, 3))):
            acc += _anisotropic_blob((h, w), rng, 5.0, 13.0, float(rng.uniform(150, 205)),
                                     size_scale=1.4)
        acc = cv2.GaussianBlur(acc, (0, 0), 3.0) * cues.specular
        inner = inner + acc[..., None]

    # 5. label / sticker
    if cues.label > 0 and rng.random() < 0.55:
        ys, xs = np.nonzero(mask)
        if len(xs):
            hx0, hx1 = xs.min(), xs.max()
            lw = int((hx1 - hx0) * rng.uniform(0.35, 0.7))
            lh = int(lw * rng.uniform(0.28, 0.55))
            if lw > 5 and lh > 4:
                lx = int(rng.integers(hx0, max(hx0 + 1, hx1 - lw)))
                ly = int(rng.integers(ys.min(), max(ys.min() + 1, ys.max() - lh)))
                colour = rng.integers(20, 240, 3).astype(np.float32)
                sub = inner[ly:ly + lh, lx:lx + lw]
                inner[ly:ly + lh, lx:lx + lw] = sub * 0.25 + colour * 0.75
                for _ in range(int(rng.integers(3, 9))):  # printed marks
                    x0 = int(rng.integers(lx + 1, max(lx + 2, lx + lw - 2)))
                    y0 = int(rng.integers(ly + 1, max(ly + 2, ly + lh - 2)))
                    cv2.line(inner, (x0, y0), (x0 + int(rng.integers(2, 6)), y0), 25.0, 1)
    return inner


def render_sample(material: str,
                  object_kind: str,
                  background: str,
                  rng: np.random.Generator,
                  size: int = 160,
                  cues: Optional[CueStrength] = None) -> np.ndarray:
    """Render one synthetic image (BGR uint8)."""
    cues = cues or CueStrength()
    bg = _background(rng, size, background)
    mask, _bbox = _object_mask(rng, size, object_kind)

    if material == "glass":
        inner = _render_glass(bg, mask, rng, cues)
    else:
        inner = _render_plastic(bg, mask, rng, cues)

    # chromatic aberration on strong edges (before compositing so it lives on the object)
    if cues.chromatic > 0:
        shift = max(1, int(round(rng.uniform(1, 2.6) * cues.chromatic)))
        inner[..., 2] = np.roll(inner[..., 2], shift, axis=1)   # R
        inner[..., 0] = np.roll(inner[..., 0], -shift, axis=1)  # B

    # occasional transparent plastic / frosted glass ambiguity: applies to both classes
    alpha = _feather(mask, sigma=float(rng.uniform(0.8, 1.8)))
    out = bg.astype(np.float32) * (1.0 - alpha[..., None]) + inner.astype(np.float32) * alpha[..., None]

    # global degradations
    if rng.random() < 0.35:
        out = cv2.GaussianBlur(out, (0, 0), float(rng.uniform(0.4, 1.3)))
    if rng.random() < 0.3:  # soft shadow under the object
        sh = cv2.GaussianBlur(alpha, (0, 0), 4.0)
        out = out * (1.0 - 0.22 * sh[..., None])
    out = np.clip(out, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        out = np.clip(out.astype(np.float32) + rng.normal(0, rng.uniform(1.0, 4.0), out.shape), 0, 255).astype(np.uint8)
    return out


# --------------------------------------------------------------------------------------
# Domain shifts (test-time degradation / distribution shift)
# --------------------------------------------------------------------------------------


def apply_shift(img: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    """Apply an acquisition-time domain shift. Used for robustness sweeps."""
    out = img.astype(np.float32)
    if kind == "blur":
        out = cv2.GaussianBlur(out, (0, 0), float(rng.uniform(1.8, 3.2)))
    elif kind == "dark":
        out = out * rng.uniform(0.30, 0.50)
    elif kind == "noise":
        out = out + rng.normal(0, rng.uniform(18, 34), out.shape)
    elif kind == "jpeg":
        ok, enc = cv2.imencode(".jpg", np.clip(out, 0, 255).astype(np.uint8),
                               [int(cv2.IMWRITE_JPEG_QUALITY), int(rng.integers(8, 22))])
        if ok:
            out = cv2.imdecode(enc, cv2.IMREAD_COLOR).astype(np.float32)
    elif kind == "lowcontrast":
        out = (out - out.mean()) * rng.uniform(0.35, 0.55) + rng.uniform(110, 150)
    elif kind == "clutter":
        h, w = out.shape[:2]
        # debris and other items on the belt: mostly near the periphery, plus glare streaks
        # across the frame. Deliberately does NOT bury the object -- heavy occlusion is a
        # different experiment (and a different pipeline: detect-then-classify).
        for _ in range(int(rng.integers(3, 9))):
            x0, y0 = rng.integers(0, w, 2)
            r = int(rng.integers(w // 24, w // 9))
            cx0, cy0 = w / 2, h / 2
            central = (abs(x0 - cx0) < w * 0.22) and (abs(y0 - cy0) < h * 0.22)
            if central and rng.random() < 0.75:
                continue  # keep the central object area usable
            colour = rng.integers(20, 240, 3).astype(np.float32)
            cv2.circle(out, (int(x0), int(y0)), r, colour.tolist(), -1)
        for _ in range(int(rng.integers(1, 4))):  # glare / reflection streaks
            p0 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            cv2.line(out, p0, p1, (250, 250, 255), int(rng.integers(2, 7)))
        out = cv2.GaussianBlur(out, (0, 0), 0.8)
    elif kind == "background_removal":  # unused placeholder, kept out of SHIFT_KINDS
        out = out
    else:
        raise ValueError(f"unknown shift {kind!r}; expected one of {SHIFT_KINDS}")
    return np.clip(out, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------------------
# Dataset driver
# --------------------------------------------------------------------------------------


def build_proxy_dataset(out_dir: str,
                        n_per_class: int = 300,
                        size: int = 160,
                        seed: int = 0,
                        cues: Optional[CueStrength] = None,
                        shifts: Sequence[str] = (),
                        n_shift: int = 120,
                        ambiguous_fraction: float = 0.18,
                        backgrounds: Sequence[str] = SCENE_BACKGROUNDS,
                        save_images: bool = True) -> Dict[str, str]:
    """Write a proxy dataset and its manifest.

    Returns a dict of ``{split_name: manifest_csv_path}``.

    Splits
    ------
    train / val / test : in-distribution scenes, disjoint object instances, disjoint seeds.
    shift_<kind>       : fresh scenes (new seed + background palette perturbation) with an
                         acquisition shift applied -- used as a *cross-domain* test set.
    """
    cues = cues or CueStrength()
    os.makedirs(out_dir, exist_ok=True)
    img_dir = os.path.join(out_dir, "images")
    if save_images:
        os.makedirs(img_dir, exist_ok=True)

    def _emit(rows: List[dict], name: str) -> str:
        path = os.path.join(out_dir, f"manifest_{name}.csv")
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def _generate(n_per_class_local: int, tag: str, rng: np.random.Generator,
                  shift: Optional[str], bgs: Sequence[str]) -> List[dict]:
        rows: List[dict] = []
        for material in ("glass", "plastic"):
            for i in range(n_per_class_local):
                rng_local = np.random.default_rng(rng.integers(0, 2**63 - 1))
                obj = OBJECT_KINDS[int(rng_local.integers(0, len(OBJECT_KINDS)))]
                # the whitebg shift is a *scene* change (re-render on a flat backdrop);
                # every other shift is a post-hoc acquisition degradation.
                bg = bgs[int(rng_local.integers(0, len(bgs)))]
                if shift == "whitebg":
                    bg = FLAT_WHITE_BG
                # ambiguity: render with weakened class-typical cues
                local_cues = cues
                ambiguous = rng_local.random() < ambiguous_fraction
                if ambiguous:
                    mixed = cues.as_dict()
                    for k in CUE_KEYS:
                        mixed[k] = float(mixed[k]) * float(rng_local.uniform(0.15, 0.5))
                    local_cues = CueStrength(**mixed)
                img = render_sample(material, obj, bg, rng_local, size=size, cues=local_cues)
                if shift and shift != "whitebg":
                    img = apply_shift(img, shift, rng_local)
                sid = f"{tag}_{material}_{i:05d}"
                path = os.path.join("images", f"{sid}.png")
                if save_images:
                    cv2.imwrite(os.path.join(out_dir, path), img)
                rows.append({
                    "sample_id": sid,
                    "path": path,
                    "label": 1 if material == "glass" else 0,
                    "material": material,
                    "object": obj,
                    "background": bg,
                    "size": size,
                    "split": tag,
                    "shift": shift or "none",
                    "ambiguous": int(ambiguous),
                    **{f"cue_{k}": float(getattr(local_cues, k)) for k in CUE_KEYS},
                })
        return rows

    n_train = int(round(n_per_class * 0.70))
    n_val = int(round(n_per_class * 0.15))
    n_test = n_per_class - n_train - n_val

    manifests: Dict[str, str] = {}
    rng = _rng(seed)
    manifests["train"] = _emit(_generate(n_train, "train", rng, None, backgrounds), "train")
    manifests["val"] = _emit(_generate(n_val, "val", rng, None, backgrounds), "val")
    manifests["test"] = _emit(_generate(n_test, "test", rng, None, backgrounds), "test")

    for k, shift in enumerate(shifts):
        # OOD scenes: different background palette + different seed stream
        shifted_bgs = list(np.random.default_rng(seed + 100 + k).permutation(np.array(backgrounds)))
        rows = _generate(max(2, n_shift // 2), f"shift_{shift}", _rng(seed + 1000 + k), shift, shifted_bgs)
        manifests[f"shift_{shift}"] = _emit(rows, f"shift_{shift}")

    print(f"[synth] wrote {len(manifests)} manifests to {out_dir}")
    return manifests


def load_manifest(path: str) -> List[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))
