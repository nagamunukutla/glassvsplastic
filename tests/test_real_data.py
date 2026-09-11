"""Tests for the real-data pipeline: dataset conversion, controls, crop builder.

These run without network access and without the TrashNet download: they synthesise tiny fixtures
that exercise the code paths that the real-data study depends on.
"""

from __future__ import annotations

import csv
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import cv2  # noqa: E402

from gvp import classical  # noqa: E402
from gvp.features import FEATURE_BLOCKS, extract_features  # noqa: E402

import real_data_study as rds  # noqa: E402

import build_measured_comparison as bmc  # noqa: E402
import merge_deep_results as mdr  # noqa: E402
import prepare_real_dataset as prep  # noqa: E402


def _fake_scene(bg_bgr, size=(128, 96), obj=True, obj_bgr=(60, 60, 60)):
    img = np.zeros((size[1], size[0], 3), np.float32)
    img[:] = bg_bgr
    img += np.random.default_rng(0).normal(0, 3, img.shape).astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if obj:
        cv2.rectangle(img, (size[0] // 3, size[1] // 4), (2 * size[0] // 3, 3 * size[1] // 4),
                      obj_bgr, -1)
    return img


class TestBackgroundTyper(unittest.TestCase):
    def test_white_backdrop_detected(self):
        kind, stats = prep.background_type(_fake_scene((245, 245, 245)))
        self.assertEqual(kind, "white")
        self.assertGreater(stats["bg_brightness"], 232)

    def test_cardboard_backdrop_detected(self):
        # warm, mid-brightness, saturated: the TrashNet cardboard backdrop
        kind, _ = prep.background_type(_fake_scene((120, 160, 190)))
        self.assertEqual(kind, "cardboard")

    def test_grey_backdrop_detected(self):
        kind, _ = prep.background_type(_fake_scene((190, 192, 195)))
        self.assertEqual(kind, "grey")

    def test_colour_backdrop_detected(self):
        kind, _ = prep.background_type(_fake_scene((60, 180, 60)))  # saturated green
        self.assertEqual(kind, "colour")

    def test_stats_are_complete(self):
        _, stats = prep.background_type(_fake_scene((245, 245, 245)))
        for k in ("bg_brightness", "bg_saturation", "bg_std", "bg_b", "bg_g", "bg_r", "bg_kind"):
            self.assertIn(k, stats)


class TestSplitFileParsing(unittest.TestCase):
    """The index-base trap: 1-indexed means glass=1, plastic=4."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="gvp_real_")
        for cls in ("glass", "plastic", "metal"):
            os.makedirs(os.path.join(self.tmp, "imgs", cls), exist_ok=True)
        cv2.imwrite(os.path.join(self.tmp, "imgs", "glass", "g1.jpg"), _fake_scene((240, 240, 240)))
        cv2.imwrite(os.path.join(self.tmp, "imgs", "plastic", "p1.jpg"), _fake_scene((240, 240, 240)))
        cv2.imwrite(os.path.join(self.tmp, "imgs", "metal", "m1.jpg"), _fake_scene((240, 240, 240)))
        # the one-indexed variant that TrashNet actually ships: glass=1, paper=2,
        # cardboard=3, plastic=4, metal=5, trash=6
        with open(os.path.join(self.tmp, "split_train.txt"), "w") as fh:
            fh.write("g1.jpg 1\np1.jpg 4\nm1.jpg 5\n")
        with open(os.path.join(self.tmp, "split_val.txt"), "w") as fh:
            fh.write("g1.jpg 1\n")
        with open(os.path.join(self.tmp, "split_test.txt"), "w") as fh:
            fh.write("p1.jpg 4\n")
        # the zero-indexed variant: glass=0, paper=1, cardboard=2, plastic=3, metal=4, trash=5
        for split, body in (("z_train", "g1.jpg 0\np1.jpg 3\nm1.jpg 4\n"),
                            ("z_val", "g1.jpg 0\n"), ("z_test", "p1.jpg 3\n")):
            with open(os.path.join(self.tmp, f"split_{split}.txt"), "w") as fh:
                fh.write(body)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_one_indexed_mapping_keeps_glass_and_plastic(self):
        paths = prep.prepare_splitfiles(self.tmp, "imgs",
                                        ["split_train", "split_val", "split_test"], 1,
                                        "glass", "plastic", self.tmp)
        self.assertTrue(os.path.exists(paths["train"]))
        with open(paths["train"], newline="") as fh:
            rows = list(csv.DictReader(fh))
        # metal (label 5) must be filtered out; glass=1 and plastic=4 must be kept
        self.assertEqual({r["material"] for r in rows}, {"glass", "plastic"})
        labels = {r["material"]: int(r["label"]) for r in rows}
        self.assertEqual(labels, {"glass": 1, "plastic": 0})

    def test_zero_indexed_variant_works_with_base_zero(self):
        paths = prep.prepare_splitfiles(self.tmp, "imgs",
                                        ["split_z_train", "split_z_val", "split_z_test"], 0,
                                        "glass", "plastic", os.path.join(self.tmp, "z"))
        with open(paths["train"], newline="") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual({r["material"] for r in rows}, {"glass", "plastic"})

    def test_wrong_index_base_silently_drops_a_class(self):
        """The documented trap, asserted rather than described.

        Passing ``--index-base 0`` against a one-indexed split file reads label 4 as 'metal' and
        label 1 as 'paper': the plastic images vanish and glass is silently lost. The corpus
        shrinks and the run *succeeds*, which is exactly why ``prepare_real_dataset.py`` reports
        the class histograms it derived and why this test pins the behaviour.
        """
        with self.assertRaises(SystemExit) as ctx:
            prep.prepare_splitfiles(self.tmp, "imgs",
                                    ["split_train", "split_val", "split_test"], 0,
                                    "glass", "plastic", os.path.join(self.tmp, "b0"))
        msg = str(ctx.exception)
        self.assertIn("index-base", msg)
        self.assertIn("fails loudly", msg)


class TestControlTransforms(unittest.TestCase):
    def test_outer_ring_keeps_only_the_border(self):
        img = _fake_scene((200, 200, 200), obj_bgr=(10, 10, 200))
        out = classical.mask_outer_ring(keep=0.05)(img)
        h, w = img.shape[:2]
        m = max(2, int(min(h, w) * 0.05))
        self.assertTrue(np.array_equal(out[m:-m, m:-m], np.full_like(out[m:-m, m:-m], 127)))
        self.assertTrue(np.array_equal(out[:m], img[:m]))
        self.assertEqual(out.shape, img.shape)

    def test_mask_non_object_keeps_object_region(self):
        img = _fake_scene((235, 235, 235), obj_bgr=(20, 20, 20))
        out = classical.mask_non_object(dilate=9)(img)
        self.assertEqual(out.shape, img.shape)
        # some object pixels survive, and some background is replaced
        self.assertTrue((out.reshape(-1, 3) == np.array([127, 127, 127])).all(1).any())
        self.assertGreater(float((out == img).all(2).mean()), 0.0)

    def test_features_stay_finite_after_transforms(self):
        img = _fake_scene((240, 240, 240))
        for tf in (classical.mask_centre(0.9), classical.mask_outer_ring(0.03),
                   classical.mask_non_object(dilate=9), classical.mask_border(0.25)):
            f = extract_features(tf(img), FEATURE_BLOCKS)
            self.assertTrue(all(np.isfinite(v) for v in f.values()))


class TestCropBuilder(unittest.TestCase):
    def test_cropped_copy_is_smaller_and_manifest_consistent(self):
        import deep_real_study as drs

        src = tempfile.mkdtemp(prefix="gvp_crop_src_")
        dst = tempfile.mkdtemp(prefix="gvp_crop_dst_")
        try:
            for split, n in (("train", 3), ("val", 2), ("test", 2)):
                rows = []
                for i in range(n):
                    img = _fake_scene((235, 235, 235), size=(160, 120))
                    rel = os.path.join("images", f"{split}_{i}.jpg")
                    os.makedirs(os.path.join(src, "images"), exist_ok=True)
                    cv2.imwrite(os.path.join(src, rel), img)
                    rows.append({"sample_id": f"{split}_{i}", "path": rel,
                                 "label": 1 if i % 2 else 0, "material": "glass" if i % 2 else "plastic",
                                 "object": "unknown", "background": "white", "split": split,
                                 "shift": "none", "ambiguous": 0, "width": 160, "height": 120})
                with open(os.path.join(src, f"manifest_{split}.csv"), "w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)
            paths = drs.build_cropped_copy(src, dst, margin=0.12)
            self.assertEqual(set(paths), {"train", "val", "test"})
            with open(paths["test"], newline="") as fh:
                out_rows = list(csv.DictReader(fh))
            for r in out_rows:
                self.assertTrue(os.path.exists(os.path.join(dst, r["path"])))
                self.assertEqual(int(r["label"]), 1 if r["sample_id"].endswith(("1",)) else 0)
                # crops must not be larger than the source frame
                self.assertLessEqual(int(r["crop_w"]), 160)
                self.assertLessEqual(int(r["crop_h"]), 120)
        finally:
            shutil.rmtree(src, ignore_errors=True)
            shutil.rmtree(dst, ignore_errors=True)


class TestMeasuredComparison(unittest.TestCase):
    """The fused comparison must be reproducible from the CSVs alone, with no invented cells."""

    def _write(self, path, rows):
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    def test_fuses_rules_classical_and_deep(self):
        d = tempfile.mkdtemp()
        try:
            self._write(os.path.join(d, "real_rule_baseline.csv"), [{
                "rule": "low_tex_lap_var_inside", "balanced_accuracy": 0.50,
                "glass_recall": 0.52, "plastic_recall": 0.47,
                "glass_to_plastic_rate": 0.48, "plastic_to_glass_rate": 0.53,
                "split": "test"}])
            self._write(os.path.join(d, "real_model_comparison.csv"), [{
                "model": "hist_gbdt", "split": "test", "balanced_accuracy": 0.88,
                "ba_ci_lo": 0.82, "ba_ci_hi": 0.92, "glass_recall": 0.85,
                "plastic_recall": 0.91, "glass_to_plastic_rate": 0.15,
                "plastic_to_glass_rate": 0.09, "clf_ms_per_image": 0.04}])
            self._write(os.path.join(d, "deep_results.csv"), [{
                "model": "tv:resnet18", "variant": "object_crop", "split": "test",
                "balanced_accuracy": 0.868, "glass_recall": 0.817, "plastic_recall": 0.919,
                "glass_to_plastic_rate": 0.183, "plastic_to_glass_rate": 0.081,
                "params_m": 11.18, "macs_g": 1.33, "latency_ms_per_image": 27.1,
                "epochs": 8, "notes": "real TrashNet glass-vs-plastic, object_crop"}])
            out = bmc.build(d)
            text = open(out).read()
            for needle in ("hist_gbdt", "resnet18", "low tex lap var inside", "object crop"):
                self.assertIn(needle, text)
            # the rules have no timed latency: the cell must be an em dash, never "None"
            self.assertNotIn("None", text)
            # both inputs must be represented in the table
            self.assertIn("full frame", text)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_absent_deep_csv_degrades_gracefully(self):
        d = tempfile.mkdtemp()
        try:
            self._write(os.path.join(d, "real_rule_baseline.csv"), [{
                "rule": "r", "balanced_accuracy": 0.5, "glass_recall": 0.5,
                "plastic_recall": 0.5, "glass_to_plastic_rate": 0.5,
                "plastic_to_glass_rate": 0.5, "split": "test"}])
            text = open(bmc.build(d)).read()
            self.assertIn("r", text)
            self.assertIn("Controls", text + "Controls")  # section headings present
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestDeepResultMerge(unittest.TestCase):
    """A second backbone must never erase the first one's rows (it did once)."""

    def _rows(self, path):
        with open(path, newline="") as fh:
            return list(csv.DictReader(fh))

    def test_union_across_runs_and_reruns_replace_only_their_own_row(self):
        d = tempfile.mkdtemp()
        try:
            os.makedirs(os.path.join(d, "deep"))
            base = {"model": "", "accuracy": "0.9", "balanced_accuracy": "0.90",
                    "split": "test", "epochs": "8", "img_size": "160",
                    "notes": "real TrashNet, full_frame"}
            # run 1 wrote only into the per-run accumulator (older schema, no `variant`)
            with open(os.path.join(d, "deep", "deep_results.csv"), "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(base))
                w.writeheader()
                w.writerow({**base, "model": "tv:resnet18"})
            # run 2 wrote a curated copy with a variant column
            with open(os.path.join(d, "deep_results.csv"), "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(base) + ["variant"])
                w.writeheader()
                w.writerow({**base, "model": "tv:mobilenet_v3_large", "variant": "full_frame"})
            dst = mdr.merge(d)
            rows = self._rows(dst)
            self.assertEqual({r["model"] for r in rows},
                             {"tv:resnet18", "tv:mobilenet_v3_large"})
            # variant recovered from notes where the column was absent
            for r in rows:
                self.assertEqual(r["variant"], "full_frame")
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestDeepRowsTolerance(unittest.TestCase):
    """A run that did not record an optional field must still appear in the table."""

    def test_missing_optional_timing_does_not_delete_the_row(self):
        d = rds._deep_rows([
            {"model": "tv:resnet18", "variant": "full_frame", "balanced_accuracy": "0.93",
             "accuracy": "0.93", "glass_recall": "0.93", "plastic_recall": "0.93", "roc_auc": "0.98",
             "params_m": "11.18", "macs_g": "1.33", "latency_ms_per_image": "24.5",
             "epochs": "8", "wall_seconds": "",          # <- empty, not absent
             "img_size": "192"},
            {"model": "tv:x", "variant": "object_crop", "balanced_accuracy": "",   # no accuracy
             "epochs": "8"},
        ])
        self.assertEqual(len(d), 1)                       # only the row with an accuracy survives
        self.assertEqual(d[0]["model"], "resnet18")
        self.assertIsNone(d[0]["train (s)"])              # rendered as an em dash, not dropped
        self.assertEqual(d[0]["img (px)"], 192)

if __name__ == "__main__":
    unittest.main(verbosity=2)
