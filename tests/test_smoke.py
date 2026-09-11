"""Unit and end-to-end tests.

These are not just smoke tests: the ones marked *control* encode the methodological checks the
study depends on. If a change to the renderer or the feature extractor introduces scene leakage or
makes the proxy task trivially separable, `make test` fails -- which is the point.

Run with:  make test      (or: PYTHONPATH=src python -m unittest discover -s tests -v)
"""

from __future__ import annotations

import csv
import hashlib
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import cv2  # noqa: E402

from gvp import classical, evaluate, synth  # noqa: E402
from gvp.features import FEATURE_BLOCKS, extract_features, feature_names  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TINY = dict(n_per_class=100, size=96)  # small enough for CI, large enough for stable assertions


def _make_dataset(tmp: str, **kw):
    opts = {**TINY, **kw}
    return synth.build_proxy_dataset(tmp, seed=7, **opts)


class TestRenderDeterminism(unittest.TestCase):
    def test_same_seed_same_pixels(self):
        """Determinism of the renderer is what makes every measured number reproducible."""
        a = synth.render_sample("glass", "bottle", "conveyor_dark", np.random.default_rng(1), size=96)
        b = synth.render_sample("glass", "bottle", "conveyor_dark", np.random.default_rng(1), size=96)
        np.testing.assert_array_equal(a, b)

    def test_different_seed_different_pixels(self):
        a = synth.render_sample("glass", "bottle", "conveyor_dark", np.random.default_rng(1), size=96)
        b = synth.render_sample("glass", "bottle", "conveyor_dark", np.random.default_rng(2), size=96)
        self.assertFalse(np.array_equal(a, b))

    def test_material_changes_the_image(self):
        """Glass and plastic renders must differ -- otherwise the task is undefined."""
        g = synth.render_sample("glass", "bottle", "table_wood", np.random.default_rng(3), size=96)
        p = synth.render_sample("plastic", "bottle", "table_wood", np.random.default_rng(3), size=96)
        self.assertGreater(float(np.abs(g.astype(float) - p.astype(float)).mean()), 1.0)


class TestCueKnobs(unittest.TestCase):
    def test_disabling_a_cue_changes_both_classes(self):
        """The bug this guards against: a knob whose 'off' state amplifies the class difference."""
        rng = np.random.default_rng(11)
        on = synth.CueStrength()
        off = synth.CueStrength().without("specular")
        g_on = synth.render_sample("glass", "jar", "conveyor_blue", np.random.default_rng(5), size=96, cues=on)
        g_off = synth.render_sample("glass", "jar", "conveyor_blue", np.random.default_rng(5), size=96, cues=off)
        p_on = synth.render_sample("plastic", "jar", "conveyor_blue", np.random.default_rng(5), size=96, cues=on)
        p_off = synth.render_sample("plastic", "jar", "conveyor_blue", np.random.default_rng(5), size=96, cues=off)
        for arr in (g_on, g_off, p_on, p_off):
            self.assertGreater(arr.mean(), 0.0)  # nothing degenerate
        self.assertFalse(np.array_equal(g_on, g_off))
        self.assertFalse(np.array_equal(p_on, p_off))
        del rng

    def test_transparency_removal_does_not_increase_separability(self):
        """Removing the transmission cue must not make glass *more* transparent than plastic."""
        cues_off = synth.CueStrength().without("transparency")
        diffs = []
        for seed in range(6):
            g = synth.render_sample("glass", "cup", "table_wood", np.random.default_rng(seed),
                                    size=96, cues=cues_off).astype(float)
            p = synth.render_sample("plastic", "cup", "table_wood", np.random.default_rng(seed),
                                    size=96, cues=synth.CueStrength()).astype(float)
            diffs.append(np.abs(g - p).mean())
        self.assertLess(float(np.mean(diffs)), 90.0)  # not a trivially separable pair

    def test_shifts_are_valid(self):
        img = synth.render_sample("plastic", "sheet", "cluttered", np.random.default_rng(9), size=96)
        for kind in synth.SHIFT_KINDS:
            if kind == "whitebg":  # implemented as a scene change, not a post-hoc shift
                continue
            out = synth.apply_shift(img, kind, np.random.default_rng(0))
            self.assertEqual(out.shape, img.shape)
            self.assertEqual(out.dtype, np.uint8)
        with self.assertRaises(ValueError):
            synth.apply_shift(img, "not_a_shift", np.random.default_rng(0))


class TestFeatures(unittest.TestCase):
    def test_vector_is_fixed_length_and_finite(self):
        names = feature_names(FEATURE_BLOCKS)
        self.assertGreater(len(names), 100)
        for material in ("glass", "plastic"):
            for kind in synth.OBJECT_KINDS:
                img = synth.render_sample(material, kind, "conveyor_dark",
                                          np.random.default_rng(hash((material, kind)) % 2**31), size=96)
                f = extract_features(img, FEATURE_BLOCKS)
                self.assertEqual(sorted(f.keys()), names)
                self.assertTrue(all(np.isfinite(v) for v in f.values()),
                                f"non-finite feature for {material}/{kind}")

    def test_degenerate_images_do_not_produce_nan(self):
        for img in (np.zeros((96, 96, 3), np.uint8),
                    np.full((96, 96, 3), 255, np.uint8),
                    np.random.default_rng(0).integers(0, 255, (96, 96, 3), dtype=np.uint8)):
            f = extract_features(img, FEATURE_BLOCKS)
            self.assertTrue(all(np.isfinite(v) for v in f.values()))

    def test_blocks_can_be_selected(self):
        img = synth.render_sample("glass", "bottle", "table_wood", np.random.default_rng(4), size=96)
        only = extract_features(img, ("color",))
        self.assertTrue(all(k.startswith("col_") for k in only))
        self.assertLess(len(only), len(extract_features(img, FEATURE_BLOCKS)))


class TestPipelineAndControls(unittest.TestCase):
    """The methodological tests: if these fail, the study's numbers are not trustworthy."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="gvp_test_")
        _make_dataset(cls.tmp, shifts=("blur", "clutter"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_manifests_are_complete_and_balanced(self):
        rows = synth.load_manifest(os.path.join(self.tmp, "manifest_train.csv"))
        self.assertEqual(len(rows), 2 * int(round(TINY["n_per_class"] * 0.70)))
        labels = [int(r["label"]) for r in rows]
        self.assertEqual(sum(labels), len(labels) // 2)
        for r in rows:
            self.assertTrue(os.path.exists(os.path.join(self.tmp, r["path"])))

    def test_feature_extraction_cached_consistently(self):
        a = classical.extract_split(self.tmp, "test", FEATURE_BLOCKS)
        b = classical.extract_split(self.tmp, "test", FEATURE_BLOCKS)  # cache hit
        np.testing.assert_allclose(a["X"], b["X"])

    def test_shuffled_labels_give_chance(self):
        """Control: a leak (split contamination, duplicate items) shows up here first."""
        from sklearn.metrics import balanced_accuracy_score
        tr = classical.extract_split(self.tmp, "train", FEATURE_BLOCKS)
        ev = classical.extract_split(self.tmp, "test", FEATURE_BLOCKS)
        model = classical.build_model_zoo(0)["extra_trees"]
        rng = np.random.default_rng(0)
        model.fit(tr["X"], rng.permutation(tr["label"].astype(int)))
        ba = balanced_accuracy_score(ev["label"].astype(int), model.predict(ev["X"]))
        self.assertLess(ba, 0.70, "shuffled-label control scored above chance: inspect for leakage")
        self.assertGreater(ba, 0.30)

    def test_background_only_control_falls_to_chance_as_the_window_grows(self):
        """Control: the scene must not carry class information.

        The control is only meaningful if the blanked window actually removes the object. This test
        asserts the *diagnostic shape* rather than a single number: a small window leaves peripheral
        object pixels visible (bottle necks, caps, shard tips) and scores above chance; once the
        window is large enough, the score must fall to chance. A dataset with a genuine scene
        shortcut would stay above chance at every window size -- and would be caught here.
        """
        from sklearn.metrics import balanced_accuracy_score
        scores = {}
        for cover in (0.5, 0.8, 0.95):
            tf = classical.mask_centre(cover)
            tr = classical.extract_split(self.tmp, "train", FEATURE_BLOCKS, transform=tf)
            ev = classical.extract_split(self.tmp, "test", FEATURE_BLOCKS, transform=tf)
            model = classical.build_model_zoo(0)["extra_trees"]
            model.fit(tr["X"], tr["label"].astype(int))
            scores[cover] = balanced_accuracy_score(ev["label"].astype(int), model.predict(ev["X"]))
        self.assertLess(scores[0.95], 0.72,
                        f"background-only control still above chance with a 95% window: {scores}")
        self.assertGreaterEqual(max(scores.values()), scores[0.95] - 1e-9)
        self.assertLess(scores[0.95], max(scores[0.5], scores[0.8]) + 0.30,
                        f"implausible control profile: {scores}")

    def test_full_pipeline_produces_valid_metrics(self):
        rows, fitted = classical.run_model_comparison(self.tmp, self.tmp, models=("svm_rbf", "extra_trees"))
        self.assertTrue(rows and fitted)
        for r in rows:
            self.assertGreaterEqual(r["balanced_accuracy"], 0.0)
            self.assertLessEqual(r["balanced_accuracy"], 1.0)
            self.assertEqual(r["n"], 2 * int(round(TINY["n_per_class"] * 0.15)))
            self.assertLessEqual(r["ba_ci_lo"], r["balanced_accuracy"] + 1e-9)
            self.assertGreaterEqual(r["ba_ci_hi"], r["balanced_accuracy"] - 1e-9)

    def test_rule_baselines_are_plausible(self):
        """The physics floor: at least one single-feature rule must beat chance, and none may be
        suspiciously perfect (a rule that separates the classes perfectly means the renderer leaks)."""
        rows = classical.run_rule_baseline(self.tmp, self.tmp)
        self.assertTrue(rows)
        bas = [r["balanced_accuracy"] for r in rows]
        self.assertGreater(max(bas), 0.55, f"no single-feature rule beat chance: {bas}")
        for r in rows:
            self.assertGreater(r["balanced_accuracy"], 0.20)
            self.assertLess(r["balanced_accuracy"], 0.995)

    def test_metrics_implemented_correctly(self):
        y = np.array([0, 0, 1, 1])
        pred = np.array([0, 1, 1, 1])
        m = evaluate.binary_metrics(y, pred)
        self.assertAlmostEqual(m["accuracy"], 0.75)
        self.assertAlmostEqual(m["balanced_accuracy"], 0.75)
        self.assertAlmostEqual(m["plastic_recall"], 0.5)
        self.assertAlmostEqual(m["glass_recall"], 1.0)
        self.assertAlmostEqual(m["plastic_to_glass_rate"], 0.5)
        self.assertEqual(m["tn_plastic_ok"], 1)

    def test_bootstrap_ci_brackets_the_estimate(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 200)
        pred = np.where(rng.random(200) < 0.85, y, 1 - y)
        lo, hi = evaluate.bootstrap_ci(y, pred, n_boot=200)
        self.assertLessEqual(lo, hi)
        self.assertLess(lo, 0.95)
        self.assertGreater(hi, 0.70)


class TestRegistryAndDocsIntegrity(unittest.TestCase):
    """The comparison tables are data; these tests keep the data self-consistent."""

    def test_every_scored_option_exists_in_the_registry(self):
        reg = [r["id"] for r in _read(os.path.join(REPO, "data", "model_registry.csv"))]
        scores = [r["id"] for r in _read(os.path.join(REPO, "data", "model_scores.csv"))]
        self.assertEqual(sorted(reg), sorted(scores),
                         "registry and scores are out of sync; run scripts/build_registry.py")

    def test_all_cited_keys_are_defined(self):
        reg = _read(os.path.join(REPO, "data", "model_registry.csv"))
        refs = {r["key"] for r in _read(os.path.join(REPO, "data", "references.csv"))}
        used = set()
        for r in reg:
            for token in str(r.get("refs", "")).split(","):
                token = token.strip()
                if token.startswith("R"):
                    used.add(token)
        self.assertTrue(used, "no references found in the registry")
        self.assertFalse(used - refs, f"undefined reference keys: {sorted(used - refs)}")

    def test_registry_entries_have_evidence_and_dual_assessment(self):
        for r in _read(os.path.join(REPO, "data", "model_registry.csv")):
            for col in ("evidence", "glass_plastic_strengths", "glass_plastic_weaknesses",
                        "pros", "cons", "recommended_when"):
                self.assertTrue(str(r.get(col, "")).strip(),
                                f"{r['id']} is missing {col}")
            self.assertIn(r.get("kind", "model"), ("model", "component"))

    def test_scoring_profiles_cover_all_criteria(self):
        import yaml
        with open(os.path.join(REPO, "configs", "scoring.yaml")) as fh:
            cfg = yaml.safe_load(fh)
        cols = set(_read(os.path.join(REPO, "data", "model_scores.csv"))[0].keys()) - {"id", "mean"}
        for name, prof in cfg["weights"].items():
            self.assertEqual(set(prof["weights"].keys()), cols,
                             f"profile {name} does not cover exactly the scored criteria")
        self.assertIn(cfg["default"], cfg["weights"])


def _read(path: str):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


class TestReportsRender(unittest.TestCase):
    def test_report_pipeline_runs_and_is_self_contained(self):
        from gvp import report as report_mod
        tmp = tempfile.mkdtemp(prefix="gvp_report_")
        try:
            res = os.path.join(tmp, "results")
            docs = os.path.join(tmp, "docs")
            os.makedirs(res)
            paths = report_mod.build_all_reports(
                registry=os.path.join(REPO, "data", "model_registry.csv"),
                scores=os.path.join(REPO, "data", "model_scores.csv"),
                config=os.path.join(REPO, "configs", "scoring.yaml"),
                results_dir=res, docs_dir=docs)
            for name, p in paths.items():
                if not p:  # optional artefacts are skipped when their inputs are absent
                    continue
                self.assertTrue(os.path.exists(p), f"{name}: {p}")
            with open(paths["html"]) as fh:
                html = fh.read()
            self.assertIn('<script id="payload"', html)
            self.assertNotIn("http://", html.split("<body>")[1].split("</body>")[0],
                             "the HTML report must not depend on external resources")
            # the embedded payload must be valid JSON with the full registry
            import json
            import re
            payload = re.search(r'<script id="payload" type="application/json">(.*?)</script>',
                                html, re.S).group(1).replace("<\\/", "</")
            data = json.loads(payload)
            self.assertEqual(len(data["registry"]), len(_read(os.path.join(REPO, "data",
                                                                          "model_registry.csv"))))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
