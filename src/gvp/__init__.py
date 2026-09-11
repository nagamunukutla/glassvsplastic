"""gvp — Glass vs Plastic: algorithms and model comparison for material discrimination.

This package accompanies the study in ``docs/``. It provides:

* ``gvp.synth``      — a synthetic glass/plastic *proxy* image generator (smoke tests only)
* ``gvp.features``   — hand-crafted, interpretable image features (colour / specular / texture / edges / transparency)
* ``gvp.classical``  — classical ML model zoo + ablations + robustness sweeps
* ``gvp.deep``       — optional PyTorch/timm transfer-learning runner (guarded import)
* ``gvp.benchmark``  — latency / size / FLOPs measurement
* ``gvp.evaluate``   — metrics, confusion matrices, plots
* ``gvp.report``     — renders the model-comparison registry into Markdown + self-contained HTML
* ``gvp.cli``        — single entry point for every experiment

The synthetic generator exists so that the *pipeline* is verifiable end to end without a
dataset download. It is **not** evidence about real glass and plastic. See
``docs/03_datasets_and_benchmarks.md`` and ``results/SYNTHETIC_RESULTS.md``.
"""

__version__ = "0.1.0"

MATERIALS = ("glass", "plastic")
