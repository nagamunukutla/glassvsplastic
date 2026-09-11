# Glass vs Plastic — algorithms and models for material discrimination

A study of **how to tell glass from plastic from an image**, and of when an image is not enough.
It contains a citable comparison of 38 model families and sensor configurations scored on ten
criteria, a runnable classical-ML tier you can reproduce on a laptop in minutes, an optional
PyTorch/timm tier, and an evaluation protocol written so that a claim made with it can be audited.

![Pipeline: sensing → cues → models → operating policy](docs/assets/pipeline.svg)

```
                      ┌──────────────────────────────────────────────────────────┐
   RGB image ────────►│  cue families: transmission · specular · texture ·        │
                      │                silhouette · chromatic fringing            │
                      └───────────────┬──────────────────────────────────────────┘
                                      │
        ┌─────────────────────────────┼───────────────────────────────┐
        ▼                             ▼                               ▼
  physics rules                 classical ML                     deep tier
  (1-D thresholds)         (142 descriptors →              (ResNet/DenseNet/MobileNet/
  floor ≈ 0.70             SVM/RF/GBDT ≈ 0.93)              ConvNeXt/ViT/MobileNetV4)
        │                             │                               │
        └─────────────────────────────┴───────────────┬───────────────┘
                                                      ▼
                     calibration + abstention → purity/recovery operating point
                                                      │
                                    when RGB plateaus on transparent items:
                                    ┌─────────────────┴──────────────────┐
                                    ▼                                    ▼
                          NIR/SWIR · FTIR · HSI                 polarimetry · thermal · 3D
                          (polymer ID, F1 0.98–1.00)           (glass vs transparent plastic)
```

## The three findings that matter

1. **The published accuracy race is a dataset artefact.** Studio-lit household-waste benchmarks are
   saturated (90–99.6%, [R7](#references),[R8](#references),[R12](#references)), and *glass* is
   the recurring exception class in their confusion matrices. Binary **glass-vs-plastic is not what
   those benchmarks measure** — it removes the easy context and keeps the hard boundary: two
   transparent, shiny, similarly shaped materials whose governing difference is chemical.
2. **Within RGB, architecture choice is worth a few points; robustness and data are worth more.**
   Measured here on identical features and splits: tree ensembles hold 0.82–0.92 balanced accuracy
   under sensor noise, JPEG artefacts and deep shadow where an RBF SVM collapses to chance, and
   *every* global-descriptor model collapses under background clutter (edge density ×6.4,
   saturated pixels ×3.8). Model families differ far more in robustness than in peak accuracy.
3. **Transparent-vs-transparent is solved by sensors, not by bigger networks.** NIR/FTIR models
   reach F1 0.98–1.00 on polymer identification, hyperspectral PLS-DA sorts glass colour classes at
   sensitivity/specificity 0.910–1.000, polarimetry is reported specifically for glass vs
   transparent plastics, and multi-scale RGB+HSI fusion was required for transparent PET/PP on a
   black belt. Budget a sensor before you budget a bigger backbone.

Full reasoning: **[`docs/index.md`](docs/index.md)** · the table: **[`docs/06_model_comparison.md`](docs/06_model_comparison.md)**
· interactive: **[`results/model_comparison.html`](results/model_comparison.html)**

## Measured on real data (TrashNet, official splits)

Beyond the cited literature, the repository trains and evaluates everything itself on a public
dataset: TrashNet, restricted to the glass and plastic classes, on the authors' own test split
(156 images).

| Configuration (test n = 156) | Balanced accuracy |
|---|---|
| **MobileNetV3-Large, fine-tuned, full frame** | **0.936** |
| MobileNetV3-Large, fine-tuned, object cropped | 0.943 |
| ResNet-18, fine-tuned, full frame | 0.930 |
| EfficientNet-B0, fine-tuned, full frame | 0.929 |
| Classical models on 142 descriptors, full image | 0.763 – 0.880 |
| **Backdrop ring only** (3% strip, no object pixels) | **0.800 – 0.815** |
| Object only (background removed) | 0.827 – 0.846 |
| 6 scene statistics (logistic regression, no object) | 0.725 |
| Single-feature physics rules (1-D threshold) | 0.487 – 0.556 |
| Shuffled labels (control) | 0.449 |

Three findings, in the order they matter for a real deployment:

1. **What the input contains moves the number more than which model you pick.** Every fine-tuned
   backbone lands at 0.93 on full frames; the spread between the best and worst *classical* model on
   the same features is larger than the spread between the three backbones. But a model that sees
   only a 3% strip of studio backdrop — nothing of the object — already scores 0.80. On TrashNet, a
   large part of a published "glass vs plastic" accuracy is the room.
2. **Whether a model leans on that backdrop is architecture-specific, and must be measured.** Removing
   the backdrop costs ResNet-18 6.2 points (0.930 → 0.868) and nearly triples its
   glass-passed-as-plastic rate (7.3% → 18.3%), while MobileNetV3-Large and EfficientNet-B0 barely
   move (−0.006 to +0.007). So "fine-tuned CNN" does not predict behaviour here: crop one backbone
   and re-measure.
3. **Cost separates the backbones that accuracy does not.** MobileNetV3-Large reaches the top score
   with 4.2M parameters, 0.11G MACs and 8.6 ms/image — 12× fewer MACs and 2.8× faster than
   ResNet-18 for the same accuracy. On this evidence it is the default choice for an edge device.

Honest limits: 156 test images (95% CIs ±0.05–0.08, so a few points is not a resolved difference),
single seed per configuration, ResNet-18 fine-tuned at 192 px while the other two ran at 160 px
(memory budget — noted in the report), and the crop also upsamples the object, a confound this
design does not isolate.

* [`results/real/MEASURED_MODEL_COMPARISON.md`](results/real/MEASURED_MODEL_COMPARISON.md) — **the
  comparison table**: every measured configuration with accuracy, per-class recall, error direction,
  parameters, MACs, latency, and the advantage/disadvantage its own measurement supports.
* [`results/real/REAL_DATA_RESULTS.md`](results/real/REAL_DATA_RESULTS.md) — the full real-data study
  (audit, leakage controls, backdrop-stratified accuracy, synthetic-vs-real gap, deep tier).

## Quickstart

```bash
pip install -r requirements.txt          # numpy, scipy, scikit-learn, opencv, scikit-image, ...

make proxy        # render the synthetic proxy dataset            (~10 s)
make classical    # classical tier: models, rules, controls, shifts (~2 min)
make cues         # cue ablation with confidence intervals         (~3 min)
make report       # regenerate docs/06 + results/model_comparison.html
make test         # unit + end-to-end smoke tests
```

Deep tier (optional):

```bash
pip install -r requirements-deep.txt
python -m gvp.cli deep  --model tv:resnet18 --epochs 20 --splits test,shift_blur,shift_clutter
python -m gvp.cli bench --models tv:resnet18,tv:mobilenet_v3_large,tv:convnext_tiny
```

All commands run as `PYTHONPATH=src python -m gvp.cli …` (the Makefile handles this).

## ⚠️ Two kinds of numbers, never mixed

| | Source | Describes | Where |
|---|---|---|---|
| **Cited** | the 45 papers in [`docs/12_references.md`](docs/12_references.md) | real datasets, real deployements | `data/model_registry.csv`, all of `docs/01–08`, `10–11` |
| **Measured** | this repository's synthetic renderer (`gvp.synth`) | *method behaviour*: robustness, failure modes, statistical power | `results/MEASURED_RESULTS.md`, `results/*.csv` |

The proxy renderer encodes the physical cue structure (transmission, highlight morphology, surface
finish, silhouette geometry) with per-cue strength knobs. It is a test bench, **not evidence about
real glass and plastic**. Every measured table carries that caveat in the same document that
reports it, and the study's own underpowered cue ablation is published as a negative result
(`results/MEASURED_RESULTS.md` §5) rather than quietly dropped.

## Repository layout

```
docs/                      the study (start at index.md)
  01_problem_definition     six problems inside one question, acceptance criteria
  02_physics_and_cues      what physically differs, and how each cue breaks
  03_datasets_and_benchmarks  TrashNet/RealWaste/ZeroWaste/…, what their numbers mean
  04_classical_pipelines   142 descriptors, 10 shallow models, 3 measured failure modes
  05_deep_learning_zoo     CNNs, transformers, attention hybrids, recipes that work
  06_model_comparison      GENERATED. 38 options × 10 criteria × 4 weight profiles
  07_beyond_rgb            NIR/SWIR/FTIR, hyperspectral, polarimetry, thermal, 3D
  08_foundation_models_and_vlm  zero-shot, prompt engineering, training-free adapters
  09_evaluation_protocol   how to produce a defensible number (+ power analysis)
  10_deployment            latency budgets, optics, operating points, drift, playbook
  11_recommendations       conclusions, decision guide, 90-day plan, open problems
  12_references            GENERATED. 45 numbered sources
src/gvp/
  synth.py       synthetic proxy renderer with cue strength knobs and 7 domain shifts
  features.py    colour / specular / texture / edge / transparency descriptor blocks
  classical.py   model zoo, feature + cue ablations, shift sweeps, leakage controls
  deep.py        optional PyTorch/timm tier, ONNX export, params/MACs/latency accounting
  evaluate.py    metrics, bootstrap CIs, plots, Markdown rendering
  report.py      renders the registry into Markdown + a self-contained interactive HTML page
  cli.py         one entry point for every experiment
data/
  model_registry.csv   architecture facts + cited evidence with [Rn] keys
  model_scores.csv     ten criteria scored 1–5 per option
  references.csv       the 45 sources
configs/scoring.yaml   the weights; four profiles, including an edge-budget and a cold-start one
results/               measured CSVs, figures, MEASURED_RESULTS.md, model_comparison.html
scripts/build_registry.py   provenance of the registry (reviewable, regenerable)
tests/                 unit + end-to-end tests, including the leakage controls as CI checks
```

## The comparison, in one table

Top-ranked options under the default *balanced commercial* profile (see
[`docs/06_model_comparison.md`](docs/06_model_comparison.md) for all 37 ranked models, the
`rejection layer` component, per-criterion leaders, and the four alternative profiles):

| Rank | Option | Score | Why it ranks there |
|---|---|---|---|
| 1 | NIR/FTIR spectroscopy + PLS-DA or 1D-CNN | 3.97 | the only family that resolves the transparent case on chemistry; F1 0.98–1.00 |
| 2 | DenseNet-121/-169 transfer learning | 3.82 | most reproducible waste-imagery track record; 8M params |
| 3 | Hyperspectral imaging + ML | 3.78 | glass colour sorting at 0.91–1.00 sens/spec; ~99% purity industrially |
| 4 | LDA/QDA on hand-crafted features | 3.75 | 0.004 ms inference, seconds to train, fully auditable |
| 5 | MobileNetV4 (Conv-S / Hybrid-L) | 3.74 | 3.8M params, 0.2 GMACs, 2.4 ms on a phone CPU |
| — | *(component, not ranked)* uncertainty + conformal rejection | 4.36 | converts silent failures into a managed reject stream |

And the empirical core of the classical tier, measured here (see
[`results/MEASURED_RESULTS.md`](results/MEASURED_RESULTS.md)):

| Model | Balanced accuracy (proxy) | Under sensor noise | Under clutter |
|---|---|---|---|
| HistGradientBoosting | 0.944 | 0.817 | **0.508** |
| ExtraTrees | 0.933 | 0.842 | **0.500** |
| SVM-RBF | 0.933 | **0.500** | **0.508** |
| Physics rule (1 threshold) | 0.700 | — | — |

Leakage controls pass: shuffled labels 0.48–0.51; background-only 0.51–0.52 once the blanking
window is large enough to remove the object (the first, too-small window reported a false alarm of
0.78 — documented as a lesson in control design).

## Extending the study

* **Add a model or sensor option** → append to `MODELS` in `scripts/build_registry.py` (with a
  `[Rn]` evidence key), score it in `SCORES`, run `python scripts/build_registry.py && make report`.
* **Disagree with the weights** → edit `configs/scoring.yaml`; the HTML report also lets you drag the
  ten criteria live.
* **Add a feature block** → implement it in `src/gvp/features.py`, add it to `FEATURE_BLOCKS`; the
  ablation and report layers pick it up automatically.
* **Try a backbone** → `python -m gvp.cli deep --model <timm-or-tv-name>`; the manifests are shared
  with the classical tier, so results are directly comparable.
* **Bring real data** → follow `docs/09_evaluation_protocol.md` and point the loader at your own
  manifest CSVs with the same columns (see `gvp.synth.build_proxy_dataset` for the schema).

## Citation

See [`CITATION.cff`](CITATION.cff). If you use the comparison registry, cite it together with the
primary sources listed in `data/references.csv` — the registry aggregates *their* evidence, and the
`refs` column exists so that credit lands where it belongs.

## Licence

Code: MIT ([`LICENSE`](LICENSE)). Documentation, registry and generated results: CC BY 4.0. The
cited works retain their own licences; datasets referenced here (TrashNet, RealWaste, ZeroWaste,
TACO, …) are **not** redistributed by this repository — download them from their own sources.
