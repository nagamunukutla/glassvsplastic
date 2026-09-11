# 9. Evaluation protocol: how to produce a glass-vs-plastic result that survives scrutiny

Most published waste-classification numbers cannot be compared to each other because the splits,
the class granularity, the metrics and the selection criterion all differ. This chapter is the
protocol this repository tries to follow, written so it can be copied into a paper's methods
section. It is also the checklist for auditing somebody else's claim.

## 9.1 Define the decision before the model

Write down, before any training:

| Question | Why it matters | Example answer |
|---|---|---|
| What exactly is being separated? | "Glass vs plastic" hides a taxonomy | `soda-lime container glass` vs `PET/PP/HDPE packaging plastic`, films included? |
| Where does the label come from? | Label quality bounds accuracy | material specification from the plant, not visual guessing |
| What is the unit of prediction? | Object, pixel, or bag? | one object on the belt, one prediction per object |
| What happens after a mistake? | Sets the operating point | glass in the plastic stream lowers bale value; plastic in the glass stream can crack the furnace — asymmetric costs |
| What accuracy is required? | Determines feasibility | e.g. ≥98% glass purity with ≥90% recovery |

The last two rows are the ones normally skipped, and they are the ones that decide everything: an
85%-accurate classifier with a rejection policy can be deployable if the reject stream is recycled
back, and a 99%-accurate classifier with no reject option can be undeployable if it fails on one
systematic sub-class.

## 9.2 Build the dataset

* **Sample by physical object.** Photograph each item from several angles; assign all its views to
  one split. Splitting by image leaks identity and inflates scores by a large margin.
* **Stratify by the cue table** (chapter 2), not by product name. Minimum useful strata:
  transparent × {clean, labelled, dirty}, opaque-coloured, film/flexible, shard/broken, multi-layer.
* **Include the confusable negatives deliberately**: clear PET film, frosted glass, sleeved bottles,
  glass jars with plastic lids, ceramic (which is *not* glass and is a real sorting target [R31]).
* **Capture ≥2 acquisition domains** (sites, lightings, belts, cameras) and hold one out entirely.
  Cross-domain degradation is the single best predictor of deployment failure, and this repository's
  shift sweep shows models surviving in-domain can drop 40 points out of domain.
* **Pair a subset with a reference modality** (NIR/HSI or polarimetry). Without it you cannot
  distinguish "the model is bad" from "the information is not in the image" — which is the central
  question of this whole field.
* **Keep the ambiguity labels.** Items that even a human expert cannot classify from a photo should
  be marked as such and reported separately, not forced into a class. They define your Bayes floor.

Realistic sizes: 1,500–3,000 objects (four views each) is enough for a publishable binary study;
under ~800 objects, expect confidence intervals wide enough to hide the effect you are looking for.

## 9.3 Splits and leakage controls

```
object-level split:      train 60% | val 15% | test 15%   (stratified by cue stratum)
cross-domain split:      train/val from domain A, test from domain B (untouched)
temporal split:          for plant data, train on weeks 1-8, test on weeks 9-12
```

Then run these controls and *publish* them:

| Control | What it detects | Pass criterion |
|---|---|---|
| Shuffled labels | split leakage, duplicated items | ≈0.50 balanced accuracy |
| Background-only (object blanked out) | scene/lighting shortcuts | ≈0.50; if not, quote the blanking window |
| Object-only (background replaced) | whether the model needs context | informative either way — this repository measured models *improving* without context |
| Per-class recall | hidden minority-class failure | report for every class, both directions |
| Group-disjoint check | same object/plant batch in two splits | zero overlap by construction |
| Label audit (n ≥ 100, second annotator) | label noise ceiling | report agreement (%) |

The background-only control deserves emphasis: it is cheap, and its absence is the most common
reason a waste-classification number is inflated. This repository's own first implementation of it
was mis-specified (the window was too small, leaving peripherally visible object pixels) and
produced a false alarm of 0.78 — see `results/MEASURED_RESULTS.md` §3. Size the window from the
object bounding boxes you already have.

## 9.4 Metrics to report

| Metric | Why |
|---|---|
| **Balanced accuracy** (headline) | immune to class imbalance, which waste streams always have [R37] |
| **Per-class recall** (both directions) | `glass→plastic` and `plastic→glass` fail differently and cost differently |
| **Macro F1** | comparable to the classification literature |
| **ROC AUC** | threshold-independent ranking quality; also what "AUC 1.0 except glass and trash" [R34] means |
| **Calibration** (reliability curve / ECE) | decides whether a reject threshold is meaningful |
| **Coverage vs accuracy at the operating point** | the actual production contract: "accept 90% of items, purity 99%" |
| **Bootstrap 95% CI on every headline number** | n is always small; a bare 0.933 with n=90 is a range of 0.89–0.98 |
| **Per-stratum breakdown** | the aggregate hides the transparent cell, which is the point |
| **Latency (ms/image, p50 and p95), params, MACs, memory** | deployment reality |

**Do not report only accuracy.** With 80/20 class imbalance, predicting the majority class gives
80% accuracy and 0.50 balanced accuracy.

## 9.5 Power: how many test items do you need?

Approximate number of **test items per split** to detect a difference between two classifiers
(unpaired two-proportion comparison, α = 0.05 two-sided, power 0.80), around accuracies of 0.90:

| Difference to detect | n per split |
|---|---|
| 10 percentage points | ~125 |
| 5 percentage points | ~490 |
| 3 percentage points | ~1,360 |
| 2 percentage points | ~3,050 |

For paired comparisons (same test items, two models) the requirement is roughly halved. This is the
concrete reason the repository's own cue ablation cannot resolve single-cue effects: at the
`make cues` default (400 items per class -> n = 120 test items per condition) the measured 95% CI
widths were 0.087–0.122, and an earlier, smaller run (200 per class -> n = 60) produced widths of
0.11–0.20 — while the largest single-cue effect observed was 0.022 (see
`results/MEASURED_RESULTS.md` §5). **A cue-importance study needs a factorial design with ≥1,000
items per cell, or a dose-response design with graded cue strengths and the same total n.**
Publishing a 2-point ablation difference at n = 100 is not a finding.

## 9.6 Ablations worth running (and their cost)

| Ablation | Question answered | Cost |
|---|---|---|
| Input resolution: 128 / 224 / 384 / native | is the cue fine-grained (highlights) or coarse (texture)? | low |
| Crop around the object vs full frame | is the model using the object or the scene? | low |
| Colour: RGB / grayscale / edge-only | how much is colour-driven? | low |
| Background: real belt / flat white / cluttered | scene dependence | low (this repo's `whitebg`, `clutter` shifts) |
| Illumination: diffuse dome vs point source | specular-cue dependence | medium (physical) |
| Cue strength (rendered or physically graded) | dose-response of the decisive cue | medium |
| Model family at matched FLOPs | is the architecture doing work? | high |
| Cross-domain test | the number that predicts deployment | high (data) |
| Rejection threshold sweep | the purity/recovery frontier | low (post-hoc) |

The single highest-value ablation for this problem is **object-crop versus full-frame**. If cropping
to the object changes accuracy by more than a couple of points, the published number is a scene
classifier.

## 9.7 Reporting checklist

- [ ] Task definition, taxonomy, unit of prediction, and the cost structure of errors (§9.1)
- [ ] Dataset: object counts (not image counts), capture settings ×2+ domains, licences
- [ ] Split protocol, object-level, with the leakage controls of §9.3 and their outcomes
- [ ] Headline: balanced accuracy + 95% CI; per-class recalls in both directions; macro F1; AUC
- [ ] Calibration and the operating point actually proposed for deployment
- [ ] Per-stratum results, including the transparent × clean stratum where available
- [ ] Cross-domain result (trained on A, tested on B) — even if it is bad
- [ ] Latency, parameters, MACs, memory, and the hardware used
- [ ] Baselines: a physics rule, a shallow model, and at least two backbones
- [ ] Seed variance (≥3 seeds) for any deep-learning number quoted to two decimals
- [ ] Code + weights + prompt versions (for VLM-based components)
- [ ] Honest statement of what the dataset cannot support

## 9.8 Reproducibility in this repository

```bash
make proxy        # deterministic proxy data (seeded renderer)
make classical    # model zoo, feature/rule baselines, controls, shift sweep
make cues         # cue ablation with CIs
make report       # registry + measured results -> docs + HTML
make test         # CI: unit + end-to-end smoke tests (including the leak controls)
```

* every run writes CSV, not just console output;
* the feature cache key includes a hash of the data manifest *and* of the feature code, so a stale
  cache can never silently serve old numbers (this bug was hit and fixed during development — see
  the git history of `src/gvp/classical.py`);
* every reported measured number carries its caveat in the same document that reports it.
