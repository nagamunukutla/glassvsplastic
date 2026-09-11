# 1. Problem definition: what exactly are we classifying?

"Differentiate glass and plastic" can mean six different engineering problems. They have different
data, different models, different sensors and different success criteria. Getting this wrong is the
most expensive mistake available, so this chapter fixes the scope of the study.

## 1.1 The six problems hiding inside the question

| # | Task | Input | Output | Typical system | Difficulty driver |
|---|---|---|---|---|---|
| 1 | **Binary material decision** | one crop of one item | `glass` / `plastic` | this study's primary target | transparency, specularity, visual similarity |
| 2 | Material + product classification | one crop | `glass bottle` / `PET bottle` / `glass jar` / … | sorting robotics | taxonomy depth (+11 points cost from 6 to 36 classes [R13]) |
| 3 | Full waste taxonomy | one crop | 6–60 categories | municipal sorting | class imbalance, label noise [R4,R5] |
| 4 | Object detection / segmentation | whole belt frame | boxes/masks + material | industrial sorters | clutter, occlusion, transparency |
| 5 | Polymer identification | spectrum (point) | PET / PE / PP / PS / … | NIR/FTIR sorters | spectral overlap, black plastics [R22,R24] |
| 6 | Contaminant / defect detection | image of glass cullet | ceramic, stone, metal fragments | glass-recycling QC | rare, subtle classes [R36] |

This repository studies **(1)**, and treats (4) and (5) as the two things that most often decide
whether (1) is deployable: (4) because material classifiers are scene-sensitive, (5) because it is
the reference modality for the transparent case.

## 1.2 Scope decisions

| Decision | This study | Why |
|---|---|---|
| Input modality | RGB primary; spectral, polarimetric, thermal, 3D surveyed and compared | the practical question is "what does RGB buy me and when must I add a sensor" |
| Output | binary, with an explicit reject/abstain option | production needs an operating point and a destination for rejects (chapter 10) |
| Unit of prediction | one object (crop), with a note on where full-frame models go wrong | pixel and bag-level tasks have different evaluation protocols |
| Classes | `glass` (soda-lime container glass) vs `plastic` (PET/PP/HDPE/PS packaging, films included) | the confusable pair; ceramics and metals are separate decisions |
| Human-level target | not "human parity" — humans use motion [R41] | a single frame does not show what a person would use |
| Success criteria | balanced accuracy with CI, per-class recall both directions, calibration, latency, and a purity/recovery operating point | accuracy alone is meaningless under imbalance [R37] |

## 1.3 Why this is not just "another waste-classification benchmark"

Three properties make glass-vs-plastic different from the datasets that dominate the literature
(TrashNet, RealWaste, TACO, …):

1. **The classes are not visually distinct.** In household-waste taxonomies the dominant confusions
   are paper/cardboard, plastic/metal and trash/anything [R33,R34,R35] — i.e. *objects* that differ
   in shape, colour and texture. Glass vs plastic, restricted to the confusable sub-population,
   differs in almost nothing visible: both can be transparent, colourless, shiny, thin-walled and
   bottle-shaped.
2. **The discriminative signal is chemical, and it is measurable — just not in RGB.** NIR/FTIR models
   reach F1 0.98–1.00 on polymer ID [R25,R26,R27]; polarimetry separates glass from transparent
   plastic [R31]. This gives the study a ground-truth reference against which the RGB ceiling can be
   established rather than asserted.
3. **The error costs are asymmetric and large.** Glass contamination limits the value of recycled
   plastic bales; plastic or ceramic contamination changes glass-furnace chemistry. A 95%-accurate
   binary classifier with balanced errors and a 95%-accurate classifier that is wrong only on clear
   PET are completely different products.

## 1.4 Formal problem statement

Given an image region $x$ containing (part of) an object, and a reference modality $s$ (spectrum,
polarisation state, or none), predict

$$ y \in \{\text{glass}, \text{plastic}\}, \quad \text{or } \bot \text{ (abstain)} $$

with a decision rule that also reports a calibrated confidence $\hat p$, chosen to maximise an
economic objective

$$ \max \; U(\hat y; \tau) = \text{value}(\text{recovery}) - \text{cost}(\text{contamination}) - \text{cost}(\text{throughput}) $$

subject to a constraint of the form $\Pr(y = \text{plastic} \mid \hat y = \text{glass}) \le \epsilon$
(a purity guarantee, which is what a conformal prediction layer buys — chapter 10 §10.4).

**Acceptance criteria for a research claim in this area:**

| Criterion | Threshold this study uses |
|---|---|
| Balanced accuracy reported with 95% CI | mandatory |
| Per-class recall in both directions | mandatory |
| Object-level split with leakage controls | mandatory (shuffled-label and background-only) |
| Cross-domain test (≥2 acquisition domains) | mandatory |
| The transparent × clean stratum reported separately | mandatory |
| A physics baseline (single-feature rule) and a shallow baseline | mandatory |
| Latency/params/MACs on named hardware | mandatory for any deployment claim |
| Effect claimed larger than the CI width | mandatory |

## 1.5 What "good" looks like

| Application | Realistic target | Reference point |
|---|---|---|
| Glass cullet colour sorting (material pure) | sensitivity/specificity 0.91–1.00 | hierarchical PLS-DA on VIS-NIR HSI [R21] |
| Polymer identification in mixed plastics | F1 0.98–1.00, except black items | NIR/FTIR deep models [R25,R26,R27] |
| Industrial purity target | ~99% purity PP/PE/PET | hyperspectral sorting practice [R23] |
| RGB household-waste classification | 90–99% on studio benchmarks; expect much less in the field | [R7,R8,R9,R12,R37] |
| Zero-shot with no labels | 76–90% | OpenCLIP ± prompt engineering [R28,R29] |
| **Binary glass vs plastic on transparent items, RGB only** | **no reliable published number exists — and that is the finding** | chapter 3, chapter 7 |

## 1.6 Repository map (what implements what)

| Scope item | Implementation |
|---|---|
| Cue taxonomy (§1.2) | `src/gvp/features.py` (five descriptor blocks) + `src/gvp/synth.py` (cue strength knobs) |
| Binary decision + abstention | `src/gvp/evaluate.py` (metrics), `docs/10_deployment.md` §10.4 (policy) |
| Model comparison | `data/model_registry.csv`, `data/model_scores.csv`, `configs/scoring.yaml` → `docs/06_model_comparison.md` |
| Classical tier experiments | `src/gvp/classical.py` → `results/*.csv`, `results/MEASURED_RESULTS.md` |
| Deep tier | `src/gvp/deep.py` (timm/torchvision, ONNX export, latency/MACs) |
| Evaluation protocol | `docs/09_evaluation_protocol.md` |
| Reference modality survey | `docs/07_beyond_rgb.md` |

Next: `docs/02_physics_and_cues.md` for the physics that constrains everything else.
