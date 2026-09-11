# 3. Datasets and benchmarks: what you can actually measure on

No glass-vs-plastic *binary* benchmark exists as a first-class dataset. What exists is a set of
household-waste image datasets with a `glass` and a `plastic` class among others, in which the
glass/plastic boundary is one of several, and is usually the hardest one. This chapter is the
inventory, with the caveats that decide whether a number is meaningful.

## 3.1 The public image datasets

| Dataset | Images | Classes | Capture setting | Glass / plastic counts | Notes for this study |
|---|---|---|---|---|---|
| **TrashNet** [R1] | 2,527 | 6 (cardboard, glass, metal, paper, plastic, trash) | Single object on a **white/cardboard backdrop**, 512×384 | glass ≈501, plastic ≈482 | The de-facto benchmark: small, studio-lit, saturated. Not a proxy for a conveyor |
| **RealWaste** [R2] | 4,752 | 9 (incl. glass, plastic, metal, organics, textile, vegetation) | **Working landfill**, 524×524 | glass 378, plastic 831 | The honest counterpart to TrashNet. Its authors explicitly scaled to 524×524 to handle "transparent objects in plastic and glass classes" and "similarities between specific objects (e.g. glass and plastic bottles)" |
| **ZeroWaste** [R3] | 26,766 total (4,503 fully annotated + 6,212 unlabelled + 1,410 weakly labelled) | 4 material classes | Real recycling-plant conveyor | — | Designed for exactly the deployment setting; includes pose/instance metadata |
| **TACO** [R4] | 4,617 | 60 litter categories (COCO-style) | In the wild | — | Extreme class imbalance; detection-oriented |
| **TrashBox** [R33] | 7 primary classes | 7 (incl. glass, plastic) + subclasses | Curated, multi-source | — | Used for the federated-learning study; the confusion matrix shows visually similar classes dominating errors |
| **GlobalWasteData / catalogue** [R5,R6] | 20+ datasets surveyed | varies | varies | varies | Consolidated tables of sizes/classes; useful for spotting that most sets are small and single-setting |
| **Kaggle household sets** | 19,762–25,077 | 2–10 | Mixed web scraped | — | Larger but noisy; label quality is the limiting factor |
| **FPWaste / MultiWaste** [R30] | purpose-built for taxonomy drift | packaging-focused | industrial probes | — | Built to test *semantic* boundaries (food vs non-food packaging), not materials |

### The three properties that decide everything

1. **Scale.** TrashNet is 2,527 images [R1]; the DeepGarbage/RealWaste tier is ~4.7k [R2]; the
   web-scraped plastics study used 9,440 [R37]. Below ~5k images, transfer learning is not a
   convenience but a requirement — from-scratch CNNs lose to fine-tuned backbones by 5 points in
   the controlled comparison (90.61% vs 95.51%/96.00%) [R45].
2. **Setting.** Studio-white backgrounds make the transmission and background-continuity cues
   trivially available and identical for every image, which both inflates accuracy and lets models
   learn scene identity. RealWaste was created precisely because pristine studio items do not
   predict landfill performance [R2].
3. **Class granularity.** "Plastic" is not one material. Reported per-class behaviour within plastic
   varies from perfect for PET to catastrophic for polypropylene in the same experiment (PET 100%,
   PS 95.2%, PP 0% with a single measurement per sample) [R26].

## 3.2 What the benchmark numbers actually say

Aggregating the numbers cited in `docs/12_references.md`:

**Household-waste classification (6–9 classes, studio or curated imagery):**

| Model | Dataset | Accuracy | Reference |
|---|---|---|---|
| Optimised DenseNet-121 | TrashNet (6 cls) | 99.60% | [R12] |
| WasNet | TrashNet | 96.10% | [R11] |
| DenseNet-121 + Adam + augmentation | TrashNet | ~95% | [R8] |
| EfficientNetV2-S | garbage classification | 96.41% (also lowest carbon) | [R16] |
| DNN-TC | TrashNet / VN-trash | 94% / 98% | [R7] |
| Focus-RCNet-KD (0.525M params) | TrashNet | 92% | [R9] |
| Beluga-optimised InceptionV3 | TrashNet | 92.62% | [R15] |
| RWNet-101 | TrashNet | 89.9% (plastic worst) | [R14] |
| RecycleNet (3M params) | TrashNet | 81% | [R10] |
| Cascaded DP-CNN-En-ELM | 2 / 9 / 36 classes | 96% / 91% / 85.25% | [R13] |

**Polymer / material identification (spectral, 4–6 polymer classes):**

| Method | Modality | Result | Reference |
|---|---|---|---|
| Improved CNN + trainable preprocessing | FTIR / NIR / H-NIR | F1 0.981 / 0.978 / 1.000 | [R27] |
| Transformer + trainable preprocessing | FTIR / NIR / H-NIR | F1 0.921 / 0.972 / 1.000 | [R27] |
| PLS-DA / LDA | FTIR–NIR–H-NIR | F1 0.517–0.971 / 0.506–0.987 | [R27] |
| 1D-ResNet + augmentation | FTIR | 0.991 accuracy | [R25] |
| GAN-augmented classifier | 6 recycled polymers | 96.2% balanced accuracy | [R26] |
| P1CH (pixel-level CNN) | HSI, HDPE/PET/PP/PS | 97.44% (99.94% excl. borders) | [R19] |
| ANN on HSI features | NIR HSI, 4 polymers | 89.5% | [R20] |
| N-BEATS ensemble | NIR, black + coloured plastics | F1 0.79 overall; 0.90 coloured, 0.67 black | [R24] |
| Hierarchical PLS-DA | VIS-NIR HSI, glass colour classes | sensitivity/specificity 0.910–1.000 | [R21] |

**Zero-shot / few-shot foundation models:**

| Method | Setting | Result | Reference |
|---|---|---|---|
| OpenCLIP ViT-L/14-336 | zero-shot, TrashNet | 76.30% (2.83 FPS) | [R29] |
| OpenCLIP ViT-L/14-2B | zero-shot → prompt-engineered → supervised | 82.71% → 90.48% → 97.18% | [R28] |
| Frozen VLM + training-free adapter | taxonomy drift | recommended route | [R30] |
| MLLMs (GPT-4o, LLaVA-OneVision) | zero-shot waste | competitive zero-shot; textual few-shot *hurt* | [R30] |

### Four readings of that table

1. **The studio benchmark is saturated.** 99.6% on TrashNet [R12] says more about the dataset than
   about the model. The same architecture family scores 89.9–95% under stricter protocols
   [R8,R14,R15].
2. **Spectral beats RGB on material identity, by a wide and consistent margin** — F1 0.98–1.00 for
   polymer ID [R25,R26,R27] versus the RGB confusion patterns described in chapter 2.
3. **Even spectral methods have a documented hole:** carbon-black plastics, where NIR absorption
   defeats the sensor (F1 0.67 vs 0.90 for coloured items) [R22,R24].
4. **Zero-shot VLMs are now better than a small CNN trained on ten images per class, and worse than
   a fine-tuned CNN.** Prompt engineering bought 7.8 points for free [R28]; that is the cheapest
   accuracy in the entire literature. But it is not a substitute for fine-tuning, and the prompt
   sensitivity it exposes [R29] is a maintenance liability.

## 3.2b Measured: the backdrop carries most of the glass-vs-plastic signal

Everything above quotes the literature. This repository also ran the experiment on **real TrashNet
data**, using the authors' official train/val/test split files, restricted to the glass and plastic
classes (983 images). Full details in [`results/real/REAL_DATA_RESULTS.md`](../results/real/REAL_DATA_RESULTS.md) and the one-table view in [`results/real/MEASURED_MODEL_COMPARISON.md`](../results/real/MEASURED_MODEL_COMPARISON.md);
the headline:

| Probe (test split, 156 images) | Balanced accuracy |
|---|---|
| MobileNetV3-Large fine-tuned, full frame | **0.936** |
| ResNet-18 fine-tuned, full frame | 0.930 |
| EfficientNet-B0 fine-tuned, full frame | 0.929 |
| Classical models on 142 descriptors, full image | 0.763 – 0.880 |
| Backdrop ring only (3% strip, no object pixels) | 0.800 – 0.815 |
| Object only (background removed) | 0.827 – 0.846 |
| Best single scene attribute (`bg_r` threshold) | 0.666 |
| Shuffled labels (control) | 0.449 – 0.450 |

Backdrop type and material class are associated with Cramér's V = 0.279 (χ² = 76.7, p ≈ 1e-16),
and the dataset is 65% grey studio backdrop / 33% cardboard.

**Interpretation.** On this benchmark, a model that never sees the object — only a 3%-wide strip of
backdrop — reaches ~0.80 of the ~0.88–0.94 that full-image models achieve. The object contributes
real signal too (~0.83 on its own), but the two are heavily redundant, and a large part of what a
published "glass vs plastic" number measures on TrashNet is the studio, not the material. This is
the concrete, measured version of the shortcut warning in §3.1 and the reason
`docs/09_evaluation_protocol.md` makes the scene-only control mandatory. It is also the reason this
repository refuses to quote TrashNet accuracies as if they were material-classification accuracy.

Note the design lesson, since this repository got it wrong first: **rectangular blanking is not a
valid scene-only control on this dataset** — TrashNet objects are rotated and reach into the frame
corners, so a "90% blanked" image still contains object fragments (see the montage in
`results/real/figures/control_visual_check.png`). Only the thin-ring probe, verified visually,
isolates the scene.

**Cropping the backdrop away is the cheapest test of what a model learned — and its result is
architecture-specific.** Repeating each backbone's fine-tune with the object cropped out of the
frame (backdrop removed, nothing else changed) gives:

| Backbone | Full frame | Object crop | Change |
|---|---|---|---|
| MobileNetV3-Large | 0.936 | 0.943 | +0.007 |
| EfficientNet-B0 | 0.929 | 0.923 | −0.006 |
| ResNet-18 | 0.930 | 0.868 | **−0.062** |

So it is not true that "fine-tuned CNNs read the studio": one widely used backbone loses most of a
publication-sized margin when the backdrop goes, and two others do not move. The practical rule for
this project is therefore *measure the crop test per candidate backbone* rather than reasoning about
the family — it costs one extra training run and it is the only evidence that separates a model that
learned the material from one that learned the room. Two limits, stated rather than hidden: the crop
also upsamples the object, and each configuration is a single seed.

## 3.3 The benchmark this project uses, and why

Because no redistributable glass-vs-plastic set exists inside a repository, `gvp.synth` renders a
**physically-motivated proxy dataset** with the cue structure of §2.2, per-cue strength knobs, an
ambiguity fraction, and seven acquisition-shift test sets (blur, dark, noise, JPEG, low contrast,
clutter, background removal).

It is honest to use it for exactly three purposes:

* **Method behaviour**: which model classes hold up under shift, how wide the confidence intervals
  really are, what a cue ablation can and cannot resolve, where pipelines break.
* **Pipeline verification**: the code path from data → features → model → report is exercised end to
  end, in CI, on every commit.
* **Power analysis**: the measured CI widths tell you how many test items a *real* study needs.

It must not be used for:

* absolute accuracy claims about glass and plastic;
* ranking models on the criterions that matter at the margin (a 2-point difference is noise);
* claims that a cue is or is not used (see the underpowered cue ablation in
  `results/MEASURED_RESULTS.md` §5).

## 3.4 How to build a real benchmark for this problem

If the goal is a publishable glass-vs-plastic result, the dataset does not exist yet and building it
is most of the contribution. The protocol in `docs/09_evaluation_protocol.md` specifies it; the
short version:

* **Sample by object, not by image.** Several views of the same physical item must never straddle
  a split.
* **Stratify explicitly by the cue table.** Transparent/opaque × coloured/clear × clean/labelled ×
  rigid/film. Report per-stratum results; an aggregate number hides the only interesting cells.
* **Include the confusable negatives.** Clear plastic film, frosted glass, dirty containers,
  shards, multi-layer and sleeved packaging.
* **Capture at two or more sites/lightings** and keep one out entirely as a cross-domain test.
* **Capture paired spectral data** for a subset, so the RGB ceiling can be measured against a
  physics reference rather than asserted.
* **Publish the failures.** The most valuable rows in any real glass/plastic dataset are the items
  no RGB model can separate.
