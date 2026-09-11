# 11. Recommendations: what to do, in what order, and what to expect

This chapter is the executive summary of the study. It assumes you have a glass/plastic decision to
make and a limited budget of time, data and hardware.

## 11.1 Seven conclusions

1. **The published accuracy race is mostly a dataset artefact.** On studio-lit, single-object
   household-waste benchmarks, everything from a 0.5M-parameter distilled CNN (92% [R9]) to an
   optimised DenseNet-121 (99.6% [R12]) "works". The same pipelines lose 20+ points on harder,
   imbalanced or real-world data (96% → 71.8% for the same ResNet-34 pipeline [R37]; 6 → 36 classes
   costs 11 points [R13]). **Choose your model on robustness, cost and evidence quality — not on
   benchmark ranking position.**

2. **Binary glass-vs-plastic is not what any of those benchmarks measure, and it is much harder.**
   It removes the easy context (colour, shape, product familiarity) and keeps exactly the confusable
   boundary: two transparent, shiny, similarly shaped materials whose main physical difference is
   chemical. Repeated confusions in published confusion matrices concentrate on precisely these
   pairs [R34,R35], and the sensor literature says outright that transparent plastic can be
   "almost indistinguishable" from glass to vision alone [R31].

3. **Within RGB, the architecture choice is a cost/benefit decision worth a few points at most.**
   ConvNeXt-Tiny leads a controlled six-domain comparison at 28.6M parameters [R38]; MobileNetV3
   matches much larger models on augmented material data at a fraction of the training time [R17];
   DenseNet-121 has the best documented waste-imagery track record [R7,R8,R12]; MobileNetV4 is the
   current frontier for edge latency [R39]. Pick one, spend the remaining effort on data.

4. **Robustness, not peak accuracy, is where model families genuinely differ.** Measured in this
   repository on identical features and splits: under sensor noise, JPEG artefacts and deep shadow,
   a tree ensemble holds 0.82–0.92 balanced accuracy where an RBF SVM collapses to chance; under
   background clutter, *every* global-descriptor model collapses because bright debris injects
   glass-like highlight statistics everywhere (edge density ×6.4, saturated pixels ×3.8). The fix is
   architectural — put a detector/segmenter in front, or use a model with spatial selectivity.

5. **The methods that actually solve transparent-vs-transparent are spectral or polarimetric, not
   deeper CNNs.** NIR/FTIR classifiers reach F1 0.98–1.00 on polymer identification [R25,R26,R27];
   hierarchical PLS-DA on VIS-NIR hyperspectral data sorts five glass colour classes at
   sensitivity/specificity 0.910–1.000 [R21]; polarimetry is reported as specifically effective for
   glass versus transparent plastics [R31]; and multi-scale RGB+HSI fusion was necessary to separate
   transparent PET, blue PET and transparent PP [R32]. Budget a sensor before you budget a bigger
   network.

6. **Every approach has a documented blind spot; a working system needs a rejection policy.**
   Carbon-black plastics defeat NIR (F1 0.67 vs 0.90 [R24]); black PS was entirely undetected by a
   97%-accurate hyperspectral model [R19]; extreme class imbalance silently destroys per-class
   recall [R37]; single-measurement spectral baselines can fail completely for one polymer while
   scoring 100% on another [R26]. A calibrated "unknown → divert" route converts all of these from
   silent errors into a managed cost.

7. **The cheapest accuracy improvements are not models.** Prompt engineering bought 7.8 points
   zero-shot with no training [R28]; diffusion-based synthetic minority data lifted an imbalanced
   glass-detection model by 15 points [R36]; 2× spectral augmentation lifted every classifier by
   ≥3 points [R25]; and the physical arrangement of the capture cell (textured background, controlled
   specular source) decides which cues exist at all.

## 11.2 Decision guide

| Your situation | Start here | Then | Expected |
|---|---|---|---|
| **No labels, need something this week** | OpenCLIP/SigLIP zero-shot with physically descriptive prompts (chapter 8 §8.5) | frozen features + k-NN/training-free adapter; label only the ambiguous stratum | 76–90% depending on prompt quality and class mix [R28,R29] |
| **200–2,000 labelled objects, one GPU, standard RGB line** | ResNet-18 and DenseNet-121 fine-tuned; plus a tree ensemble on hand-crafted features as the floor | ConvNeXt-Tiny or EfficientNetV2-S if data allows; per-stratum and cross-domain evaluation | in-domain high, cross-domain often 10–25 points lower; measure it, don't hope |
| **Strict CPU/NPU budget on the machine** | MobileNetV3-Large or MobileNetV4-Conv-S, INT8 | verify the quantised model on the transparent stratum; add detection to remove scene dependence | within a few points of much larger models [R17,R39] |
| **Transparent items are the crux** | Measure the RGB ceiling honestly (object-crop vs full-frame, transparent stratum only) | add SWIR/NIR hiperspectral or a polarisation head; treat RGB as shape/context | RGB alone will plateau; fusion is the documented fix [R31,R32] |
| **Purity is money (high-value stream)** | HSI + hierarchical PLS-DA or a small 1D-CNN with careful preprocessing | conformal rejection to a manual/second-sensor path; report purity/recovery, not accuracy | 0.91–1.00 sensitivity/specificity class-wise [R21]; ~99% purity PP/PE/PET industrially [R23] |
| **Black plastics in the stream** | SWIR with wide spectral range; test early | explicit reject path; black items are the documented weak case [R22,R24] | expect materially worse performance on black than coloured |
| **Research contribution wanted** | Polarimetric or RGB+HSI fusion for transparent-vs-transparent; a properly powered cue study; a public binary glass/plastic benchmark with a spectral reference branch | | the gaps are: no public binary benchmark, underpowered cue ablations everywhere, and almost no polarimetry datasets [R31] |

## 11.3 A 90-day plan

**Weeks 1–2 — establish the floor and the ceiling of your data.**
Build a stratified, object-level test set (chapter 9 §9.2). Run the physics rule (one threshold on
highlight statistics), a tree ensemble on hand-crafted features, and a ResNet-18 fine-tune.
Run the shuffled-label and background-only controls. Publish nothing until the controls pass.

**Weeks 3–6 — find out whether RGB can do the job at all.**
Object-crop vs full-frame ablation; transparent-stratum-only evaluation; three acquisition domains,
one held out. If the transparent stratum stalls while the rest improves, stop optimising the RGB
model and start procuring a second modality.

**Weeks 7–10 — add the second modality (if needed) and the operating policy.**
Spectral or polarimetric branch, fusion at feature level, plus calibration and a conformal reject
rule. Quantise and deploy the RGB branch on the target hardware and re-verify on the golden set.

**Weeks 11–13 — productionise the boring parts.**
Golden-set monitoring, drift statistics, per-stratum dashboards, the reject-stream destination, and
documentation good enough that a second engineer can retrain the model without you.

## 11.4 What this repository does *not* establish

Being explicit about the limits matters more than the summary:

* **All measured numbers are on synthetic proxy data.** `results/MEASURED_RESULTS.md` measures
  method behaviour (robustness, failure modes, statistical power), not real glass/plastic accuracy.
  The renderer cannot reproduce real materials, dirt, optics or camera physics.
* **The cue ablation is underpowered** (95% CI ≈ ±0.10 at n=120) and therefore cannot rank cue
  importance. A factorial or dose-response design with ≥1,000 items per cell is required.
* **Only classical models were actually trained here.** The deep tier is implemented
  (`src/gvp/deep.py`) and shares the splits, but no GPU run is included in the released results.
* **The composite ranking is a rubric, not a measurement.** It is a transparent weighted mean of
  human judgements on ten criteria, ships with five weighting profiles, and is meant to be argued
  with — `configs/scoring.yaml` is where you disagree.

## 11.5 Three open problems worth someone's time

1. **A public binary glass-vs-plastic benchmark with a spectral reference branch.** Nothing in the
   current dataset landscape [R1,R2,R3,R4,R5,R6] isolates this decision, and the absence of paired
   spectral data makes it impossible to say whether a failure is the model's or the sensor's.
2. **Polarisation imaging datasets and fusion architectures.** The physics is unambiguous and the
   literature is thin; the glass/transparent-plastic case is precisely where it should win [R31].
3. **Properly powered cue-importance studies.** The field is full of single-seed, few-hundred-item
   ablation claims — including, deliberately documented, this repository's own underpowered one.
