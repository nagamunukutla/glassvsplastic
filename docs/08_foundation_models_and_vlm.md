# 8. Foundation models and vision-language models

The 2024–2026 wave changed the economics of this problem: you can now get a usable classifier
before you own a single labelled image, and re-target it to a new taxonomy by editing text. This
chapter covers what that is actually worth, what it costs, and where it silently fails.

## 8.1 The reported numbers

| Approach | Setting | Result | Reference |
|---|---|---|---|
| OpenCLIP ViT-L/14-336 | zero-shot, TrashNet (6 classes) | 76.30% accuracy, 427.94M params, 2.83 FPS | [R29] |
| OpenCLIP ViT-L/14-2B | zero-shot, 6-class industrial set (14,310 images) | 82.71% | [R28] |
| OpenCLIP ViT-L/14-2B | + targeted prompt engineering | **90.48%** (+7.77 points, no training) | [R28] |
| OpenCLIP ViT-L/14-2B | fully supervised | 97.18%, 3.79 ms/image (~263 FPS batched GPU) | [R28] |
| OWL-ViT / OpenCLIP | zero-shot TrashNet, hierarchical prompts | competitive per-class, below supervised ViT-B/16 and MobileNetV3-S | [R29] |
| Frozen VLM features + training-free adapter (Tip-Adapter, k-NN) | zero/few-shot under taxonomy drift | recommended practical route | [R30] |
| DINOv3 1-NN | many exemplars available | strong training-free baseline | [R30] |
| MLLMs (GPT-4o, LLaVA-OneVision) | zero-shot waste | performed well; textual few-shot *reduced* accuracy; image few-shot helped GPT-4o at high cost | [R30] |
| Grounding DINO + CLIP + LoRA (EcoVision) | cluttered field images | few-shot adaptation improves accuracy while preserving zero-shot generalisation | [R44] |

## 8.2 What these models are genuinely good at here

* **Cold start.** 76–83% zero-shot on waste taxonomies is worse than a fine-tuned CNN and vastly
  better than nothing. It is enough to bootstrap: pre-label a batch, correct the errors, train a
  small supervised model on the corrected set, and keep the VLM for the long tail.
* **Prompt engineering is the cheapest accuracy in the field.** +7.8 points for editing text
  [R28]. Specific, visually descriptive prompts beat abstract labels ("a picture of a glass jar and
  beer bottles" over "broken pieces of glass") [R28].
* **Taxonomy drift.** Plants change product mix; category definitions change; a new resin code
  appears. A text-conditioned head absorbs that as an ops action rather than a re-training project
  [R30]. This is a real operational advantage, not a benchmark curiosity.
* **Feature extraction.** Frozen foundation features + a k-NN or training-free adapter trained on
  1–16 images per class is the highest-label-efficiency route available, and it fits on cached
  features in seconds [R30].
* **Hard-case triage and dataset auditing.** An MLLM asked to describe *why* an item is ambiguous
  is useful for building a stratified test set, even if it never ships as the classifier [R30].

## 8.3 Where they fail, specifically

* **They do not solve the transparent-transparent case.** CLIP's text prior knows "glass" and
  "plastic bottle" as concepts, but the visual evidence for a clear PET bottle versus a clear glass
  bottle is exactly what chapter 2 says is missing. Prompt engineering cannot add sensor physics.
* **Prompt sensitivity is a maintenance liability.** Reported zero-shot accuracy moves by several
  points with wording [R29], which means your production behaviour changes when someone edits a
  string. Version prompts like code and re-validate on a frozen test set after any edit.
* **Latency and cost.** ViT-L/14-class models run at 2.83 FPS in a CPU-limited study [R29] and
  3.79 ms/image on a capable GPU at batch [R28]; a 0.5M-parameter distilled CNN does the same job
  in under 2 ms on a phone CPU [R9,R39]. On a single-lane belt that may be fine; on a high-throughput
  line it will not be.
* **Calibration.** Contrastive models are not calibrated classifiers; thresholds do not transfer
  between prompt sets. Any rejection policy must be recalibrated per prompt version.
* **Licensing and provenance.** "Open" weights are not always openly licensed data, and regulated
  waste-handling deployments increasingly need to document training-data provenance [R29].

## 8.4 Recommended deployment patterns

**Pattern A — cold start, then distillation.**
```
zero-shot VLM labels 2–5k unlabelled plant images
   → human correction on the ambiguous stratum only (~10–20% of items)
   → train a small supervised CNN (MobileNetV3/V4 or DenseNet-121)
   → keep the VLM as the fallback for low-confidence items and novel categories
```
This is the pattern the evidence supports: zero-shot gets you moving [R28,R29], fine-tuning gets you
the last 5–7 points [R28], and the small model gets you the latency [R9,R39].

**Pattern B — detector first, then a frozen material head.**
```
Grounding DINO / YOLO detector  → crop  →  frozen CLIP/DINOv2 features  →  training-free adapter
```
Motivated directly by this repository's measurement that global image statistics collapse under
background clutter: the detector restores the spatial selectivity that the classical tier lacks and
that a full-frame classifier never has. EcoVision reports exactly this composition with few-shot
LoRA adaptation and field-image validation [R44].

**Pattern C — never let the VLM make the material call.**
Use it for object-ness, category language, and explanations; let spectra, polarisation or a
task-trained model make the glass/plastic decision. There is no evidence in the literature that a
VLM outperforms a fine-tuned CNN on this binary question, and considerable evidence that it is
slower and less stable.

## 8.5 A note on prompt design for material classification

The waste-VLM work reports measurable gains from prompts that are *physically descriptive* rather
than nominal [R28]. For glass vs plastic, the cues of chapter 2 can be written into the prompt
directly:

```
"a photo of a clear glass bottle with sharp bright highlights and a visible,
 undistorted background seen through the glass"
"a photo of a plastic bottle, translucent body, hazy background visible
 through the wall, soft broad sheen, mould seam"
```

Two cautions. First, prompt engineering is a hyper-parameter search on your test set unless you
hold out a validation split — do it properly (chapter 9). Second, a prompt that lists the cues the
model should use is also a documented way to *measure* them: prompt ablations (with/without the
"hazy background" clause) tell you whether the model responds to the physics or to spurious
wording, which is a cheap interpretability probe unique to language-conditioned models.
