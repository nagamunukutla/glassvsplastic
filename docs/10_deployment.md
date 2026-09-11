# 10. Deployment: from a notebook metric to a working sorting line

A model that scores 0.95 in a notebook and 0.74 on the belt is the normal outcome, not a bug. This
chapter covers the gap: latency budgets, hardware, optics, drift, and the operating policy that
turns a probability into an actuator decision.

## 10.1 Start with the latency budget, not the model

Work backwards from the mechanics:

```
objects per second  =  belt speed (m/s) / mean object pitch (m)
decision budget     =  pitch / speed          (time between the camera and the actuator)
per-object budget   =  decision budget - tracking/actuation overhead
```

| Belt speed | Object pitch | Throughput | Total budget | Comfortable per-image inference |
|---|---|---|---|---|
| 0.3 m/s | 0.20 m | 1.5 obj/s | 660 ms | anything |
| 1.0 m/s | 0.20 m | 5 obj/s | 200 ms | a large ViT is viable |
| 2.0 m/s | 0.15 m | 13 obj/s | 75 ms | a mid GPU CNN |
| 3.0 m/s | 0.10 m | 30 obj/s | 33 ms | MobileNet-class on GPU/NPU; HSI needs band selection |
| 4.0 m/s | 0.08 m | 50 obj/s | 20 ms | INT8 mobile CNN on fixed-function NPU, or FPGA |

Practical numbers from this study's own accounting: a classical pipeline spends ~18 ms/image on
descriptor extraction and 0.004–1.2 ms on the classifier; classic CNNs range from sub-millisecond to
~25 ms per image on CPU and 1–8 ms on a mid GPU; MobileNetV4-Conv-S runs at 2.4 ms on a phone CPU
and its distilled Hybrid-L variant at 3.8 ms on an EdgeTPU in INT8 [R39]. Industrial HSI systems do
not do per-pixel deep inference in real time; they use band selection or dedicated hardware [R20-adjacent].

**Rule of thumb:** reserve at most one third of the budget for material classification. Detection,
tracking, actuator timing and safety checks consume the rest, and they are not optional.

## 10.2 Hardware tiers

| Tier | Hardware | Realistic model | Notes |
|---|---|---|---|
| MCU / DSP | Cortex-M + NPU, few hundred kB | MobileNetV3-Small / MNv4-Conv-S INT8 | needs a fixed object pitch and a very clean scene |
| Embedded SoC | Jetson Orin Nano-class, 8–15 W | MobileNetV3-L, EfficientNet-B0, 224 px | the sweet spot for a single-lane sorter |
| Industrial PC + GPU | RTX A2000/A4000-class | ConvNeXt-T, EfficientNetV2-S, ViT-B/16 frozen + head | allows multi-camera and fusion |
| Server / edge rack | A100/L40S-class | HSI cubes, VLM fallback, ensembles | usually for training and offline re-labelling, not per-object decisions |
| Spectral sorter (commercial) | NIR/HSI camera + FPGA | band ratios, PLS-DA, small 1D CNN | purity near 100% achievable [R23] |
| Polarimetric add-on | polarisation camera + RGB | feature-level fusion | specifically for glass vs transparent plastics [R31] |

Quantisation is the usual lever: INT8 typically costs ~0.5–1 point of accuracy on a material task
and halves latency/memory, but **verify the quantised model on the highlight cue specifically** —
fine, high-dynamic-range reflections are exactly what per-tensor quantisation damages, and they are
the cue chapter 2 says matters.

## 10.3 Optics and illumination (usually the highest-leverage change)

* **Illumination geometry decides which cues exist at all.** Diffuse dome light suppresses the
  specular difference between glass and plastic; a small, sharp source restores it. The correct
  industrial choice is usually *both*: broad diffuse light for a stable base image (line-scan
  geometry with consistent shadows), plus a controlled specular source to keep the highlight cue
  alive. This repository's proxy experiments are a cautionary tale in the other direction: a
  featureless white background removed the transmission cue entirely, and models did not care —
  because they had other cues. Do not assume which cue your model uses; measure it (chapter 9 §9.6).
* **Background**: a textured, matte, non-glossy belt surface preserves transmission cues and avoids
  specular contamination. This is the single cheapest upgrade to a glass/plastic RGB deployment.
* **Polarise the illumination**, not just the camera: a polariser on the light plus a cross-polarised
  camera removes most surface glare and reveals subsurface detail — at the cost of killing the
  specular cue. Choose deliberately; you cannot have both.
* **Backlighting** turns transparent objects into silhouettes and is excellent for detection and
  shape, useless for material.
* **Resolution**: target ≥80–120 px across the smallest object of interest. RealWaste's authors
  scaled to 524×524 precisely because transparency and glass-vs-plastic-bottle distinctions need the
  pixels [R2]; downscaling to 224 px to fit a mobile model may be deleting your signal.

## 10.4 Operating policy: accuracy is not the deliverable

What the plant buys is a **purity/recovery operating point**:

```
accept if p_glass > τ        →  BOM: higher τ means higher purity, lower recovery
reject/divert otherwise      →  the reject stream is the cost of the policy
```

Procedure:

1. Train the classifier; hold out a **calibration set** that matches production (not the test set,
   which you will have used for model selection).
2. Compute the reliability curve. If it is not monotone (contrastive/VLM heads often are not),
   apply temperature scaling on the calibration set first.
3. Sweep τ and plot purity/recovery versus τ for the *plausible* deployment distribution, including
   the sub-classes you care about (glass shards, clear PET film, black plastic).
4. Choose τ from the economics, not from the F1 maximum.
5. Wrap the decision in a **conformal prediction** layer if you need a distribution-free coverage
   guarantee (e.g. "at most 1% of accepted items are mislabelled"), which is what a certification
   audit asks for.
6. Route the reject stream somewhere useful: back to the feeding conveyor, to a second sensor
   (NIR/polarimeter), or to a manual station. A rejection policy without a destination is just a
   smaller yield.

For reference on how the industry quantifies this: hyperspectral glass studies report
sensitivity/specificity pairs rather than accuracy [R21], and industrial HSI vendors quote purity
close to 100% for PP/PE/PET with NIR [R23]. That is the format your results should be in if the
output is a sorting machine and not a paper.

## 10.5 Drift, calibration and maintenance

| Failure | Symptom | Mitigation |
|---|---|---|
| Lamp ageing / spectral shift | slow accuracy decay | reference target in frame; monitor feature-distribution drift, not just accuracy |
| Belt wear / new belt colour | sudden drop (background-continuity features) | keep the background in the calibration set; re-calibrate thresholds |
| New product designs (sleeves, colour) | class-specific recall collapse | per-stratum dashboards; add exemplars; k-NN/adapters update in minutes |
| Camera lens contamination | global contrast loss | scheduled cleaning; monitor high-frequency energy of a reference patch |
| Seasonality of the feed | composition shift, apparent accuracy change without model change | report metrics per stratum, never only aggregate |
| Prompt/model version change (VLM) | quiet behaviour change | version prompts and re-validate on a frozen probe set |
| Quantisation/deployment build change | small accuracy change | run a fixed "golden set" through the deployed artifact, not the notebook |

A minimal production monitor: **fixed golden set** re-scored on every deployment (200–500 items,
~50 of them transparent), plus drift statistics on the descriptor/feature distributions, plus
per-stratum recall. Accuracy alone is too coarse and too late to catch a lamp ageing out.

## 10.6 Failure-mode playbook

| Symptom | Likely cause | First thing to try |
|---|---|---|
| Great offline, bad on the line | scene shortcut: background/lighting differ | background-only control; retrain with the real belt in frame; crop to object |
| Works on bottles, fails on film and shards | shape-based decision | stratify test set; add flexible items to training; lower resolution dependency |
| Works on coloured, fails on clear items | the transparent stratum is unlearnable in RGB | add spectral/polarimetric branch or an explicit "unknown-clear" class [R31,R32] |
| Confidence is high but wrong | miscalibration (especially VLM heads) | temperature scaling on a matched calibration set; conformal wrapper |
| Accuracy decays slowly and steadily | sensor/illumination drift | hardware check first, model second |
| Collapses on cluttered belts | global descriptors with no spatial selectivity (see `results/MEASURED_RESULTS.md` §6) | add a detector/segmenter in front |
| Black plastics misclassified | NIR absorption | SWIR range hardware or an explicit reject path [R22,R24] |

## 10.7 Documentation and compliance

Whatever you deploy, record: dataset provenance and licences, the exact split (object-level), the
selected operating point and its purity/recovery, the calibration procedure, the golden set and its
version, the model artifact hash, and the prompt versions for any language-conditioned component.
For regulated streams, the rejection rate and the destination of rejected items are part of the
record — an auditable pipeline is usually worth a point of accuracy.
