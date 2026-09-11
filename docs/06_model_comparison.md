# Model comparison: what to use for glass vs plastic, and why

This document is generated (`python -m gvp.cli report`) from three data files:

- `data/model_registry.csv` — architecture and deployment facts, each reported result carrying a `[Rn]` reference key

- `data/model_scores.csv` — ten criteria scored 1–5 per model (5 = best; cost and latency are scored as *efficiency*)

- `configs/scoring.yaml` — the weights that turn those ten criteria into one ranking, including alternative profiles (accuracy-first, edge budget, cold start)


> **How to read this.** The registry is a *literature-and-architecture* comparison: `evidence` fields quote numbers from the papers cited in `docs/12_references.md`, not numbers produced by this repository. Repository measurements (on the synthetic proxy task) are in `results/MEASURED_RESULTS.md` and are, deliberately, reported separately. Never mix the two.


## 1. Composite ranking (default profile: *Balanced commercial deployment*)


37 architectural options are ranked below. Cross-cutting *components* (1) are deliberately excluded — ranking a rejection layer against a model would be a category error — and appear in §1c instead.


Weights: `accuracy_ceiling` 0.18, `transparency_robustness` 0.20, `data_efficiency` 0.12, `contamination_robustness` 0.12, `cost_efficiency` 0.10, `throughput` 0.08, `explainability` 0.07, `tooling_maturity` 0.08, `calibration_abstention` 0.05, `zero_shot_capability` 0.03

| rank | model | family | score | acc | transp | data | contam | cost | thr | expl | tool | reject | zero | tier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | NIR / FTIR spectroscopy + PLS-DA or 1D-CNN | Spectral sensor + learned model | 3.97 | 5 | 5 | 4 | 3 | 2 | 4 | 4 | 3 | 5 | 2 | production (reference modality) |
| 2 | DenseNet-121 / -169 (transfer learning) | CNN transfer learning | 3.82 | 5 | 4 | 4 | 3 | 4 | 3 | 2 | 5 | 3 | 2 | baseline / production |
| 3 | Hyperspectral imaging (VIS-NIR / SWIR) + ML | Spectral imaging + learned model | 3.78 | 5 | 5 | 3 | 4 | 2 | 3 | 3 | 3 | 4 | 2 | production (industrial) / research |
| 4 | LDA / QDA on hand-crafted features | Classical ML on features | 3.75 | 3 | 3 | 5 | 2 | 5 | 5 | 5 | 5 | 4 | 1 | baseline / production (linear reference) |
| 5 | MobileNetV4 (Conv-S / Hybrid-L) | CNN transfer learning (mobile, 2024) | 3.74 | 4 | 4 | 4 | 3 | 5 | 5 | 2 | 3 | 3 | 2 | production (edge, new) |
| 6 | Random Forest / ExtraTrees on hand-crafted features | Classical ML on features | 3.73 | 4 | 3 | 5 | 2 | 5 | 4 | 4 | 5 | 3 | 1 | production (small-data) |
| 7 | EfficientNet-B0 / -B3 | CNN transfer learning | 3.72 | 4 | 4 | 4 | 3 | 4 | 4 | 2 | 5 | 3 | 2 | production |
| 8 | MobileNetV2 / ShuffleNetV2 (1.0x) | CNN transfer learning (mobile) | 3.70 | 4 | 3 | 4 | 3 | 5 | 5 | 2 | 5 | 3 | 2 | production (edge) |
| 9 | MobileNetV3-Large / -Small | CNN transfer learning (mobile) | 3.70 | 4 | 3 | 4 | 3 | 5 | 5 | 2 | 5 | 3 | 2 | production (edge) |
| 10 | Specular-highlight statistics (1-D rule) | Heuristic / physics-first | 3.69 | 3 | 3 | 5 | 2 | 5 | 5 | 5 | 4 | 2 | 5 | baseline |
| 11 | Gradient-boosted trees (HistGradientBoosting/XGBoost/LightGBM) | Classical ML on features | 3.67 | 4 | 3 | 4 | 2 | 5 | 5 | 3 | 5 | 4 | 1 | production (small-mid data) |
| 12 | Open-vocabulary detect-then-classify (Grounding DINO + CLIP, EcoVision) | Foundation model pipeline | 3.65 | 4 | 4 | 4 | 5 | 2 | 2 | 3 | 4 | 3 | 4 | research / production |
| 13 | ConvNeXt-Tiny / -Small | CNN transfer learning (modern CNN) | 3.64 | 5 | 4 | 3 | 4 | 3 | 3 | 2 | 4 | 3 | 2 | research / production |
| 14 | Swin-T/SwinV2-T / MaxViT-T (hierarchical attention) | Hybrid transformer | 3.64 | 5 | 5 | 2 | 4 | 2 | 2 | 3 | 4 | 3 | 3 | research / production (server) |
| 15 | Frozen foundation features + training-free adapter (Tip-Adapter / kNN / linear probe) | Foundation model / VLM (few-shot) | 3.64 | 4 | 3 | 5 | 3 | 3 | 3 | 3 | 5 | 4 | 4 | production (bootstrap) |
| 16 | SVM (RBF) on hand-crafted features | Classical ML on features | 3.63 | 3 | 3 | 5 | 2 | 5 | 5 | 4 | 5 | 3 | 1 | production (small-data) |
| 17 | Polarisation imaging (glass vs transparent plastic) | Heuristic / physics-first (non-RGB sensor) | 3.59 | 4 | 5 | 4 | 2 | 3 | 4 | 3 | 2 | 3 | 4 | research |
| 18 | k-NN / Gaussian mixture / Mahalanobis on features | Classical ML on features | 3.55 | 3 | 3 | 5 | 2 | 5 | 4 | 4 | 4 | 4 | 2 | production (small-data) |
| 19 | ResNet-18 (transfer learning) | CNN transfer learning | 3.52 | 4 | 3 | 4 | 3 | 4 | 4 | 2 | 5 | 3 | 2 | baseline |
| 20 | EfficientNetV2-S / -M | CNN transfer learning | 3.52 | 5 | 4 | 3 | 3 | 3 | 2 | 2 | 5 | 3 | 2 | research / production (server) |
| 21 | Attention-augmented CNN (SE / CBAM / attention-AlexNet) and multi-scale fusion | Hybrid / attention | 3.52 | 5 | 4 | 3 | 4 | 3 | 3 | 3 | 2 | 3 | 1 | research |
| 22 | DeiT-Tiny / DeiT-Small (distilled ViT) | Transformer (data-efficient) | 3.51 | 4 | 4 | 3 | 3 | 3 | 3 | 4 | 4 | 3 | 3 | research |
| 23 | Custom sensor fusion (RGB+HSI, RGB+polarisation, RGB+thermal, RGB+depth/3D) | Multisensor fusion | 3.50 | 5 | 5 | 2 | 5 | 1 | 2 | 3 | 2 | 4 | 1 | production (industrial) / research |
| 24 | RegNetY-3.2GF / InceptionNeXt-T | CNN transfer learning | 3.47 | 4 | 4 | 3 | 4 | 3 | 3 | 2 | 4 | 3 | 2 | research |
| 25 | CLIP / OpenCLIP / SigLIP — zero-shot classification | Foundation model / VLM | 3.40 | 3 | 3 | 5 | 3 | 3 | 3 | 3 | 5 | 2 | 5 | production (bootstrap) / research |
| 26 | ResNet-50 / ResNeXt-50 32x4d (transfer learning) | CNN transfer learning | 3.35 | 4 | 3 | 4 | 3 | 3 | 3 | 2 | 5 | 3 | 2 | baseline / production |
| 27 | Inception-v3 / Xception / InceptionResNetV2 | CNN transfer learning | 3.35 | 4 | 4 | 3 | 3 | 3 | 3 | 2 | 4 | 3 | 2 | baseline / production |
| 28 | ViT-B/16 (supervised fine-tune) | Transformer | 3.35 | 4 | 5 | 2 | 3 | 2 | 2 | 3 | 4 | 3 | 3 | research |
| 29 | Purpose-built lightweight CNNs (RecycleNet, Focus-RCNet-KD, WasNet) | CNN from scratch (waste-specific designs) | 3.32 | 4 | 3 | 3 | 3 | 5 | 5 | 2 | 2 | 3 | 1 | production (edge) |
| 30 | Thermal IR and 3D/structured-light sensing for transparent objects | Non-RGB sensing | 3.28 | 3 | 4 | 3 | 4 | 3 | 3 | 3 | 3 | 3 | 2 | research |
| 31 | Shallow MLP on hand-crafted features | Classical ML on features | 3.18 | 3 | 3 | 3 | 2 | 5 | 5 | 2 | 4 | 3 | 1 | research |
| 32 | Transmission / haze index (background-continuation) | Heuristic / physics-first | 3.14 | 2 | 3 | 4 | 1 | 5 | 5 | 4 | 3 | 2 | 5 | baseline |
| 33 | Edge + texture descriptors (Canny/Sobel, LBP, GLCM) | Heuristic / physics-first | 3.05 | 2 | 2 | 4 | 1 | 5 | 5 | 4 | 5 | 1 | 5 | baseline |
| 34 | Colour/intensity thresholding (industrial VIS sorter) | Heuristic / physics-first | 3.04 | 2 | 1 | 5 | 1 | 5 | 5 | 5 | 5 | 1 | 5 | baseline |
| 35 | Small CNN trained from scratch (4-8 conv layers) | CNN from scratch | 3.00 | 3 | 3 | 2 | 3 | 4 | 4 | 3 | 3 | 3 | 1 | baseline / research |
| 36 | Multimodal LLM as a classifier (GPT-4o / LLaVA-OneVision) | Foundation model / VLM (language interface) | 2.92 | 3 | 3 | 5 | 2 | 1 | 1 | 4 | 4 | 2 | 5 | research / tooling |
| 37 | VGG-16 / VGG-19 (transfer learning) | CNN transfer learning | 2.47 | 3 | 3 | 2 | 3 | 1 | 1 | 2 | 4 | 2 | 2 | legacy baseline |

*(criterion columns are the raw 1–5 scores: `acc` accuracy ceiling, `transp` transparency robustness, `data` data efficiency, `contam` contamination robustness, `cost` cost efficiency, `thr` throughput, `expl` explainability, `tool` tooling maturity, `reject` calibration/abstention, `zero` useful with zero labels.)*


### 1c. Cross-cutting components (not ranked against models)


These are not alternatives to a model; they are things to add *on top of* whichever model you pick. Scores shown for reference.

| component | score | why it matters | use it when |
| --- | --- | --- | --- |
| Uncertainty + rejection layer (conformal prediction, MC-dropout, ensembles, OOD) | 4.3204 | Directly monetises the glass/plastic ambiguity: route low-confidence items to a second sensor (NIR/polarimeter) or to a manual station instead of guessing. Conformal prediction gives a *distribution-free* coverage guarantee, which is what a certification process asks for. | Any real deployment - especially anything feeding a purity-sensitive product stream. |

### 1b. Does the ranking survive a change of priorities?


A ranking that only holds under one weight vector is not a finding. Top 8 per profile:

| profile | #1 | #2 | #3 | #4 | #5 | #6 | #7 | #8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Balanced commercial deployment | Uncertainty + rejection layer (conformal prediction, MC-dropout, ensembles, OOD) | NIR / FTIR spectroscopy + PLS-DA or 1D-CNN | DenseNet-121 / -169 (transfer learning) | Hyperspectral imaging (VIS-NIR / SWIR) + ML | LDA / QDA on hand-crafted features | MobileNetV4 (Conv-S / Hybrid-L) | Random Forest / ExtraTrees on hand-crafted features | EfficientNet-B0 / -B3 |
| Accuracy first (purity-driven) | Uncertainty + rejection layer (conformal prediction, MC-dropout, ensembles, OOD) | NIR / FTIR spectroscopy + PLS-DA or 1D-CNN | Hyperspectral imaging (VIS-NIR / SWIR) + ML | Custom sensor fusion (RGB+HSI, RGB+polarisation, RGB+thermal, RGB+depth/3D) | Swin-T/SwinV2-T / MaxViT-T (hierarchical attention) | ConvNeXt-Tiny / -Small | DenseNet-121 / -169 (transfer learning) | Open-vocabulary detect-then-classify (Grounding DINO + CLIP, EcoVision) |
| Edge / embedded budget | Uncertainty + rejection layer (conformal prediction, MC-dropout, ensembles, OOD) | MobileNetV4 (Conv-S / Hybrid-L) | Specular-highlight statistics (1-D rule) | LDA / QDA on hand-crafted features | MobileNetV2 / ShuffleNetV2 (1.0x) | MobileNetV3-Large / -Small | Random Forest / ExtraTrees on hand-crafted features | SVM (RBF) on hand-crafted features |
| Cold start / no labels | Uncertainty + rejection layer (conformal prediction, MC-dropout, ensembles, OOD) | Specular-highlight statistics (1-D rule) | CLIP / OpenCLIP / SigLIP — zero-shot classification | Frozen foundation features + training-free adapter (Tip-Adapter / kNN / linear probe) | Open-vocabulary detect-then-classify (Grounding DINO + CLIP, EcoVision) | Polarisation imaging (glass vs transparent plastic) | Transmission / haze index (background-continuation) | Colour/intensity thresholding (industrial VIS sorter) |
| Research benchmark | Uncertainty + rejection layer (conformal prediction, MC-dropout, ensembles, OOD) | NIR / FTIR spectroscopy + PLS-DA or 1D-CNN | Hyperspectral imaging (VIS-NIR / SWIR) + ML | Swin-T/SwinV2-T / MaxViT-T (hierarchical attention) | LDA / QDA on hand-crafted features | DenseNet-121 / -169 (transfer learning) | Custom sensor fusion (RGB+HSI, RGB+polarisation, RGB+thermal, RGB+depth/3D) | Open-vocabulary detect-then-classify (Grounding DINO + CLIP, EcoVision) |

## 2. Who wins each criterion?

| criterion | 1st | 2nd | 3rd |
| --- | --- | --- | --- |
| Accuracy ceiling on glass-vs-plastic | DenseNet-121 / -169 (transfer learning) | EfficientNetV2-S / -M | ConvNeXt-Tiny / -Small |
| Transparency robustness (clear item vs clear item) | Polarisation imaging (glass vs transparent plastic) | ViT-B/16 (supervised fine-tune) | Swin-T/SwinV2-T / MaxViT-T (hierarchical attention) |
| Data efficiency | Colour/intensity thresholding (industrial VIS sorter) | Specular-highlight statistics (1-D rule) | SVM (RBF) on hand-crafted features |
| Contamination / clutter robustness | Open-vocabulary detect-then-classify (Grounding DINO + CLIP, EcoVision) | Custom sensor fusion (RGB+HSI, RGB+polarisation, RGB+thermal, RGB+depth/3D) | ConvNeXt-Tiny / -Small |
| Cost efficiency | Colour/intensity thresholding (industrial VIS sorter) | Edge + texture descriptors (Canny/Sobel, LBP, GLCM) | Specular-highlight statistics (1-D rule) |
| Throughput | Colour/intensity thresholding (industrial VIS sorter) | Edge + texture descriptors (Canny/Sobel, LBP, GLCM) | Specular-highlight statistics (1-D rule) |
| Explainability / auditability | Colour/intensity thresholding (industrial VIS sorter) | Specular-highlight statistics (1-D rule) | LDA / QDA on hand-crafted features |
| Tooling maturity | Colour/intensity thresholding (industrial VIS sorter) | Edge + texture descriptors (Canny/Sobel, LBP, GLCM) | SVM (RBF) on hand-crafted features |
| Calibration & abstention | NIR / FTIR spectroscopy + PLS-DA or 1D-CNN | LDA / QDA on hand-crafted features | Gradient-boosted trees (HistGradientBoosting/XGBoost/LightGBM) |
| Useful with zero labels | Colour/intensity thresholding (industrial VIS sorter) | Edge + texture descriptors (Canny/Sobel, LBP, GLCM) | Specular-highlight statistics (1-D rule) |

## 3. Cost of each option (facts, not judgements)

| model | family | params (M) | MACs (G) | input | pretraining | ImageNet top-1 | labels needed | train cost | inference | licence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NIR / FTIR spectroscopy + PLS-DA or 1D-CNN | Spectral sensor + learned model | 0.01-2 (1D models) | negligible | 1D spectrum (hundreds-2,000 bins) | none (1D nets)  | n/a | 40-1,000 spectra per class | CPU: seconds-minutes | 0.5-1.6 ms/spectrum (model only); sensor-limited in practice | open algorithms (chemometrics); sensor vendor-dependent |
| DenseNet-121 / -169 (transfer learning) | CNN transfer learning | 8.0 | 2.9 | 224 | ImageNet-1k | 74.4 (121) / 75.6 (169) | ~200-2,000 labelled crops | GPU: ~10-15 min | ~4-12 ms/image CPU | BSD-3 |
| Hyperspectral imaging (VIS-NIR / SWIR) + ML | Spectral imaging + learned model | 0.1-30 (CNN) / 0.01 (PLS-DA) | 0.1-20 | cube (e.g. 100-300 bands) | ImageNet (2D) or none (3D/1D nets) | n/a | a few hundred labelled objects (pixel-level: thousands of spectra) | GPU: minutes-hours (cube data is large) | not real-time per-pixel on CPU; industrial systems use band-selection/FPGA | algorithms open; cameras commercial |
| LDA / QDA on hand-crafted features | Classical ML on features | — | — | feature vector or spectrum | none | n/a | 50-500 labelled samples | <1 s CPU | microseconds/image | scikit-learn (BSD-3) |
| MobileNetV4 (Conv-S / Hybrid-L) | CNN transfer learning (mobile, 2024) | 3.8 (Conv-S) / 32.5 (Hybrid-L) | 0.2 (Conv-S) | 224 | ImageNet-1k (+JFT for distilled variants) | 73.8 (Conv-S) / 87.0 (Hybrid-L distilled) | ~100-2,000 labelled crops | CPU minutes to GPU hours | 2.4 ms on a Pixel 6 CPU (Conv-S); 3.8 ms on a Pixel 8 EdgeTPU (Hybrid-L, INT8) | Apache-2.0 |
| Random Forest / ExtraTrees on hand-crafted features | Classical ML on features | — | — | feature vector | none | n/a | ~200-1,000 labelled images | <1 s CPU | 0.1-1.2 ms/image (clf only) | scikit-learn (BSD-3) |
| EfficientNet-B0 / -B3 | CNN transfer learning | 5.3 / 12.2 | 0.39 / 1.8 | 224 / 300 | ImageNet-1k | 77.1 (B0) / 81.6 (B3); 79.4 for B0 with the RA4-E3600 recipe | ~300-3,000 labelled crops | GPU minutes | 2-8 ms/image CPU | Apache-2.0 |
| MobileNetV2 / ShuffleNetV2 (1.0x) | CNN transfer learning (mobile) | 3.5 | 0.3 | 224 | ImageNet-1k | 71.9 / 69.4 | ~100-2,000 labelled crops | CPU: minutes; GPU: <5 min | 1-3 ms/image CPU; sub-ms on a DSP/NPU | Apache-2.0 / BSD-3 |
| MobileNetV3-Large / -Small | CNN transfer learning (mobile) | 5.5 / 2.5 | 0.22 / 0.06 | 224 | ImageNet-1k | 75.3 / 67.7 | ~100-2,000 labelled crops | CPU minutes | 1-2 ms/image CPU | Apache-2.0 |
| Specular-highlight statistics (1-D rule) | Heuristic / physics-first | — | — | full frame | none | n/a | 0 | 0 | very high | n/a (own code) |
| Gradient-boosted trees (HistGradientBoosting/XGBoost/LightGBM) | Classical ML on features | — | — | feature vector | none | n/a | ~500-5,000 labelled images | 1-30 s CPU | 0.04 ms/image (clf only) | BSD/MIT (sklearn/LightGBM/XGBoost) |
| Open-vocabulary detect-then-classify (Grounding DINO + CLIP, EcoVision) | Foundation model pipeline | ~250+ | high | variable (800-1333) | grounding + CLIP | n/a | 0 up to ~1,000 for LoRA adaptation | hours GPU for LoRA; 0 for zero-shot | ~2-10 FPS on a mid GPU | Apache-2.0 components (check Grounding DINO licence) |
| ConvNeXt-Tiny / -Small | CNN transfer learning (modern CNN) | 28.6 / 50.0 | 4.5 / 8.7 | 224 | ImageNet-1k/-21k | 82.1-82.5 (T) / 83.1 (S) | ~500-5,000 labelled crops | GPU: ~15-45 min | ~8-25 ms/image CPU | MIT (timm) / BSD-3 |
| Swin-T/SwinV2-T / MaxViT-T (hierarchical attention) | Hybrid transformer | 28.3 / 28.4 / 30.9 | 4.5 / 5.9 / 5.6 | 224 | ImageNet-1k/-21k | 81.3 / 82.1 / 83.4 | ~1,000-10,000 labelled crops | GPU: ~30-90 min | ~15-40 ms/image CPU | MIT / Apache-2.0 |
| Frozen foundation features + training-free adapter (Tip-Adapter / kNN / linear probe) | Foundation model / VLM (few-shot) | 86-1000 (frozen backbone) | backbone only, once | 224-518 | DINOv2/DINOv3, EVA-CLIP, SigLIP | kNN 82-86 (DINOv2 B->g) | 1-16 labelled images per class | seconds-minutes (adapter fit on cached features) | backbone-bound; ~5-30 ms/image GPU | varies (DINOv2 Apache-2.0; check others) |
| SVM (RBF) on hand-crafted features | Classical ML on features | — | — | feature vector (no image at inference beyond features) | none | n/a | ~200-1,000 labelled images | <1 s CPU | 0.04 ms/image (clf only) | scikit-learn (BSD-3) |
| Polarisation imaging (glass vs transparent plastic) | Heuristic / physics-first (non-RGB sensor) | — | — | full frame | none | n/a | small (calibration) or a few hundred for a learned head | minutes of CPU | high (camera-limited) | hardware-dependent (commercial polarisation cameras) |
| k-NN / Gaussian mixture / Mahalanobis on features | Classical ML on features | — | — | feature vector | none | n/a | 50-500 labelled samples | milliseconds | 0.1-1 ms/image | scikit-learn (BSD-3) |
| ResNet-18 (transfer learning) | CNN transfer learning | 11.7 | 1.8 | 224 | ImageNet-1k | 69.8 | ~200-2,000 labelled crops | GPU: ~5 min (2-class, 20 epochs) | ~2-6 ms/image on a modern CPU, <1 ms on a mid GPU (batch 1) | BSD-3 (torchvision/timm) |
| EfficientNetV2-S / -M | CNN transfer learning | 21.5 | 8.4 | 384 (S) | ImageNet-1k/-21k | 83.9-84.2 | ~1,000-10,000 labelled crops | GPU: ~1-3 h (full fine-tune) | ~10-30 ms/image CPU | Apache-2.0 |
| Attention-augmented CNN (SE / CBAM / attention-AlexNet) and multi-scale fusion | Hybrid / attention | ~25-60 (backbone + modules) | ~5-15 | 224-384 | ImageNet-1k | backbone-dependent (no clean public number) | ~500-5,000 labelled crops | GPU: ~20-60 min | ~10-30 ms/image CPU | varies (paper code / timplike reimplementations) |
| DeiT-Tiny / DeiT-Small (distilled ViT) | Transformer (data-efficient) | 5.7 / 22.1 | 1.3 / 4.6 | 224 | ImageNet-1k (distilled) | 72.2 / 79.8 | ~1,000-10,000 labelled crops | GPU: ~30-90 min | ~8-30 ms/image CPU | Apache-2.0 |
| Custom sensor fusion (RGB+HSI, RGB+polarisation, RGB+thermal, RGB+depth/3D) | Multisensor fusion | backbone-dependent | sum of branches | aligned streams | ImageNet branches + sensor-specific encoders | n/a | a few hundred-aligned-object samples | GPU: hours | limited by the slowest sensor + alignment | components vary; alignment code often proprietary |
| RegNetY-3.2GF / InceptionNeXt-T | CNN transfer learning | 19.4 / 28.0 | 3.2 / 4.2 | 224 | ImageNet-1k | 82.0 / 82.3 | ~500-5,000 labelled crops | GPU: ~15-45 min | ~10-20 ms/image CPU | Apache-2.0 / MIT |
| CLIP / OpenCLIP / SigLIP — zero-shot classification | Foundation model / VLM | 151 (CLIP ViT-B/32) / 428 (ViT-L/14) / 878+ (SigLIP SO400M) | ~40 (L/14 at 336) | 224-336 | 400M-5B image-text pairs (LAION/WebLI) | 76.0 (B/32 zero-shot) / ~80 (L/14) | 0 labelled images (few-shot: 1-16 per class helps) | 0 for zero-shot; minutes for a linear head on cached features | 3.8 ms/image reported for OpenCLIP ViT-L/14 (~263 FPS, batched GPU); 2.83 FPS for ViT-L/14-336 in a CPU-limited study | MIT (CLIP/OpenCLIP code) - check each checkpoint's data terms |
| ResNet-50 / ResNeXt-50 32x4d (transfer learning) | CNN transfer learning | 25.6 | 4.1 | 224 | ImageNet-1k | 76.1 (classic recipe) / 81.8 (RA4-E3600 recipe @288) | ~500-5,000 labelled crops | GPU: ~10-20 min | ~5-20 ms/image CPU, ~1-3 ms/image GPU | BSD-3 / Apache-2.0 |
| Inception-v3 / Xception / InceptionResNetV2 | CNN transfer learning | 27.2 / 22.9 / 55.8 | 5.7 / 8.4 / 13.2 | 299 | ImageNet-1k | 77.3 / 79.0 / 80.3 | ~500-5,000 labelled crops | GPU: ~15-45 min | ~8-40 ms/image CPU | Apache-2.0 / BSD-3 |
| ViT-B/16 (supervised fine-tune) | Transformer | 86.0 | 17.6 | 224 (384 for best recipes) | ImageNet-1k or -21k | 79.9 (IN-1k from scratch; needs heavy aug) / 84.9 (IN-21k AugReg) | ~5,000-100,000 labelled crops (or 21k pretraining) | GPU: hours (fine-tune), much more from scratch | ~30-100 ms/image CPU; 3-8 ms/image GPU | Apache-2.0 (timm) |
| Purpose-built lightweight CNNs (RecycleNet, Focus-RCNet-KD, WasNet) | CNN from scratch (waste-specific designs) | 0.525 | 0.06 | 224 | none / distillation | n/a | ~2,000-10,000 images | hours CPU / minutes GPU | 1-3 ms/image CPU | mixed (paper code, check repo licences) |
| Thermal IR and 3D/structured-light sensing for transparent objects | Non-RGB sensing | small | small | full frame | ImageNet or none | n/a | hundreds of objects | GPU: minutes | sensor-limited | vendor-dependent |
| Shallow MLP on hand-crafted features | Classical ML on features | 0.01 | — | feature vector | none | n/a | ~1,000+ labelled images | 1-60 s CPU | 0.1 ms/image | scikit-learn / PyTorch (BSD/MIT) |
| Transmission / haze index (background-continuation) | Heuristic / physics-first | — | — | full frame | none | n/a | 0 | 0 | very high | n/a (own code) |
| Edge + texture descriptors (Canny/Sobel, LBP, GLCM) | Heuristic / physics-first | — | — | full frame | none | n/a | 0 labelled images (thresholds) or ~100 for calibration | seconds of CPU | high (1-10 ms/frame CPU) | n/a (OpenCV/skimage) |
| Colour/intensity thresholding (industrial VIS sorter) | Heuristic / physics-first | — | — | full frame | none | n/a | 0 labelled images (hand-tuned) | 0 (manual tuning) | very high (>10k fps on FPGA/line-scan) | n/a (implement in-house; see vendor toolkits) |
| Small CNN trained from scratch (4-8 conv layers) | CNN from scratch | 0.5 | 0.05 | 96-224 | none | n/a | ~5,000+ images (or heavy augmentation) | CPU hours / GPU minutes | 1-5 ms/image CPU | own code (MIT) |
| Multimodal LLM as a classifier (GPT-4o / LLaVA-OneVision) | Foundation model / VLM (language interface) | proprietary / 7B+ | very high | variable | web-scale multimodal | n/a | 0-5 labelled images | none (prompting) or fine-tune at high cost | <1-5 FPS typical; API latency seconds | proprietary API / research licences |
| VGG-16 / VGG-19 (transfer learning) | CNN transfer learning | 138.4 | 15.5 | 224 | ImageNet-1k | 71.6 | ~1,000-10,000 labelled crops | GPU: ~30-60 min | >20 ms/image CPU (slow) | CC BY 4.0 (weights) / own impl. |

## 4. Family-by-family detail: evidence, pros, cons, glass/plastic specifics


### Heuristic / physics-first


#### Colour/intensity thresholding (industrial VIS sorter)  —  composite 3.04/5 (profile: Balanced commercial deployment)


* **Cost:** —M params, —G MACs, input full frame, pretraining none, licence n/a (implement in-house; see vendor toolkits)

* **Evidence on real waste / glass-plastic data:** Deployed in real plants: PICVISA VIS + neural networks classify glass by colour and shape for colour sorting [R31,R23]. Works where the *material stream is already separated* and only colour/tone classes remain (brown/white/green cullet) [R21].

* **Glass/plastic strengths:** Extremely cheap, deterministic, audit-friendly; handles glass colour classes (amber/green/white) reliably when the stream is clean.

* **Glass/plastic weaknesses:** Cannot separate clear glass from clear plastic at all - identical achromatic appearance; dies on mixed-colour streams, dirt and labels.

* **Pros:** Zero data, zero training, sub-millisecond, certifiable logic; the right baseline to beat.

* **Cons:** No notion of material; brittle to illumination changes; every new stream needs re-tuning.

* **Use it when:** Colour sorting of an already-material-pure glass stream; as an interpretable baseline in any study.

* **Scores (1–5):** accuracy_ceiling 2, transparency_robustness 1, data_efficiency 5, contamination_robustness 1, cost_efficiency 5, throughput 5, explainability 5, tooling_maturity 5, calibration_abstention 1, zero_shot_capability 5  | tier: baseline


#### Edge + texture descriptors (Canny/Sobel, LBP, GLCM)  —  composite 3.05/5 (profile: Balanced commercial deployment)


* **Cost:** —M params, —G MACs, input full frame, pretraining none, licence n/a (OpenCV/skimage)

* **Evidence on real waste / glass-plastic data:** The classical glass-CV toolkit: Canny/Sobel contours plus texture analysis differentiate glass from other materials by surface characteristics [R31]. Measured in this repo: texture-only features reach 0.94 balanced accuracy in the proxy task.

* **Glass/plastic strengths:** Cheap proxy for surface finish: glass is smooth and shows rim gradients; plastic shows mould ribs, grain and film wrinkles.

* **Glass/plastic weaknesses:** Global descriptors: any bright/deformed item elsewhere in the frame corrupts them (measured here: Canny density rises 6.4x under background clutter and accuracy collapses to chance).

* **Pros:** Interpretable, dependency-free, runs on a microcontroller; good cue engineering tool.

* **Cons:** No spatial selectivity -> needs a detector/segmenter in front; threshold sensitivity.

* **Use it when:** Embedded pre-screening, first-pass prototyping, or when the object is guaranteed isolated in the frame.

* **Scores (1–5):** accuracy_ceiling 2, transparency_robustness 2, data_efficiency 4, contamination_robustness 1, cost_efficiency 5, throughput 5, explainability 4, tooling_maturity 5, calibration_abstention 1, zero_shot_capability 5  | tier: baseline


#### Specular-highlight statistics (1-D rule)  —  composite 3.69/5 (profile: Balanced commercial deployment)


* **Cost:** —M params, —G MACs, input full frame, pretraining none, licence n/a (own code)

* **Evidence on real waste / glass-plastic data:** Measured in this repo on the synthetic proxy: a single threshold on highlight spikiness gives 0.70 balanced accuracy - i.e. the physics prior explains a large part of the signal, and gives a floor any learned model must beat.

* **Glass/plastic strengths:** Encodes the most cited physical difference: glass returns few small near-saturated highlights, plastic returns broad soft sheen (lower refractive index contrast + diffuse body).

* **Glass/plastic weaknesses:** Confounded by illumination: a single point light on glossy PET mimics glass; a diffuse light dome erases the cue for both.

* **Pros:** One threshold, explainable to a line operator, no training, trivially auditable.

* **Cons:** Only meaningful under controlled lighting; colourless text/labels create false highlights.

* **Use it when:** Lighting-controlled capture cell as an interpretable baseline / fallback rule.

* **Scores (1–5):** accuracy_ceiling 3, transparency_robustness 3, data_efficiency 5, contamination_robustness 2, cost_efficiency 5, throughput 5, explainability 5, tooling_maturity 4, calibration_abstention 2, zero_shot_capability 5  | tier: baseline


#### Transmission / haze index (background-continuation)  —  composite 3.14/5 (profile: Balanced commercial deployment)


* **Cost:** —M params, —G MACs, input full frame, pretraining none, licence n/a (own code)

* **Evidence on real waste / glass-plastic data:** The 'does the scene continue through the object' test underlies classical transparent-object work; measured here at 0.58 balanced accuracy alone, contributing to a 0.94 ensemble. Related physics led to polarimetry, which is exactly this idea with a polariser [R31].

* **Glass/plastic strengths:** Directly measures transmission vs diffuse attenuation, the one cue that is material-specific rather than shape- or lighting-specific.

* **Glass/plastic weaknesses:** Requires textured background behind the object (measured here: it is the cue a featureless white studio backdrop removes - with whitebg shift the composite model still held 0.95, showing the model is not actually relying on it).

* **Pros:** Physically grounded, cheap; tells you when your imaging station is uninformative.

* **Cons:** Useless on a blank backdrop; needs a foreground estimate to be computed at all.

* **Use it when:** Conveyor/table capture with visible texture behind items; design input for the capture cell.

* **Scores (1–5):** accuracy_ceiling 2, transparency_robustness 3, data_efficiency 4, contamination_robustness 1, cost_efficiency 5, throughput 5, explainability 4, tooling_maturity 3, calibration_abstention 2, zero_shot_capability 5  | tier: baseline


### Heuristic / physics-first (non-RGB sensor)


#### Polarisation imaging (glass vs transparent plastic)  —  composite 3.59/5 (profile: Balanced commercial deployment)


* **Cost:** —M params, —G MACs, input full frame, pretraining none, licence hardware-dependent (commercial polarisation cameras)

* **Evidence on real waste / glass-plastic data:** Polarimetry 'leverages the polarization properties of light reflected and transmitted by glass', exploiting the dielectric nature of glass to alter polarisation state, and 'has proven useful, for example, in differentiating glass from transparent plastics' [R31]. Still thin in the literature - an open research gap.

* **Glass/plastic strengths:** Attacks the exact failure case of RGB (clear glass vs clear PET/PP) with a physical property rather than appearance.

* **Glass/plastic weaknesses:** Sensitive to surface orientation, stress birefringence in moulded plastics and to dirt; adds a sensor and calibration burden.

* **Pros:** Highest physics-per-euro for the transparent/transparent case; passive and fast.

* **Cons:** Immature tooling, few public datasets, needs fusion logic with RGB for shape/colour.

* **Use it when:** When the stream is dominated by transparent items and RGB has plateaued; a strong candidate for a research contribution.

* **Scores (1–5):** accuracy_ceiling 4, transparency_robustness 5, data_efficiency 4, contamination_robustness 2, cost_efficiency 3, throughput 4, explainability 3, tooling_maturity 2, calibration_abstention 3, zero_shot_capability 4  | tier: research


### Classical ML on features


#### SVM (RBF) on hand-crafted features  —  composite 3.63/5 (profile: Balanced commercial deployment)


* **Cost:** —M params, —G MACs, input feature vector (no image at inference beyond features), pretraining none, licence scikit-learn (BSD-3)

* **Evidence on real waste / glass-plastic data:** Measured here: 0.933 balanced accuracy on the proxy task with 142 features, 0.89-0.98 bootstrap CI, but collapses to chance under sensor noise and clutter. In spectroscopy, SVM/RF are the most stable choice for small samples while deep models win on large samples [R25]; RF/SVM with grid-search/Bayesian tuning are the standard strong baselines for glass CV [R31].

* **Glass/plastic strengths:** Works well with engineered transparency/specular features; strong in the small-data regime typical of sorting plants.

* **Glass/plastic weaknesses:** Depends entirely on the feature engineering; not spatially selective; RBF kernels are acutely sensitive to input distribution shift (measured here: 0.50 on noise/jpeg/deep-shadow shifts where tree ensembles retained 0.82-0.92).

* **Pros:** Seconds to train, kilobytes to store, fully inspectable decision surface; excellent data-efficiency; auditable for certification.

* **Cons:** Feature engineering is where the accuracy lives; scaling assumptions break under shift; no notion of shape or context.

* **Use it when:** Small labelled sets, embedded deployment, or when you must justify every decision to an auditor.

* **Scores (1–5):** accuracy_ceiling 3, transparency_robustness 3, data_efficiency 5, contamination_robustness 2, cost_efficiency 5, throughput 5, explainability 4, tooling_maturity 5, calibration_abstention 3, zero_shot_capability 1  | tier: production (small-data)


#### Random Forest / ExtraTrees on hand-crafted features  —  composite 3.73/5 (profile: Balanced commercial deployment)


* **Cost:** —M params, —G MACs, input feature vector, pretraining none, licence scikit-learn (BSD-3)

* **Evidence on real waste / glass-plastic data:** Measured here: 0.908-0.933 balanced accuracy, and by far the most robust tier under acquisition shift (0.82-0.92 under noise/jpeg/dark, vs 0.50 for the RBF SVM). The hyperspectral review finds ResNet-50 best among deep classifiers and RF/SVM strong among conventional ones [R22].

* **Glass/plastic strengths:** Handles the mixed, non-Gaussian statistics of specular/texture features; supplies feature importances, which is how you discover which cue the model is actually using.

* **Glass/plastic weaknesses:** Same blindness to spatial context as any global-descriptor model; large ensembles cost memory on MCUs.

* **Pros:** Robust, fast, interpretable, no scaling assumptions, trains on a laptop in under a second.

* **Cons:** Cannot exceed its features; importances are biased toward high-cardinality features.

* **Use it when:** Default workhorse for a classical pipeline; the model to beat before reaching for a CNN.

* **Scores (1–5):** accuracy_ceiling 4, transparency_robustness 3, data_efficiency 5, contamination_robustness 2, cost_efficiency 5, throughput 4, explainability 4, tooling_maturity 5, calibration_abstention 3, zero_shot_capability 1  | tier: production (small-data)


#### LDA / QDA on hand-crafted features  —  composite 3.75/5 (profile: Balanced commercial deployment)


* **Cost:** —M params, —G MACs, input feature vector or spectrum, pretraining none, licence scikit-learn (BSD-3)

* **Evidence on real waste / glass-plastic data:** The linear reference model for engineered features (measured here: 0.922 balanced accuracy, second only to boosting in the classical tier). Its *spectral* counterpart - PLS-DA on NIR/HSI/FTIR spectra - is the industrial workhorse and is covered under the two spectral entries in this registry, where it reaches sensitivity/specificity 0.910-1.000 for glass colour classes [R21] and F1 0.517-0.971 on plastic spectra [R27].

* **Glass/plastic strengths:** Optimal when the classes are linearly separable in the chosen feature space; gives signed class scores that an operator or auditor can inspect, and needs very few samples to stabilise.

* **Glass/plastic weaknesses:** A single linear hyperplane in feature space cannot represent 'bright AND elongated AND smooth' interactions; sensitive to feature scaling and outliers in specular statistics.

* **Pros:** Standard, explainable, tiny, extremely fast; the score itself is a linear audit trail.

* **Cons:** Underfits interaction structure (boosting beats it here by 2 points); no spatial context.

* **Use it when:** As the linear reference for a feature-based model; use the spectral entries for the PLS-DA-on-spectra pipeline.

* **Scores (1–5):** accuracy_ceiling 3, transparency_robustness 3, data_efficiency 5, contamination_robustness 2, cost_efficiency 5, throughput 5, explainability 5, tooling_maturity 5, calibration_abstention 4, zero_shot_capability 1  | tier: baseline / production (linear reference)


#### Gradient-boosted trees (HistGradientBoosting/XGBoost/LightGBM)  —  composite 3.67/5 (profile: Balanced commercial deployment)


* **Cost:** —M params, —G MACs, input feature vector, pretraining none, licence BSD/MIT (sklearn/LightGBM/XGBoost)

* **Evidence on real waste / glass-plastic data:** Measured here: best shallow model overall (0.944 balanced accuracy, ROC AUC 0.986) and strong under dark/noise/jpeg shifts (0.82-0.93) - but still collapses under background clutter, like every global-descriptor model.

* **Glass/plastic strengths:** Captures interactions between cues (e.g. 'bright AND large highlight AND low texture' -> plastic), which matters because no single cue is reliable for glass vs plastic.

* **Glass/plastic weaknesses:** Needs more data than LDA/linear models; less interpretable than a single rule; still blind to spatial context.

* **Pros:** Best accuracy-per-second in the classical tier; handles mixed feature types and missing values; feature importances + SHAP for audits.

* **Cons:** Hyper-parameter sensitive; CPU-bound training on huge feature sets; black-box-ish.

* **Use it when:** Tabular features of any sensor modality; when a CNN cannot be justified by data volume or latency budget.

* **Scores (1–5):** accuracy_ceiling 4, transparency_robustness 3, data_efficiency 4, contamination_robustness 2, cost_efficiency 5, throughput 5, explainability 3, tooling_maturity 5, calibration_abstention 4, zero_shot_capability 1  | tier: production (small-mid data)


#### k-NN / Gaussian mixture / Mahalanobis on features  —  composite 3.55/5 (profile: Balanced commercial deployment)


* **Cost:** —M params, —G MACs, input feature vector, pretraining none, licence scikit-learn (BSD-3)

* **Evidence on real waste / glass-plastic data:** Measured here: k-NN 0.90 balanced accuracy. In industrial hyperspectral work, similarity/nearest-neighbour heads over frozen embeddings are competitive when many exemplars are available, and DINOv3 1-NN is the strongest training-free baseline in the waste-VLM study [R30].

* **Glass/plastic strengths:** Non-parametric, so it tracks multimodal appearance (many bottle/ jar sub-types) better than a single linear boundary; GMM gives an explicit 'novelty' score.

* **Glass/plastic weaknesses:** Degrades with dimensionality and with class imbalance; needs a feature space where distance is meaningful (i.e. you must normalise well).

* **Pros:** Zero training, instantly updatable (add exemplars for a new product), supports OOD detection via likelihood.

* **Cons:** Inference grows with the exemplar set; sensitive to uninformative features.

* **Use it when:** Frequently changing product mix where re-training is impractical; OOD/reject logic.

* **Scores (1–5):** accuracy_ceiling 3, transparency_robustness 3, data_efficiency 5, contamination_robustness 2, cost_efficiency 5, throughput 4, explainability 4, tooling_maturity 4, calibration_abstention 4, zero_shot_capability 2  | tier: production (small-data)


#### Shallow MLP on hand-crafted features  —  composite 3.18/5 (profile: Balanced commercial deployment)


* **Cost:** 0.01M params, —G MACs, input feature vector, pretraining none, licence scikit-learn / PyTorch (BSD/MIT)

* **Evidence on real waste / glass-plastic data:** Measured here: 0.90 - no better than SVM/trees at this sample size, with a larger variance. Consistent with the classical-vs-deep result that deep models need sample scale before they pay off [R25,R18].

* **Glass/plastic strengths:** Can learn mild non-linear interactions of features; trivially portable (a few hundred weights).

* **Glass/plastic weaknesses:** Data-hungry relative to its capacity, no accuracy advantage over trees/SVM on the same features, extra tuning surface.

* **Pros:** Small, fast, familiar; useful stepping stone to a CNN pipeline.

* **Cons:** Usually strictly dominated by boosting on tabular features - include it in a benchmark mainly to document that fact.

* **Use it when:** Research comparison / teaching; middle ground between linear models and CNNs.

* **Scores (1–5):** accuracy_ceiling 3, transparency_robustness 3, data_efficiency 3, contamination_robustness 2, cost_efficiency 5, throughput 5, explainability 2, tooling_maturity 4, calibration_abstention 3, zero_shot_capability 1  | tier: research


### CNN from scratch


#### Small CNN trained from scratch (4-8 conv layers)  —  composite 3.00/5 (profile: Balanced commercial deployment)


* **Cost:** 0.5M params, 0.05G MACs, input 96-224, pretraining none, licence own code (MIT)

* **Evidence on real waste / glass-plastic data:** A 6-layer baseline CNN reached 90.61% on a 2-class waste set, versus 95.51% (MobileNetV2) and 96.00% (VGG16) with transfer learning in the same study - the canonical 'transfer learning beats from-scratch at small n' result.

* **Glass/plastic strengths:** Can learn local texture/gradient cues in a spatially selective way, which global descriptors cannot.

* **Glass/plastic weaknesses:** Needs far more data than a fine-tuned backbone; learns dataset-specific shortcuts (background, lighting) easily, especially on small sets.

* **Pros:** Full architectural control and total transparency; no licence or download constraints; cheap inference.

* **Cons:** Poor sample efficiency; usually loses to a fine-tuned pretrained model of similar size; more hyper-parameter work.

* **Use it when:** Huge in-domain datasets, on-device constraints with no pretrained weights available, or when you must own the whole stack.

* **Scores (1–5):** accuracy_ceiling 3, transparency_robustness 3, data_efficiency 2, contamination_robustness 3, cost_efficiency 4, throughput 4, explainability 3, tooling_maturity 3, calibration_abstention 3, zero_shot_capability 1  | tier: baseline / research


### CNN from scratch (waste-specific designs)


#### Purpose-built lightweight CNNs (RecycleNet, Focus-RCNet-KD, WasNet)  —  composite 3.32/5 (profile: Balanced commercial deployment)


* **Cost:** 0.525M params, 0.06G MACs, input 224, pretraining none / distillation, licence mixed (paper code, check repo licences)

* **Evidence on real waste / glass-plastic data:** RecycleNet: 3M params, 81% TrashNet [R10]. Focus-RCNet-KD: 0.525M params, 92% TrashNet [R9]. WasNet: 96.10% TrashNet, 82.5% Huawei, 64.5% ImageNet [R11].

* **Glass/plastic strengths:** Knowledge distillation into a sub-million-parameter model is the documented route to on-belt inference with acceptable accuracy.

* **Glass/plastic weaknesses:** All the usual small-model caveats, plus: the residual errors concentrate exactly where the physics is hard (transparent, deformed, contaminated items).

* **Pros:** Mobile-grade cost with respectable accuracy; designed with waste taxonomies in mind.

* **Cons:** Non-standard, harder to maintain than a timm/torchvision model; benchmark numbers mostly come from TrashNet, which is saturated and studio-lit.

* **Use it when:** Retrofitting legacy sorting lines with CPU-only inference.

* **Scores (1–5):** accuracy_ceiling 4, transparency_robustness 3, data_efficiency 3, contamination_robustness 3, cost_efficiency 5, throughput 5, explainability 2, tooling_maturity 2, calibration_abstention 3, zero_shot_capability 1  | tier: production (edge)


### CNN transfer learning


#### ResNet-18 (transfer learning)  —  composite 3.52/5 (profile: Balanced commercial deployment)


* **Cost:** 11.7M params, 1.8G MACs, input 224, pretraining ImageNet-1k, licence BSD-3 (torchvision/timm)

* **Evidence on real waste / glass-plastic data:** The default reference in waste studies; ResNet-34 transfer learning reached 96% on web-scraped plastics when the classes were balanced but 71.8% on a harder, imbalanced set (F1 71.1%) [R37]; ResNet-50 was the best of four conventional waste classifiers in the hyperspectral review [R22].

* **Glass/plastic strengths:** Cheap to fine-tune, learns texture/gradient cues that survive mild colour shifts; robust enough for a first real-data benchmark.

* **Glass/plastic weaknesses:** Residual blocks are good at texture but not at reasoning about specularity; will happily exploit background/lighting shortcuts, which is why a background-only control is mandatory.

* **Pros:** Small, well-supported, fast to train, easy to export (ONNX/TorchScript), abundant pre-trained recipes.

* **Cons:** Lowest ImageNet accuracy of the modern CNNs in this table; limited capacity for fine-grained material cues.

* **Use it when:** First transfer-learning baseline on your own glass/plastic data; always report it alongside a heavier backbone.

* **Scores (1–5):** accuracy_ceiling 4, transparency_robustness 3, data_efficiency 4, contamination_robustness 3, cost_efficiency 4, throughput 4, explainability 2, tooling_maturity 5, calibration_abstention 3, zero_shot_capability 2  | tier: baseline


#### ResNet-50 / ResNeXt-50 32x4d (transfer learning)  —  composite 3.35/5 (profile: Balanced commercial deployment)


* **Cost:** 25.6M params, 4.1G MACs, input 224, pretraining ImageNet-1k, licence BSD-3 / Apache-2.0

* **Evidence on real waste / glass-plastic data:** ResNet-50 was the strongest of four conventional waste classifiers (>90%) in the hyperspectral plastic review [R22]; ResNeXt-101 is the backbone used for the TrashBox federated study [R33]; the 12-backbone comparison puts ResNet-50 at 76.13% versus ConvNeXt-Tiny 82.52% at similar size [R38].

* **Glass/plastic strengths:** Strong feature extractor for a modest cost; ResNeXt grouping adds capacity at iso-FLOPs; well-understood failure modes.

* **Glass/plastic weaknesses:** Same image-level shortcut risk as any classifier head; no built-in attention to the object vs the belt.

* **Pros:** The de-facto standard for reproducibility; huge prior literature for comparison; trivially quantisable (INT8) for edge.

* **Cons:** Dominated on accuracy-per-FLOP by ConvNeXt/RegNet/EfficientNet at the same budget [R38].

* **Use it when:** When you need comparability with published waste papers; as the second baseline after ResNet-18.

* **Scores (1–5):** accuracy_ceiling 4, transparency_robustness 3, data_efficiency 4, contamination_robustness 3, cost_efficiency 3, throughput 3, explainability 2, tooling_maturity 5, calibration_abstention 3, zero_shot_capability 2  | tier: baseline / production


#### DenseNet-121 / -169 (transfer learning)  —  composite 3.82/5 (profile: Balanced commercial deployment)


* **Cost:** 8.0M params, 2.9G MACs, input 224, pretraining ImageNet-1k, licence BSD-3

* **Evidence on real waste / glass-plastic data:** The most consistent winner on waste imagery in the literature: ~95% TrashNet [R8], 91% under a stricter split [R7], and 99.60% with hyper-parameter optimisation [R12]; also used in the HSI plastic review [R22].

* **Glass/plastic strengths:** Feature reuse across scales suits the multi-scale nature of material cues (rim highlight vs body texture vs label); small parameter count for its accuracy.

* **Glass/plastic weaknesses:** Memory-hungry activations at high resolution; like all of these, TrashNet numbers are inflated by a studio-white background.

* **Pros:** Best documented accuracy/parameter ratio for waste classification; widely available.

* **Cons:** Slower than MobileNet-class models; concatenation memory limits batch size.

* **Use it when:** Default strong CNN baseline for a static-image glass/plastic benchmark.

* **Scores (1–5):** accuracy_ceiling 5, transparency_robustness 4, data_efficiency 4, contamination_robustness 3, cost_efficiency 4, throughput 3, explainability 2, tooling_maturity 5, calibration_abstention 3, zero_shot_capability 2  | tier: baseline / production


#### VGG-16 / VGG-19 (transfer learning)  —  composite 2.47/5 (profile: Balanced commercial deployment)


* **Cost:** 138.4M params, 15.5G MACs, input 224, pretraining ImageNet-1k, licence CC BY 4.0 (weights) / own impl.

* **Evidence on real waste / glass-plastic data:** VGG16 reached 96.00% on a 2-class waste set, beating MobileNetV2 (95.51%) and a 6-layer CNN (90.61%) in the same study [R45]; RealWaste and TrashNet comparisons routinely include VGG-16 [R2,R15].

* **Glass/plastic strengths:** Fine texture modelling via stacked 3x3 kernels - useful for mould lines, grain and surface finish.

* **Glass/plastic weaknesses:** Enormous for the accuracy delivered; the fully-connected head invites overfitting on small waste datasets.

* **Pros:** Simple, still occasionally top of a small held-out split; many worked examples.

* **Cons:** 93x the parameters of MobileNetV2 for ~4 points of ImageNet accuracy; poor latency; obsolete for deployment.

* **Use it when:** Reproducing older published baselines; not recommended for new deployments.

* **Scores (1–5):** accuracy_ceiling 3, transparency_robustness 3, data_efficiency 2, contamination_robustness 3, cost_efficiency 1, throughput 1, explainability 2, tooling_maturity 4, calibration_abstention 2, zero_shot_capability 2  | tier: legacy baseline


#### Inception-v3 / Xception / InceptionResNetV2  —  composite 3.35/5 (profile: Balanced commercial deployment)


* **Cost:** 27.2 / 22.9 / 55.8M params, 5.7 / 8.4 / 13.2G MACs, input 299, pretraining ImageNet-1k, licence Apache-2.0 / BSD-3

* **Evidence on real waste / glass-plastic data:** A beluga-whale-optimised InceptionV3 reached 92.62% in 0.63 s on TrashNet, beating MobileNetV2, VGG16 and AlexNet under one protocol [R15]; InceptionResNetV2 and Inception-v3 are standard entries in TrashNet/RealWaste comparisons [R2,R8].

* **Glass/plastic strengths:** Multi-scale receptive fields (Inception) match material cues that span rim-to-texture scales; depthwise separable convolutions (Xception) give a better accuracy/latency point.

* **Glass/plastic weaknesses:** 299x299 input costs latency; separable-convolution models are more sensitive to low-light noise in our proxy sweep.

* **Pros:** Mature, well-supported, strong accuracy for a moderate FLOP budget.

* **Cons:** Superseded at equal cost by ConvNeXt/EfficientNetV2; more preprocessing quirks.

* **Use it when:** Reproducing published waste benchmarks; otherwise prefer ConvNeXt-T/EffNetV2-S.

* **Scores (1–5):** accuracy_ceiling 4, transparency_robustness 4, data_efficiency 3, contamination_robustness 3, cost_efficiency 3, throughput 3, explainability 2, tooling_maturity 4, calibration_abstention 3, zero_shot_capability 2  | tier: baseline / production


#### EfficientNet-B0 / -B3  —  composite 3.72/5 (profile: Balanced commercial deployment)


* **Cost:** 5.3 / 12.2M params, 0.39 / 1.8G MACs, input 224 / 300, pretraining ImageNet-1k, licence Apache-2.0

* **Evidence on real waste / glass-plastic data:** EfficientNetB2 with an attention module reached 93.38% on a 4-class garbage set; EfficientNetB0 improved from 83.95% to 84.85% overall on glass-defect detection when synthetic minority data was added, with recall 0.35->0.65 [R36]; MobileNetV3/EfficientNet were joint top on microplastic images [R17].

* **Glass/plastic strengths:** Compound scaling gives a clean accuracy/cost dial (B0->B3) without changing the code; MBConv blocks respond well to fine texture.

* **Glass/plastic weaknesses:** Sensitive to input resolution - downscaling to save latency can erase the thin-highlight cue; group/depthwise normalisation shows more low-light noise sensitivity in our proxy sweep.

* **Pros:** Excellent accuracy per FLOP, one-line scale change, huge deployment support.

* **Cons:** Superseded by EfficientNetV2 and ConvNeXt at equal cost; recipe-dependent ImageNet numbers [R40].

* **Use it when:** Balanced accuracy/latency target on a mid-range GPU or a strong CPU.

* **Scores (1–5):** accuracy_ceiling 4, transparency_robustness 4, data_efficiency 4, contamination_robustness 3, cost_efficiency 4, throughput 4, explainability 2, tooling_maturity 5, calibration_abstention 3, zero_shot_capability 2  | tier: production


#### EfficientNetV2-S / -M  —  composite 3.52/5 (profile: Balanced commercial deployment)


* **Cost:** 21.5M params, 8.4G MACs, input 384 (S), pretraining ImageNet-1k/-21k, licence Apache-2.0

* **Evidence on real waste / glass-plastic data:** EfficientNetV2S was identified as the most sustainable/accurate option (96.41%) in a garbage-classification comparison that also accounted for carbon footprint [R16]; ranked in the top-3 backbones across six domains in a controlled backbone study [R38].

* **Glass/plastic strengths:** High effective resolution (384) preserves rim highlights and small labels; progressive learning recipe trains fast on small sets.

* **Glass/plastic weaknesses:** Much more compute than B0 for a few points; fine-tuning instability on very small datasets without a low LR.

* **Pros:** One of the best accuracy-per-FLOP models available; strong published waste evidence.

* **Cons:** Not the most efficient at very low latency; larger memory footprint.

* **Use it when:** Server/GPU-side classifier or the accuracy champion in a benchmark table.

* **Scores (1–5):** accuracy_ceiling 5, transparency_robustness 4, data_efficiency 3, contamination_robustness 3, cost_efficiency 3, throughput 2, explainability 2, tooling_maturity 5, calibration_abstention 3, zero_shot_capability 2  | tier: research / production (server)


#### RegNetY-3.2GF / InceptionNeXt-T  —  composite 3.47/5 (profile: Balanced commercial deployment)


* **Cost:** 19.4 / 28.0M params, 3.2 / 4.2G MACs, input 224, pretraining ImageNet-1k, licence Apache-2.0 / MIT

* **Evidence on real waste / glass-plastic data:** RegNetY-3.2GF ranked among the top backbones overall and specifically strong on plant/remote-sensing domains [R38]; InceptionNeXt-T reported 82.3% at 4.2 GFLOPs with 901 img/s training throughput [R38-adjacent benchmark].

* **Glass/plastic strengths:** Regular, hardware-friendly design: high training throughput per FLOP lets you run more augmentations/seeds within a fixed budget - which usually buys more robustness than a bigger model on a small dataset.

* **Glass/plastic weaknesses:** No architectural feature aimed at specular/transparent cues; same shortcut risks as any ImageNet backbone.

* **Pros:** Best throughput-per-accuracy for repeated experimentation; SE blocks give channel attention cheaply.

* **Cons:** Less waste-domain evidence than DenseNet/ResNet families.

* **Use it when:** Large hyper-parameter/seed sweeps where wall-clock matters.

* **Scores (1–5):** accuracy_ceiling 4, transparency_robustness 4, data_efficiency 3, contamination_robustness 4, cost_efficiency 3, throughput 3, explainability 2, tooling_maturity 4, calibration_abstention 3, zero_shot_capability 2  | tier: research


### CNN transfer learning (mobile)


#### MobileNetV2 / ShuffleNetV2 (1.0x)  —  composite 3.70/5 (profile: Balanced commercial deployment)


* **Cost:** 3.5M params, 0.3G MACs, input 224, pretraining ImageNet-1k, licence Apache-2.0 / BSD-3

* **Evidence on real waste / glass-plastic data:** MobileNetV2 reaches 95.51% on a 2-class waste set, within 0.5 points of VGG16 at 1/40th the parameters; an improved MobileNetV2 reached 90.7% on a 4-class challenge set [R16]; MobileNetV2 is the standard edge choice in waste-classification surveys.

* **Glass/plastic strengths:** Cheapest credible accuracy on a CPU/DSP; depthwise separable convolutions learn local texture cues efficiently.

* **Glass/plastic weaknesses:** A single 224x224 image-level label gives no spatial evidence: one bright belt reflection can dominate a global-pooled feature.

* **Pros:** Runs on an MCU-class NPU; tiny memory; easy quantisation to INT8.

* **Cons:** Several points below ConvNeXt-T/EffNetV2-S accuracy; limited capacity for rare sub-types.

* **Use it when:** On-belt/edge deployment with a strict wattage or cost budget.

* **Scores (1–5):** accuracy_ceiling 4, transparency_robustness 3, data_efficiency 4, contamination_robustness 3, cost_efficiency 5, throughput 5, explainability 2, tooling_maturity 5, calibration_abstention 3, zero_shot_capability 2  | tier: production (edge)


#### MobileNetV3-Large / -Small  —  composite 3.70/5 (profile: Balanced commercial deployment)


* **Cost:** 5.5 / 2.5M params, 0.22 / 0.06G MACs, input 224, pretraining ImageNet-1k, licence Apache-2.0

* **Evidence on real waste / glass-plastic data:** EfficientNet_b7, Inception_v3 and MobileNet_v3 all achieved 98% on augmented microplastic data, with MobileNet_v3 reaching it in the least training time and with the smallest model - the clearest published argument for mobile-grade models in material classification [R17]; the same study notes MobileNet's suitability for smartphone deployment.

* **Glass/plastic strengths:** Squeeze-and-excite blocks give it a light channel-attention mechanism - useful for suppressing a dominant specular highlight channel.

* **Glass/plastic weaknesses:** Compressed capacity can lose rare but decisive cues (thin rim highlights, small resin codes).

* **Pros:** Best accuracy/watt in the mobile tier; tflite/torchscript/ONNX well supported.

* **Cons:** Behind modern mobile CNNs (MobileNetV4, EfficientViT) at equal latency [R39].

* **Use it when:** Real-time belt or handheld app; the pragmatic default for edge glass/plastic.

* **Scores (1–5):** accuracy_ceiling 4, transparency_robustness 3, data_efficiency 4, contamination_robustness 3, cost_efficiency 5, throughput 5, explainability 2, tooling_maturity 5, calibration_abstention 3, zero_shot_capability 2  | tier: production (edge)


### CNN transfer learning (mobile, 2024)


#### MobileNetV4 (Conv-S / Hybrid-L)  —  composite 3.74/5 (profile: Balanced commercial deployment)


* **Cost:** 3.8 (Conv-S) / 32.5 (Hybrid-L)M params, 0.2 (Conv-S)G MACs, input 224, pretraining ImageNet-1k (+JFT for distilled variants), licence Apache-2.0

* **Evidence on real waste / glass-plastic data:** MNv4 is 'mostly Pareto-optimal' across mobile CPUs, GPUs, DSPs and accelerators, and the distilled Hybrid-L reaches 87.0% ImageNet top-1 with 39x fewer MACs than its teacher [R39] - the current best target for on-device inference.

* **Glass/plastic strengths:** Universal Inverted Bottleneck + Mobile MQA give transformer-like mixing at mobile cost - helpful when the decisive cue is a small highlight or a fine mould mark rather than a global colour.

* **Glass/plastic weaknesses:** Very new: few waste-specific published results, and JFT-distilled variants carry unclear provenance for regulated deployment.

* **Pros:** Best known accuracy/latency frontier for edge hardware; INT8-friendly.

* **Cons:** Ecosystem (tflite/timm parity) is less mature than ResNet/MobileNetV2-V3.

* **Use it when:** New edge deployments where you can afford to validate the toolchain.

* **Scores (1–5):** accuracy_ceiling 4, transparency_robustness 4, data_efficiency 4, contamination_robustness 3, cost_efficiency 5, throughput 5, explainability 2, tooling_maturity 3, calibration_abstention 3, zero_shot_capability 2  | tier: production (edge, new)


### CNN transfer learning (modern CNN)


#### ConvNeXt-Tiny / -Small  —  composite 3.64/5 (profile: Balanced commercial deployment)


* **Cost:** 28.6 / 50.0M params, 4.5 / 8.7G MACs, input 224, pretraining ImageNet-1k/-21k, licence MIT (timm) / BSD-3

* **Evidence on real waste / glass-plastic data:** ConvNeXt-Tiny ranked *best overall* across six domains (natural, texture, remote sensing, plant, astronomy, medical) in a controlled 12-backbone comparison, ahead of EfficientNetV2-S and Swin-Tiny [R38] - the strongest architecture-level prior for a new material-classification task.

* **Glass/plastic strengths:** Large-kernel depthwise convolutions + LayerNorm give wide receptive fields, useful for judging surface finish and transmission over a whole object rather than a patch; transformer-like features at CNN runtime cost.

* **Glass/plastic weaknesses:** Still an image-level label: needs a detector or attention to localise the object on a busy belt; ImageNet-1k vs -21k weights differ noticeably in transfer quality.

* **Pros:** Best all-round transfer backbone in independent comparisons; excellent tooling (timm) and quantisation support.

* **Cons:** 2-3x the latency of a MobileNet; more parameters to fine-tune on tiny sets.

* **Use it when:** The default 'strong model' in a new glass/plastic benchmark.

* **Scores (1–5):** accuracy_ceiling 5, transparency_robustness 4, data_efficiency 3, contamination_robustness 4, cost_efficiency 3, throughput 3, explainability 2, tooling_maturity 4, calibration_abstention 3, zero_shot_capability 2  | tier: research / production


### Transformer


#### ViT-B/16 (supervised fine-tune)  —  composite 3.35/5 (profile: Balanced commercial deployment)


* **Cost:** 86.0M params, 17.6G MACs, input 224 (384 for best recipes), pretraining ImageNet-1k or -21k, licence Apache-2.0 (timm)

* **Evidence on real waste / glass-plastic data:** ViT-B/16 and MobileNetV3-Small are the supervised reference models in the zero-shot waste study [R29]; transformers reached F1 0.921-1.00 on plastic spectroscopic data [R27]; the waste-VLM study finds the practical value of ViTs is as frozen feature extractors for few-shot heads [R30].

* **Glass/plastic strengths:** Global self-attention can compare distant image regions - the natural operation for 'is the texture behind the object sharp and continuing?' (the transmission cue).

* **Glass/plastic weaknesses:** Patch-tokenisation at 224 tends to downsample exactly the small, high-frequency cues (thin highlights, rim gradients) that separate glass from plastic; needs strong augmentation on small datasets.

* **Pros:** Best-in-class scaling behaviour when data or 21k pretraining is available; excellent as a frozen backbone.

* **Cons:** Data-hungry, slow, large; poor choice for <2k images without heavy regularisation.

* **Use it when:** You have (or can generate) tens of thousands of in-domain crops, or you use it frozen with a linear/ProtoNet head.

* **Scores (1–5):** accuracy_ceiling 4, transparency_robustness 5, data_efficiency 2, contamination_robustness 3, cost_efficiency 2, throughput 2, explainability 3, tooling_maturity 4, calibration_abstention 3, zero_shot_capability 3  | tier: research


### Transformer (data-efficient)


#### DeiT-Tiny / DeiT-Small (distilled ViT)  —  composite 3.51/5 (profile: Balanced commercial deployment)


* **Cost:** 5.7 / 22.1M params, 1.3 / 4.6G MACs, input 224, pretraining ImageNet-1k (distilled), licence Apache-2.0

* **Evidence on real waste / glass-plastic data:** DeiT is the canonical demonstration that transformer fine-tuning can work on ImageNet-scale data only when distillation + strong augmentation are used; the waste-VLM study's conclusion that pretrained models beat non-pretrained across the board (0.54-0.93 vs 0.41-0.78) [R18] is the same lesson for material classification.

* **Glass/plastic strengths:** Transformer inductive bias with CNN-like parameter count; attention maps are directly inspectable for audit.

* **Glass/plastic weaknesses:** Attention distillation quality varies; the fixed 16x16 patch granularity limits very fine texture reasoning.

* **Pros:** Best parameter/accuracy point among transformers; interpretable attention rollout.

* **Cons:** Typically beaten by ConvNeXt-Tiny at the same budget because it lacks the hierarchical multi-scale structure [R38].

* **Use it when:** When you specifically need attention-based interpretability at manageable cost.

* **Scores (1–5):** accuracy_ceiling 4, transparency_robustness 4, data_efficiency 3, contamination_robustness 3, cost_efficiency 3, throughput 3, explainability 4, tooling_maturity 4, calibration_abstention 3, zero_shot_capability 3  | tier: research


### Hybrid transformer


#### Swin-T/SwinV2-T / MaxViT-T (hierarchical attention)  —  composite 3.64/5 (profile: Balanced commercial deployment)


* **Cost:** 28.3 / 28.4 / 30.9M params, 4.5 / 5.9 / 5.6G MACs, input 224, pretraining ImageNet-1k/-21k, licence MIT / Apache-2.0

* **Evidence on real waste / glass-plastic data:** Swin-Tiny and SwinV2-Tiny are standard comparison entries in cross-domain backbone studies (81.5-82.1% ImageNet at ~28M params) [R38]; windowed/hybrid attention is the architecture family used in most recent HSI material-classification work (RVT/transformers on spectral-spatial data) [R27].

* **Glass/plastic strengths:** Hierarchical windows keep fine local detail while still allowing global comparison - a better inductive bias than plain ViT for mixed local(highlight)+global(transmission) evidence.

* **Glass/plastic weaknesses:** Same data appetite as ViTs; sliding-window attention is latency-unfriendly on fixed-function edge NPUs.

* **Pros:** Typically the best accuracy among ~30M-parameter models together with ConvNeXt-T.

* **Cons:** Edge-unfriendly; more hyper-parameters (window size, drop path) to tune.

* **Use it when:** Accuracy-first server-side deployment where a small CNN has plateaued.

* **Scores (1–5):** accuracy_ceiling 5, transparency_robustness 5, data_efficiency 2, contamination_robustness 4, cost_efficiency 2, throughput 2, explainability 3, tooling_maturity 4, calibration_abstention 3, zero_shot_capability 3  | tier: research / production (server)


### Hybrid / attention


#### Attention-augmented CNN (SE / CBAM / attention-AlexNet) and multi-scale fusion  —  composite 3.52/5 (profile: Balanced commercial deployment)


* **Cost:** ~25-60 (backbone + modules)M params, ~5-15G MACs, input 224-384, pretraining ImageNet-1k, licence varies (paper code / timplike reimplementations)

* **Evidence on real waste / glass-plastic data:** An attention-enhanced AlexNet achieved the highest accuracy of three architectures it was compared against with the fewest confusions (e.g. 2,877 correct plastic-bottle vs 2,727 for plain CNN), explicitly because attention 'minimized misclassifications between visually confusing waste classes'; EfficientNetB2 + PMAM reached 93.38% on a 4-class set; multi-scale RGB+HSI fusion (RHFF-SOLOv1) was required for transparent PET/PP on a black belt [R32].

* **Glass/plastic strengths:** Attention is the natural fix for the core difficulty: learn to weight the rim highlight and the through-object texture instead of the background or a label; channel attention can suppress a saturating specular channel.

* **Glass/plastic weaknesses:** Attention maps are a *soft* fix - they do not guarantee the model ignores a bright belt reflection, and they add parameters to an already data-hungry model.

* **Pros:** Small architectural change with the best published confusion-matrix improvements on visually confusable class pairs (plastic/paper/glass).

* **Cons:** Non-standard code; benefits are dataset-specific; harder to benchmark fairly.

* **Use it when:** After a baseline CNN plateaus on glass/plastic confusions and you have the data to train it.

* **Scores (1–5):** accuracy_ceiling 5, transparency_robustness 4, data_efficiency 3, contamination_robustness 4, cost_efficiency 3, throughput 3, explainability 3, tooling_maturity 2, calibration_abstention 3, zero_shot_capability 1  | tier: research


### Foundation model / VLM


#### CLIP / OpenCLIP / SigLIP — zero-shot classification  —  composite 3.40/5 (profile: Balanced commercial deployment)


* **Cost:** 151 (CLIP ViT-B/32) / 428 (ViT-L/14) / 878+ (SigLIP SO400M)M params, ~40 (L/14 at 336)G MACs, input 224-336, pretraining 400M-5B image-text pairs (LAION/WebLI), licence MIT (CLIP/OpenCLIP code) - check each checkpoint's data terms

* **Evidence on real waste / glass-plastic data:** Zero-shot waste classification on TrashNet: OpenCLIP ViT-L/14-336 reaches 76.30% with no training, versus 82.71% on a 6-class industrial set, improving to 90.48% purely by prompt engineering and 97.18% when fully supervised [R28,R29]; prompt sensitivity is the dominant failure mode [R29].

* **Glass/plastic strengths:** Zero-shot 'glass jar' vs 'plastic bottle' prompts work surprisingly well because the language prior includes material semantics; can be re-targeted to a new taxonomy (e.g. 'amber glass', 'transparent PET') by editing text only - no retraining, no downtime [R30].

* **Glass/plastic weaknesses:** Prompt-dependent and poorly calibrated; fails precisely on visually ambiguous items (clear glass vs clear PET); large models are slow on CPU and their text prior can dominate weak visual evidence.

* **Pros:** No annotation cost, instant taxonomy changes, strong baseline for rare classes, and an excellent feature extractor for few-shot heads.

* **Cons:** Accuracy below a fine-tuned CNN in-domain; big models violate real-time edge budgets; licence/data-provenance questions for regulated settings.

* **Use it when:** Cold start with no labels, taxonomy drift, or bootstrapping annotations for a supervised model.

* **Scores (1–5):** accuracy_ceiling 3, transparency_robustness 3, data_efficiency 5, contamination_robustness 3, cost_efficiency 3, throughput 3, explainability 3, tooling_maturity 5, calibration_abstention 2, zero_shot_capability 5  | tier: production (bootstrap) / research


### Foundation model / VLM (few-shot)


#### Frozen foundation features + training-free adapter (Tip-Adapter / kNN / linear probe)  —  composite 3.64/5 (profile: Balanced commercial deployment)


* **Cost:** 86-1000 (frozen backbone)M params, backbone only, onceG MACs, input 224-518, pretraining DINOv2/DINOv3, EVA-CLIP, SigLIP, licence varies (DINOv2 Apache-2.0; check others)

* **Evidence on real waste / glass-plastic data:** Frozen-foundation-model + training-free adapters with fixed hyper-parameters are the recommended practical route under taxonomy drift; DINOv3 1-NN is a strong baseline when many exemplars exist, and textual few-shot descriptions actually *reduce* MLLM accuracy while image-based few-shot helps [R30].

* **Glass/plastic strengths:** Extremely label-efficient - the industrial reality is that a plant can label 5-20 examples per new product; cached features make re-training an ops action rather than a project.

* **Glass/plastic weaknesses:** Ceiling is set by the frozen representation, which was never trained to reason about specularity/transmission; nearest-neighbour heads inherit the embedding's confusion between shiny plastics and glass.

* **Pros:** Fastest path from 'new stream' to 'working classifier'; no GPU needed at fit time; supports conformal/OOD wrappers.

* **Cons:** Needs a vision foundation model in the inference path (memory/legal), and the representation may not encode the decisive material cue.

* **Use it when:** Rapid deployment, frequent taxonomy change, scarce labels - combined with an abstention policy.

* **Scores (1–5):** accuracy_ceiling 4, transparency_robustness 3, data_efficiency 5, contamination_robustness 3, cost_efficiency 3, throughput 3, explainability 3, tooling_maturity 5, calibration_abstention 4, zero_shot_capability 4  | tier: production (bootstrap)


### Foundation model / VLM (language interface)


#### Multimodal LLM as a classifier (GPT-4o / LLaVA-OneVision)  —  composite 2.92/5 (profile: Balanced commercial deployment)


* **Cost:** proprietary / 7B+M params, very highG MACs, input variable, pretraining web-scale multimodal, licence proprietary API / research licences

* **Evidence on real waste / glass-plastic data:** Zero-shot MLLMs performed well on the tested waste datasets; adding textual few-shot descriptions *reduced* accuracy, while image-based few-shot improved GPT-4o at high inference cost [R30].

* **Glass/plastic strengths:** Can be asked to reason explicitly ('is the background visible and undistorted through the object?') - useful for auditing and for generating labels, and handles long-tail novelty better than a closed classifier.

* **Glass/plastic weaknesses:** Orders of magnitude too slow and expensive for a belt; opaque failure modes; no calibration for a reject/accept decision.

* **Pros:** Zero-shot flexibility, natural-language explanations, excellent for weak labelling and dataset bootstrapping.

* **Cons:** Latency, cost, non-determinism, and no evidence of beating a fine-tuned CNN on the binary glass/plastic decision.

* **Use it when:** Label generation, dataset auditing, hard-case triage, operator assistance - not the production classifier.

* **Scores (1–5):** accuracy_ceiling 3, transparency_robustness 3, data_efficiency 5, contamination_robustness 2, cost_efficiency 1, throughput 1, explainability 4, tooling_maturity 4, calibration_abstention 2, zero_shot_capability 5  | tier: research / tooling


### Foundation model pipeline


#### Open-vocabulary detect-then-classify (Grounding DINO + CLIP, EcoVision)  —  composite 3.65/5 (profile: Balanced commercial deployment)


* **Cost:** ~250+M params, highG MACs, input variable (800-1333), pretraining grounding + CLIP, licence Apache-2.0 components (check Grounding DINO licence)

* **Evidence on real waste / glass-plastic data:** EcoVision combines Grounding DINO detection with CLIP material classification, reports that few-shot LoRA adaptation of the visual encoder improves cluttered field-image accuracy while preserving zero-shot generalisation, and is designed to bolt onto existing mechanical segregation lines [R44].

* **Glass/plastic strengths:** Solves the failure mode we measured directly: a detector removes the background so the material classifier sees only the object - which is what fixes the clutter collapse of global descriptors.

* **Glass/plastic weaknesses:** Two failure surfaces (missed detections, then material confusion); transparent objects are also hard for detectors; slower.

* **Pros:** Composable, retargetable by text prompt, and the natural architecture for a real belt.

* **Cons:** Latency and complexity; open-vocabulary detectors are weak on transparent, low-contrast items - exactly glass.

* **Use it when:** Multi-object streams and cluttered scenes where an image-level classifier is not deployable.

* **Scores (1–5):** accuracy_ceiling 4, transparency_robustness 4, data_efficiency 4, contamination_robustness 5, cost_efficiency 2, throughput 2, explainability 3, tooling_maturity 4, calibration_abstention 3, zero_shot_capability 4  | tier: research / production


### Spectral sensor + learned model


#### NIR / FTIR spectroscopy + PLS-DA or 1D-CNN  —  composite 3.97/5 (profile: Balanced commercial deployment)


* **Cost:** 0.01-2 (1D models)M params, negligibleG MACs, input 1D spectrum (hundreds-2,000 bins), pretraining none (1D nets) , licence open algorithms (chemometrics); sensor vendor-dependent

* **Evidence on real waste / glass-plastic data:** Benchmark table across FTIR/NIR/hyperspectral-NIR: PLS-DA F1 0.517/0.580/0.971, LDA 0.506/0.578/0.987, 1D-CNN 0.691/0.938/0.877, improved-CNN 0.981/0.978/1.000, transformer 0.921/0.972/1.000 [R27]; 1D-ResNet reaches 0.991 on FTIR with 2x augmentation [R25]; GAN augmentation lifts six-polymer balanced accuracy to 96.2% [R26]; contrast this with RGB, which cannot see polymer chemistry at all.

* **Glass/plastic strengths:** Answers the *chemistry* question directly: glass is a silicate (no C-H/NIR absorption bands), plastics are hydrocarbon polymers with specific overtone bands - the single most reliable glass-vs-plastic discriminator known. Also resolves polymer identity (PET/PE/PP/PS).

* **Glass/plastic weaknesses:** Point-measurement geometry (needs the item under the probe); black carbon-filled plastics absorb NIR light and defeat it [R22,R24]; sensitive to dirt, moisture, thickness and surface roughness; requires preprocessing (SNV/MSC/Savitzky-Golay) to be reproducible [R27].

* **Pros:** Very high discriminative power including the transparent-transparent case; tiny models, millisecond decisions; mature industrial hardware.

* **Cons:** Costly sensors, needs presentation/scanning mechanics, immune to the shape/context cues that matter for handling; performance depends strongly on preprocessing and augmentation.

* **Use it when:** Any application where the transparent-glass-vs-transparent-plastic decision must be correct - i.e. as the reference modality, or the arbiter when RGB confidence is low.

* **Scores (1–5):** accuracy_ceiling 5, transparency_robustness 5, data_efficiency 4, contamination_robustness 3, cost_efficiency 2, throughput 4, explainability 4, tooling_maturity 3, calibration_abstention 5, zero_shot_capability 2  | tier: production (reference modality)


### Spectral imaging + learned model


#### Hyperspectral imaging (VIS-NIR / SWIR) + ML  —  composite 3.78/5 (profile: Balanced commercial deployment)


* **Cost:** 0.1-30 (CNN) / 0.01 (PLS-DA)M params, 0.1-20G MACs, input cube (e.g. 100-300 bands), pretraining ImageNet (2D) or none (3D/1D nets), licence algorithms open; cameras commercial

* **Evidence on real waste / glass-plastic data:** Hierarchical PLS-DA on VIS-NIR HSI classifies five industrial glass colour/tone classes with sensitivity/specificity 0.910-1.000 [R21]; industrial HSI pushes recycled purity near 100% with PP/PE/PET ~99% [R23]; pixel-level CNN (P1CH) reaches 97.44% on HDPE/PET/PP/PS (99.94% excluding border pixels) while HybridSN scores 21.81% [R19]; NIR HSI + ANN reaches 89.5% on an unknown mixed plastic stream [R20]; black plastics remain hard (F1 0.67 vs 0.90 for coloured) [R24].

* **Glass/plastic strengths:** Combines the chemical specificity of spectra with spatial context: can sort glass colour classes *and* reject ceramics/stones, which RGB cannot; operates on whole objects on a belt, unlike a point probe.

* **Glass/plastic weaknesses:** Cube data is bulky and slow; strong illumination dependence; black carbon-filled plastics still fail; per-pixel metrics can be inflated by ignoring background/border pixels (the same paper drops from 97.44% to 39.69% under a stricter metric) [R19].

* **Pros:** Highest demonstrated accuracy for material *and* colour sorting in one sensor; deployed industrially, so the reliability evidence is real.

* **Cons:** 5-50x the cost and complexity of an RGB camera; calibration drift; heavy compute; explaining a spectral decision to an operator is harder than showing a highlight.

* **Use it when:** High-value streams where purity drives price, or as the labelling oracle for an RGB-only deployment.

* **Scores (1–5):** accuracy_ceiling 5, transparency_robustness 5, data_efficiency 3, contamination_robustness 4, cost_efficiency 2, throughput 3, explainability 3, tooling_maturity 3, calibration_abstention 4, zero_shot_capability 2  | tier: production (industrial) / research


### Multisensor fusion


#### Custom sensor fusion (RGB+HSI, RGB+polarisation, RGB+thermal, RGB+depth/3D)  —  composite 3.50/5 (profile: Balanced commercial deployment)


* **Cost:** backbone-dependentM params, sum of branchesG MACs, input aligned streams, pretraining ImageNet branches + sensor-specific encoders, licence components vary; alignment code often proprietary

* **Evidence on real waste / glass-plastic data:** Spectral-conversion autoencoders lifted accuracy from 0.933 (unimodal) to 0.970; a combined HSI-RGB system 'significantly outperforms the individual methods' on a municipal sorting line [R22-adjacent]; multi-scale RGB+HSI fusion (RHFF-SOLOv1) was needed to separate transparent PET, blue PET and transparent PP on a black belt [R32]; thermal+RGB solves glass segmentation and 3D active vision targets dark glass and contaminants [R31,R43].

* **Glass/plastic strengths:** Sensor complementarity is the theoretical answer to the glass/plastic problem: RGB gives shape/colour, NIR/polarisation gives material, thermal/3D gives objectness and dark-glass detection. Fusion is the only approach with *published* evidence of solving the transparent-vs-transparent case in an industrial setting.

* **Glass/plastic weaknesses:** Alignment/registration between streams, calibration drift, doubled failure surfaces, cost, and much harder deployment/maintenance.

* **Pros:** Best achievable accuracy and the strongest evidence base for hard transparent items; each branch covers the other's blind spot.

* **Cons:** Engineering-heavy; datasets are rare; hard to benchmark fairly or reproduce.

* **Use it when:** Industrial deployment where a few points of purity have a direct monetary value, or a research contribution on the transparent-transparent problem.

* **Scores (1–5):** accuracy_ceiling 5, transparency_robustness 5, data_efficiency 2, contamination_robustness 5, cost_efficiency 1, throughput 2, explainability 3, tooling_maturity 2, calibration_abstention 4, zero_shot_capability 1  | tier: production (industrial) / research


### Non-RGB sensing


#### Thermal IR and 3D/structured-light sensing for transparent objects  —  composite 3.28/5 (profile: Balanced commercial deployment)


* **Cost:** smallM params, smallG MACs, input full frame, pretraining ImageNet or none, licence vendor-dependent

* **Evidence on real waste / glass-plastic data:** Thermal+RGB imaging is used for 'reliable glass segmentation'; RGB stereo gives 3D characterisation of recyclables; 3D active vision with structured light targets dark glass and contaminants; UV fluorescence is patented for glass with additives (lead) [R31,R43].

* **Glass/plastic strengths:** Solves the *detection/segmentation* half of the problem for transparent objects (structured light and thermal boundaries do not rely on RGB contrast), and thermal emissivity can differ between thin films and rigid glass.

* **Glass/plastic weaknesses:** Weak on the material half: emissivity depends more on surface finish/coating than on glass-vs-plastic; 3D shape is shared by both material classes.

* **Pros:** Complementary, mature sensors; robust segmentation of items that RGB cannot even outline.

* **Cons:** Does not by itself separate glass from plastic; adds hardware only justified when segmentation, not material ID, is the bottleneck.

* **Use it when:** As the detection stage in front of an RGB/spectral material classifier.

* **Scores (1–5):** accuracy_ceiling 3, transparency_robustness 4, data_efficiency 3, contamination_robustness 4, cost_efficiency 3, throughput 3, explainability 3, tooling_maturity 3, calibration_abstention 3, zero_shot_capability 2  | tier: research


### Method layer (model-agnostic)


#### Uncertainty + rejection layer (conformal prediction, MC-dropout, ensembles, OOD)  —  composite 4.32/5 (profile: Balanced commercial deployment)


* **Cost:** +0M params, x N for ensemblesG MACs, input n/a, pretraining n/a, licence open (MAPIE, Torch/uncertainty-toolbox, custom)

* **Evidence on real waste / glass-plastic data:** The waste-VLM study shows that frozen-feature pipelines need a confidence policy to be usable under taxonomy drift [R30]; the industrial HSI literature reports sensitivity/specificity pairs rather than accuracy precisely because plants operate at a chosen operating point [R21]; black-plastic NIR results (F1 0.67) are a case where the correct answer is 'reject and divert' [R24].

* **Glass/plastic strengths:** Directly monetises the glass/plastic ambiguity: route low-confidence items to a second sensor (NIR/polarimeter) or to a manual station instead of guessing. Conformal prediction gives a *distribution-free* coverage guarantee, which is what a certification process asks for.

* **Glass/plastic weaknesses:** Needs a representative calibration set; a miscalibrated model produces confidently wrong labels; adds latency and complexity.

* **Pros:** Turns an accuracy number into an operating policy (purity vs recovery trade-off); gives auditable coverage guarantees; cheap.

* **Cons:** Requires honest evaluation of calibration under shift (the classical models measured in this repo are *not* calibrated under noise/clutter shifts).

* **Use it when:** Any real deployment - especially anything feeding a purity-sensitive product stream.

* **Scores (1–5):** accuracy_ceiling 4, transparency_robustness 5, data_efficiency 5, contamination_robustness 4, cost_efficiency 4, throughput 3, explainability 5, tooling_maturity 4, calibration_abstention 5, zero_shot_capability 3  | tier: production (mandatory in practice)


## 5. What the evidence says when you put it together


Five patterns recur across the sources in `docs/12_references.md` and are the practical core of this study:

1. **On the published benchmarks, almost everything works.** Reported accuracies on 6-class
   household-waste sets are 90–99.6% (`DenseNet-121` ~95% [R8], optimised `DenseNet-121` 99.6%
   [R12], `DNN-TC` 94% [R7], `WasNet` 96.1% [R11]). Those datasets are studio-lit, single-object,
   white-backdrop and heavily saturated — `TrashNet` has only 2,527 images [R1] and an AUC of 1.0
   for all classes except glass and trash [R34]. The **glass** class is the recurring exception.

2. **The accuracy gap between model families is small next to the accuracy gap between datasets.**
   The same `ResNet-34` transfer-learning pipeline reaches 96% on a balanced web-scraped plastic
   set and 71.8% on a harder, imbalanced one [R37]. Taxonomy depth moves the number even more:
   96% (2 classes) → 91% (9) → 85% (36) in one cascaded system [R13].

3. **The binary glass-vs-plastic decision is not what benchmarks measure.** Benchmarks test six
   or nine mutually distinguishable classes, where paper/cardboard, plastic/metal and
   glass/metal are the confusions that dominate [R33,R34,R35]. Removing the easy context and
   asking *only* "is this clear object glass or PET?" is a much harder, barely-benchmarked task.
   Two independent lines of evidence say RGB alone cannot do it reliably: the glass-waste sensor
   review states that a transparent plastic sheet may be "almost indistinguishable" from glass
   [R31], and dedicated multi-scale RGB+HSI fusion was required for transparent PET vs
   transparent PP on a black belt [R32].

4. **The methods that do solve it are spectral or polarimetric, not bigger CNNs.** NIR/FTIR
   models reach F1 0.98–1.00 on polymer identification [R25,R26,R27]; polarimetry is *specifically*
   reported as useful for glass vs transparent plastics [R31]; industrial HSI pushes recycled
   purity close to 100% [R23]. Where black carbon-filled plastics defeat NIR, the honest answer in
   the literature is a reject/divert policy, not a better classifier (F1 0.67 on black plastics
   [R24]).

5. **Model choice within the RGB family is a cost/benefit decision, not an accuracy hunt.**
   Independent backbone comparisons put `ConvNeXt-Tiny` first overall across six domains at 28.6M
   params [R38]; `MobileNetV3` matches much larger models on augmented material data at a
   fraction of the training time [R17]; sub-million-parameter distilled models are within a few
   points of the state of the art on waste images [R9]. ImageNet top-1 numbers are themselves
   recipe-dependent — the same `ResNet50d` scores 76.1% or 81.8% depending on the training recipe
   [R40] — so architecture ranking is not a substitute for benchmarking on your own data.


## 6. Recommended stack by scenario

| scenario | model | why |
| --- | --- | --- |
| Prototype on TrashNet/RealWaste, need a number fast | ResNet-18 or DenseNet-121 transfer learning | Cheap, comparable with the published literature, and DenseNet-121 is the most reproducibly strong waste backbone [R7,R8,R12] |
| Best accuracy per unit of effort on your own labelled crops | ConvNeXt-Tiny (or EfficientNetV2-S if data is plentiful) | Top-ranked backbone in a controlled 6-domain comparison [R38]; 82.1–83.9% ImageNet at 4.5–8.4 GMACs |
| Belt-speed inference on CPU/NPU | MobileNetV3-Large or MobileNetV4-Conv-S, INT8 | 98% on augmented material data at minimal training cost and size [R17]; MNv4-Conv-S is 3.8M params / 0.2 GMACs at 2.4 ms on a phone CPU [R39] |
| No labels yet / taxonomy will change | OpenCLIP or SigLIP zero-shot + prompt engineering, then a frozen-feature adapter | 76.3% zero-shot on TrashNet and 82.7→90.5% from prompt engineering alone [R28,R29]; training-free adapters are the recommended route under taxonomy drift [R30] |
| Transparent items must be separated correctly (the real problem) | RGB + NIR/HSI or polarisation fusion; RGB as the shape/context branch only | The only approach with published success on transparent-vs-transparent [R31,R32]; spectral models reach F1 0.98–1.00 on polymer ID [R25,R27] |
| Industrial line, high value, must certify purity | HSI + hierarchical PLS-DA / 1D-CNN, with RGB as pre-sort; conformal rejection routing to manual QA | Sensitivity/specificity 0.910–1.000 for glass colour classes [R21]; near-100% purity in industrial practice [R23]; distribution-free coverage from conformal prediction suits certification |

---

*Generated by `gvp.report`. Edit `data/model_registry.csv`, `data/model_scores.csv` or `configs/scoring.yaml` and re-run `python -m gvp.cli report` — every table above updates.*
