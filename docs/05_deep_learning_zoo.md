# 5. Deep-learning zoo: architecture families, costs and how to choose

This chapter is the *reasoning* behind the cost and evidence columns of
`docs/06_model_comparison.md`. Every number quoted here comes from the sources in
`docs/12_references.md`; the repository's own measured deep-learning code lives in
`src/gvp/deep.py` (optional PyTorch/timm tier, running on exactly the same manifests as the
classical tier).

## 5.1 The five architecture families that matter

### (a) CNNs trained from scratch
From-scratch CNNs are the control group. The canonical result: a 6-layer baseline reaches 90.61%
where MobileNetV2 reaches 95.51% and VGG16 96.00% on the same 2-class waste data [R45]. Purpose-built
lightweight waste networks sit here too — RecycleNet (3M params, 81% on TrashNet) [R10],
Focus-RCNet-KD (0.525M params, 92%) [R9], WasNet (96.10% TrashNet, 82.5% Huawei, 64.5% ImageNet)
[R11]. They win on parameter count and lose on accuracy per unit of *engineering effort*.
**Use when** you have a large in-domain dataset, a distillation target, or must own the whole stack.

### (b) CNN transfer learning — the workhorses
Fine-tuning ImageNet backbones is where most published waste results live, and the family
differences are smaller than the dataset differences:

| Backbone | Params (M) | MACs (G) | ImageNet top-1 | Where it earns its place |
|---|---|---|---|---|
| ResNet-18 | 11.7 | 1.8 | 69.8 | fastest credible baseline |
| ResNet-50 / ResNeXt-50 32×4d | 25.6 / 25.0 | 4.1 / 4.3 | 76.1 / 77.6–81.2 | comparability with published waste papers [R22,R33] |
| DenseNet-121 / -169 | 8.0 / 14.1 | 2.9 / 3.4 | 74.4 / 75.6 | most reproducible winner on waste imagery [R7,R8,R12] |
| VGG-16 | 138.4 | 15.5 | 71.6 | reproducing old baselines only |
| Inception-v3 / Xception / IRv2 | 27.2 / 22.9 / 55.8 | 5.7 / 8.4 / 13.2 | 77.3 / 79.0 / 80.3 | multi-scale receptive fields [R15] |
| MobileNetV2 / V3-L / V4-Conv-S | 3.5 / 5.5 / 3.8 | 0.30 / 0.22 / 0.20 | 71.9 / 75.3 / 73.8 | on-belt, on-device inference [R17,R39] |
| EfficientNet-B0 / B3 | 5.3 / 12.2 | 0.39 / 1.8 | 77.1 / 81.6 | the accuracy/cost dial |
| EfficientNetV2-S | 21.5 | 8.4 | 83.9–84.2 | accuracy champion with published waste evidence [R16] |
| ConvNeXt-Tiny / -Small | 28.6 / 50.0 | 4.5 / 8.7 | 82.1 / 83.1 | best all-round backbone in a controlled 6-domain study [R38] |
| RegNetY-3.2GF / InceptionNeXt-T | 19.4 / 28.0 | 3.2 / 4.2 | 82.0 / 82.3 | throughput-per-FLOP for big sweeps [R38] |

Three practical warnings:

* **ImageNet top-1 is recipe-dependent.** The same `ResNet50d` scores 76.1% or 81.8% depending on
  the training recipe [R40]. Comparing papers by top-1 alone is comparing recipes.
* **Parameter count is not latency.** MobileNetV4-Conv-S (3.8M params) runs in 2.4 ms on a phone
  CPU; a VGG-16 has 36× the parameters and is not 36× slower in wall-clock, but it is 93× the
  parameters of MobileNetV2 for ~4 points of ImageNet accuracy — and those points do not survive
  contact with a 2-class material decision.
* **DenseNet's waste reputation is real but dataset-bound.** Its TrashNet numbers (95–99.6%
  [R8,R12]) were produced on a studio-white, single-object benchmark. Expect the ranking to shuffle
  on conveyor data.

### (c) Transformers and hierarchical attention
Plain ViT underperforms at small data (ViT-B/16: 79.9% ImageNet *from scratch with heavy
augmentation*, 84.9% with ImageNet-21k AugReg). DeiT-Ti (5.7M, 72.2%) and DeiT-S (22.1M, 79.8%)
narrow the gap; Swin-T (28.3M, 81.3%), SwinV2-T (82.1%) and MaxViT-T (30.9M, 83.4%) beat same-size
CNNs on accuracy but are latency-unfriendly on edge NPUs. Independently, a controlled backbone
comparison ranks ConvNeXt-Tiny (a CNN) ahead of Swin-Tiny overall [R38].

The *interesting* use of transformers here is not as a classifier but as a **frozen feature
extractor**: the waste-VLM study finds frozen foundation features with a training-free adapter to be
the practical route under taxonomy drift [R30], and transformer heads reach F1 0.92–1.00 on plastic
spectroscopic data [R27]. Attention also has a physical justification in this problem: judging
transmission means comparing the background *through* the object with the background *outside* it —
a global, long-range operation that self-attention performs natively and a small convolution stack
does not.

### (d) Attention-augmented and multi-scale hybrids
Adding attention to a CNN is the cheapest architectural intervention with *published confusion-matrix
evidence* on exactly the confusable classes at issue here: an attention-augmented AlexNet improved
per-class correct counts on pictures of plastic bottles (2,877 vs 2,727) and reduced cross-class
confusions among visually similar waste categories; EfficientNetB2+PMAM reached 93.38% on a 4-class
set; and multi-scale RGB+HSI fusion (RHFF-SOLOv1) was *required* to separate transparent PET, blue
PET and transparent PP on a black belt [R32]. Channel attention also gives a principled way to
handle a saturating specular channel.

### (e) Foundation models and vision-language models
Covered in `docs/08_foundation_models_and_vlm.md`; the headline is that prompt engineering alone
moved zero-shot waste accuracy from 82.71% to 90.48% on a 6-class industrial set, and full
supervision reached 97.18% [R28], while zero-shot on TrashNet reaches 76.30% at 2.83 FPS [R29].

## 5.2 Training recipe: what actually moves the number

Ordered by measured effect size in the literature cited here:

1. **Data augmentation and synthetic minority data.** A diffusion-model augmentation lifted
   ResNet50V2 from 78% to 93% accuracy and precision from 0.53 to 1.00 on imbalanced
   glass-defect data — a 15-point change, larger than any architecture choice available to you
   [R36]. 2× spectral augmentation lifted every classifier by ≥3 points [R25]. GAN-augmented
   spectra reached 96.2% balanced accuracy where single-measurement baselines failed outright
   [R26].
2. **Class imbalance handling.** Web-scraped plastic data scores 96% when balanced and 71.8% when
   not, with per-class recall collapsing for the minority polymers [R37]. Use class weights,
   resampling, or stratified evaluation — and always report per-class recall, never only accuracy.
3. **Resolution.** RealWaste was scaled to 524×524 explicitly to make transparent objects and
   glass-vs-plastic-bottle distinctions learnable [R2]. Conversely, downscaling to save latency can
   delete the thin-highlight cue that chapter 2 identifies as the signal.
4. **Preprocessing/normalisation for spectral inputs.** Savitzky-Golay consistently outperformed MSC
   and was competitive with SNV on spectral data; trainable preprocessing modules were best of all
   ([R27] and the tables therein).
5. **Distillation.** MNv4-Conv-L trained as a distilled student reaches 85.9% ImageNet with 15×
   fewer parameters and 48× fewer MACs than its teacher, losing 1.6 points [R39] — the standard
   route from a research model to a line-deployable one.
6. **Backbone choice.** Real, but the smallest lever of the list, and largely swamped by
   dataset-setting effects [R38,R40].

## 5.3 The code in this repository

`src/gvp/deep.py` is deliberately thin and *comparable*:

* the same manifest CSVs drive the same 70/15/15 split and the same seven shift test sets as the
  classical tier (`data/processed/proxy/manifest_*.csv`);
* `build_backbone()` accepts `tv:resnet18`-style torchvision names or any timm name;
* `train_and_eval()` runs AdamW + OneCycle + label smoothing, optional freezing, AMP, class
  balancing, and logs per-epoch validation accuracy;
* `count_params()` / `count_macs()` give cost accounting without `fvcore`/`thop` dependencies;
* `export_onnx()` and `benchmark_latency()` cover the deployment questions in `docs/10_deployment.md`;
* evaluation reports balanced accuracy, per-class recall, ROC AUC, confusion counts and per-image
  latency on the *same* metrics as the classical tier, so the comparison in
  `docs/06_model_comparison.md` is not hand-waved.

Run it with:

```bash
pip install -r requirements-deep.txt
python -m gvp.cli deep --model tv:resnet18 --epochs 20 --splits test,shift_blur,shift_clutter
python -m gvp.cli bench --models tv:resnet18,tv:mobilenet_v3_large,tv:convnext_tiny
```

## 5.4 How to choose, in one paragraph

Start with the **physics floor** (a one-threshold rule) and a **cheap transfer-learning baseline**
(ResNet-18 or DenseNet-121) on your own data with per-class recall and a background-only control.
If the glass/plastic boundary is the limiting factor and both classes contain transparent items, no
amount of architecture shopping will fix it — add a spectral or polarimetric branch (chapter 7).
If the bottleneck is latency, cut to MobileNetV3/V4 with INT8 and verify that the quantised model
still resolves the highlight cue. If the bottleneck is labels, use a frozen foundation model with a
training-free adapter behind a rejection policy, and spend your annotation budget on the ambiguous
stratum (transparent × clean × unlabelled) rather than on more of the same.
