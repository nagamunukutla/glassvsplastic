# 7. Beyond RGB: spectral, polarimetric and 3D sensing

Chapter 2 concluded that the decisive physical cues for glass-vs-plastic are not visual. This
chapter is the evidence, the numbers, and the engineering trade-offs of the sensors that *do* see
the difference. If you read one chapter of this repository before starting a project, read this one
and chapter 6 §5 together.

## 7.1 Why a second modality is not optional for transparent items

Three independent lines of evidence:

1. **The glass-waste sensor review** states that materials with very different composition can share
   visual features — "a sheet of transparent plastic may appear almost indistinguishable from glass"
   — and concludes that relying on visual information alone compromises accuracy, making
   complementary or hybrid technologies necessary [R31].
2. **Multi-scale RGB+HSI fusion** was required to separate transparent PET, blue PET and transparent
   PP bottles on a black conveyor belt: RGB confused the colours, spectra could not resolve colour,
   and only the fused model worked [R32].
3. **Spectral models reach F1 0.98–1.00** on polymer identification [R25,R26,R27] — a different
   regime from the RGB confusion matrices reported for waste imagery [R34,R35].

## 7.2 NIR / SWIR and FTIR spectroscopy

The physics: polymers have C–H, O–H, C=O and C–O absorption bands in the near- and mid-infrared.
Soda-lime glass has none in those windows. This is a *chemical* discriminator, not an appearance
one, and it is why industrial sorting uses NIR.

| Method | Input | Reported performance | Reference |
|---|---|---|---|
| PLS-DA | FTIR / NIR / H-NIR | F1 0.517 / 0.580 / 0.971 | [R27] |
| LDA | FTIR / NIR / H-NIR | F1 0.506 / 0.578 / 0.987 | [R27] |
| 1D-CNN | FTIR / NIR | F1 0.691 / 0.938 | [R27] |
| Improved CNN + trainable preprocessing | FTIR / NIR / H-NIR | F1 0.981 / 0.978 / 1.000 | [R27] |
| Transformer + trainable preprocessing | FTIR / NIR / H-NIR | F1 0.921 / 0.972 / 1.000 | [R27] |
| 1D-ResNet + 2× augmentation | FTIR | 0.991 accuracy | [R25] |
| GAN-augmented classifier | 6 recycled polymers | 96.2% balanced accuracy | [R26] |
| ANN on hand-crafted spectral features | HSI NIR 900–1700 nm | 89.5% on an unknown mixed stream | [R20] |
| N-BEATS ensemble | NIR, industrial black + coloured plastics | F1 0.79 (0.90 coloured / 0.67 black) | [R24] |

Practical notes that decide success:

* **Preprocessing is not optional.** Baseline shifts, scatter and noise dominate raw spectra;
  Savitzky-Golay consistently beat MSC and matched SNV, and *trainable* preprocessing modules were
  best of all [R27]. Report your preprocessing with the same seriousness as your architecture.
* **Small samples favour chemometrics.** SVM/RF show stability and accuracy within a couple of
  points of deep models when data is scarce; deep models win as sample size grows [R25]. With 40–50
  spectra per class, balanced accuracy of 95–96% is achievable for six polymers [R26].
* **Class-specific failure is common and must be reported.** In the same experiment: PET 100%,
  PS 95.2%, PE 90.9%, ABS 90.0%, PC 83.3%, PP **0%** with a single measurement per sample [R26].
  A single aggregate number would have hidden a total failure on one polymer.
* **Black carbon-filled plastics are the documented hole.** Carbon black absorbs across the UV-IR
  range; the hyperspectral review concludes NIR cannot capture their spectrum [R22], and an
  industrial benchmark reports F1 0.67 on black items versus 0.90 on coloured ones — while noting
  that most literature had deemed black-polymer differentiation in NIR infeasible [R24].
  Counterpoint: lab measurements with a wide-range SWIR camera (Specim FX50) *did* sort ABS, PS and
  PE black samples [R23], so the limitation is sensor-dependent, not absolute.

## 7.3 Hyperspectral imaging (VIS-NIR / SWIR)

HSI adds spatial context to spectroscopy: an entire belt can be imaged, each pixel carrying a
spectrum. Industrial systems push recycled-material purity close to 100%, with PP/PE/PET near 99%
purity [R23].

* **Glass colour/tone sorting works at production quality.** A hierarchical PLS-DA framework on
  VIS-NIR HSI (400–1000 nm) classifies five industrially meaningful glass classes — brown, light
  green, dark green, half-white, white — with sensitivity and specificity between 0.910 and 1.000,
  using an object-based strategy to stabilise pixel-level decisions [R21].
* **Pixel-level polymer classification is strong but metrics are treacherous.** A lightweight
  hyperspectral CNN (P1CH) reaches 97.44% overall and 99.94% when border pixels are excluded, versus
  21.81% for HybridSN — but only 39.69% when the background is excluded from the metric, and it
  fails completely on black PS (all pixels classified as background) [R19]. Quote the metric
  definition or the number is meaningless.
* **Cube data is heavy.** Per-pixel inference on CPU is not real-time; production systems use band
  selection, dimensionality reduction or FPGA pipelines. Band-selection research reports k-NN/SVM/RF/
  MLP above 99% using only 2–10 selected bands [R20-adjacent], which is the pragmatic route.

## 7.4 Polarisation imaging

The physics is the cleanest of all: glass is a dielectric, so reflection and transmission change the
polarisation state of light. The glass-waste review reports polarimetry as useful specifically for
**differentiating glass from transparent plastics** [R31] — i.e. exactly the case that defeats RGB.
It is passive, fast and adds one camera.

Why you have probably not seen it deployed: few public datasets, calibration burden, sensitivity to
surface orientation and to stress birefringence in moulded plastics, and no mature fusion tooling
with RGB. For a research group this is one of the highest-value unexplored directions in the field.

## 7.5 Thermal, 3D and structured light

* **Thermal + RGB** gives "reliable glass segmentation" in industrial settings [R31] — thermal
  boundaries do not depend on RGB contrast, so transparent objects become findable.
* **RGB stereo / 3D active vision** targets 3D characterisation of recyclables, dark glass and
  contaminants; structured-light systems with HSI sensors are the commercial end of this [R31,R43].
* **UV fluorescence** is patented for special glass with additives (e.g. lead crystal) [R31] — a
  reminder that "glass" is itself a family with sub-types worth separating.

These modalities are strongest at the **detection/segmentation** half of the problem: they establish
*where* an object is and *that* it is a solid transparent item. Material identity still needs spectra
or polarisation.

## 7.6 Sensor fusion: what actually beats what

| Configuration | Reported outcome | Reference |
|---|---|---|
| RGB alone | colour ambiguity for transparent PET vs PP | [R32] |
| HSI alone | no colour discrimination | [R32] |
| RGB + HSI multi-scale fusion | separates transparent PET, blue PET, transparent PP on a black belt | [R32] |
| Spectral-conversion autoencoder fusion | 0.933 → 0.970 accuracy | [R22-adjacent] |
| HSI + RGB system on a municipal line | "significantly outperforms the individual methods" under challenging conditions | [R22-adjacent] |
| Thermal + RGB | reliable glass segmentation | [R31] |
| Polarimetry + RGB | glass vs transparent plastics | [R31] |

The pattern is consistent: **the two modalities fail on disjoint subsets**, which is the definition
of a good fusion. Fusion engineering costs (registration, calibration drift, two failure surfaces) are
real but bounded; the cost of *not* fusing is a class of objects you can never separate.

## 7.7 Recommended sensor strategy, by problem

| Situation | Strategy |
|---|---|
| Coloured glass cullet, material already pure | VIS camera + colour/tone classifier (mature, cheap) [R21] |
| Mixed recyclables, RGB works well enough | RGB CNN + a rejection policy; monitor the transparent stratum |
| Transparent glass vs transparent PET/PP is the crux | Add SWIR/NIR HSI or polarisation; treat RGB as the shape/context branch [R31,R32] |
| Black automotive/electronic plastics in the stream | SWIR with wide spectral range; expect degraded performance and design the reject path [R23,R24] |
| High-value stream, purity is money | HSI + hierarchical PLS-DA/1D-CNN with conformal rejection to manual QA [R21,R23] |
| No budget for new hardware | Refuse the transparent-vs-transparent decision explicitly, or restructure the stream (e.g. separate by shape/weight first) |

Continue to `docs/08_foundation_models_and_vlm.md` for the labelling-cost side of the same story, or
to `docs/09_evaluation_protocol.md` to design the experiment that proves any of this on your data.
