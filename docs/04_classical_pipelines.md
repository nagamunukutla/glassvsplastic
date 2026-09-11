# 4. Classical and shallow pipelines (and why they are still on the short list)

The classical tier matters here for reasons that have nothing to do with nostalgia: on a real
sorting line it is the cheapest thing that can work, it is the only tier a line operator can audit,
and — as this repository measures — **some of its members are markedly more robust to acquisition
shift than a deep model with the same features**. It also defines the floor that any deep model
must beat.

Everything in this chapter is implemented in:
`src/gvp/features.py` (five descriptor blocks), `src/gvp/classical.py` (model zoo, ablations,
shift sweeps, controls) and `results/MEASURED_RESULTS.md` (measured tables).

## 4.1 The two classical architectures

```
(A) descriptor + shallow classifier          (B) descriptor + spectral/sensor classifier
    image ──► hand-crafted features ──► SVM/RF/GBDT      spectra ──► preprocessing ──► PLS-DA/1D-CNN
              (colour, specular, texture,                 (SNV/MSC/Savitzky-Golay,
               edges, transparency)                        baseline correction)
```

(A) is what this repository benchmarks on proxy data. (B) is the industrial workhorse for actual
material identification and is covered in `docs/07_beyond_rgb.md`; the numbers there
(F1 0.98–1.00 for polymer ID [R25,R26,R27]) are the strongest in the entire study and justify
treating (B) as the reference modality rather than a competitor.

## 4.2 The descriptor blocks, mapped to the cue table

| Block | Features | Cue it encodes (§2.2) | Cost |
|---|---|---|---|
| `color` | HSV/Lab histograms, saturation/illumination statistics, cool/warm tint, colour entropy | weak proxy: dyed vs achromatic | ~1 ms |
| `specular` | highlight fraction, saturated-pixel share, blob count/area, peak spikiness, upper-tail ratio, elongation | highlight morphology (cue 2) | ~2 ms |
| `texture` | LBP (P=8,R=1 and P=16,R=2), GLCM contrast/dissimilarity/homogeneity/energy/correlation at two distances, Laplacian variance, high-frequency energy and kurtosis, local-std percentiles, mid/low frequency-band ratio | surface finish, mould marks, film wrinkles (cue 4) | ~9 ms |
| `edge` | Canny density, Sobel magnitude statistics, 8-bin orientation histogram + entropy, contour circularity/aspect/solidity/extent, Hough line counts and lengths, corner count | silhouette and rim geometry (cue 5) | ~5 ms |
| `transparency` | interior vs annulus gradient ratio, contrast ratio, haze index (interior detail retention), background continuity, interior gradient-orientation coherence, estimated object area | transmission and haze (cue 1) | ~2 ms |

Total: **142 features, ~18 ms/image on one CPU core** (measured; dominates the whole classical
pipeline — the classifiers themselves take 0.004–1.2 ms).

Design constraint worth stating: every feature is computed from the raw image plus an *estimated*
object region (`estimate_foreground`, a border-colour/Otsu heuristic). No ground-truth masks are
used, because a deployment would not have them. The cost of that choice is visible in the results:
under background clutter the estimator and the global descriptors both degrade, and the pipeline
collapses (see §4.5).

## 4.3 The model zoo and what it buys

Measured on the proxy task (420 train / 90 test images, balanced accuracy with 95% bootstrap CI):

| Model | Balanced acc. | Fit time | Classifier latency | Notes |
|---|---|---|---|---|
| `hist_gbdt` | 0.944 | 0.96 s | 0.046 ms | best overall; interactions between cues |
| `svm_rbf` | 0.933 | 0.01 s | 0.031 ms | fastest to fit; least robust under shift |
| `extra_trees` | 0.933 | 0.66 s | 1.17 ms | most robust under shift |
| `lda_shrinkage` | 0.922 | 0.01 s | 0.005 ms | cheapest at inference; linear reference |
| `svm_linear`, `logreg_l2` | 0.911 | 0.01 s | 0.004 ms | interpretable coefficients |
| `knn_5` | 0.900 | 0 s | 0.019 ms | zero training; updates by adding exemplars |
| `random_forest` | 0.900 | 0.79 s | 0.711 ms | feature importances |
| `mlp_64_32` | 0.889 | 0.04 s | 0.007 ms | no advantage at this sample size |
| `gaussian_nb` | 0.867 | 0 s | 0.006 ms | weakest; Gaussian assumption violated |

Two observations the literature corroborates:

* **On small samples, shallow models and deep models are close, and the shallow ones are more
  stable.** The spectroscopy comparison reaches the same conclusion explicitly: "for small sample
  datasets, traditional machine learning algorithms such as SVM and RF demonstrated high stability
  and accuracy, with only minimal differences compared to deep learning algorithms. However, on
  large sample datasets, deep learning algorithms showed a stronger advantage" [R25]. The general
  version of this — pretrained models (0.54–0.93) beat non-pretrained (0.41–0.78), and CNNs beat
  plain NNs — is reported for microplastic imagery too [R18].
* **1-D/tabular inputs beat 2-D reformulations of the same information** in the spectroscopy study
  [R25], which is a warning against "just turn the features into an image and use a CNN".

## 4.4 Interpretability: the physics floor and the feature ranking

A single threshold on the highlight-spikiness feature — with the threshold calibrated on the *train*
split only — reaches **0.70 balanced accuracy** on the proxy task. That is the number to keep in
mind: one explainable rule captures a large fraction of a 142-feature model's performance, and any
learned model that cannot clearly beat it on your data is not paying for its complexity.

The extra-trees importances (figure `results/figures/top_features.png`) concentrate on specular,
texture and transparency descriptors rather than colour — the cue ordering the physics predicts,
and a good first check that a pipeline is looking at the right things.

## 4.5 The three failure modes you should expect (measured)

Trained on clean data, evaluated on fresh scenes with one acquisition change each:

| Shift | ExtraTrees | HistGBDT | SVM-RBF | Diagnosis |
|---|---|---|---|---|
| clean test | 0.933 | 0.944 | 0.933 | — |
| blur | 0.883 | 0.842 | 0.883 | fine-detail texture cues degrade |
| dark | 0.825 | 0.883 | 0.742 | global statistics shift |
| sensor noise | 0.842 | 0.817 | **0.500** | SVM standardisation + RBF geometry do not survive the shift |
| JPEG artefacts | 0.917 | 0.925 | **0.542** | as above |
| low contrast | 0.717 | 0.767 | 0.817 | LBP/GLCM respond to contrast normalisation |
| **background clutter** | **0.500** | **0.508** | **0.508** | every global descriptor contaminated |
| background removed (white) | 0.950 | 0.958 | 0.900 | the transmission cue was *not* load-bearing in this renderer |

Diagnostics for the clutter collapse, from the same run: Canny edge density ×6.4, saturated-pixel
fraction ×3.8, `edge_mag_p95` ×4.1 versus the clean test split. Debris and glare inject glass-like
highlight statistics *everywhere*, and a global descriptor has no way to know that the bright blob
it is measuring is a bolt on the belt rather than the rim of the object.

**The fix is architectural, not statistical**: restrict the descriptors to the object (a detector
or segmenter), which is exactly what an attention/detector-prior network does implicitly. Any
classical deployment on a busy line should be preceded by detection; the residual errors of all
classical models on this task will otherwise be dominated by scene content.

## 4.6 Where the classical tier is the right answer

* **Few labels, hard deadline.** Everything here trains in under a second and needs hundreds, not
  thousands, of images. A hand-designed rule plus a tree ensemble is often deployable in days.
* **Certification and audit.** Every decision decomposes into named descriptors with thresholds.
  For a plant that must justify a purity claim, a
  `glass: specular_spikiness > 0.44 ∧ haze_index < 0.48`-style rule is a feature, not a limitation.
* **Extreme edge constraints.** 0.004 ms of classifier latency and a few hundred weights, on a
  device with no GPU, in a language with no deep-learning runtime.
* **As the floor in any deep-learning project.** Report it. A CNN that scores 2 points above a
  GBDT on 142 descriptors is a much weaker result than it looks.
* **Spectral pipelines.** The descendent of the classical tier (PLS-DA/chemometrics) remains the
  industrial standard for material identification, and for good reasons [R21,R27].

## 4.7 What not to do

* Do not use global descriptors without detection/segmentation on a busy belt (§4.5).
* Do not ship an RBF-kernel model without testing it under the degradations your camera actually
  produces — it was the least shift-robust model in the entire classical tier.
* Do not tune thresholds on the test split. Every threshold in this repository is calibrated on
  train only, which is why the numbers are as unimpressive (and as trustworthy) as they are.
* Do not treat "142 features" as free: ~18 ms/image of descriptor computation is the real cost
  driver, and at belt speed it is the argument for an end-to-end model or an FPGA implementation.
