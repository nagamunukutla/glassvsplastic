# 2. Physics and cues: what actually differs between glass and plastic in an image

The entire model comparison in this repository is downstream of one question: **is there enough
signal in a single RGB frame to separate glass from plastic?** This chapter fixes the vocabulary of
cues that the rest of the study scores models against, and states plainly which cues survive
contact with reality.

## 2.1 Why the question is hard

Glass (soda-lime) is a silicate: optically transparent, amorphous, hard, smooth, non-porous,
isotropic, with refractive index ≈ 1.5 and negligible diffuse albedo. Plastics are polymers:
many are transparent (PET, PS, PC, PP), many are translucent or opaque (HDPE, PP, PVC), most are
softer with visible mould marks, and all are *chemically* distinguishable by their C–H and
functional-group absorption bands.

The overlap is what hurts: **a clear PET bottle and a clear glass bottle share shape, transparency,
specularity and often colour.** The glass-waste sensor review states the problem without hedging:
materials with very different compositions can share visual features, and a sheet of transparent
plastic may appear "almost indistinguishable from glass" to computer vision alone [R31].

Human vision does not solve it from a still image either. The psychophysics is unusually clear:
discriminating mirror-like from refractive (glass) surfaces depends substantially on **motion** —
study participants needed video to read the refractive index of thick transparent objects, using
"motion transparency" from the rear surface as the decisive cue [R41]. A production classifier that
sees one frame has thrown away the strongest cue humans use.

## 2.2 The cue inventory

Five visual cue families are separable, physically motivated, and are the ones the feature blocks in
`src/gvp/features.py` implement. For each: what it is, how glass and plastic differ, and how it
breaks.

| # | Cue | Physical origin | Glass, typically | Plastic, typically | Primary failure mode |
|---|-----|-----------------|------------------|--------------------|----------------------|
| 1 | **Transmission / haze** | Light passes through a transparent medium; a diffuse polymer body scatters it | Background visible through the object, sharp, mildly refracted; refraction fringes at thick rims | Background attenuated and blurred behind the wall; strong diffuse component from the body, pigment or filler | Needs visible background texture; a white studio backdrop destroys the cue (measured in this repo: the `whitebg` shift removes it) |
| 2 | **Specular highlights** | Fresnel reflection at the air–material interface | Few, small, near-saturated, elongated, hugging the silhouette and rims | Fewer, larger, softer, lower peak intensity (diffuse body raises the ambient floor) | Entirely determined by illumination geometry: a single point source on glossy PET mimics glass; a diffuse dome erases the cue for both |
| 3 | **Chromatic/refractive fringing** | Dispersion at steeply curved surfaces | Colour separation on strong edges and thick rims | Negligible | Sub-pixel at typical belt resolutions; destroyed by compression |
| 4 | **Surface texture / finish** | Moulding vs. forming process | Smooth, near-uniform; grinding/polishing marks only | Ribs, mould lines, gate marks, grain, film wrinkles, print | Labels, dirt and wear dominate the texture of both materials in the real world |
| 5 | **Silhouette and rim geometry** | Wall thickness, manufacturing tolerances | Thin, high-contrast rim; crisp silhouette | Thicker walls, softer edge, sometimes double-wall or sleeved | Deformation, crushing and shard geometry destroy it; shape is a *product* cue, not a *material* cue |

Two further cues are available but are **not** RGB cues, which is the whole point of chapter 7:

| Cue | Signal | Where it is decisive |
|-----|--------|----------------------|
| **Spectral (NIR/SWIR/FTIR)** | Overtone and combination bands of C–H, O–H, C=O, C–O; glass has none in the polymer windows | Separating *any* polymer from glass and from each other; the single most reliable discriminator known [R25,R27] |
| **Polarisation** | Glass is a dielectric: reflection/transmission change the polarisation state | Explicitly reported as useful for **glass vs transparent plastics** [R31] |

## 2.3 What the machine-learning community keeps rediscovering

**Shortcuts are the default, not the exception.** A CNN trained to separate "glass" from "plastic"
on a dataset where those classes were photographed under different conditions learns the *scene*.
This is not hypothetical: benchmark datasets are studio-lit, single-object and white-backdrop
(TrashNet, 2,527 images, collected on a white or cardboard background [R1]), and per-class AUC of
1.0 is routinely reported except for the classes that genuinely overlap — glass and trash [R34].
Hence this repository ships a background-only control (`gvp.classical.run_sanity_checks`) and
insists it runs before any accuracy is believed.

**Depth and granularity change the problem.** A cascaded system reports 96% on a 2-way decision,
91% on 9 classes and 85.25% on 36 [R13]. Binary glass-vs-plastic is *not* the task the literature
benchmarks; the literature benchmarks 6–9 mutually distinguishable household classes where the
dominant confusions are paper/cardboard, plastic/metal and trash/anything [R34,R35].

**Where models actually fail is predictable from the cue table.** Recurrent confusions in
published confusion matrices are exactly the cue-overlap pairs: 14 glass samples predicted as
plastic in one multi-class study [R34]; plastic/paper/glass dominating the fine-grained
confusions of another [R35]; plastic repeatedly singled out as the worst-served class [R14];
web-scraped plastics misclassified largely where class imbalance was worst [R37]. None of this is
a coincidence — it is the cue table being played back at you.

**Illumination is part of the model.** Every cue except spectral/polarimetric is a function of the
lighting and the background as much as of the material. A deployment that changes lamps, camera
angle or belt colour without re-validating the model has changed the input distribution of the
most important features. The domain-shift sweep in `results/MEASURED_RESULTS.md` quantifies how
badly the classical tier degrades when the acquisition changes.

## 2.4 Engineering consequences

1. **Control the background.** A textured, non-glossy belt surface behind the objects preserves the
   transmission cue. A white studio backdrop deletes it. (This is why TrashNet results transfer
   badly to a real line — see `docs/03_datasets_and_benchmarks.md`.)
2. **Prefer diffuse, large-area illumination with a *small* bright element.** Diffuse light
   suppresses the specular difference; a single sharp highlight source restores it while keeping
   the rest of the scene clean.
3. **Do not expect shape or colour to separate materials.** Both are product cues. A shard of
   green glass and a green PET bottle fragment are the same shape and colour.
4. **Add a second modality before adding a bigger network.** Every point of accuracy in the
   transparent-vs-transparent regime has come from spectra, polarisation, thermal or 3D sensing,
   not from a deeper CNN [R31,R32,R43].
5. **Design for rejection.** Where NIR itself fails (carbon-black plastics, F1 0.67 [R24]),
   the correct behaviour is "unknown → divert", not a confident wrong label.

Continue to `docs/03_datasets_and_benchmarks.md` for the data reality check, or jump straight to the
comparison in `docs/06_model_comparison.md`.
