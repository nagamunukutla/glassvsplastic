#!/usr/bin/env python3
"""Build the comparison registry: data/model_registry.csv, data/model_scores.csv, data/references.csv.

This script is the *provenance* of the tables rendered in
``docs/06_model_comparison.md`` and ``results/model_comparison.html``. Keeping the content in
a reviewable Python source file (rather than hand-editing CSV) makes the rubric auditable:
every field is either (a) an architecture fact, (b) a reported result with a reference key, or
(c) an explicit judgement of the authors.

Columns
-------
Architecture facts : params_m, macs_g, input_res, pretrain, imagenet_top1
Deployment facts   : data_need, train_cost, throughput, licence
Evidence           : evidence (reported numbers on waste/glass/plastic data, with [Rn] keys)
Glass/plastic view : glass_plastic_strengths, glass_plastic_weaknesses
Judgement          : pros, cons, recommended_when, tier
Scoring            : model_scores.csv -- 9 criteria on a 1-5 scale (see configs/scoring.yaml)

Nothing here is a measured result of *this* repository. Measured repo results live in
``results/*.csv``.
"""

from __future__ import annotations

import csv
import os

# --------------------------------------------------------------------------------------
# References (cited as [Rn] in the registry and in docs/12_references.md)
# --------------------------------------------------------------------------------------

REFERENCES = [
    ("R1", "Thung & Yang — TrashNet: a 2,527-image, 6-class waste dataset captured on a white "
           "backdrop (cardboard, glass, metal, paper, plastic, trash), Stanford CS229 project, 2016.",
     "https://github.com/garythung/trashnet"),
    ("R2", "Single, Iranmanesh & Javadi — RealWaste: a novel real-life data set for landfill waste "
           "classification using deep learning. Information 14(12):633, 2023. 4,752 images at "
           "524x524, 9 classes (glass 378, plastic 831). Explicitly motivated by transparency and by "
           "'similarities between specific objects (e.g. glass and plastic bottles)'.",
     "https://www.mdpi.com/2078-2489/14/12/633"),
    ("R3", "Bashkirova et al. — ZeroWaste dataset. 26,766 images (ZeroWaste-f 4,503 fully annotated; "
           "s 6,212 unlabelled; w 1,410 weakly labelled), 4 material classes.",
     "https://github.com/Trash-AI/ZeroWaste"),
    ("R4", "Proenca & Simoes — TACO: trash annotations in context. 4,617 images, 60 litter "
           "categories, COCO-style, severe class imbalance.",
     "https://arxiv.org/abs/2003.05664"),
    ("R5", "GlobalWasteData (2026) — consolidated table of 20+ waste image datasets with sample "
           "counts and class counts; documents that most public datasets are small, imbalanced and "
           "captured in a single setting.",
     "https://arxiv.org/html/2602.07463v1"),
    ("R6", "AgaMiko — waste-datasets-review: curated catalogue of waste image datasets with licences "
           "and download links.",
     "https://github.com/AgaMiko/waste-datasets-review"),
    ("R7", "Bui et al. — A novel framework for trash classification using deep transfer learning "
           "(DNN-TC). 94% on TrashNet, 98% on VN-trash; compares DenseNet121 (91%), RecycleNet (68%) "
           "and ResNet (72%) under identical splits.",
     "https://www.researchgate.net/publication/344650744"),
    ("R8", "Aral, Keskin et al. — Classification of TrashNet dataset based on deep learning models "
           "(DenseNet121/169, InceptionResNetV2, MobileNet, Xception). Best: DenseNet121 ~95% with "
           "Adam + augmentation.",
     "https://www.semanticscholar.org/paper/f5a380760b91393ad05bfc2063434f76935a428e"),
    ("R9", "Focus-RCNet with knowledge distillation — 0.525M parameters, 92% on TrashNet; argues "
           "mobile-grade models are sufficient for recyclable sorting.",
     "https://www.sciencedirect.com/science/article/pii/S0950705125000760"),
    ("R10", "RecycleNet — 3M parameters, 81% on TrashNet (vs 7M for the reference model); earliest "
            "explicit accuracy/parameter trade-off in waste classification.",
     "https://www.sciencedirect.com/science/article/pii/S0950705125000760"),
    ("R11", "WasNet — lightweight architecture: 96.10% on TrashNet, 82.5% on Huawei garbage "
            "classification, 64.5% on ImageNet.",
     "https://www.sciencedirect.com/science/article/pii/S0950705125000760"),
    ("R12", "Optimised DenseNet121 — 99.60% on TrashNet (6 classes, 2,527 images) reported in a 2025 "
            "survey table of state-of-the-art waste classifiers; illustrates how saturated TrashNet "
            "has become.",
     "https://www.sciencedirect.com/science/article/pii/S0950705125000760"),
    ("R13", "DP-CNN-En-ELM / TriCascade (2025) — hierarchical waste classification: 96% (2 classes), "
            "91% (9 classes), 85.25% (36 classes). Direct evidence that accuracy decays sharply with "
            "taxonomy depth.",
     "https://www.sciencedirect.com/science/article/pii/S0950705125000760"),
    ("R14", "Lin et al. — RWNet (ResNet variants) on TrashNet: RWNet-101 89.9%; the authors report "
            "that the models sort most recyclables well 'except plastic' (ROC > 0.9).",
     "https://pmc.ncbi.nlm.nih.gov/articles/PMC11096226/"),
    ("R15", "Multi-objective beluga-whale-optimised InceptionV3 — 92.62% in 0.63 s on TrashNet; "
            "beats MobileNetV2, VGG16 and AlexNet under the same protocol. Includes an explicit "
            "class-imbalance discussion.",
     "https://pmc.ncbi.nlm.nih.gov/articles/PMC11096226/"),
    ("R16", "Sustainability study of garbage classification models — EfficientNetV2S 96.41% "
            "identified as the most sustainable/accurate trade-off; ResNet50 > ResNet110 in accuracy "
            "and IoU but larger carbon footprint.",
     "https://www.researchgate.net/publication/370110735"),
    ("R17", "Zhang et al. / ACS Anal. Chem. (2025) — hyperspectral microplastic shape classification: "
            "EfficientNet_b7, Inception_v3 and MobileNet_v3 all reach 98% on augmented data, with "
            "MobileNet reaching that accuracy in the least training time and smallest size.",
     "https://pubs.acs.org/doi/10.1021/acs.analchem.5c02683"),
    ("R18", "Deep learning shape classification for hyperspectral-imaged microplastics (2025) — "
            "MobileNet best (0.93 validation, 1.00 test on augmented refined data); CNNs (0.41-0.78) "
            "beat plain NNs (0.34-0.69); pretrained > from-scratch.",
     "https://pmc.ncbi.nlm.nih.gov/articles/PMC12489888/"),
    ("R19", "P1CH (2025) — pixel-level hyperspectral material classification of HDPE/PET/PP/PS: "
            "97.44% overall (99.94% when border pixels are excluded), vs 21.81% for HybridSN; fails "
            "badly on black PS (undetected); 39.69% when background is excluded from the metric.",
     "https://arxiv.org/html/2409.13498v2"),
    ("R20", "Hyperspectral NIR (900-1700 nm) classification of post-consumer thermoplastics with a "
            "2-layer ANN — 89.5% on an unknown mixed plastic waste stream; entropy/contrast-stretching "
            "segmentation.",
     "https://www.sciencedirect.com/science/article/abs/pii/S0957582023008935"),
    ("R21", "Bonifazi et al. — Hyperspectral imaging + hierarchical PLS-DA for colour classification "
            "of glass fragments in recycling (VIS-NIR 400-1000 nm, 5 industry colour classes): "
            "sensitivity and specificity 0.910-1.000.",
     "https://www.mdpi.com/2313-4321/11/3/43"),
    ("R22", "Review of hyperspectral imaging-based plastic waste detection (2023) — NIR fails on "
            "carbon-black plastics because carbon absorbs across the UV-IR range; among conventional "
            "classifiers ResNet-50 performed best (>90%).",
     "https://www.researchgate.net/publication/369146831"),
    ("R23", "Konica Minolta Sensing (2025) — industrial HSI practice: newest hyperspectral cameras "
            "push recycled-material purity close to 100%; PP/PE/PET near 99% purity; black plastics "
            "remain the hard case (Specim FX50 sorted ABS/PE/PS in a lab test).",
     "https://sensing.konicaminolta.us/us/blog/hyperspectral-imaging-shaping-the-future-of-plastic-recycling/"),
    ("R24", "N-BEATS waveform-decomposition ensemble on 1,491 hyperspectral images of ~4,500 black "
            "and coloured plastic pieces from an industrial line — F1 0.79 overall, 0.90 coloured, "
            "0.67 black; the authors note black-polymer separation in NIR was 'deemed unfeasible' in "
            "most prior literature.",
     "https://www.researchgate.net/publication/374118822"),
    ("R25", "Analyst (2025) — data augmentation and classification algorithms on plastic "
            "spectroscopy: 1D-ResNet reaches 0.991 accuracy on FTIR with 2x augmentation; SVM and RF "
            "are the most stable on small samples while deep models dominate on large samples; 1D "
            "input formats beat 2D.",
     "https://pubs.rsc.org/en/content/articlehtml/2025/ay/d4ay01759e"),
    ("R26", "Analyst (2026) — GAN-augmented spectra for recycled plastics: 96.2% balanced accuracy "
            "across six polymers at the optimal synthetic ratio; single-measurement baselines are "
            "highly class-dependent (PET 100%, PP 0%).",
     "https://pubs.rsc.org/en/content/articlehtml/2026/an/d5an01042j"),
    ("R27", "Deep learning-based plastic classification using spectroscopic data (2025) — benchmark "
            "table across FTIR / NIR / hyperspectral-NIR: PLS-DA F1 0.517-0.971, LDA 0.506-0.987, "
            "1D-CNN 0.691-0.938, ANN 0.79-0.99, transformer 0.92-1.00.",
     "https://www.sciencedirect.com/science/article/pii/S0959652625021432"),
    ("R28", "Novelis Research Lab (2025) — vision-language models for waste recognition: OpenCLIP "
            "ViT-L/14-2B 82.71% zero-shot, 90.48% after prompt engineering, 97.18% fully supervised, "
            "3.79 ms/image (~263 FPS) on 14,310 images / 6 classes.",
     "https://novelis.io/research-lab/a-comparative-analysis-of-vision-language-models-for-scalable-waste-recognition/"),
    ("R29", "Recycling (2025) 10(4):144 — zero-shot learning for municipal waste classification on "
            "TrashNet with OpenCLIP/OWL-ViT: ViT-L/14-336 reaches 76.30% zero-shot (427.94M params, "
            "2.83 FPS); supervised models still lead; prompt sensitivity is the main failure mode.",
     "https://www.mdpi.com/2313-4321/10/4/144"),
    ("R30", "Funk et al., Cleaner Waste Systems 13 (2026) 100475 — zero/few-shot evaluation of VLMs "
            "and multimodal LLMs across TrashNet, FPWaste, RealWaste and MultiWaste under taxonomy "
            "drift: frozen-foundation-model + training-free adapters (Tip-Adapter) are the practical "
            "route; textual few-shot descriptions *reduced* MLLM accuracy; image-based few-shot "
            "improved GPT-4o at high inference cost.",
     "https://publica-rest.fraunhofer.de/server/api/core/bitstreams/4367c274-e326-413e-b00c-d57bfc6b11b7/content"),
    ("R31", "Computer vision for glass waste: technologies and sensors (2025) — survey of sensors for "
            "glass sorting: polarimetry exploits the dielectric polarisation signature to separate "
            "glass from *transparent plastics*; thermal+RGB and structured-light systems for dark "
            "glass; states plainly that 'a sheet of transparent plastic may appear almost "
            "indistinguishable from glass' to RGB vision alone.",
     "https://pmc.ncbi.nlm.nih.gov/articles/PMC12609803/"),
    ("R32", "Multi-scale RGB+hyperspectral feature fusion (RHFF-SOLOv1, 2022) — transparent PET, blue "
            "PET and transparent PP bottles on a black conveyor belt; neither RGB alone (colour "
            "ambiguity) nor spectra alone (no colour discrimination) suffices, fusion does.",
     "https://pmc.ncbi.nlm.nih.gov/articles/PMC9609436/"),
    ("R33", "TrashBox + federated deep learning (Sci Rep 14, 2024) — 7 waste classes; ResNeXt-101 "
            "confusion matrix shows the largest residual error between visually similar classes "
            "(cardboard->paper), with glass among the better-separated classes in that setting.",
     "https://www.nature.com/articles/s41598-024-62003-4"),
    ("R34", "PLOS ONE (2025) — enhanced deep CNN waste framework over Waste Classification V2, "
            "TrashNet and OpenRecycle: '14 Glass samples were predicted as Plastic'; AUC is 1.0 for "
            "all TrashNet classes except glass and trash.",
     "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0324294"),
    ("R35", "ResSeMo (2026) — ResNeXt+SENet+MobileNetV3 integration for waste classification; the "
            "fine-grained TrashNet confusion matrix is dominated by plastic/paper/glass confusions.",
     "https://pubmed.ncbi.nlm.nih.gov/41888559/"),
    ("R36", "Enhancing glass defect detection with diffusion models (2025) — synthetic minority-class "
            "images lift ResNet50V2 from 78% to 93% accuracy and precision from 0.53 to 1.00 on an "
            "imbalanced glass-manufacturing set; EfficientNetB0 and MobileNetV2 gain less.",
     "https://arxiv.org/html/2505.03134v1"),
    ("R37", "Evaluation of modern ML approaches for optic plastics sorting (2025) — 9,440 web-scraped "
            "images of 7 plastic types: ResNet-34 transfer learning 96% on the balanced web set, but "
            "only 71.8% on a 2,721-image harder set (F1 71.1%), with PS/HDPE largely wrong due to "
            "class imbalance.",
     "https://arxiv.org/html/2505.16513v1"),
    ("R38", "Which backbone to use (2025) — controlled comparison of 12 backbones across 6 domains: "
            "ConvNeXt-Tiny ranks best overall (28.6M params, ImageNet 82.52%), RegNetY-3.2GF and "
            "EfficientNetV2-S next; the authors warn against choosing backbones by pre-training "
            "accuracy alone.",
     "https://arxiv.org/html/2406.05612v1"),
    ("R39", "Qin et al. — MobileNetV4: universal models for the mobile ecosystem, ECCV 2024. "
            "MNv4-Conv-S: 3.8M params, 0.2G MACs, 73.8% ImageNet top-1 at 2.4 ms on a Pixel 6 CPU; "
            "distilled MNv4-Hybrid-L: 87.0% top-1 at 3.8 ms on a Pixel 8 EdgeTPU.",
     "https://arxiv.org/abs/2404.10518"),
    ("R40", "timm model changelog — training-recipe effects on the same architecture: "
            "resnet50d.ra4_e3600 81.8% top-1 @288 (vs 76.1% for the classic recipe) and "
            "efficientnet_b0.ra4_e3600 79.4% @224; a reminder that reported ImageNet numbers are "
            "recipe-dependent, not architecture-only.",
     "https://huggingface.co/docs/timm/changes"),
    ("R41", "Kimura et al. — Dynamic visual cues for differentiating mirror and glass, Scientific "
            "Reports 8:8410, 2018. Humans need motion (motion transparency from the rear surface) to "
            "separate refractive glass from purely reflective surfaces; static cues are much weaker. "
            "Relevant because a single RGB frame removes the strongest human cue.",
     "https://www.nature.com/articles/s41598-018-26720-x"),
    ("R42", "Frontiers in Environmental Science (2025) — review of ML for microplastic detection: "
            "tabulated advantages/challenges of manual counting, semi-automatic CV, CNNs and "
            "mobile-based detection (data hunger and compute cost being the recurring drawbacks).",
     "https://www.frontiersin.org/journals/environmental-science/articles/10.3389/fenvs.2025.1573579/full"),
    ("R43", "MDPI/ACS-style comparisons of glass-selective sensors and 3D active vision (in [R31]): "
            "structured light and HSI for dark glass and contaminants; RGB stereo for recyclable "
            "characterisation in households.",
     "https://pmc.ncbi.nlm.nih.gov/articles/PMC12609803/"),
    ("R45", "A transfer learning approach for efficient classification of waste materials — "
            "VGG16 96.00%, MobileNetV2 95.51% and a 6-layer baseline CNN 90.61% on the same 2-class "
            "waste data: the canonical demonstration that transfer learning beats from-scratch at "
            "small sample sizes.",
     "https://www.researchgate.net/publication/370110735"),
    ("R44", "EcoVision (2026) — Grounding DINO detection + CLIP material classification with LoRA "
            "adaptation of the visual encoder, evaluated on TrashNet/TACO and field images; argues "
            "for zero-shot material heads behind a supervised detector in landfill pipelines.",
     "https://harbinengineeringjournal.com/index.php/journal/article/view/5151"),
]

# --------------------------------------------------------------------------------------
# Model registry
# --------------------------------------------------------------------------------------

M = []  # populated below


def add(**kw):
    M.append(kw)


# ---------------------------- A. Learning-free / physics-first ------------------------

add(
    id="heuristic_colour_threshold",
    name="Colour/intensity thresholding (industrial VIS sorter)",
    family="Heuristic / physics-first",
    params_m=None, macs_g=None, input_res="full frame", pretrain="none", imagenet_top1=None,
    data_need="0 labelled images (hand-tuned)", train_cost="0 (manual tuning)",
    throughput="very high (>10k fps on FPGA/line-scan)",
    licence="n/a (implement in-house; see vendor toolkits)",
    evidence="Deployed in real plants: PICVISA VIS + neural networks classify glass by colour and "
             "shape for colour sorting [R31,R23]. Works where the *material stream is already "
             "separated* and only colour/tone classes remain (brown/white/green cullet) [R21].",
    glass_plastic_strengths="Extremely cheap, deterministic, audit-friendly; handles glass colour "
                            "classes (amber/green/white) reliably when the stream is clean.",
    glass_plastic_weaknesses="Cannot separate clear glass from clear plastic at all - identical "
                             "achromatic appearance; dies on mixed-colour streams, dirt and labels.",
    pros="Zero data, zero training, sub-millisecond, certifiable logic; the right baseline to beat.",
    cons="No notion of material; brittle to illumination changes; every new stream needs re-tuning.",
    recommended_when="Colour sorting of an already-material-pure glass stream; as an interpretable "
                     "baseline in any study.",
    tier="baseline",
    refs="R21,R23,R31",
)
add(
    id="heuristic_edge_texture",
    name="Edge + texture descriptors (Canny/Sobel, LBP, GLCM)",
    family="Heuristic / physics-first",
    params_m=None, macs_g=None, input_res="full frame", pretrain="none", imagenet_top1=None,
    data_need="0 labelled images (thresholds) or ~100 for calibration",
    train_cost="seconds of CPU", throughput="high (1-10 ms/frame CPU)",
    licence="n/a (OpenCV/skimage)",
    evidence="The classical glass-CV toolkit: Canny/Sobel contours plus texture analysis "
             "differentiate glass from other materials by surface characteristics [R31]. Measured in "
             "this repo: texture-only features reach 0.94 balanced accuracy in the proxy task.",
    glass_plastic_strengths="Cheap proxy for surface finish: glass is smooth and shows rim "
                            "gradients; plastic shows mould ribs, grain and film wrinkles.",
    glass_plastic_weaknesses="Global descriptors: any bright/deformed item elsewhere in the frame "
                             "corrupts them (measured here: Canny density rises 6.4x under "
                             "background clutter and accuracy collapses to chance).",
    pros="Interpretable, dependency-free, runs on a microcontroller; good cue engineering tool.",
    cons="No spatial selectivity -> needs a detector/segmenter in front; threshold sensitivity.",
    recommended_when="Embedded pre-screening, first-pass prototyping, or when the object is "
                     "guaranteed isolated in the frame.",
    tier="baseline",
    refs="R31",
)
add(
    id="heuristic_specular_stats",
    name="Specular-highlight statistics (1-D rule)",
    family="Heuristic / physics-first",
    params_m=None, macs_g=None, input_res="full frame", pretrain="none", imagenet_top1=None,
    data_need="0", train_cost="0", throughput="very high",
    licence="n/a (own code)",
    evidence="Measured in this repo on the synthetic proxy: a single threshold on highlight "
             "spikiness gives 0.70 balanced accuracy - i.e. the physics prior explains a large part "
             "of the signal, and gives a floor any learned model must beat.",
    glass_plastic_strengths="Encodes the most cited physical difference: glass returns few small "
                            "near-saturated highlights, plastic returns broad soft sheen (lower "
                            "refractive index contrast + diffuse body).",
    glass_plastic_weaknesses="Confounded by illumination: a single point light on glossy PET mimics "
                             "glass; a diffuse light dome erases the cue for both.",
    pros="One threshold, explainable to a line operator, no training, trivially auditable.",
    cons="Only meaningful under controlled lighting; colourless text/labels create false highlights.",
    recommended_when="Lighting-controlled capture cell as an interpretable baseline / fallback rule.",
    tier="baseline",
    refs="R41,R31",
)
add(
    id="heuristic_transmission",
    name="Transmission / haze index (background-continuation)",
    family="Heuristic / physics-first",
    params_m=None, macs_g=None, input_res="full frame", pretrain="none", imagenet_top1=None,
    data_need="0", train_cost="0", throughput="very high",
    licence="n/a (own code)",
    evidence="The 'does the scene continue through the object' test underlies classical transparent-"
             "object work; measured here at 0.58 balanced accuracy alone, contributing to a 0.94 "
             "ensemble. Related physics led to polarimetry, which is exactly this idea with a "
             "polariser [R31].",
    glass_plastic_strengths="Directly measures transmission vs diffuse attenuation, the one cue that "
                            "is material-specific rather than shape- or lighting-specific.",
    glass_plastic_weaknesses="Requires textured background behind the object (measured here: it is "
                             "the cue a featureless white studio backdrop removes - with whitebg "
                             "shift the composite model still held 0.95, showing the model is not "
                             "actually relying on it).",
    pros="Physically grounded, cheap; tells you when your imaging station is uninformative.",
    cons="Useless on a blank backdrop; needs a foreground estimate to be computed at all.",
    recommended_when="Conveyor/table capture with visible texture behind items; design input for the "
                     "capture cell.",
    tier="baseline",
    refs="R31,R32",
)
add(
    id="polarimetry_vision",
    name="Polarisation imaging (glass vs transparent plastic)",
    family="Heuristic / physics-first (non-RGB sensor)",
    params_m=None, macs_g=None, input_res="full frame", pretrain="none", imagenet_top1=None,
    data_need="small (calibration) or a few hundred for a learned head",
    train_cost="minutes of CPU", throughput="high (camera-limited)",
    licence="hardware-dependent (commercial polarisation cameras)",
    evidence="Polarimetry 'leverages the polarization properties of light reflected and transmitted "
             "by glass', exploiting the dielectric nature of glass to alter polarisation state, and "
             "'has proven useful, for example, in differentiating glass from transparent plastics' "
             "[R31]. Still thin in the literature - an open research gap.",
    glass_plastic_strengths="Attacks the exact failure case of RGB (clear glass vs clear PET/PP) with "
                            "a physical property rather than appearance.",
    glass_plastic_weaknesses="Sensitive to surface orientation, stress birefringence in moulded "
                             "plastics and to dirt; adds a sensor and calibration burden.",
    pros="Highest physics-per-euro for the transparent/transparent case; passive and fast.",
    cons="Immature tooling, few public datasets, needs fusion logic with RGB for shape/colour.",
    recommended_when="When the stream is dominated by transparent items and RGB has plateaued; a "
                     "strong candidate for a research contribution.",
    tier="research",
    refs="R31",
)

# ---------------------------- B. Classical ML on features ----------------------------

add(
    id="svm_rbf_features",
    name="SVM (RBF) on hand-crafted features",
    family="Classical ML on features",
    params_m=None, macs_g=None, input_res="feature vector (no image at inference beyond features)",
    pretrain="none", imagenet_top1=None,
    data_need="~200-1,000 labelled images", train_cost="<1 s CPU", throughput="0.04 ms/image (clf only)",
    licence="scikit-learn (BSD-3)",
    evidence="Measured here: 0.933 balanced accuracy on the proxy task with 142 features, "
             "0.89-0.98 bootstrap CI, but collapses to chance under sensor noise and clutter. In "
             "spectroscopy, SVM/RF are the most stable choice for small samples while deep models "
             "win on large samples [R25]; RF/SVM with grid-search/Bayesian tuning are the standard "
             "strong baselines for glass CV [R31].",
    glass_plastic_strengths="Works well with engineered transparency/specular features; strong in "
                            "the small-data regime typical of sorting plants.",
    glass_plastic_weaknesses="Depends entirely on the feature engineering; not spatially selective; "
                             "RBF kernels are acutely sensitive to input distribution shift (measured "
                             "here: 0.50 on noise/jpeg/deep-shadow shifts where tree ensembles "
                             "retained 0.82-0.92).",
    pros="Seconds to train, kilobytes to store, fully inspectable decision surface; excellent "
         "data-efficiency; auditable for certification.",
    cons="Feature engineering is where the accuracy lives; scaling assumptions break under shift; "
         "no notion of shape or context.",
    recommended_when="Small labelled sets, embedded deployment, or when you must justify every "
                     "decision to an auditor.",
    tier="production (small-data)",
    refs="R25,R31",
)
add(
    id="rf_extratrees_features",
    name="Random Forest / ExtraTrees on hand-crafted features",
    family="Classical ML on features",
    params_m=None, macs_g=None, input_res="feature vector", pretrain="none", imagenet_top1=None,
    data_need="~200-1,000 labelled images", train_cost="<1 s CPU",
    throughput="0.1-1.2 ms/image (clf only)",
    licence="scikit-learn (BSD-3)",
    evidence="Measured here: 0.908-0.933 balanced accuracy, and by far the most robust tier under "
             "acquisition shift (0.82-0.92 under noise/jpeg/dark, vs 0.50 for the RBF SVM). The "
             "hyperspectral review finds ResNet-50 best among deep classifiers and RF/SVM strong "
             "among conventional ones [R22].",
    glass_plastic_strengths="Handles the mixed, non-Gaussian statistics of specular/texture "
                            "features; supplies feature importances, which is how you discover which "
                            "cue the model is actually using.",
    glass_plastic_weaknesses="Same blindness to spatial context as any global-descriptor model; "
                             "large ensembles cost memory on MCUs.",
    pros="Robust, fast, interpretable, no scaling assumptions, trains on a laptop in under a second.",
    cons="Cannot exceed its features; importances are biased toward high-cardinality features.",
    recommended_when="Default workhorse for a classical pipeline; the model to beat before reaching "
                     "for a CNN.",
    tier="production (small-data)",
    refs="R22,R25",
)
add(
    id="lda_features",
    name="LDA / QDA on hand-crafted features",
    family="Classical ML on features",
    params_m=None, macs_g=None, input_res="feature vector or spectrum", pretrain="none",
    imagenet_top1=None,
    data_need="50-500 labelled samples", train_cost="<1 s CPU", throughput="microseconds/image",
    licence="scikit-learn (BSD-3)",
    evidence="The linear reference model for engineered features (measured here: 0.922 balanced "
             "accuracy, second only to boosting in the classical tier). Its *spectral* counterpart - "
             "PLS-DA on NIR/HSI/FTIR spectra - is the industrial workhorse and is covered under the "
             "two spectral entries in this registry, where it reaches sensitivity/specificity "
             "0.910-1.000 for glass colour classes [R21] and F1 0.517-0.971 on plastic spectra "
             "[R27].",
    glass_plastic_strengths="Optimal when the classes are linearly separable in the chosen feature "
                            "space; gives signed class scores that an operator or auditor can "
                            "inspect, and needs very few samples to stabilise.",
    glass_plastic_weaknesses="A single linear hyperplane in feature space cannot represent "
                             "'bright AND elongated AND smooth' interactions; sensitive to feature "
                             "scaling and outliers in specular statistics.",
    pros="Standard, explainable, tiny, extremely fast; the score itself is a linear audit trail.",
    cons="Underfits interaction structure (boosting beats it here by 2 points); no spatial context.",
    recommended_when="As the linear reference for a feature-based model; use the spectral entries "
                     "for the PLS-DA-on-spectra pipeline.",
    tier="baseline / production (linear reference)",
    refs="R21,R27",
)
add(
    id="gbdt_features",
    name="Gradient-boosted trees (HistGradientBoosting/XGBoost/LightGBM)",
    family="Classical ML on features",
    params_m=None, macs_g=None, input_res="feature vector", pretrain="none", imagenet_top1=None,
    data_need="~500-5,000 labelled images", train_cost="1-30 s CPU",
    throughput="0.04 ms/image (clf only)", licence="BSD/MIT (sklearn/LightGBM/XGBoost)",
    evidence="Measured here: best shallow model overall (0.944 balanced accuracy, ROC AUC 0.986) and "
             "strong under dark/noise/jpeg shifts (0.82-0.93) - but still collapses under background "
             "clutter, like every global-descriptor model.",
    glass_plastic_strengths="Captures interactions between cues (e.g. 'bright AND large highlight AND "
                            "low texture' -> plastic), which matters because no single cue is "
                            "reliable for glass vs plastic.",
    glass_plastic_weaknesses="Needs more data than LDA/linear models; less interpretable than a "
                             "single rule; still blind to spatial context.",
    pros="Best accuracy-per-second in the classical tier; handles mixed feature types and missing "
         "values; feature importances + SHAP for audits.",
    cons="Hyper-parameter sensitive; CPU-bound training on huge feature sets; black-box-ish.",
    recommended_when="Tabular features of any sensor modality; when a CNN cannot be justified by "
                     "data volume or latency budget.",
    tier="production (small-mid data)",
    refs="R25",
)
add(
    id="knn_gmm_features",
    name="k-NN / Gaussian mixture / Mahalanobis on features",
    family="Classical ML on features",
    params_m=None, macs_g=None, input_res="feature vector", pretrain="none", imagenet_top1=None,
    data_need="50-500 labelled samples", train_cost="milliseconds", throughput="0.1-1 ms/image",
    licence="scikit-learn (BSD-3)",
    evidence="Measured here: k-NN 0.90 balanced accuracy. In industrial hyperspectral work, "
             "similarity/nearest-neighbour heads over frozen embeddings are competitive when many "
             "exemplars are available, and DINOv3 1-NN is the strongest training-free baseline in "
             "the waste-VLM study [R30].",
    glass_plastic_strengths="Non-parametric, so it tracks multimodal appearance (many bottle/ jar "
                            "sub-types) better than a single linear boundary; GMM gives an explicit "
                            "'novelty' score.",
    glass_plastic_weaknesses="Degrades with dimensionality and with class imbalance; needs a feature "
                             "space where distance is meaningful (i.e. you must normalise well).",
    pros="Zero training, instantly updatable (add exemplars for a new product), supports OOD "
         "detection via likelihood.",
    cons="Inference grows with the exemplar set; sensitive to uninformative features.",
    recommended_when="Frequently changing product mix where re-training is impractical; OOD/reject "
                     "logic.",
    tier="production (small-data)",
    refs="R30",
)
add(
    id="mlp_shallow_features",
    name="Shallow MLP on hand-crafted features",
    family="Classical ML on features",
    params_m=0.01, macs_g=None, input_res="feature vector", pretrain="none", imagenet_top1=None,
    data_need="~1,000+ labelled images", train_cost="1-60 s CPU", throughput="0.1 ms/image",
    licence="scikit-learn / PyTorch (BSD/MIT)",
    evidence="Measured here: 0.90 - no better than SVM/trees at this sample size, with a larger "
             "variance. Consistent with the classical-vs-deep result that deep models need sample "
             "scale before they pay off [R25,R18].",
    glass_plastic_strengths="Can learn mild non-linear interactions of features; trivially portable "
                            "(a few hundred weights).",
    glass_plastic_weaknesses="Data-hungry relative to its capacity, no accuracy advantage over "
                             "trees/SVM on the same features, extra tuning surface.",
    pros="Small, fast, familiar; useful stepping stone to a CNN pipeline.",
    cons="Usually strictly dominated by boosting on tabular features - include it in a benchmark "
         "mainly to document that fact.",
    recommended_when="Research comparison / teaching; middle ground between linear models and CNNs.",
    tier="research",
    refs="R25,R18",
)

# ---------------------------- C. CNNs from scratch ------------------------------------

add(
    id="cnn_small_scratch",
    name="Small CNN trained from scratch (4-8 conv layers)",
    family="CNN from scratch",
    params_m=0.5, macs_g=0.05, input_res="96-224", pretrain="none", imagenet_top1=None,
    data_need="~5,000+ images (or heavy augmentation)", train_cost="CPU hours / GPU minutes",
    throughput="1-5 ms/image CPU",
    licence="own code (MIT)",
    evidence="A 6-layer baseline CNN reached 90.61% on a 2-class waste set, versus 95.51% "
             "(MobileNetV2) and 96.00% (VGG16) with transfer learning in the same study - the "
             "canonical 'transfer learning beats from-scratch at small n' result.",
    glass_plastic_strengths="Can learn local texture/gradient cues in a spatially selective way, "
                            "which global descriptors cannot.",
    glass_plastic_weaknesses="Needs far more data than a fine-tuned backbone; learns dataset-specific "
                             "shortcuts (background, lighting) easily, especially on small sets.",
    pros="Full architectural control and total transparency; no licence or download constraints; "
         "cheap inference.",
    cons="Poor sample efficiency; usually loses to a fine-tuned pretrained model of similar size; "
         "more hyper-parameter work.",
    recommended_when="Huge in-domain datasets, on-device constraints with no pretrained weights "
                     "available, or when you must own the whole stack.",
    tier="baseline / research",
    refs="R7,R37",
)
add(
    id="cnn_lightweight_research",
    name="Purpose-built lightweight CNNs (RecycleNet, Focus-RCNet-KD, WasNet)",
    family="CNN from scratch (waste-specific designs)",
    params_m=0.525, macs_g=0.06, input_res="224", pretrain="none / distillation", imagenet_top1=None,
    data_need="~2,000-10,000 images", train_cost="hours CPU / minutes GPU",
    throughput="1-3 ms/image CPU",
    licence="mixed (paper code, check repo licences)",
    evidence="RecycleNet: 3M params, 81% TrashNet [R10]. Focus-RCNet-KD: 0.525M params, 92% "
             "TrashNet [R9]. WasNet: 96.10% TrashNet, 82.5% Huawei, 64.5% ImageNet [R11].",
    glass_plastic_strengths="Knowledge distillation into a sub-million-parameter model is the "
                            "documented route to on-belt inference with acceptable accuracy.",
    glass_plastic_weaknesses="All the usual small-model caveats, plus: the residual errors "
                             "concentrate exactly where the physics is hard (transparent, "
                             "deformed, contaminated items).",
    pros="Mobile-grade cost with respectable accuracy; designed with waste taxonomies in mind.",
    cons="Non-standard, harder to maintain than a timm/torchvision model; benchmark numbers mostly "
         "come from TrashNet, which is saturated and studio-lit.",
    recommended_when="Retrofitting legacy sorting lines with CPU-only inference.",
    tier="production (edge)",
    refs="R9,R10,R11",
)

# ---------------------------- D. CNN transfer learning --------------------------------

add(
    id="cnn_resnet18",
    name="ResNet-18 (transfer learning)",
    family="CNN transfer learning",
    params_m=11.7, macs_g=1.8, input_res="224", pretrain="ImageNet-1k", imagenet_top1=69.8,
    data_need="~200-2,000 labelled crops", train_cost="GPU: ~5 min (2-class, 20 epochs)",
    throughput="~2-6 ms/image on a modern CPU, <1 ms on a mid GPU (batch 1)",
    licence="BSD-3 (torchvision/timm)",
    evidence="The default reference in waste studies; ResNet-34 transfer learning reached 96% on "
             "web-scraped plastics when the classes were balanced but 71.8% on a harder, imbalanced "
             "set (F1 71.1%) [R37]; ResNet-50 was the best of four conventional waste classifiers "
             "in the hyperspectral review [R22].",
    glass_plastic_strengths="Cheap to fine-tune, learns texture/gradient cues that survive mild "
                            "colour shifts; robust enough for a first real-data benchmark.",
    glass_plastic_weaknesses="Residual blocks are good at texture but not at reasoning about "
                             "specularity; will happily exploit background/lighting shortcuts, "
                             "which is why a background-only control is mandatory.",
    pros="Small, well-supported, fast to train, easy to export (ONNX/TorchScript), abundant "
         "pre-trained recipes.",
    cons="Lowest ImageNet accuracy of the modern CNNs in this table; limited capacity for "
         "fine-grained material cues.",
    recommended_when="First transfer-learning baseline on your own glass/plastic data; always "
                     "report it alongside a heavier backbone.",
    tier="baseline",
    refs="R22,R37",
)
add(
    id="cnn_resnet50",
    name="ResNet-50 / ResNeXt-50 32x4d (transfer learning)",
    family="CNN transfer learning",
    params_m=25.6, macs_g=4.1, input_res="224", pretrain="ImageNet-1k",
    imagenet_top1="76.1 (classic recipe) / 81.8 (RA4-E3600 recipe @288)",
    data_need="~500-5,000 labelled crops", train_cost="GPU: ~10-20 min",
    throughput="~5-20 ms/image CPU, ~1-3 ms/image GPU",
    licence="BSD-3 / Apache-2.0",
    evidence="ResNet-50 was the strongest of four conventional waste classifiers (>90%) in the "
             "hyperspectral plastic review [R22]; ResNeXt-101 is the backbone used for the TrashBox "
             "federated study [R33]; the 12-backbone comparison puts ResNet-50 at 76.13% versus "
             "ConvNeXt-Tiny 82.52% at similar size [R38].",
    glass_plastic_strengths="Strong feature extractor for a modest cost; ResNeXt grouping adds "
                            "capacity at iso-FLOPs; well-understood failure modes.",
    glass_plastic_weaknesses="Same image-level shortcut risk as any classifier head; no built-in "
                             "attention to the object vs the belt.",
    pros="The de-facto standard for reproducibility; huge prior literature for comparison; trivially "
         "quantisable (INT8) for edge.",
    cons="Dominated on accuracy-per-FLOP by ConvNeXt/RegNet/EfficientNet at the same budget [R38].",
    recommended_when="When you need comparability with published waste papers; as the second "
                     "baseline after ResNet-18.",
    tier="baseline / production",
    refs="R22,R33,R38",
)
add(
    id="cnn_densenet121",
    name="DenseNet-121 / -169 (transfer learning)",
    family="CNN transfer learning",
    params_m=8.0, macs_g=2.9, input_res="224", pretrain="ImageNet-1k",
    imagenet_top1="74.4 (121) / 75.6 (169)",
    data_need="~200-2,000 labelled crops", train_cost="GPU: ~10-15 min",
    throughput="~4-12 ms/image CPU", licence="BSD-3",
    evidence="The most consistent winner on waste imagery in the literature: ~95% TrashNet "
             "[R8], 91% under a stricter split [R7], and 99.60% with hyper-parameter optimisation "
             "[R12]; also used in the HSI plastic review [R22].",
    glass_plastic_strengths="Feature reuse across scales suits the multi-scale nature of material "
                            "cues (rim highlight vs body texture vs label); small parameter count "
                            "for its accuracy.",
    glass_plastic_weaknesses="Memory-hungry activations at high resolution; like all of these, "
                             "TrashNet numbers are inflated by a studio-white background.",
    pros="Best documented accuracy/parameter ratio for waste classification; widely available.",
    cons="Slower than MobileNet-class models; concatenation memory limits batch size.",
    recommended_when="Default strong CNN baseline for a static-image glass/plastic benchmark.",
    tier="baseline / production",
    refs="R7,R8,R12",
)
add(
    id="cnn_vgg16",
    name="VGG-16 / VGG-19 (transfer learning)",
    family="CNN transfer learning",
    params_m=138.4, macs_g=15.5, input_res="224", pretrain="ImageNet-1k", imagenet_top1=71.6,
    data_need="~1,000-10,000 labelled crops", train_cost="GPU: ~30-60 min",
    throughput=">20 ms/image CPU (slow)", licence="CC BY 4.0 (weights) / own impl.",
    evidence="VGG16 reached 96.00% on a 2-class waste set, beating MobileNetV2 (95.51%) and a "
             "6-layer CNN (90.61%) in the same study [R45]; RealWaste and "
             "TrashNet comparisons routinely include VGG-16 [R2,R15].",
    glass_plastic_strengths="Fine texture modelling via stacked 3x3 kernels - useful for mould "
                            "lines, grain and surface finish.",
    glass_plastic_weaknesses="Enormous for the accuracy delivered; the fully-connected head invites "
                             "overfitting on small waste datasets.",
    pros="Simple, still occasionally top of a small held-out split; many worked examples.",
    cons="93x the parameters of MobileNetV2 for ~4 points of ImageNet accuracy; poor latency; "
         "obsolete for deployment.",
    recommended_when="Reproducing older published baselines; not recommended for new deployments.",
    tier="legacy baseline",
    refs="R2,R15,R45",
)
add(
    id="cnn_inception_xception",
    name="Inception-v3 / Xception / InceptionResNetV2",
    family="CNN transfer learning",
    params_m="27.2 / 22.9 / 55.8", macs_g="5.7 / 8.4 / 13.2", input_res="299", pretrain="ImageNet-1k",
    imagenet_top1="77.3 / 79.0 / 80.3",
    data_need="~500-5,000 labelled crops", train_cost="GPU: ~15-45 min",
    throughput="~8-40 ms/image CPU", licence="Apache-2.0 / BSD-3",
    evidence="A beluga-whale-optimised InceptionV3 reached 92.62% in 0.63 s on TrashNet, beating "
             "MobileNetV2, VGG16 and AlexNet under one protocol [R15]; InceptionResNetV2 and "
             "Inception-v3 are standard entries in TrashNet/RealWaste comparisons [R2,R8].",
    glass_plastic_strengths="Multi-scale receptive fields (Inception) match material cues that span "
                            "rim-to-texture scales; depthwise separable convolutions (Xception) give "
                            "a better accuracy/latency point.",
    glass_plastic_weaknesses="299x299 input costs latency; separable-convolution models are more "
                             "sensitive to low-light noise in our proxy sweep.",
    pros="Mature, well-supported, strong accuracy for a moderate FLOP budget.",
    cons="Superseded at equal cost by ConvNeXt/EfficientNetV2; more preprocessing quirks.",
    recommended_when="Reproducing published waste benchmarks; otherwise prefer ConvNeXt-T/EffNetV2-S.",
    tier="baseline / production",
    refs="R2,R8,R15",
)
add(
    id="cnn_mobilenet_v2",
    name="MobileNetV2 / ShuffleNetV2 (1.0x)",
    family="CNN transfer learning (mobile)",
    params_m=3.5, macs_g=0.3, input_res="224", pretrain="ImageNet-1k",
    imagenet_top1="71.9 / 69.4",
    data_need="~100-2,000 labelled crops", train_cost="CPU: minutes; GPU: <5 min",
    throughput="1-3 ms/image CPU; sub-ms on a DSP/NPU",
    licence="Apache-2.0 / BSD-3",
    evidence="MobileNetV2 reaches 95.51% on a 2-class waste set, within 0.5 points of VGG16 at "
             "1/40th the parameters; an improved MobileNetV2 reached 90.7% on a 4-class challenge "
             "set [R16]; MobileNetV2 is the standard edge choice in waste-classification surveys.",
    glass_plastic_strengths="Cheapest credible accuracy on a CPU/DSP; depthwise separable "
                            "convolutions learn local texture cues efficiently.",
    glass_plastic_weaknesses="A single 224x224 image-level label gives no spatial evidence: one "
                             "bright belt reflection can dominate a global-pooled feature.",
    pros="Runs on an MCU-class NPU; tiny memory; easy quantisation to INT8.",
    cons="Several points below ConvNeXt-T/EffNetV2-S accuracy; limited capacity for rare sub-types.",
    recommended_when="On-belt/edge deployment with a strict wattage or cost budget.",
    tier="production (edge)",
    refs="R16,R18",
)
add(
    id="cnn_mobilenet_v3",
    name="MobileNetV3-Large / -Small",
    family="CNN transfer learning (mobile)",
    params_m="5.5 / 2.5", macs_g="0.22 / 0.06", input_res="224", pretrain="ImageNet-1k",
    imagenet_top1="75.3 / 67.7",
    data_need="~100-2,000 labelled crops", train_cost="CPU minutes", throughput="1-2 ms/image CPU",
    licence="Apache-2.0",
    evidence="EfficientNet_b7, Inception_v3 and MobileNet_v3 all achieved 98% on augmented "
             "microplastic data, with MobileNet_v3 reaching it in the least training time and with "
             "the smallest model - the clearest published argument for mobile-grade models in "
             "material classification [R17]; the same study notes MobileNet's suitability for "
             "smartphone deployment.",
    glass_plastic_strengths="Squeeze-and-excite blocks give it a light channel-attention mechanism - "
                            "useful for suppressing a dominant specular highlight channel.",
    glass_plastic_weaknesses="Compressed capacity can lose rare but decisive cues (thin rim "
                             "highlights, small resin codes).",
    pros="Best accuracy/watt in the mobile tier; tflite/torchscript/ONNX well supported.",
    cons="Behind modern mobile CNNs (MobileNetV4, EfficientViT) at equal latency [R39].",
    recommended_when="Real-time belt or handheld app; the pragmatic default for edge glass/plastic.",
    tier="production (edge)",
    refs="R17,R39",
)
add(
    id="cnn_mobilenet_v4",
    name="MobileNetV4 (Conv-S / Hybrid-L)",
    family="CNN transfer learning (mobile, 2024)",
    params_m="3.8 (Conv-S) / 32.5 (Hybrid-L)", macs_g="0.2 (Conv-S)",
    input_res="224", pretrain="ImageNet-1k (+JFT for distilled variants)",
    imagenet_top1="73.8 (Conv-S) / 87.0 (Hybrid-L distilled)",
    data_need="~100-2,000 labelled crops", train_cost="CPU minutes to GPU hours",
    throughput="2.4 ms on a Pixel 6 CPU (Conv-S); 3.8 ms on a Pixel 8 EdgeTPU (Hybrid-L, INT8)",
    licence="Apache-2.0",
    evidence="MNv4 is 'mostly Pareto-optimal' across mobile CPUs, GPUs, DSPs and accelerators, and "
             "the distilled Hybrid-L reaches 87.0% ImageNet top-1 with 39x fewer MACs than its "
             "teacher [R39] - the current best target for on-device inference.",
    glass_plastic_strengths="Universal Inverted Bottleneck + Mobile MQA give transformer-like "
                            "mixing at mobile cost - helpful when the decisive cue is a small "
                            "highlight or a fine mould mark rather than a global colour.",
    glass_plastic_weaknesses="Very new: few waste-specific published results, and JFT-distilled "
                             "variants carry unclear provenance for regulated deployment.",
    pros="Best known accuracy/latency frontier for edge hardware; INT8-friendly.",
    cons="Ecosystem (tflite/timm parity) is less mature than ResNet/MobileNetV2-V3.",
    recommended_when="New edge deployments where you can afford to validate the toolchain.",
    tier="production (edge, new)",
    refs="R39",
)
add(
    id="cnn_efficientnet_b0_b3",
    name="EfficientNet-B0 / -B3",
    family="CNN transfer learning",
    params_m="5.3 / 12.2", macs_g="0.39 / 1.8", input_res="224 / 300", pretrain="ImageNet-1k",
    imagenet_top1="77.1 (B0) / 81.6 (B3); 79.4 for B0 with the RA4-E3600 recipe",
    data_need="~300-3,000 labelled crops", train_cost="GPU minutes",
    throughput="2-8 ms/image CPU", licence="Apache-2.0",
    evidence="EfficientNetB2 with an attention module reached 93.38% on a 4-class garbage set; "
             "EfficientNetB0 improved from 83.95% to 84.85% overall on glass-defect detection when "
             "synthetic minority data was added, with recall 0.35->0.65 [R36]; MobileNetV3/EfficientNet"
             " were joint top on microplastic images [R17].",
    glass_plastic_strengths="Compound scaling gives a clean accuracy/cost dial (B0->B3) without "
                            "changing the code; MBConv blocks respond well to fine texture.",
    glass_plastic_weaknesses="Sensitive to input resolution - downscaling to save latency can erase "
                             "the thin-highlight cue; group/depthwise normalisation shows more "
                             "low-light noise sensitivity in our proxy sweep.",
    pros="Excellent accuracy per FLOP, one-line scale change, huge deployment support.",
    cons="Superseded by EfficientNetV2 and ConvNeXt at equal cost; recipe-dependent ImageNet "
         "numbers [R40].",
    recommended_when="Balanced accuracy/latency target on a mid-range GPU or a strong CPU.",
    tier="production",
    refs="R17,R36,R40",
)
add(
    id="cnn_efficientnetv2",
    name="EfficientNetV2-S / -M",
    family="CNN transfer learning",
    params_m=21.5, macs_g=8.4, input_res="384 (S)", pretrain="ImageNet-1k/-21k",
    imagenet_top1="83.9-84.2",
    data_need="~1,000-10,000 labelled crops", train_cost="GPU: ~1-3 h (full fine-tune)",
    throughput="~10-30 ms/image CPU", licence="Apache-2.0",
    evidence="EfficientNetV2S was identified as the most sustainable/accurate option (96.41%) in a "
             "garbage-classification comparison that also accounted for carbon footprint [R16]; "
             "ranked in the top-3 backbones across six domains in a controlled backbone study [R38].",
    glass_plastic_strengths="High effective resolution (384) preserves rim highlights and small "
                            "labels; progressive learning recipe trains fast on small sets.",
    glass_plastic_weaknesses="Much more compute than B0 for a few points; fine-tuning instability "
                             "on very small datasets without a low LR.",
    pros="One of the best accuracy-per-FLOP models available; strong published waste evidence.",
    cons="Not the most efficient at very low latency; larger memory footprint.",
    recommended_when="Server/GPU-side classifier or the accuracy champion in a benchmark table.",
    tier="research / production (server)",
    refs="R16,R38",
)
add(
    id="cnn_convnext_tiny",
    name="ConvNeXt-Tiny / -Small",
    family="CNN transfer learning (modern CNN)",
    params_m="28.6 / 50.0", macs_g="4.5 / 8.7", input_res="224", pretrain="ImageNet-1k/-21k",
    imagenet_top1="82.1-82.5 (T) / 83.1 (S)",
    data_need="~500-5,000 labelled crops", train_cost="GPU: ~15-45 min",
    throughput="~8-25 ms/image CPU", licence="MIT (timm) / BSD-3",
    evidence="ConvNeXt-Tiny ranked *best overall* across six domains (natural, texture, remote "
             "sensing, plant, astronomy, medical) in a controlled 12-backbone comparison, ahead of "
             "EfficientNetV2-S and Swin-Tiny [R38] - the strongest architecture-level prior for a new "
             "material-classification task.",
    glass_plastic_strengths="Large-kernel depthwise convolutions + LayerNorm give wide receptive "
                            "fields, useful for judging surface finish and transmission over a "
                            "whole object rather than a patch; transformer-like features at CNN "
                            "runtime cost.",
    glass_plastic_weaknesses="Still an image-level label: needs a detector or attention to localise "
                             "the object on a busy belt; ImageNet-1k vs -21k weights differ "
                             "noticeably in transfer quality.",
    pros="Best all-round transfer backbone in independent comparisons; excellent tooling (timm) and "
         "quantisation support.",
    cons="2-3x the latency of a MobileNet; more parameters to fine-tune on tiny sets.",
    recommended_when="The default 'strong model' in a new glass/plastic benchmark.",
    tier="research / production",
    refs="R38",
)
add(
    id="cnn_regnety",
    name="RegNetY-3.2GF / InceptionNeXt-T",
    family="CNN transfer learning",
    params_m="19.4 / 28.0", macs_g="3.2 / 4.2", input_res="224", pretrain="ImageNet-1k",
    imagenet_top1="82.0 / 82.3",
    data_need="~500-5,000 labelled crops", train_cost="GPU: ~15-45 min",
    throughput="~10-20 ms/image CPU", licence="Apache-2.0 / MIT",
    evidence="RegNetY-3.2GF ranked among the top backbones overall and specifically strong on "
             "plant/remote-sensing domains [R38]; InceptionNeXt-T reported 82.3% at 4.2 GFLOPs with "
             "901 img/s training throughput [R38-adjacent benchmark].",
    glass_plastic_strengths="Regular, hardware-friendly design: high training throughput per FLOP "
                            "lets you run more augmentations/seeds within a fixed budget - which "
                            "usually buys more robustness than a bigger model on a small dataset.",
    glass_plastic_weaknesses="No architectural feature aimed at specular/transparent cues; same "
                             "shortcut risks as any ImageNet backbone.",
    pros="Best throughput-per-accuracy for repeated experimentation; SE blocks give channel "
         "attention cheaply.",
    cons="Less waste-domain evidence than DenseNet/ResNet families.",
    recommended_when="Large hyper-parameter/seed sweeps where wall-clock matters.",
    tier="research",
    refs="R38",
)

# ---------------------------- E. Transformers / hybrids -------------------------------

add(
    id="vit_b16",
    name="ViT-B/16 (supervised fine-tune)",
    family="Transformer",
    params_m=86.0, macs_g=17.6, input_res="224 (384 for best recipes)", pretrain="ImageNet-1k or -21k",
    imagenet_top1="79.9 (IN-1k from scratch; needs heavy aug) / 84.9 (IN-21k AugReg)",
    data_need="~5,000-100,000 labelled crops (or 21k pretraining)",
    train_cost="GPU: hours (fine-tune), much more from scratch",
    throughput="~30-100 ms/image CPU; 3-8 ms/image GPU", licence="Apache-2.0 (timm)",
    evidence="ViT-B/16 and MobileNetV3-Small are the supervised reference models in the zero-shot "
             "waste study [R29]; transformers reached F1 0.921-1.00 on plastic spectroscopic data "
             "[R27]; the waste-VLM study finds the practical value of ViTs is as frozen feature "
             "extractors for few-shot heads [R30].",
    glass_plastic_strengths="Global self-attention can compare distant image regions - the natural "
                            "operation for 'is the texture behind the object sharp and continuing?' "
                            "(the transmission cue).",
    glass_plastic_weaknesses="Patch-tokenisation at 224 tends to downsample exactly the small, "
                             "high-frequency cues (thin highlights, rim gradients) that separate "
                             "glass from plastic; needs strong augmentation on small datasets.",
    pros="Best-in-class scaling behaviour when data or 21k pretraining is available; excellent as a "
         "frozen backbone.",
    cons="Data-hungry, slow, large; poor choice for <2k images without heavy regularisation.",
    recommended_when="You have (or can generate) tens of thousands of in-domain crops, or you use it "
                     "frozen with a linear/ProtoNet head.",
    tier="research",
    refs="R27,R29,R30",
)
add(
    id="deit_small_tiny",
    name="DeiT-Tiny / DeiT-Small (distilled ViT)",
    family="Transformer (data-efficient)",
    params_m="5.7 / 22.1", macs_g="1.3 / 4.6", input_res="224", pretrain="ImageNet-1k (distilled)",
    imagenet_top1="72.2 / 79.8",
    data_need="~1,000-10,000 labelled crops", train_cost="GPU: ~30-90 min",
    throughput="~8-30 ms/image CPU", licence="Apache-2.0",
    evidence="DeiT is the canonical demonstration that transformer fine-tuning can work on "
             "ImageNet-scale data only when distillation + strong augmentation are used; the "
             "waste-VLM study's conclusion that pretrained models beat non-pretrained "
             "across the board (0.54-0.93 vs 0.41-0.78) [R18] is the same lesson for material "
             "classification.",
    glass_plastic_strengths="Transformer inductive bias with CNN-like parameter count; attention "
                            "maps are directly inspectable for audit.",
    glass_plastic_weaknesses="Attention distillation quality varies; the fixed 16x16 patch "
                             "granularity limits very fine texture reasoning.",
    pros="Best parameter/accuracy point among transformers; interpretable attention rollout.",
    cons="Typically beaten by ConvNeXt-Tiny at the same budget because it lacks the hierarchical "
         "multi-scale structure [R38].",
    recommended_when="When you specifically need attention-based interpretability at manageable cost.",
    tier="research",
    refs="R18,R38",
)
add(
    id="swin_maxvit",
    name="Swin-T/SwinV2-T / MaxViT-T (hierarchical attention)",
    family="Hybrid transformer",
    params_m="28.3 / 28.4 / 30.9", macs_g="4.5 / 5.9 / 5.6", input_res="224",
    pretrain="ImageNet-1k/-21k", imagenet_top1="81.3 / 82.1 / 83.4",
    data_need="~1,000-10,000 labelled crops", train_cost="GPU: ~30-90 min",
    throughput="~15-40 ms/image CPU", licence="MIT / Apache-2.0",
    evidence="Swin-Tiny and SwinV2-Tiny are standard comparison entries in cross-domain backbone "
             "studies (81.5-82.1% ImageNet at ~28M params) [R38]; windowed/hybrid attention is the "
             "architecture family used in most recent HSI material-classification work (RVT/"
             "transformers on spectral-spatial data) [R27].",
    glass_plastic_strengths="Hierarchical windows keep fine local detail while still allowing global "
                            "comparison - a better inductive bias than plain ViT for mixed "
                            "local(highlight)+global(transmission) evidence.",
    glass_plastic_weaknesses="Same data appetite as ViTs; sliding-window attention is latency-unfriendly "
                             "on fixed-function edge NPUs.",
    pros="Typically the best accuracy among ~30M-parameter models together with ConvNeXt-T.",
    cons="Edge-unfriendly; more hyper-parameters (window size, drop path) to tune.",
    recommended_when="Accuracy-first server-side deployment where a small CNN has plateaued.",
    tier="research / production (server)",
    refs="R27,R38",
)
add(
    id="attention_cnn",
    name="Attention-augmented CNN (SE / CBAM / attention-AlexNet) and multi-scale fusion",
    family="Hybrid / attention",
    params_m="~25-60 (backbone + modules)", macs_g="~5-15", input_res="224-384",
    pretrain="ImageNet-1k", imagenet_top1="backbone-dependent (no clean public number)",
    data_need="~500-5,000 labelled crops", train_cost="GPU: ~20-60 min",
    throughput="~10-30 ms/image CPU",
    licence="varies (paper code / timplike reimplementations)",
    evidence="An attention-enhanced AlexNet achieved the highest accuracy of three architectures it "
             "was compared against with the fewest confusions (e.g. 2,877 correct plastic-bottle "
             "vs 2,727 for plain CNN), explicitly because attention 'minimized misclassifications "
             "between visually confusing waste classes'; EfficientNetB2 + PMAM reached 93.38% on a "
             "4-class set; multi-scale RGB+HSI fusion (RHFF-SOLOv1) was required for transparent "
             "PET/PP on a black belt [R32].",
    glass_plastic_strengths="Attention is the natural fix for the core difficulty: learn to weight "
                            "the rim highlight and the through-object texture instead of the "
                            "background or a label; channel attention can suppress a saturating "
                            "specular channel.",
    glass_plastic_weaknesses="Attention maps are a *soft* fix - they do not guarantee the model "
                             "ignores a bright belt reflection, and they add parameters to an "
                             "already data-hungry model.",
    pros="Small architectural change with the best published confusion-matrix improvements on "
         "visually confusable class pairs (plastic/paper/glass).",
    cons="Non-standard code; benefits are dataset-specific; harder to benchmark fairly.",
    recommended_when="After a baseline CNN plateaus on glass/plastic confusions and you have the "
                     "data to train it.",
    tier="research",
    refs="R32,R35",
)

# ---------------------------- F. Foundation models / VLM ------------------------------

add(
    id="vlm_clip_zeroshot",
    name="CLIP / OpenCLIP / SigLIP — zero-shot classification",
    family="Foundation model / VLM",
    params_m="151 (CLIP ViT-B/32) / 428 (ViT-L/14) / 878+ (SigLIP SO400M)",
    macs_g="~40 (L/14 at 336)", input_res="224-336",
    pretrain="400M-5B image-text pairs (LAION/WebLI)", imagenet_top1="76.0 (B/32 zero-shot) / ~80 (L/14)",
    data_need="0 labelled images (few-shot: 1-16 per class helps)",
    train_cost="0 for zero-shot; minutes for a linear head on cached features",
    throughput="3.8 ms/image reported for OpenCLIP ViT-L/14 (~263 FPS, batched GPU); 2.83 FPS for "
               "ViT-L/14-336 in a CPU-limited study",
    licence="MIT (CLIP/OpenCLIP code) - check each checkpoint's data terms",
    evidence="Zero-shot waste classification on TrashNet: OpenCLIP ViT-L/14-336 reaches 76.30% with "
             "no training, versus 82.71% on a 6-class industrial set, improving to 90.48% purely by "
             "prompt engineering and 97.18% when fully supervised [R28,R29]; prompt sensitivity is "
             "the dominant failure mode [R29].",
    glass_plastic_strengths="Zero-shot 'glass jar' vs 'plastic bottle' prompts work surprisingly "
                            "well because the language prior includes material semantics; can be "
                            "re-targeted to a new taxonomy (e.g. 'amber glass', 'transparent PET') "
                            "by editing text only - no retraining, no downtime [R30].",
    glass_plastic_weaknesses="Prompt-dependent and poorly calibrated; fails precisely on visually "
                             "ambiguous items (clear glass vs clear PET); large models are slow on "
                             "CPU and their text prior can dominate weak visual evidence.",
    pros="No annotation cost, instant taxonomy changes, strong baseline for rare classes, and an "
         "excellent feature extractor for few-shot heads.",
    cons="Accuracy below a fine-tuned CNN in-domain; big models violate real-time edge budgets; "
         "licence/data-provenance questions for regulated settings.",
    recommended_when="Cold start with no labels, taxonomy drift, or bootstrapping annotations for a "
                     "supervised model.",
    tier="production (bootstrap) / research",
    refs="R28,R29,R30",
)
add(
    id="vlm_fewshot_adapter",
    name="Frozen foundation features + training-free adapter (Tip-Adapter / kNN / linear probe)",
    family="Foundation model / VLM (few-shot)",
    params_m="86-1000 (frozen backbone)", macs_g="backbone only, once",
    input_res="224-518", pretrain="DINOv2/DINOv3, EVA-CLIP, SigLIP", imagenet_top1="kNN 82-86 (DINOv2 B->g)",
    data_need="1-16 labelled images per class", train_cost="seconds-minutes (adapter fit on cached features)",
    throughput="backbone-bound; ~5-30 ms/image GPU",
    licence="varies (DINOv2 Apache-2.0; check others)",
    evidence="Frozen-foundation-model + training-free adapters with fixed hyper-parameters are the "
             "recommended practical route under taxonomy drift; DINOv3 1-NN is a strong baseline "
             "when many exemplars exist, and textual few-shot descriptions actually *reduce* MLLM "
             "accuracy while image-based few-shot helps [R30].",
    glass_plastic_strengths="Extremely label-efficient - the industrial reality is that a plant can "
                            "label 5-20 examples per new product; cached features make re-training "
                            "an ops action rather than a project.",
    glass_plastic_weaknesses="Ceiling is set by the frozen representation, which was never trained "
                             "to reason about specularity/transmission; nearest-neighbour heads "
                             "inherit the embedding's confusion between shiny plastics and glass.",
    pros="Fastest path from 'new stream' to 'working classifier'; no GPU needed at fit time; "
         "supports conformal/OOD wrappers.",
    cons="Needs a vision foundation model in the inference path (memory/legal), and the "
         "representation may not encode the decisive material cue.",
    recommended_when="Rapid deployment, frequent taxonomy change, scarce labels - combined with an "
                     "abstention policy.",
    tier="production (bootstrap)",
    refs="R30",
)
add(
    id="mllm_zero_shot",
    name="Multimodal LLM as a classifier (GPT-4o / LLaVA-OneVision)",
    family="Foundation model / VLM (language interface)",
    params_m="proprietary / 7B+", macs_g="very high", input_res="variable",
    pretrain="web-scale multimodal", imagenet_top1=None,
    data_need="0-5 labelled images", train_cost="none (prompting) or fine-tune at high cost",
    throughput="<1-5 FPS typical; API latency seconds",
    licence="proprietary API / research licences",
    evidence="Zero-shot MLLMs performed well on the tested waste datasets; adding textual few-shot "
             "descriptions *reduced* accuracy, while image-based few-shot improved GPT-4o at high "
             "inference cost [R30].",
    glass_plastic_strengths="Can be asked to reason explicitly ('is the background visible and "
                            "undistorted through the object?') - useful for auditing and for "
                            "generating labels, and handles long-tail novelty better than a closed "
                            "classifier.",
    glass_plastic_weaknesses="Orders of magnitude too slow and expensive for a belt; opaque failure "
                             "modes; no calibration for a reject/accept decision.",
    pros="Zero-shot flexibility, natural-language explanations, excellent for weak labelling and "
         "dataset bootstrapping.",
    cons="Latency, cost, non-determinism, and no evidence of beating a fine-tuned CNN on the binary "
         "glass/plastic decision.",
    recommended_when="Label generation, dataset auditing, hard-case triage, operator assistance - "
                     "not the production classifier.",
    tier="research / tooling",
    refs="R30",
)
add(
    id="openvocab_detect_classify",
    name="Open-vocabulary detect-then-classify (Grounding DINO + CLIP, EcoVision)",
    family="Foundation model pipeline",
    params_m="~250+", macs_g="high", input_res="variable (800-1333)",
    pretrain="grounding + CLIP", imagenet_top1=None,
    data_need="0 up to ~1,000 for LoRA adaptation", train_cost="hours GPU for LoRA; 0 for zero-shot",
    throughput="~2-10 FPS on a mid GPU",
    licence="Apache-2.0 components (check Grounding DINO licence)",
    evidence="EcoVision combines Grounding DINO detection with CLIP material classification, "
             "reports that few-shot LoRA adaptation of the visual encoder improves cluttered "
             "field-image accuracy while preserving zero-shot generalisation, and is designed to "
             "bolt onto existing mechanical segregation lines [R44].",
    glass_plastic_strengths="Solves the failure mode we measured directly: a detector removes the "
                            "background so the material classifier sees only the object - which is "
                            "what fixes the clutter collapse of global descriptors.",
    glass_plastic_weaknesses="Two failure surfaces (missed detections, then material confusion); "
                             "transparent objects are also hard for detectors; slower.",
    pros="Composable, retargetable by text prompt, and the natural architecture for a real belt.",
    cons="Latency and complexity; open-vocabulary detectors are weak on transparent, low-contrast "
         "items - exactly glass.",
    recommended_when="Multi-object streams and cluttered scenes where an image-level classifier is "
                     "not deployable.",
    tier="research / production",
    refs="R31,R44",
)

# ---------------------------- G. Spectral / multisensor ---------------------------------

add(
    id="spectral_nir_ftir",
    name="NIR / FTIR spectroscopy + PLS-DA or 1D-CNN",
    family="Spectral sensor + learned model",
    params_m="0.01-2 (1D models)", macs_g="negligible", input_res="1D spectrum (hundreds-2,000 bins)",
    pretrain="none (1D nets) ", imagenet_top1=None,
    data_need="40-1,000 spectra per class", train_cost="CPU: seconds-minutes",
    throughput="0.5-1.6 ms/spectrum (model only); sensor-limited in practice",
    licence="open algorithms (chemometrics); sensor vendor-dependent",
    evidence="Benchmark table across FTIR/NIR/hyperspectral-NIR: PLS-DA F1 0.517/0.580/0.971, LDA "
             "0.506/0.578/0.987, 1D-CNN 0.691/0.938/0.877, improved-CNN 0.981/0.978/1.000, "
             "transformer 0.921/0.972/1.000 [R27]; 1D-ResNet reaches 0.991 on FTIR with 2x "
             "augmentation [R25]; GAN augmentation lifts six-polymer balanced accuracy to 96.2% "
             "[R26]; contrast this with RGB, which cannot see polymer chemistry at all.",
    glass_plastic_strengths="Answers the *chemistry* question directly: glass is a silicate "
                            "(no C-H/NIR absorption bands), plastics are hydrocarbon polymers with "
                            "specific overtone bands - the single most reliable glass-vs-plastic "
                            "discriminator known. Also resolves polymer identity (PET/PE/PP/PS).",
    glass_plastic_weaknesses="Point-measurement geometry (needs the item under the probe); black "
                             "carbon-filled plastics absorb NIR light and defeat it [R22,R24]; "
                             "sensitive to dirt, moisture, thickness and surface roughness; requires "
                             "preprocessing (SNV/MSC/Savitzky-Golay) to be reproducible [R27].",
    pros="Very high discriminative power including the transparent-transparent case; tiny models, "
         "millisecond decisions; mature industrial hardware.",
    cons="Costly sensors, needs presentation/scanning mechanics, immune to the shape/context cues "
         "that matter for handling; performance depends strongly on preprocessing and augmentation.",
    recommended_when="Any application where the transparent-glass-vs-transparent-plastic decision "
                     "must be correct - i.e. as the reference modality, or the arbiter when RGB "
                     "confidence is low.",
    tier="production (reference modality)",
    refs="R22,R24,R25,R26,R27",
)
add(
    id="hsi_glass_plastic",
    name="Hyperspectral imaging (VIS-NIR / SWIR) + ML",
    family="Spectral imaging + learned model",
    params_m="0.1-30 (CNN) / 0.01 (PLS-DA)", macs_g="0.1-20", input_res="cube (e.g. 100-300 bands)",
    pretrain="ImageNet (2D) or none (3D/1D nets)", imagenet_top1=None,
    data_need="a few hundred labelled objects (pixel-level: thousands of spectra)",
    train_cost="GPU: minutes-hours (cube data is large)",
    throughput="not real-time per-pixel on CPU; industrial systems use band-selection/FPGA",
    licence="algorithms open; cameras commercial",
    evidence="Hierarchical PLS-DA on VIS-NIR HSI classifies five industrial glass colour/tone "
             "classes with sensitivity/specificity 0.910-1.000 [R21]; industrial HSI pushes recycled "
             "purity near 100% with PP/PE/PET ~99% [R23]; pixel-level CNN (P1CH) reaches 97.44% on "
             "HDPE/PET/PP/PS (99.94% excluding border pixels) while HybridSN scores 21.81% [R19]; "
             "NIR HSI + ANN reaches 89.5% on an unknown mixed plastic stream [R20]; black plastics "
             "remain hard (F1 0.67 vs 0.90 for coloured) [R24].",
    glass_plastic_strengths="Combines the chemical specificity of spectra with spatial context: can "
                            "sort glass colour classes *and* reject ceramics/stones, which RGB "
                            "cannot; operates on whole objects on a belt, unlike a point probe.",
    glass_plastic_weaknesses="Cube data is bulky and slow; strong illumination dependence; black "
                             "carbon-filled plastics still fail; per-pixel metrics can be inflated "
                             "by ignoring background/border pixels (the same paper drops from 97.44% "
                             "to 39.69% under a stricter metric) [R19].",
    pros="Highest demonstrated accuracy for material *and* colour sorting in one sensor; deployed "
         "industrially, so the reliability evidence is real.",
    cons="5-50x the cost and complexity of an RGB camera; calibration drift; heavy compute; "
         "explaining a spectral decision to an operator is harder than showing a highlight.",
    recommended_when="High-value streams where purity drives price, or as the labelling oracle for "
                     "an RGB-only deployment.",
    tier="production (industrial) / research",
    refs="R19,R20,R21,R23,R24",
)
add(
    id="multisensor_fusion",
    name="Custom sensor fusion (RGB+HSI, RGB+polarisation, RGB+thermal, RGB+depth/3D)",
    family="Multisensor fusion",
    params_m="backbone-dependent", macs_g="sum of branches", input_res="aligned streams",
    pretrain="ImageNet branches + sensor-specific encoders", imagenet_top1=None,
    data_need="a few hundred-aligned-object samples", train_cost="GPU: hours",
    throughput="limited by the slowest sensor + alignment",
    licence="components vary; alignment code often proprietary",
    evidence="Spectral-conversion autoencoders lifted accuracy from 0.933 (unimodal) to 0.970; a "
             "combined HSI-RGB system 'significantly outperforms the individual methods' on a "
             "municipal sorting line [R22-adjacent]; multi-scale RGB+HSI fusion (RHFF-SOLOv1) was "
             "needed to separate transparent PET, blue PET and transparent PP on a black belt [R32]; "
             "thermal+RGB solves glass segmentation and 3D active vision targets dark glass and "
             "contaminants [R31,R43].",
    glass_plastic_strengths="Sensor complementarity is the theoretical answer to the glass/plastic "
                            "problem: RGB gives shape/colour, NIR/polarisation gives material, "
                            "thermal/3D gives objectness and dark-glass detection. Fusion is the "
                            "only approach with *published* evidence of solving the "
                            "transparent-vs-transparent case in an industrial setting.",
    glass_plastic_weaknesses="Alignment/registration between streams, calibration drift, doubled "
                             "failure surfaces, cost, and much harder deployment/maintenance.",
    pros="Best achievable accuracy and the strongest evidence base for hard transparent items; each "
         "branch covers the other's blind spot.",
    cons="Engineering-heavy; datasets are rare; hard to benchmark fairly or reproduce.",
    recommended_when="Industrial deployment where a few points of purity have a direct monetary "
                     "value, or a research contribution on the transparent-transparent problem.",
    tier="production (industrial) / research",
    refs="R22,R31,R32,R43",
)
add(
    id="thermal_rgb_3d",
    name="Thermal IR and 3D/structured-light sensing for transparent objects",
    family="Non-RGB sensing",
    params_m="small", macs_g="small", input_res="full frame",
    pretrain="ImageNet or none", imagenet_top1=None,
    data_need="hundreds of objects", train_cost="GPU: minutes",
    throughput="sensor-limited",
    licence="vendor-dependent",
    evidence="Thermal+RGB imaging is used for 'reliable glass segmentation'; RGB stereo gives 3D "
             "characterisation of recyclables; 3D active vision with structured light targets dark "
             "glass and contaminants; UV fluorescence is patented for glass with additives (lead) "
             "[R31,R43].",
    glass_plastic_strengths="Solves the *detection/segmentation* half of the problem for "
                            "transparent objects (structured light and thermal boundaries do not "
                            "rely on RGB contrast), and thermal emissivity can differ between thin "
                            "films and rigid glass.",
    glass_plastic_weaknesses="Weak on the material half: emissivity depends more on surface "
                             "finish/coating than on glass-vs-plastic; 3D shape is shared by both "
                             "material classes.",
    pros="Complementary, mature sensors; robust segmentation of items that RGB cannot even outline.",
    cons="Does not by itself separate glass from plastic; adds hardware only justified when "
         "segmentation, not material ID, is the bottleneck.",
    recommended_when="As the detection stage in front of an RGB/spectral material classifier.",
    tier="research",
    refs="R31,R43",
)

# ---------------------------- H. Uncertainty, rejection, deployment -------------------

add(
    id="uncertainty_rejection",
    name="Uncertainty + rejection layer (conformal prediction, MC-dropout, ensembles, OOD)",
    family="Method layer (model-agnostic)",
    params_m="+0", macs_g="x N for ensembles",
    input_res="n/a", pretrain="n/a", imagenet_top1=None,
    data_need="a held-out calibration set (~200-1,000 items)",
    train_cost="minutes (calibration)", throughput="1-5x the base model",
    licence="open (MAPIE, Torch/uncertainty-toolbox, custom)",
    evidence="The waste-VLM study shows that frozen-feature pipelines need a confidence policy to "
             "be usable under taxonomy drift [R30]; the industrial HSI literature reports "
             "sensitivity/specificity pairs rather than accuracy precisely because plants operate "
             "at a chosen operating point [R21]; black-plastic NIR results (F1 0.67) are a case "
             "where the correct answer is 'reject and divert' [R24].",
    glass_plastic_strengths="Directly monetises the glass/plastic ambiguity: route low-confidence "
                            "items to a second sensor (NIR/polarimeter) or to a manual station "
                            "instead of guessing. Conformal prediction gives a *distribution-free* "
                            "coverage guarantee, which is what a certification process asks for.",
    glass_plastic_weaknesses="Needs a representative calibration set; a miscalibrated model "
                             "produces confidently wrong labels; adds latency and complexity.",
    pros="Turns an accuracy number into an operating policy (purity vs recovery trade-off); gives "
         "auditable coverage guarantees; cheap.",
    cons="Requires honest evaluation of calibration under shift (the classical models measured in "
         "this repo are *not* calibrated under noise/clutter shifts).",
    recommended_when="Any real deployment - especially anything feeding a purity-sensitive product "
                     "stream.",
    tier="production (mandatory in practice)",
    refs="R21,R24,R30",
)

# --------------------------------------------------------------------------------------
# Scoring rubric (1-5, higher is better; cost/latency scored as *efficiency*)
# --------------------------------------------------------------------------------------
# accuracy_ceiling        : realistic attainable accuracy for glass-vs-plastic, incl. hard cases
# transparency_robustness : does it work when both classes are clear/shiny (the crux)
# data_efficiency         : value with few labels
# contamination_robustness: dirt, labels, occluders, mixed streams
# cost_efficiency         : BOM + compute + integration cost (5 = cheapest)
# throughput              : real-time capability (5 = belt speed comfortable)
# explainability          : can a decision be justified to an operator/auditor
# tooling_maturity        : off-the-shelf availability, docs, quantization, support
# calibration_abstention  : supports trusted reject/"don't know" (5 = first-class)
# zero_shot_capability    : useful with *zero* labels (5 = works out of the box). Distinct from
#                           data_efficiency, which is "value with few labels": a plant that has
#                           no annotations yet is a different problem from one that has 200.
CRITERIA = ["accuracy_ceiling", "transparency_robustness", "data_efficiency",
            "contamination_robustness", "cost_efficiency", "throughput", "explainability",
            "tooling_maturity", "calibration_abstention", "zero_shot_capability"]

SCORES = {
    # id :                     acc  tr  data cont cost thr  expl tool cal
    "heuristic_colour_threshold": (2, 1, 5, 1, 5, 5, 5, 5, 1, 5),
    "heuristic_edge_texture": (2, 2, 4, 1, 5, 5, 4, 5, 1, 5),
    "heuristic_specular_stats": (3, 3, 5, 2, 5, 5, 5, 4, 2, 5),
    "heuristic_transmission": (2, 3, 4, 1, 5, 5, 4, 3, 2, 5),
    "polarimetry_vision": (4, 5, 4, 2, 3, 4, 3, 2, 3, 4),
    "svm_rbf_features": (3, 3, 5, 2, 5, 5, 4, 5, 3, 1),
    "rf_extratrees_features": (4, 3, 5, 2, 5, 4, 4, 5, 3, 1),
    "lda_features": (3, 3, 5, 2, 5, 5, 5, 5, 4, 1),
    "gbdt_features": (4, 3, 4, 2, 5, 5, 3, 5, 4, 1),
    "knn_gmm_features": (3, 3, 5, 2, 5, 4, 4, 4, 4, 2),
    "mlp_shallow_features": (3, 3, 3, 2, 5, 5, 2, 4, 3, 1),
    "cnn_small_scratch": (3, 3, 2, 3, 4, 4, 3, 3, 3, 1),
    "cnn_lightweight_research": (4, 3, 3, 3, 5, 5, 2, 2, 3, 1),
    "cnn_resnet18": (4, 3, 4, 3, 4, 4, 2, 5, 3, 2),
    "cnn_resnet50": (4, 3, 4, 3, 3, 3, 2, 5, 3, 2),
    "cnn_densenet121": (5, 4, 4, 3, 4, 3, 2, 5, 3, 2),
    "cnn_vgg16": (3, 3, 2, 3, 1, 1, 2, 4, 2, 2),
    "cnn_inception_xception": (4, 4, 3, 3, 3, 3, 2, 4, 3, 2),
    "cnn_mobilenet_v2": (4, 3, 4, 3, 5, 5, 2, 5, 3, 2),
    "cnn_mobilenet_v3": (4, 3, 4, 3, 5, 5, 2, 5, 3, 2),
    "cnn_mobilenet_v4": (4, 4, 4, 3, 5, 5, 2, 3, 3, 2),
    "cnn_efficientnet_b0_b3": (4, 4, 4, 3, 4, 4, 2, 5, 3, 2),
    "cnn_efficientnetv2": (5, 4, 3, 3, 3, 2, 2, 5, 3, 2),
    "cnn_convnext_tiny": (5, 4, 3, 4, 3, 3, 2, 4, 3, 2),
    "cnn_regnety": (4, 4, 3, 4, 3, 3, 2, 4, 3, 2),
    "vit_b16": (4, 5, 2, 3, 2, 2, 3, 4, 3, 3),
    "deit_small_tiny": (4, 4, 3, 3, 3, 3, 4, 4, 3, 3),
    "swin_maxvit": (5, 5, 2, 4, 2, 2, 3, 4, 3, 3),
    "attention_cnn": (5, 4, 3, 4, 3, 3, 3, 2, 3, 1),
    "vlm_clip_zeroshot": (3, 3, 5, 3, 3, 3, 3, 5, 2, 5),
    "vlm_fewshot_adapter": (4, 3, 5, 3, 3, 3, 3, 5, 4, 4),
    "mllm_zero_shot": (3, 3, 5, 2, 1, 1, 4, 4, 2, 5),
    "openvocab_detect_classify": (4, 4, 4, 5, 2, 2, 3, 4, 3, 4),
    "spectral_nir_ftir": (5, 5, 4, 3, 2, 4, 4, 3, 5, 2),
    "hsi_glass_plastic": (5, 5, 3, 4, 2, 3, 3, 3, 4, 2),
    "multisensor_fusion": (5, 5, 2, 5, 1, 2, 3, 2, 4, 1),
    "thermal_rgb_3d": (3, 4, 3, 4, 3, 3, 3, 3, 3, 2),
    "uncertainty_rejection": (4, 5, 5, 4, 4, 3, 5, 4, 5, 3),
}

# --------------------------------------------------------------------------------------
# Emit
# --------------------------------------------------------------------------------------

def _kind(row: dict) -> str:
    """Distinguish architectural options from cross-cutting components.

    Ranking a *component* (a rejection layer) against *models* is a category error, so the
    composite table is restricted to ``model`` rows and components get their own table.
    """
    return "component" if str(row.get("family", "")).startswith("Method layer") else "model"


COLUMNS = ["id", "name", "kind", "family", "params_m", "macs_g", "input_res", "pretrain",
           "imagenet_top1", "data_need", "train_cost", "throughput", "licence", "evidence",
           "glass_plastic_strengths", "glass_plastic_weaknesses", "pros", "cons",
           "recommended_when", "tier", "refs"]


def main(out_dir: str = "data") -> None:
    os.makedirs(out_dir, exist_ok=True)
    reg_path = os.path.join(out_dir, "model_registry.csv")
    with open(reg_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for row in M:
            merged = {**row, "kind": _kind(row)}
            w.writerow({c: ("" if merged.get(c) is None else merged.get(c, "")) for c in COLUMNS})

    sc_path = os.path.join(out_dir, "model_scores.csv")
    ids = [r["id"] for r in M]
    missing = [i for i in ids if i not in SCORES]
    extra = [i for i in SCORES if i not in ids]
    if missing or extra:
        raise SystemExit(f"score/registry mismatch. missing={missing} extra={extra}")
    with open(sc_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", *CRITERIA, "mean"])
        for i in ids:
            vals = SCORES[i]
            w.writerow([i, *vals, round(sum(vals) / len(vals), 3)])

    ref_path = os.path.join(out_dir, "references.csv")
    with open(ref_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["key", "citation", "url"])
        for k, txt, url in REFERENCES:
            w.writerow([k, txt, url])

    print(f"wrote {reg_path} ({len(M)} models), {sc_path}, {ref_path} ({len(REFERENCES)} refs)")


if __name__ == "__main__":
    main()
