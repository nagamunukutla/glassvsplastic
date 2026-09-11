# References

Numbering is stable and used by `data/model_registry.csv` (column `refs`) and by every table in this documentation. Entries marked *[dataset]* are the public datasets; everything else is a method, benchmark or review.


**[R1]** Thung & Yang — TrashNet: a 2,527-image, 6-class waste dataset captured on a white backdrop (cardboard, glass, metal, paper, plastic, trash), Stanford CS229 project, 2016.  
https://github.com/garythung/trashnet


**[R2]** Single, Iranmanesh & Javadi — RealWaste: a novel real-life data set for landfill waste classification using deep learning. Information 14(12):633, 2023. 4,752 images at 524x524, 9 classes (glass 378, plastic 831). Explicitly motivated by transparency and by 'similarities between specific objects (e.g. glass and plastic bottles)'.  
https://www.mdpi.com/2078-2489/14/12/633


**[R3]** Bashkirova et al. — ZeroWaste dataset. 26,766 images (ZeroWaste-f 4,503 fully annotated; s 6,212 unlabelled; w 1,410 weakly labelled), 4 material classes.  
https://github.com/Trash-AI/ZeroWaste


**[R4]** Proenca & Simoes — TACO: trash annotations in context. 4,617 images, 60 litter categories, COCO-style, severe class imbalance.  
https://arxiv.org/abs/2003.05664


**[R5]** GlobalWasteData (2026) — consolidated table of 20+ waste image datasets with sample counts and class counts; documents that most public datasets are small, imbalanced and captured in a single setting.  
https://arxiv.org/html/2602.07463v1


**[R6]** AgaMiko — waste-datasets-review: curated catalogue of waste image datasets with licences and download links.  
https://github.com/AgaMiko/waste-datasets-review


**[R7]** Bui et al. — A novel framework for trash classification using deep transfer learning (DNN-TC). 94% on TrashNet, 98% on VN-trash; compares DenseNet121 (91%), RecycleNet (68%) and ResNet (72%) under identical splits.  
https://www.researchgate.net/publication/344650744


**[R8]** Aral, Keskin et al. — Classification of TrashNet dataset based on deep learning models (DenseNet121/169, InceptionResNetV2, MobileNet, Xception). Best: DenseNet121 ~95% with Adam + augmentation.  
https://www.semanticscholar.org/paper/f5a380760b91393ad05bfc2063434f76935a428e


**[R9]** Focus-RCNet with knowledge distillation — 0.525M parameters, 92% on TrashNet; argues mobile-grade models are sufficient for recyclable sorting.  
https://www.sciencedirect.com/science/article/pii/S0950705125000760


**[R10]** RecycleNet — 3M parameters, 81% on TrashNet (vs 7M for the reference model); earliest explicit accuracy/parameter trade-off in waste classification.  
https://www.sciencedirect.com/science/article/pii/S0950705125000760


**[R11]** WasNet — lightweight architecture: 96.10% on TrashNet, 82.5% on Huawei garbage classification, 64.5% on ImageNet.  
https://www.sciencedirect.com/science/article/pii/S0950705125000760


**[R12]** Optimised DenseNet121 — 99.60% on TrashNet (6 classes, 2,527 images) reported in a 2025 survey table of state-of-the-art waste classifiers; illustrates how saturated TrashNet has become.  
https://www.sciencedirect.com/science/article/pii/S0950705125000760


**[R13]** DP-CNN-En-ELM / TriCascade (2025) — hierarchical waste classification: 96% (2 classes), 91% (9 classes), 85.25% (36 classes). Direct evidence that accuracy decays sharply with taxonomy depth.  
https://www.sciencedirect.com/science/article/pii/S0950705125000760


**[R14]** Lin et al. — RWNet (ResNet variants) on TrashNet: RWNet-101 89.9%; the authors report that the models sort most recyclables well 'except plastic' (ROC > 0.9).  
https://pmc.ncbi.nlm.nih.gov/articles/PMC11096226/


**[R15]** Multi-objective beluga-whale-optimised InceptionV3 — 92.62% in 0.63 s on TrashNet; beats MobileNetV2, VGG16 and AlexNet under the same protocol. Includes an explicit class-imbalance discussion.  
https://pmc.ncbi.nlm.nih.gov/articles/PMC11096226/


**[R16]** Sustainability study of garbage classification models — EfficientNetV2S 96.41% identified as the most sustainable/accurate trade-off; ResNet50 > ResNet110 in accuracy and IoU but larger carbon footprint.  
https://www.researchgate.net/publication/370110735


**[R17]** Zhang et al. / ACS Anal. Chem. (2025) — hyperspectral microplastic shape classification: EfficientNet_b7, Inception_v3 and MobileNet_v3 all reach 98% on augmented data, with MobileNet reaching that accuracy in the least training time and smallest size.  
https://pubs.acs.org/doi/10.1021/acs.analchem.5c02683


**[R18]** Deep learning shape classification for hyperspectral-imaged microplastics (2025) — MobileNet best (0.93 validation, 1.00 test on augmented refined data); CNNs (0.41-0.78) beat plain NNs (0.34-0.69); pretrained > from-scratch.  
https://pmc.ncbi.nlm.nih.gov/articles/PMC12489888/


**[R19]** P1CH (2025) — pixel-level hyperspectral material classification of HDPE/PET/PP/PS: 97.44% overall (99.94% when border pixels are excluded), vs 21.81% for HybridSN; fails badly on black PS (undetected); 39.69% when background is excluded from the metric.  
https://arxiv.org/html/2409.13498v2


**[R20]** Hyperspectral NIR (900-1700 nm) classification of post-consumer thermoplastics with a 2-layer ANN — 89.5% on an unknown mixed plastic waste stream; entropy/contrast-stretching segmentation.  
https://www.sciencedirect.com/science/article/abs/pii/S0957582023008935


**[R21]** Bonifazi et al. — Hyperspectral imaging + hierarchical PLS-DA for colour classification of glass fragments in recycling (VIS-NIR 400-1000 nm, 5 industry colour classes): sensitivity and specificity 0.910-1.000.  
https://www.mdpi.com/2313-4321/11/3/43


**[R22]** Review of hyperspectral imaging-based plastic waste detection (2023) — NIR fails on carbon-black plastics because carbon absorbs across the UV-IR range; among conventional classifiers ResNet-50 performed best (>90%).  
https://www.researchgate.net/publication/369146831


**[R23]** Konica Minolta Sensing (2025) — industrial HSI practice: newest hyperspectral cameras push recycled-material purity close to 100%; PP/PE/PET near 99% purity; black plastics remain the hard case (Specim FX50 sorted ABS/PE/PS in a lab test).  
https://sensing.konicaminolta.us/us/blog/hyperspectral-imaging-shaping-the-future-of-plastic-recycling/


**[R24]** N-BEATS waveform-decomposition ensemble on 1,491 hyperspectral images of ~4,500 black and coloured plastic pieces from an industrial line — F1 0.79 overall, 0.90 coloured, 0.67 black; the authors note black-polymer separation in NIR was 'deemed unfeasible' in most prior literature.  
https://www.researchgate.net/publication/374118822


**[R25]** Analyst (2025) — data augmentation and classification algorithms on plastic spectroscopy: 1D-ResNet reaches 0.991 accuracy on FTIR with 2x augmentation; SVM and RF are the most stable on small samples while deep models dominate on large samples; 1D input formats beat 2D.  
https://pubs.rsc.org/en/content/articlehtml/2025/ay/d4ay01759e


**[R26]** Analyst (2026) — GAN-augmented spectra for recycled plastics: 96.2% balanced accuracy across six polymers at the optimal synthetic ratio; single-measurement baselines are highly class-dependent (PET 100%, PP 0%).  
https://pubs.rsc.org/en/content/articlehtml/2026/an/d5an01042j


**[R27]** Deep learning-based plastic classification using spectroscopic data (2025) — benchmark table across FTIR / NIR / hyperspectral-NIR: PLS-DA F1 0.517-0.971, LDA 0.506-0.987, 1D-CNN 0.691-0.938, ANN 0.79-0.99, transformer 0.92-1.00.  
https://www.sciencedirect.com/science/article/pii/S0959652625021432


**[R28]** Novelis Research Lab (2025) — vision-language models for waste recognition: OpenCLIP ViT-L/14-2B 82.71% zero-shot, 90.48% after prompt engineering, 97.18% fully supervised, 3.79 ms/image (~263 FPS) on 14,310 images / 6 classes.  
https://novelis.io/research-lab/a-comparative-analysis-of-vision-language-models-for-scalable-waste-recognition/


**[R29]** Recycling (2025) 10(4):144 — zero-shot learning for municipal waste classification on TrashNet with OpenCLIP/OWL-ViT: ViT-L/14-336 reaches 76.30% zero-shot (427.94M params, 2.83 FPS); supervised models still lead; prompt sensitivity is the main failure mode.  
https://www.mdpi.com/2313-4321/10/4/144


**[R30]** Funk et al., Cleaner Waste Systems 13 (2026) 100475 — zero/few-shot evaluation of VLMs and multimodal LLMs across TrashNet, FPWaste, RealWaste and MultiWaste under taxonomy drift: frozen-foundation-model + training-free adapters (Tip-Adapter) are the practical route; textual few-shot descriptions *reduced* MLLM accuracy; image-based few-shot improved GPT-4o at high inference cost.  
https://publica-rest.fraunhofer.de/server/api/core/bitstreams/4367c274-e326-413e-b00c-d57bfc6b11b7/content


**[R31]** Computer vision for glass waste: technologies and sensors (2025) — survey of sensors for glass sorting: polarimetry exploits the dielectric polarisation signature to separate glass from *transparent plastics*; thermal+RGB and structured-light systems for dark glass; states plainly that 'a sheet of transparent plastic may appear almost indistinguishable from glass' to RGB vision alone.  
https://pmc.ncbi.nlm.nih.gov/articles/PMC12609803/


**[R32]** Multi-scale RGB+hyperspectral feature fusion (RHFF-SOLOv1, 2022) — transparent PET, blue PET and transparent PP bottles on a black conveyor belt; neither RGB alone (colour ambiguity) nor spectra alone (no colour discrimination) suffices, fusion does.  
https://pmc.ncbi.nlm.nih.gov/articles/PMC9609436/


**[R33]** TrashBox + federated deep learning (Sci Rep 14, 2024) — 7 waste classes; ResNeXt-101 confusion matrix shows the largest residual error between visually similar classes (cardboard->paper), with glass among the better-separated classes in that setting.  
https://www.nature.com/articles/s41598-024-62003-4


**[R34]** PLOS ONE (2025) — enhanced deep CNN waste framework over Waste Classification V2, TrashNet and OpenRecycle: '14 Glass samples were predicted as Plastic'; AUC is 1.0 for all TrashNet classes except glass and trash.  
https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0324294


**[R35]** ResSeMo (2026) — ResNeXt+SENet+MobileNetV3 integration for waste classification; the fine-grained TrashNet confusion matrix is dominated by plastic/paper/glass confusions.  
https://pubmed.ncbi.nlm.nih.gov/41888559/


**[R36]** Enhancing glass defect detection with diffusion models (2025) — synthetic minority-class images lift ResNet50V2 from 78% to 93% accuracy and precision from 0.53 to 1.00 on an imbalanced glass-manufacturing set; EfficientNetB0 and MobileNetV2 gain less.  
https://arxiv.org/html/2505.03134v1


**[R37]** Evaluation of modern ML approaches for optic plastics sorting (2025) — 9,440 web-scraped images of 7 plastic types: ResNet-34 transfer learning 96% on the balanced web set, but only 71.8% on a 2,721-image harder set (F1 71.1%), with PS/HDPE largely wrong due to class imbalance.  
https://arxiv.org/html/2505.16513v1


**[R38]** Which backbone to use (2025) — controlled comparison of 12 backbones across 6 domains: ConvNeXt-Tiny ranks best overall (28.6M params, ImageNet 82.52%), RegNetY-3.2GF and EfficientNetV2-S next; the authors warn against choosing backbones by pre-training accuracy alone.  
https://arxiv.org/html/2406.05612v1


**[R39]** Qin et al. — MobileNetV4: universal models for the mobile ecosystem, ECCV 2024. MNv4-Conv-S: 3.8M params, 0.2G MACs, 73.8% ImageNet top-1 at 2.4 ms on a Pixel 6 CPU; distilled MNv4-Hybrid-L: 87.0% top-1 at 3.8 ms on a Pixel 8 EdgeTPU.  
https://arxiv.org/abs/2404.10518


**[R40]** timm model changelog — training-recipe effects on the same architecture: resnet50d.ra4_e3600 81.8% top-1 @288 (vs 76.1% for the classic recipe) and efficientnet_b0.ra4_e3600 79.4% @224; a reminder that reported ImageNet numbers are recipe-dependent, not architecture-only.  
https://huggingface.co/docs/timm/changes


**[R41]** Kimura et al. — Dynamic visual cues for differentiating mirror and glass, Scientific Reports 8:8410, 2018. Humans need motion (motion transparency from the rear surface) to separate refractive glass from purely reflective surfaces; static cues are much weaker. Relevant because a single RGB frame removes the strongest human cue.  
https://www.nature.com/articles/s41598-018-26720-x


**[R42]** Frontiers in Environmental Science (2025) — review of ML for microplastic detection: tabulated advantages/challenges of manual counting, semi-automatic CV, CNNs and mobile-based detection (data hunger and compute cost being the recurring drawbacks).  
https://www.frontiersin.org/journals/environmental-science/articles/10.3389/fenvs.2025.1573579/full


**[R43]** MDPI/ACS-style comparisons of glass-selective sensors and 3D active vision (in [R31]): structured light and HSI for dark glass and contaminants; RGB stereo for recyclable characterisation in households.  
https://pmc.ncbi.nlm.nih.gov/articles/PMC12609803/


**[R45]** A transfer learning approach for efficient classification of waste materials — VGG16 96.00%, MobileNetV2 95.51% and a 6-layer baseline CNN 90.61% on the same 2-class waste data: the canonical demonstration that transfer learning beats from-scratch at small sample sizes.  
https://www.researchgate.net/publication/370110735


**[R44]** EcoVision (2026) — Grounding DINO detection + CLIP material classification with LoRA adaptation of the visual encoder, evaluated on TrashNet/TACO and field images; argues for zero-shot material heads behind a supervised detector in landfill pipelines.  
https://harbinengineeringjournal.com/index.php/journal/article/view/5151



*Reference entries were compiled from the sources listed at these URLs; where a number is quoted in `docs/06_model_comparison.md` it comes from the corresponding source, not from this repository's own experiments.*
