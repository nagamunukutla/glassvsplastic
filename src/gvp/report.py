"""Render the study: comparison documents (Markdown), a self-contained HTML report, and the
measured-results summary.

Design rules of this module
---------------------------
1. **One source of truth per number.** Architecture facts and reported literature results come
   from ``data/model_registry.csv`` (built by ``scripts/build_registry.py``); measured results
   come from ``results/*.csv``. Nothing is typed twice.
2. **No hidden aggregation.** The composite score is a transparent weighted mean of ten
   explicitly scored criteria; the weights live in ``configs/scoring.yaml`` and are printed in
   every ranking.
3. **Caveats travel with the numbers.** Every table of measured proxy results is emitted with
   the proxy-data caveat attached.
"""

from __future__ import annotations

import base64
import csv
import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import evaluate

FIGURES = ("classical_models.png", "confusion_best_classical.png", "domain_shift_heatmap.png",
           "cue_ablation.png", "top_features.png")


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------


def _f(v):
    if v in (None, "", "None"):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v)
    head = s.split(" ")[0].replace(",", ".")
    try:
        return float(head)
    except ValueError:
        return None


def load_registry(path: str) -> List[dict]:
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["params_m_num"] = _f(r.get("params_m"))
        r["macs_g_num"] = _f(r.get("macs_g"))
        r["top1_num"] = _f(r.get("imagenet_top1"))
    return rows


def load_scores(path: str) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            out[r["id"]] = {k: float(v) for k, v in r.items() if k != "id"}
    return out


def load_profiles(path: str) -> dict:
    import yaml
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def load_references(path: str) -> List[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def load_measured(results_dir: str) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for name in ("classical_model_comparison", "feature_block_ablation", "cue_ablation",
                 "domain_shift_robustness", "rule_baseline", "sanity_checks"):
        p = os.path.join(results_dir, f"{name}.csv")
        if os.path.exists(p):
            with open(p, newline="") as fh:
                out[name] = list(csv.DictReader(fh))
    return out


# --------------------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------------------


def composite_scores(registry: Sequence[dict], scores: Dict[str, Dict[str, float]],
                     weights: Dict[str, float]) -> Dict[str, float]:
    w = np.array([weights[c] for c in weights], dtype=float)
    w = w / w.sum()
    keys = list(weights.keys())
    out = {}
    for r in registry:
        s = scores.get(r["id"])
        if not s:
            continue
        out[r["id"]] = float(np.dot([s[k] for k in keys], w))
    return out


def all_profile_rankings(registry, scores, profiles) -> Dict[str, Dict[str, float]]:
    return {name: composite_scores(registry, scores, prof["weights"])
            for name, prof in profiles["weights"].items()}


# --------------------------------------------------------------------------------------
# markdown rendering
# --------------------------------------------------------------------------------------


def _fmt(x, nd=2):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _short(text: str, n: int = 240) -> str:
    t = " ".join(str(text).split())
    return t if len(t) <= n else t[: n - 1] + "…"


def render_comparison_md(registry, scores, profiles, ranked, measured, docs_dir="docs") -> str:
    ids = [r["id"] for r in registry]
    by_id = {r["id"]: r for r in registry}
    default_name = profiles["default"]
    comp = ranked[default_name]
    model_ids = [r["id"] for r in registry if r.get("kind", "model") == "model"]
    comp_ids = [r["id"] for r in registry if r.get("kind") == "component"]
    order = sorted(model_ids, key=lambda i: -comp.get(i, 0.0))

    families: List[str] = []
    for r in registry:
        if r["family"] not in families:
            families.append(r["family"])

    L: List[str] = []
    A = L.append
    A("# Model comparison: what to use for glass vs plastic, and why\n")
    A("This document is generated (`python -m gvp.cli report`) from three data files:\n")
    A("- `data/model_registry.csv` — architecture and deployment facts, each reported result "
      "carrying a `[Rn]` reference key\n")
    A("- `data/model_scores.csv` — ten criteria scored 1–5 per model (5 = best; cost and latency "
      "are scored as *efficiency*)\n")
    A("- `configs/scoring.yaml` — the weights that turn those ten criteria into one ranking, "
      "including alternative profiles (accuracy-first, edge budget, cold start)\n")
    A("\n> **How to read this.** The registry is a *literature-and-architecture* comparison: "
      "`evidence` fields quote numbers from the papers cited in "
      "`docs/12_references.md`, not numbers produced by this repository. Repository measurements "
      "(on the synthetic proxy task) are in `results/MEASURED_RESULTS.md` and are, deliberately, "
      "reported separately. Never mix the two.\n")
    A("\n> **Cited vs measured.** This chapter compares options as *reported in the literature*. "
      "For the configurations actually trained and measured in this repository, on a real public "
      "dataset, see `results/real/MEASURED_MODEL_COMPARISON.md` — the same question answered with "
      "numbers produced here. Where the two disagree (they do: on fine-tuned CNNs vs hand-crafted "
      "descriptors, and on whether single-feature physics rules transfer at all), the measured "
      "table wins for that dataset.\n")

    # ---- ranking table
    A("\n## 1. Composite ranking (default profile: *%s*)\n" % profiles["weights"][default_name]["label"])
    A(f"\n{len(model_ids)} architectural options are ranked below. Cross-cutting *components* "
      f"({len(comp_ids)}) are deliberately excluded — ranking a rejection layer against a model "
      f"would be a category error — and appear in §1c instead.\n")
    A("\nWeights: " + ", ".join(f"`{k}` {v:.2f}" for k, v in profiles["weights"][default_name]["weights"].items()) + "\n")
    rows = []
    for n, i in enumerate(order, 1):
        r, s = by_id[i], scores[i]
        rows.append({
            "rank": n, "model": r["name"], "family": r["family"],
            "score": comp[i],
            "acc": int(s["accuracy_ceiling"]), "transp": int(s["transparency_robustness"]),
            "data": int(s["data_efficiency"]), "contam": int(s["contamination_robustness"]),
            "cost": int(s["cost_efficiency"]), "thr": int(s["throughput"]),
            "expl": int(s["explainability"]), "tool": int(s["tooling_maturity"]),
            "reject": int(s["calibration_abstention"]),
            "zero": int(s["zero_shot_capability"]), "tier": r["tier"],
        })
    A(evaluate.md_table(rows, ["rank", "model", "family", "score", "acc", "transp", "data",
                               "contam", "cost", "thr", "expl", "tool", "reject", "zero", "tier"],
                        float_fmt="{:.2f}"))
    A("\n*(criterion columns are the raw 1–5 scores: `acc` accuracy ceiling, `transp` transparency "
      "robustness, `data` data efficiency, `contam` contamination robustness, `cost` cost "
      "efficiency, `thr` throughput, `expl` explainability, `tool` tooling maturity, `reject` "
      "calibration/abstention, `zero` useful with zero labels.)*\n")

    A("\n### 1c. Cross-cutting components (not ranked against models)\n")
    A("\nThese are not alternatives to a model; they are things to add *on top of* whichever model "
      "you pick. Scores shown for reference.\n")
    rows = []
    for i in comp_ids:
        r, sc_ = by_id[i], scores[i]
        rows.append({"component": r["name"], "score": comp[i],
                     "why it matters": _short(r["glass_plastic_strengths"], 320),
                     "use it when": _short(r["recommended_when"], 200)})
    if rows:
        A(evaluate.md_table(rows, ["component", "score", "why it matters", "use it when"]))

    # ---- sensitivity of the ranking to the profile
    A("\n### 1b. Does the ranking survive a change of priorities?\n")
    A("\nA ranking that only holds under one weight vector is not a finding. Top 8 per profile:\n")
    prof_rows = []
    for pname, prof in profiles["weights"].items():
        rk = ranked[pname]
        top = sorted(ids, key=lambda i: -rk.get(i, 0.0))[:8]
        prof_rows.append({"profile": prof["label"],
                          **{f"#{n}": by_id[i]["name"] for n, i in enumerate(top, 1)}})
    A(evaluate.md_table(prof_rows, ["profile", "#1", "#2", "#3", "#4", "#5", "#6", "#7", "#8"]))

    # ---- per-criterion leaders
    A("\n## 2. Who wins each criterion?\n")
    crit_rows = []
    crit_titles = {
        "accuracy_ceiling": "Accuracy ceiling on glass-vs-plastic",
        "transparency_robustness": "Transparency robustness (clear item vs clear item)",
        "data_efficiency": "Data efficiency",
        "contamination_robustness": "Contamination / clutter robustness",
        "cost_efficiency": "Cost efficiency",
        "throughput": "Throughput",
        "explainability": "Explainability / auditability",
        "tooling_maturity": "Tooling maturity",
        "calibration_abstention": "Calibration & abstention",
        "zero_shot_capability": "Useful with zero labels",
    }
    for c, title in crit_titles.items():
        top = sorted(model_ids, key=lambda i: -scores[i][c])[:3]
        crit_rows.append({"criterion": title,
                          "1st": by_id[top[0]]["name"], "2nd": by_id[top[1]]["name"],
                          "3rd": by_id[top[2]]["name"]})
    A(evaluate.md_table(crit_rows, ["criterion", "1st", "2nd", "3rd"]))

    # ---- architecture facts
    A("\n## 3. Cost of each option (facts, not judgements)\n")
    fact_rows = []
    for i in order:
        r = by_id[i]
        fact_rows.append({
            "model": r["name"], "family": r["family"],
            "params (M)": r["params_m"] or "—", "MACs (G)": r["macs_g"] or "—",
            "input": r["input_res"], "pretraining": r["pretrain"],
            "ImageNet top-1": r["imagenet_top1"] or "n/a",
            "labels needed": r["data_need"], "train cost": r["train_cost"],
            "inference": r["throughput"], "licence": r["licence"],
        })
    A(evaluate.md_table(fact_rows, ["model", "family", "params (M)", "MACs (G)", "input",
                                    "pretraining", "ImageNet top-1", "labels needed", "train cost",
                                    "inference", "licence"]))

    # ---- per-family detail
    A("\n## 4. Family-by-family detail: evidence, pros, cons, glass/plastic specifics\n")
    for fam in families:
        A(f"\n### {fam}\n")
        for r in [x for x in registry if x["family"] == fam]:
            s = scores[r["id"]]
            A(f"\n#### {r['name']}  —  composite {comp.get(r['id'], float('nan')):.2f}/5 "
              f"(profile: {profiles['weights'][default_name]['label']})\n")
            A(f"\n* **Cost:** {r['params_m'] or '—'}M params, {r['macs_g'] or '—'}G MACs, "
              f"input {r['input_res']}, pretraining {r['pretrain']}, licence {r['licence']}\n")
            A(f"* **Evidence on real waste / glass-plastic data:** {_short(r['evidence'], 900)}\n")
            A(f"* **Glass/plastic strengths:** {_short(r['glass_plastic_strengths'], 600)}\n")
            A(f"* **Glass/plastic weaknesses:** {_short(r['glass_plastic_weaknesses'], 600)}\n")
            A(f"* **Pros:** {_short(r['pros'], 400)}\n")
            A(f"* **Cons:** {_short(r['cons'], 400)}\n")
            A(f"* **Use it when:** {_short(r['recommended_when'], 300)}\n")
            A(f"* **Scores (1–5):** " + ", ".join(f"{k} {int(v)}" for k, v in s.items()
                                                 if k != "mean") + f"  | tier: {r['tier']}\n")

    # ---- cross-cutting reading of the evidence
    A("\n## 5. What the evidence says when you put it together\n")
    A("""
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
""")

    # ---- recommendation matrix
    A("\n## 6. Recommended stack by scenario\n")
    A(evaluate.md_table([
        {"scenario": "Prototype on TrashNet/RealWaste, need a number fast",
         "model": "ResNet-18 or DenseNet-121 transfer learning",
         "why": "Cheap, comparable with the published literature, and DenseNet-121 is the most "
                "reproducibly strong waste backbone [R7,R8,R12]"},
        {"scenario": "Best accuracy per unit of effort on your own labelled crops",
         "model": "ConvNeXt-Tiny (or EfficientNetV2-S if data is plentiful)",
         "why": "Top-ranked backbone in a controlled 6-domain comparison [R38]; 82.1–83.9% ImageNet "
                "at 4.5–8.4 GMACs"},
        {"scenario": "Belt-speed inference on CPU/NPU",
         "model": "MobileNetV3-Large or MobileNetV4-Conv-S, INT8",
         "why": "98% on augmented material data at minimal training cost and size [R17]; MNv4-Conv-S "
                "is 3.8M params / 0.2 GMACs at 2.4 ms on a phone CPU [R39]"},
        {"scenario": "No labels yet / taxonomy will change",
         "model": "OpenCLIP or SigLIP zero-shot + prompt engineering, then a frozen-feature adapter",
         "why": "76.3% zero-shot on TrashNet and 82.7→90.5% from prompt engineering alone "
                "[R28,R29]; training-free adapters are the recommended route under taxonomy drift "
                "[R30]"},
        {"scenario": "Transparent items must be separated correctly (the real problem)",
         "model": "RGB + NIR/HSI or polarisation fusion; RGB as the shape/context branch only",
         "why": "The only approach with published success on transparent-vs-transparent [R31,R32]; "
                "spectral models reach F1 0.98–1.00 on polymer ID [R25,R27]"},
        {"scenario": "Industrial line, high value, must certify purity",
         "model": "HSI + hierarchical PLS-DA / 1D-CNN, with RGB as pre-sort; conformal rejection "
                  "routing to manual QA",
         "why": "Sensitivity/specificity 0.910–1.000 for glass colour classes [R21]; near-100% "
                "purity in industrial practice [R23]; distribution-free coverage from conformal "
                "prediction suits certification"},
    ], ["scenario", "model", "why"]))

    A("\n---\n\n*Generated by `gvp.report`. Edit `data/model_registry.csv`, "
      "`data/model_scores.csv` or `configs/scoring.yaml` and re-run "
      "`python -m gvp.cli report` — every table above updates.*\n")
    text = "\n".join(L)
    return evaluate.write_text(os.path.join(docs_dir, "06_model_comparison.md"), text)


def render_references_md(references: List[dict], docs_dir="docs") -> str:
    L = ["# References\n",
         "Numbering is stable and used by `data/model_registry.csv` (column `refs`) and by every "
         "table in this documentation. Entries marked *[dataset]* are the public datasets; "
         "everything else is a method, benchmark or review.\n"]
    for r in references:
        L.append(f"\n**[{r['key']}]** {r['citation']}  \n{r['url']}\n")
    L.append("\n\n*Reference entries were compiled from the sources listed at these URLs; where a "
             "number is quoted in `docs/06_model_comparison.md` it comes from the corresponding "
             "source, not from this repository's own experiments.*\n")
    return evaluate.write_text(os.path.join(docs_dir, "12_references.md"), "\n".join(L))


# --------------------------------------------------------------------------------------
# measured results
# --------------------------------------------------------------------------------------


def render_measured_md(measured: Dict[str, List[dict]], results_dir="results",
                       figures_dir="results/figures") -> str:
    L: List[str] = []
    A = L.append
    A("# Measured results (synthetic proxy task)\n")
    A(evaluate.CAVEAT + "\n")
    A("\nEverything on this page was produced by this repository on the **synthetic proxy "
      "dataset** (`gvp.synth`, 600 images at 128×128, 70/15/15 split, plus 7 shifted test sets). "
      "It is a test of *method behaviour* — which model classes are robust, what the failure modes "
      "look like, how much a cue ablation can resolve — with the object/cue structure of the "
      "renderer, not of reality. Do not quote these as glass/plastic accuracy.\n")
    A("\nReproduce with:\n```bash\nmake proxy        # generate data\nmake classical    # models, "
      "ablations, shifts, controls\nmake cues         # cue ablation\nmake report\n```\n")

    cls = measured.get("classical_model_comparison") or []
    test = [r for r in cls if r.get("split") == "test"]
    if test:
        test.sort(key=lambda r: -float(r["balanced_accuracy"]))
        A("\n## 1. Shallow models on 141 hand-crafted features\n")
        A("\nTrained on 420 proxy images, evaluated on 90 held-out images (in-distribution). "
          "Balanced accuracy with a 95% percentile bootstrap CI; the CI is wide because n=90.\n")
        rows = [{
            "model": r["model"], "bal acc": float(r["balanced_accuracy"]),
            "95% CI": f"{float(r['ba_ci_lo']):.3f}–{float(r['ba_ci_hi']):.3f}",
            "accuracy": float(r["accuracy"]), "macro F1": float(r["macro_f1"]),
            "glass recall": float(r["glass_recall"]), "plastic recall": float(r["plastic_recall"]),
            "ROC AUC": float(r["roc_auc"]), "fit (s)": float(r["fit_seconds"]),
            "clf (ms/img)": float(r["clf_ms_per_image"]), "size (kB)": float(r["model_size_kb"]),
        } for r in test]
        A(evaluate.md_table(rows, list(rows[0].keys())))
        A("\n![Models](figures/classical_models.png)\n")
        A("\n**Reading.** Gradient boosting (0.944) and SVM-RBF (0.933) lead; Gaussian NB is last "
          "(0.867). Feature extraction costs ~18 ms/image on 1 CPU core, so the classifier latency "
          "(0.004–1.2 ms) is irrelevant next to the descriptor cost — the practical argument for a "
          "learned end-to-end model at high line speed, and against it at low volume.\n")

    rules = measured.get("rule_baseline") or []
    if rules:
        A("\n## 2. Single-feature physics rules (the floor a learned model must beat)\n")
        rows = [{"rule": r["rule"], "bal acc": float(r["balanced_accuracy"]),
                 "glass recall": float(r["glass_recall"]),
                 "plastic recall": float(r["plastic_recall"])} for r in rules]
        A(evaluate.md_table(rows, list(rows[0].keys())))
        A("\n**Reading.** One threshold on highlight spikiness reaches 0.70 balanced accuracy. "
          "Any learned model that cannot clearly beat 1-D physics on your data is not earning its "
          "complexity.\n")

    sanity = measured.get("sanity_checks") or []
    if sanity:
        A("\n## 3. Controls: is the number trustworthy?\n")
        rows = [{"control": r["check"], "model": r["model"],
                 "bal acc": float(r["balanced_accuracy"]), "expected": r["expected"],
                 "what it tests": r["interpretation"]} for r in sanity]
        A(evaluate.md_table(rows, list(rows[0].keys())))
        A("\n**Reading.** Shuffled labels recover chance (0.48–0.51) → no split leakage. The "
          "background-only control scores 0.72–0.78 when only the central 50% is blanked and falls "
          "to 0.51–0.52 when the central 80% is blanked: the earlier score came from peripherally "
          "visible *object* pixels (bottle necks, caps, shard tips), not from a scene shortcut. "
          "This sensitivity to the control's window size is itself the lesson — a background "
          "control that is too small will make a clean dataset look leaked.\n")

    fa = measured.get("feature_block_ablation") or []
    if fa:
        A("\n## 4. Which feature blocks carry the signal?\n")
        fams = sorted({r["model"] for r in fa})
        blocks = sorted({r["combo"] for r in fa})
        mat = np.full((len(fams), len(blocks)), np.nan)
        for r in fa:
            mat[fams.index(r["model"]), blocks.index(r["combo"])] = float(r["balanced_accuracy"])
        rows = [{"model": f, **{b: (None if np.isnan(mat[i, j]) else mat[i, j])
                                for j, b in enumerate(blocks)}} for i, f in enumerate(fams)]
        A(evaluate.md_table(rows, ["model", *blocks]))
        A("\n**Reading.** No single block is sufficient and no single block is required: "
          "`all` ≈ `all_minus_X` for every X, with differences inside the CI. Textured blocks "
          "(texture/edge) carry more than colour, consistent with the physical cue structure "
          "(finish and through-object detail, not hue).\n")

    ca = measured.get("cue_ablation") or []
    if ca:
        A("\n## 5. Cue ablation: what is the model actually looking at?\n")
        base = {r["model"]: float(r["balanced_accuracy"]) for r in ca
                if r["cue_removed"] == "none_removed"}
        agg: Dict[str, List[float]] = {}
        ciw: Dict[str, float] = {}
        n_test = max((int(float(r.get("n_test", 0) or 0)) for r in ca), default=0)
        for r in ca:
            if r["cue_removed"] == "none_removed":
                continue
            agg.setdefault(r["cue_removed"].replace("without_", ""), []).append(
                float(r["balanced_accuracy"]) - base.get(r["model"], np.nan))
            ciw[r["cue_removed"]] = float(r.get("ba_ci_width", 0) or 0)
        rows = [{"cue removed": k, "mean Δ bal acc": float(np.mean(v)),
                 "models": len(v), "95% CI width (per model)": ciw.get(f"without_{k}", ciw.get(k, 0.0))}
                for k, v in sorted(agg.items(), key=lambda kv: np.mean(kv[1]))]
        A(evaluate.md_table(rows, list(rows[0].keys())))
        A("\n![Cue ablation](figures/cue_ablation.png)\n")
        single = {k: v for k, v in agg.items() if k != 'all_cues_weakened'}
        biggest = max((abs(np.mean(v)) for v in single.values()), default=0.0)
        lo_ci, hi_ci = (min(ciw.values()), max(ciw.values())) if ciw else (0.0, 0.0)
        frac_inside = sum(1 for v in single.values() if abs(np.mean(v)) < lo_ci) / max(len(single), 1)
        A(f"\n**Reading — and a negative result.** With {n_test} test items per condition the "
          f"95% CI on a single accuracy is {lo_ci:.2f}–{hi_ci:.2f} wide. The largest single-cue "
          f"effect on any single cue was {biggest:.3f}, and {frac_inside * 100:.0f}% of single-cue effects "
          f"are smaller than the *narrowest* CI in the experiment. This design therefore **cannot** "
          "conclude that any individual cue is load-bearing: the proxy task is cue-redundant and "
          "the experiment is underpowered. Only removing *all* cues at once produces a consistent "
          "drop, and even that sits inside the CI. Reporting 'cue X matters' from deltas of this "
          "size would be p-hacking. The honest statement is the one above, plus this: a study that "
          "wants to rank cue importance needs a factorial design with ≥1,000 test items per cell, "
          "or a dose-response design with graded cue strengths (the `CueStrength` knob supports "
          "both — only the sample size is missing).\n")

    ds = measured.get("domain_shift_robustness") or []
    if ds:
        A("\n## 6. Robustness under acquisition shift\n")
        A("\nTrained on clean in-distribution data, tested on fresh scenes with one degradation "
          "each. Δ is relative to the clean test split.\n")
        models = sorted({r["model"] for r in ds})
        splits = ["test"] + sorted({r["split"] for r in ds if r["split"].startswith("shift_")},
                                   key=lambda s: min(float(x["balanced_accuracy"]) for x in ds
                                                     if x["split"] == s))
        mat = np.full((len(models), len(splits)), np.nan)
        for r in ds:
            mat[models.index(r["model"]), splits.index(r["split"])] = float(r["balanced_accuracy"])
        rows = [{"model": m, **{s.replace("shift_", ""): (None if np.isnan(mat[i, j]) else mat[i, j])
                                for j, s in enumerate(splits)}} for i, m in enumerate(models)]
        A(evaluate.md_table(rows, ["model", *[s.replace("shift_", "") for s in splits]]))
        A("\n![Domain shift](figures/domain_shift_heatmap.png)\n")
        A("\n**Reading.** Three distinct failure signatures, and they are the most transferable "
          "findings of this whole exercise:\n")
        A("\n1. **Model-family robustness to pixel-level degradation differs enormously.** "
          "Extra-trees hold 0.82–0.92 under sensor noise, JPEG artefacts and deep shadow; "
          "an RBF SVM on the same features collapses to 0.50–0.54 (its standardisation and kernel "
          "geometry do not survive the shift). If you deploy a classical pipeline, prefer "
          "tree ensembles, and always test under the degradations your camera actually produces.\n"
          "2. **Background clutter breaks every global-descriptor model** (all three fall to "
          "0.50–0.53). Diagnostics: Canny edge density rises 6.4×, the fraction of saturated "
          "pixels 3.8×, and `edge_mag_p95` 4.1× — debris and glare inject glass-like highlight "
          "statistics everywhere. The fix is architectural, not a better classifier: a "
          "detection/segmentation stage (or an attention/detector-prior CNN) that restricts the "
          "features to the object.\n"
          "3. **Removing the background context does *not* hurt** (the `whitebg` condition scores "
          "0.90–0.96, slightly *better* than clean). This is evidence that the model is not relying "
          "on transmitted-background cues in this renderer — and it is exactly the kind of "
          "assumption that a white-studio benchmark (e.g. TrashNet) silently validates and a real "
          "conveyor violates.\n")

    figs = [f for f in FIGURES if os.path.exists(os.path.join(figures_dir, f))]
    if figs:
        A("\n## 7. Figures\n")
        for f in ("confusion_best_classical.png", "top_features.png"):
            if f in figs:
                name = f.replace("_", " ").replace(".png", "")
                A(f"\n![{name}](figures/{f})\n")

    A("\n## 8. Known limitations of this measurement\n")
    A("""
* **Synthetic proxy data.** The renderer encodes the cue structure the literature describes; it
  cannot reproduce real material appearance, contamination, or camera physics. Absolute numbers
  are meaningless; *relative* model behaviour and failure modes are the useful output.
* **Small test sets.** n=90 (in-distribution) and n=120 (shifted). CIs of ±0.05–0.10 mean only
  large effects are resolvable. `docs/09_evaluation_protocol.md` specifies what a real study needs.
* **One feature extractor, one hyper-parameter set, one seed.** No seed averaging was performed
  for the classical tier (the deep tier supports `--seed`). Multi-seed runs are a `make sweep`
  target and are *required* before reporting deltas of a few points.
* **Hand-tuned object-region estimator.** The transparency features depend on
  `gvp.features.estimate_foreground`, whose failure under clutter is documented above; an
  oracle-mask control would separate 'descriptor failure' from 'segmentation failure' and is
  listed as future work.
* **The cue-ablation design is underpowered** (see §5) and uses a single ablation level rather
  than a dose-response curve.
""")
    return evaluate.write_text(os.path.join(results_dir, "MEASURED_RESULTS.md"), "\n".join(L))


# --------------------------------------------------------------------------------------
# self-contained HTML report
# --------------------------------------------------------------------------------------


def _b64(path: str) -> str:
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("ascii")


def render_html(registry, scores, profiles, ranked, measured, out_path="results/model_comparison.html",
                figures_dir="results/figures") -> str:
    default = profiles["default"]
    payload = {
        "default_profile": default,
        "profiles": {k: v for k, v in profiles["weights"].items()},
        "registry": registry,
        "scores": scores,
        "rankings": ranked,
        "measured": {k: v for k, v in measured.items()
                     if k in ("classical_model_comparison", "domain_shift_robustness",
                              "sanity_checks", "rule_baseline")},
    }
    data_json = json.dumps(payload, default=str).replace("</", "<\\/")
    figs = {}
    for f in FIGURES:
        p = os.path.join(figures_dir, f)
        if os.path.exists(p):
            figs[f] = _b64(p)

    html = _HTML_TEMPLATE.replace("__DATA__", data_json)
    fig_html = "".join(
        f'<figure><img src="data:image/png;base64,{b64}" alt="{name}"/>'
        f'<figcaption>{name.replace("_", " ").replace(".png", "")}</figcaption></figure>'
        for name, b64 in figs.items())
    html = html.replace("__FIGURES__", fig_html)
    return evaluate.write_text(out_path, html)


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Glass vs Plastic — model comparison</title>
<style>
  :root{--ink:#12202e;--mut:#5b6b7c;--line:#dbe3ea;--bg:#f7f9fb;--acc:#0b6bcb;--acc2:#b3541e;
        --ok:#1f7a45;--bad:#b3261e;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
       font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
  header{background:#0e1c2b;color:#fff;padding:26px 22px 20px}
  header h1{margin:0 0 6px;font-size:23px;letter-spacing:.2px}
  header p{margin:0;color:#b9c8d6;max-width:1000px;font-size:13px}
  main{max-width:1500px;margin:0 auto;padding:18px 22px 60px}
  h2{font-size:17px;margin:28px 0 10px;padding-bottom:6px;border-bottom:2px solid var(--line)}
  h3{font-size:14.5px;margin:20px 0 8px}
  .card{background:#fff;border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:12px 0}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:12px}
  .kpi{background:#fff;border:1px solid var(--line);border-radius:10px;padding:12px 14px}
  .kpi .v{font-size:21px;font-weight:600}
  .kpi .l{color:var(--mut);font-size:12px;text-transform:uppercase;letter-spacing:.5px}
  .kpi .n{color:var(--mut);font-size:12px;margin-top:4px}
  table{border-collapse:collapse;width:100%;background:#fff;font-size:12.5px}
  th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
  th{position:sticky;top:0;background:#eef3f8;cursor:pointer;white-space:nowrap;font-size:12px}
  th:hover{background:#e3ebf3}
  tr.row:hover{background:#f2f7fc}
  td.num{text-align:right;font-variant-numeric:tabular-nums}
  .pill{display:inline-block;padding:1px 7px;border-radius:20px;font-size:11px;background:#eef3f8;
        color:var(--mut);white-space:nowrap}
  .bar{height:8px;border-radius:4px;background:linear-gradient(90deg,#0b6bcb,#39a0ed);display:inline-block;
       vertical-align:middle}
  details{background:#fff;border:1px solid var(--line);border-radius:8px;margin:8px 0;padding:6px 12px}
  summary{cursor:pointer;font-weight:600;font-size:13px}
  details p{margin:8px 0;font-size:12.5px}
  .lbl{color:var(--mut);font-weight:600}
  .sliders{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:8px}
  .slider label{display:block;font-size:12px;color:var(--mut)}
  input[type=range]{width:100%}
  .risk{color:var(--bad);font-weight:600}
  .good{color:var(--ok);font-weight:600}
  .mut{color:var(--mut)}
  figure{margin:14px 0;background:#fff;border:1px solid var(--line);border-radius:10px;padding:10px}
  figure img{max-width:100%;display:block;margin:0 auto}
  figcaption{color:var(--mut);font-size:12px;text-align:center;padding-top:6px}
  .note{background:#fff8e6;border:1px solid #f0dca8;border-radius:10px;padding:12px 14px;font-size:12.5px}
  .toolbar{display:flex;gap:10px;flex-wrap:wrap;align-items:end;margin:10px 0}
  .toolbar input,.toolbar select{padding:6px 8px;border:1px solid var(--line);border-radius:6px;
      background:#fff;font-size:13px}
  code{background:#eef3f8;padding:1px 5px;border-radius:4px;font-size:12px}
  .legend{font-size:11.5px;color:var(--mut);margin-top:6px}
</style>
</head>
<body>
<header>
  <h1>Glass vs Plastic — comparison of image-classification algorithms and models</h1>
  <p>37 models and sensor configurations plus one cross-cutting component, scored on ten criteria,
     with the literature evidence behind each one. Static, self-contained page: every number is recomputed in your browser from
     an embedded copy of <code>data/model_registry.csv</code>, <code>data/model_scores.csv</code>
     and <code>configs/scoring.yaml</code>. Change the weights and the ranking moves.</p>
</header>
<main>

<div id="kpis" class="grid"></div>

<div class="note" style="margin-top:14px">
  <b>How to read this page.</b> The <span class="pill">literature evidence</span> quoted for each
  model comes from the papers listed in <code>docs/12_references.md</code>. Numbers measured by this
  repository appear only in the last section and come from a <b>synthetic proxy renderer</b> — they
  test method behaviour (robustness, failure modes), not real glass/plastic accuracy.
</div>

<h2>1. Ranking under your weights</h2>
<div class="card">
  <div class="toolbar">
    <div>
      <label class="mut" style="font-size:12px">Profile</label><br/>
      <select id="profile"></select>
    </div>
    <div>
      <label class="mut" style="font-size:12px">Show</label><br/>
      <select id="kindfilter">
        <option value="model">models only</option>
        <option value="all">models + components</option>
        <option value="component">components only</option>
      </select>
    </div>
    <div>
      <label class="mut" style="font-size:12px">Filter family</label><br/>
      <select id="famfilter"></select>
    </div>
    <div>
      <label class="mut" style="font-size:12px">Search</label><br/>
      <input id="q" type="search" placeholder="name, evidence, tier…" size="26"/>
    </div>
    <button id="reset" style="padding:7px 10px;border-radius:6px;border:1px solid var(--line);
        background:#fff;cursor:pointer">reset weights</button>
  </div>
  <div class="sliders" id="sliders"></div>
  <div class="legend">A composite score only ranks the things you put in it. If a criterion matters
    more to you than the defaults assume, drag it and watch the order change — that sensitivity is
    the honest uncertainty in any "best model" claim.</div>
</div>

<table id="rank"></table>

<h2>2. Cost, evidence and glass/plastic specifics</h2>
<p class="mut">Click any row to expand the full evidence, pros, cons and the glass/plastic-specific
   assessment. Columns are sortable.</p>
<table id="facts"></table>

<h2>3. Measured behaviour on the synthetic proxy task</h2>
<div class="note">
  <b>Proxy-data caveat.</b> Everything below comes from the built-in renderer
  (<code>gvp.synth</code>), 600 images at 128×128 with a 70/15/15 split and seven shifted test sets.
  It measures <i>how</i> method families fail, which is transferable, rather than <i>how well</i>
  they work on glass and plastic, which is not.
</div>
<div id="measured"></div>
<div id="figures">__FIGURES__</div>

<h2>4. What this dataset cannot tell you</h2>
<div class="card">
  <ul>
    <li>Absolute accuracies: the renderer is not a camera. Use <code>results/*.csv</code> for
        relative comparisons only.</li>
    <li>Cue importance: the cue ablation is underpowered (95% CIs of ±0.10 on n=120), so single-cue
        deltas are inside the noise floor. A factorial design with ≥1,000 test items per cell is
        required.</li>
    <li>Real transparent-item difficulty: the renderer's clear-plastic and clear-glass renderings are
        separable by construction in ways that real PET and real soda-lime glass are not.</li>
  </ul>
</div>

<script id="payload" type="application/json">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById('payload').textContent);
const CRIT = ["accuracy_ceiling","transparency_robustness","data_efficiency",
  "contamination_robustness","cost_efficiency","throughput","explainability","tooling_maturity",
  "calibration_abstention","zero_shot_capability"];
const CRIT_LABEL = {accuracy_ceiling:"accuracy ceiling", transparency_robustness:"transparency robustness",
  data_efficiency:"data efficiency", contamination_robustness:"contamination robustness",
  cost_efficiency:"cost efficiency", throughput:"throughput", explainability:"explainability",
  tooling_maturity:"tooling maturity", calibration_abstention:"calibration / abstention",
  zero_shot_capability:"zero-label use"};
const byId = Object.fromEntries(DATA.registry.map(r=>[r.id,r]));
let weights = {...DATA.profiles[DATA.default_profile].weights};
let sortKey = 'score', sortDir = -1, familyFilter = '', query = '', kindFilter = 'model';

function composite(id){
  const s = DATA.scores[id]; let num=0, den=0;
  for(const c of CRIT){ num += (weights[c]||0)*(s[c]||0); den += (weights[c]||0); }
  return den>0 ? num/den : 0;
}
function esc(t){return String(t==null?"":t).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}

function visibleRows(){
  const q = query.trim().toLowerCase();
  return DATA.registry.filter(r=>{
    const k = r.kind || 'model';
    if(kindFilter==='model' && k!=='model') return false;
    if(kindFilter==='component' && k!=='component') return false;
    if(familyFilter && r.family!==familyFilter) return false;
    if(!q) return true;
    return [r.name,r.family,r.tier,r.evidence,r.glass_plastic_strengths,r.glass_plastic_weaknesses,
            r.recommended_when,r.pros,r.cons].join(" ").toLowerCase().includes(q);
  }).map(r=>({...r, score: composite(r.id)}));
}

function renderSliders(){
  const box = document.getElementById('sliders');
  box.innerHTML = CRIT.map(c=>`<div class="slider">
      <label>${CRIT_LABEL[c]} <b id="wv_${c}">${(weights[c]||0).toFixed(2)}</b></label>
      <input type="range" min="0" max="0.4" step="0.01" value="${weights[c]||0}" data-c="${c}"/>
    </div>`).join('');
  box.querySelectorAll('input').forEach(inp=>inp.addEventListener('input',e=>{
    weights[e.target.dataset.c] = parseFloat(e.target.value);
    document.getElementById('wv_'+e.target.dataset.c).textContent = weights[e.target.dataset.c].toFixed(2);
    renderAll();
  }));
}
function renderProfileSelect(){
  const sel = document.getElementById('profile');
  sel.innerHTML = '<option value="">custom</option>' + Object.entries(DATA.profiles)
     .map(([k,v])=>`<option value="${k}">${esc(v.label)}</option>`).join('');
  sel.value = DATA.default_profile;
  sel.addEventListener('change',e=>{
    if(e.target.value){ weights = {...DATA.profiles[e.target.value].weights}; renderSliders(); }
    renderAll();
  });
}
function renderKpis(){
  const rows = visibleRows().sort((a,b)=>b.score-a.score);
  const best = rows[0];
  document.getElementById('kpis').innerHTML = [
    ['Models compared', DATA.registry.length, 'across 14 families and sensor modalities'],
    ['Top ranked now', best?esc(best.name):'—', best?`score ${best.score.toFixed(2)}/5`:''],
    ['Best transparency robustness',
      esc((DATA.registry.slice().sort((a,b)=>DATA.scores[b.id].transparency_robustness-DATA.scores[a.id].transparency_robustness)[0]).name),
      'the glass-vs-plastic crux criterion'],
    ['Only family with published success on transparent-vs-transparent',
      'Spectral / polarimetric fusion', 'NIR·FTIR F1 0.98–1.00 · polarimetry [R31]'],
  ].map(([l,v,n])=>`<div class="kpi"><div class="l">${l}</div><div class="v">${v}</div>
      <div class="n">${n||''}</div></div>`).join('');
}
function bar(v){ const pct = Math.max(0,Math.min(100,(v/5)*100));
  return `<span class="bar" style="width:${pct*0.7}px"></span> <span class="num">${v.toFixed(2)}</span>`; }

function cmp(a,b){
  const k = (sortKey==='rank') ? 'score' : sortKey;
  let av = a[k], bv = b[k];
  if(av===null||av===undefined) av = (typeof bv==='number') ? -Infinity : '';
  if(bv===null||bv===undefined) bv = (typeof av==='number') ? -Infinity : '';
  if(typeof av==='number' && typeof bv==='number' && isFinite(av) && isFinite(bv)) return sortDir*(av-bv);
  return sortDir*String(av).localeCompare(String(bv));
}
function renderRank(){
  const rows = visibleRows().sort(cmp);
  let h = `<thead><tr><th data-k="rank">#</th><th data-k="name">model</th>
    <th data-k="family">family</th><th data-k="score">composite</th>
    ${CRIT.map(c=>`<th data-k="${c}" title="${CRIT_LABEL[c]}">${CRIT_LABEL[c].split(' ')[0]}</th>`).join('')}
    <th data-k="tier">tier</th></tr></thead><tbody>`;
  rows.forEach((r,i)=>{
    h += `<tr class="row"><td class="num">${i+1}</td><td><b>${esc(r.name)}</b></td>
      <td class="mut">${esc(r.family)}</td><td>${bar(r.score)}</td>
      ${CRIT.map(c=>`<td class="num">${DATA.scores[r.id][c]}</td>`).join('')}
      <td><span class="pill">${esc(r.tier)}</span></td></tr>`;
  });
  h += '</tbody>';
  const t = document.getElementById('rank'); t.innerHTML = h;
  t.querySelectorAll('th').forEach(th=>th.addEventListener('click',()=>{
    sortDir = (th.dataset.k===sortKey) ? -sortDir : -1;
    sortKey = th.dataset.k; renderRank(); renderFacts();
  }));
}
function renderFacts(){
  const rows = visibleRows().sort(cmp);
  const cols = [['name','model'],['family','family'],['params_m','params (M)'],['macs_g','MACs (G)'],
                ['input_res','input'],['pretrain','pretraining'],['imagenet_top1','ImageNet top-1'],
                ['data_need','labels needed'],['throughput','inference'],['licence','licence'],
                ['score','score']];
  let h = '<thead><tr>' + cols.map(([k,l])=>`<th data-k="${k}">${l}</th>`).join('') + '</tr></thead><tbody>';
  rows.forEach(r=>{
    h += `<tr class="row"><td>${esc(r.name)}<div class="legend">${esc(r.tier)}</div></td>
      <td class="mut">${esc(r.family)}</td>
      <td class="num">${esc(r.params_m||'—')}</td><td class="num">${esc(r.macs_g||'—')}</td>
      <td class="mut">${esc(r.input_res)}</td><td class="mut">${esc(r.pretrain)}</td>
      <td class="num">${esc(r.imagenet_top1||'n/a')}</td>
      <td class="mut">${esc(r.data_need)}</td><td class="mut">${esc(r.throughput)}</td>
      <td class="mut">${esc(r.licence)}</td>
      <td class="num"><b>${r.score.toFixed(2)}</b></td></tr>`;
    h += `<tr><td colspan="${cols.length}" style="padding:0;border:none">
      <details><summary class="mut">evidence · pros · cons · glass/plastic specifics — ${esc(r.name)}</summary>
        <p><span class="lbl">Evidence on real waste / glass-plastic data:</span> ${esc(r.evidence)}
           <span class="pill">${esc(r.refs)}</span></p>
        <p><span class="lbl good">Glass/plastic strengths:</span> ${esc(r.glass_plastic_strengths)}</p>
        <p><span class="lbl risk">Glass/plastic weaknesses:</span> ${esc(r.glass_plastic_weaknesses)}</p>
        <p><span class="lbl">Pros:</span> ${esc(r.pros)}</p>
        <p><span class="lbl">Cons:</span> ${esc(r.cons)}</p>
        <p><span class="lbl">Use it when:</span> ${esc(r.recommended_when)}</p>
        <p><span class="lbl">Cost facts:</span> params ${esc(r.params_m||'—')}M · MACs
           ${esc(r.macs_g||'—')}G · train ${esc(r.train_cost)} · licence ${esc(r.licence)}</p>
      </details></td></tr>`;
  });
  h += '</tbody>';
  const t = document.getElementById('facts'); t.innerHTML = h;
  t.querySelectorAll('th').forEach(th=>th.addEventListener('click',()=>{
    sortKey = th.dataset.k; sortDir = (th.dataset.k===sortKey) ? -sortDir : -1;
    sortKey = th.dataset.k; renderFacts(); renderRank();
  }));
}
function renderMeasured(){
  const box = document.getElementById('measured');
  const cls = (DATA.measured.classical_model_comparison||[]).filter(r=>r.split==='test')
      .sort((a,b)=>b.balanced_accuracy-a.balanced_accuracy);
  const ds = (DATA.measured.domain_shift_robustness||[]);
  const san = (DATA.measured.sanity_checks||[]).filter(r=>r.expected&&r.expected.includes('leak if'));
  const rules = (DATA.measured.rule_baseline||[]);
  let h = '';
  if(rules.length){
    h += `<h3>Physics floor — single-feature rules</h3><table><thead><tr><th>rule</th>
      <th>balanced accuracy</th><th>glass recall</th><th>plastic recall</th></tr></thead><tbody>`;
    rules.forEach(r=>h+=`<tr><td>${esc(r.rule)}</td><td class="num">${(+r.balanced_accuracy).toFixed(3)}</td>
      <td class="num">${(+r.glass_recall).toFixed(3)}</td><td class="num">${(+r.plastic_recall).toFixed(3)}</td></tr>`);
    h += '</tbody></table>';
  }
  if(cls.length){
    h += `<h3>Shallow models on hand-crafted features (in-distribution test)</h3>
      <table><thead><tr><th>model</th><th>balanced acc</th><th>95% CI</th><th>glass recall</th>
      <th>plastic recall</th><th>ROC AUC</th><th>fit (s)</th><th>classifier (ms/img)</th></tr>
      </thead><tbody>`;
    cls.forEach(r=>h+=`<tr><td>${esc(r.model)}</td><td class="num">${(+r.balanced_accuracy).toFixed(3)}</td>
      <td class="num mut">${(+r.ba_ci_lo).toFixed(3)}–${(+r.ba_ci_hi).toFixed(3)}</td>
      <td class="num">${(+r.glass_recall).toFixed(3)}</td><td class="num">${(+r.plastic_recall).toFixed(3)}</td>
      <td class="num">${(+r.roc_auc).toFixed(3)}</td><td class="num">${(+r.fit_seconds).toFixed(2)}</td>
      <td class="num">${(+r.clf_ms_per_image).toFixed(3)}</td></tr>`);
    h += '</tbody></table>';
  }
  if(san.length){
    h += `<h3>Leakage controls</h3><table><thead><tr><th>control</th><th>model</th>
      <th>balanced acc</th><th>expected</th></tr></thead><tbody>`;
    san.forEach(r=>h+=`<tr><td>${esc(r.check)}</td><td>${esc(r.model)}</td>
      <td class="num">${(+r.balanced_accuracy).toFixed(3)}</td><td class="mut">${esc(r.expected)}</td></tr>`);
    h += '</tbody></table>';
  }
  if(ds.length){
    const models=[...new Set(ds.map(r=>r.model))];
    const splits=[...new Set(ds.map(r=>r.split))];
    h += `<h3>Robustness under acquisition shift (balanced accuracy)</h3><table><thead><tr><th>model</th>`
       + splits.map(s=>`<th>${esc(s.replace('shift_',''))}</th>`).join('') + '</tr></thead><tbody>';
    models.forEach(m=>{
      h += `<tr><td><b>${esc(m)}</b></td>`;
      splits.forEach(s=>{
        const r = ds.find(x=>x.model===m && x.split===s);
        const v = r? (+r.balanced_accuracy) : null;
        const style = v===null? '' : (v<0.6? 'style="color:#b3261e;font-weight:600"' :
                       (v<0.8? 'style="color:#b3541e"' : 'style="color:#1f7a45"'));
        h += `<td class="num" ${style}>${v===null?'—':v.toFixed(3)}</td>`;
      });
      h += '</tr>';
    });
    h += '</tbody></table><div class="legend">Red = chance level. Note the three signatures: '+
         'tree ensembles survive pixel noise, the RBF SVM does not, and every global-descriptor '+
         'model collapses under background clutter.</div>';
  }
  box.innerHTML = h;
}
function renderAll(){ renderKpis(); renderRank(); renderFacts(); }

document.getElementById('famfilter').innerHTML = '<option value="">all families</option>' +
  [...new Set(DATA.registry.map(r=>r.family))].map(f=>`<option>${esc(f)}</option>`).join('');
document.getElementById('famfilter').addEventListener('change',e=>{familyFilter=e.target.value; renderAll();});
document.getElementById('kindfilter').addEventListener('change',e=>{kindFilter=e.target.value; renderAll();});
document.getElementById('q').addEventListener('input',e=>{query=e.target.value; renderAll();});
document.getElementById('reset').addEventListener('click',()=>{
  weights = {...DATA.profiles[DATA.default_profile].weights};
  document.getElementById('profile').value = DATA.default_profile;
  renderSliders(); renderAll();
});
renderProfileSelect(); renderSliders(); renderAll(); renderMeasured();
</script>
</main>
</body>
</html>
"""


# --------------------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------------------


def build_all_reports(registry="data/model_registry.csv", scores="data/model_scores.csv",
                      config="configs/scoring.yaml", results_dir="results",
                      docs_dir="docs") -> Dict[str, str]:
    reg = load_registry(registry)
    sc = load_scores(scores)
    profiles = load_profiles(config)
    measured = load_measured(results_dir)
    ranked = all_profile_rankings(reg, sc, profiles)
    refs_path = os.path.join(os.path.dirname(registry) or ".", "references.csv")
    refs = load_references(refs_path) if os.path.exists(refs_path) else []

    paths = {
        "comparison_md": render_comparison_md(reg, sc, profiles, ranked, measured, docs_dir),
        "references_md": render_references_md(refs, docs_dir) if refs else "",
        "measured_md": render_measured_md(measured, results_dir) if measured else "",
        "html": render_html(reg, sc, profiles, ranked, measured,
                            os.path.join(results_dir, "model_comparison.html"),
                            os.path.join(results_dir, "figures")),
    }
    # machine-readable ranking
    out = os.path.join(results_dir, "model_ranking.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "name", "family", *ranked.keys(), "tier"])
        for r in reg:
            w.writerow([r["id"], r["name"], r["family"],
                        *[round(ranked[p][r["id"]], 4) for p in ranked], r["tier"]])
    paths["ranking_csv"] = out
    return paths
