"""Command line entry point: ``PYTHONPATH=src python -m gvp.cli <command>``.

Commands
--------
synth      generate the synthetic proxy dataset (with optional shifted test splits)
classical  full classical tier: model comparison, feature/cue ablations, shifts, rules, sanity
cues       cue ablation only ("what is the model looking at?")
shifts     domain-shift robustness only
deep       optional PyTorch/timm fine-tuning on the same manifests
bench      latency/size benchmark for deep backbones (requires torch)
report     render the literature/registry comparison into Markdown + self-contained HTML
inspect    print dataset + feature statistics
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np


def _print_table(rows, cols, limit=25):
    from . import evaluate
    print(evaluate.md_table(rows[:limit], cols))


def cmd_synth(args) -> int:
    from . import synth
    t0 = time.perf_counter()
    synth.build_proxy_dataset(args.out, n_per_class=args.n_per_class, size=args.size,
                              seed=args.seed, shifts=tuple(args.shifts.split(",")) if args.shifts else (),
                              n_shift=args.n_shift, ambiguous_fraction=args.ambiguous)
    print(f"[synth] done in {time.perf_counter() - t0:.1f}s -> {args.out}")
    return 0


def cmd_inspect(args) -> int:
    from . import synth
    from .classical import extract_split
    from .features import FEATURE_BLOCKS, feature_names
    names = feature_names(FEATURE_BLOCKS)
    print(f"feature blocks: {FEATURE_BLOCKS}")
    print(f"total features: {len(names)}")
    from collections import Counter
    print("features per block:", dict(Counter(n.split('_')[0] for n in names)))
    for split in ("train", "val", "test"):
        p = os.path.join(args.data, f"manifest_{split}.csv")
        if os.path.exists(p):
            rows = synth.load_manifest(p)
            print(f"{split:>6}: {len(rows)} images, "
                  f"glass={sum(1 for r in rows if r['label'] == '1')}, "
                  f"ambiguous={sum(int(r['ambiguous']) for r in rows)}")
    if args.stats:
        ev = extract_split(args.data, "test", FEATURE_BLOCKS)
        X, y = ev["X"], ev["label"].astype(int)
        print(f"X shape {X.shape}; glass mean vs plastic mean for a few features:")
        for i, n in enumerate(ev["names"]):
            if n in ("spec_peak_spikiness", "spec_saturated_frac", "tr_haze_index",
                     "tex_lap_var_inside", "col_sat_mean"):
                print(f"  {n:24s} glass={X[y == 1, i].mean():8.4f}  plastic={X[y == 0, i].mean():8.4f}")
        print(f"feature extraction cost: {float(ev.get('feature_ms_per_image', [np.nan])[0]):.1f} ms/image")
    return 0


def cmd_classical(args) -> int:
    from .classical import (run_cue_ablation, run_domain_shift, run_feature_ablation,
                            run_model_comparison, run_rule_baseline, run_sanity_checks,
                            top_discriminative_features)
    from .features import FEATURE_BLOCKS
    from . import evaluate
    out = args.out
    os.makedirs(out, exist_ok=True)
    t0 = time.perf_counter()

    rows, fitted = run_model_comparison(args.data, out)
    test_rows = [r for r in rows if r["split"] == "test"]
    print("\n=== classical model comparison (in-distribution test split) ===")
    _print_table([{"model": r["model"], "balanced_acc": r["balanced_accuracy"],
                   "accuracy": r["accuracy"], "macro_f1": r["macro_f1"],
                   "glass_recall": r["glass_recall"], "plastic_recall": r["plastic_recall"],
                   "roc_auc": r["roc_auc"], "fit_s": r["fit_seconds"]} for r in test_rows],
                 ["model", "balanced_acc", "accuracy", "macro_f1", "glass_recall",
                  "plastic_recall", "roc_auc", "fit_s"], limit=20)
    evaluate.plot_bars([r["model"] for r in test_rows], [r["balanced_accuracy"] for r in test_rows],
                       os.path.join(out, "figures", "classical_models.png"),
                       "Shallow models on hand-crafted features (proxy data)")

    # confusion matrix for the best model
    best = max(test_rows, key=lambda r: r["balanced_accuracy"])
    from .classical import extract_split
    ev = extract_split(args.data, "test", FEATURE_BLOCKS)
    pred = fitted[best["model"]].predict(ev["X"])
    evaluate.plot_confusion(ev["label"].astype(int), pred,
                            os.path.join(out, "figures", "confusion_best_classical.png"),
                            f"{best['model']} — proxy test split")

    print("\n=== single-feature physics rules ===")
    rules = run_rule_baseline(args.data, out)
    _print_table([{"rule": r["rule"], "bal_acc": r["balanced_accuracy"],
                   "glass_recall": r["glass_recall"], "plastic_recall": r["plastic_recall"]}
                  for r in rules], ["rule", "bal_acc", "glass_recall", "plastic_recall"])

    print("\n=== sanity checks (leak detection) ===")
    sanity = run_sanity_checks(args.data, out)
    _print_table(sanity, ["check", "model", "balanced_accuracy", "expected"])

    print("\n=== feature-block ablation ===")
    abl = run_feature_ablation(args.data, out)
    for m in sorted({r["model"] for r in abl}):
        sub = [r for r in abl if r["model"] == m]
        sub.sort(key=lambda r: -r["balanced_accuracy"])
        print(f"  {m}: best={sub[0]['combo']} ({sub[0]['balanced_accuracy']:.3f}), "
              f"weakest={sub[-1]['combo']} ({sub[-1]['balanced_accuracy']:.3f})")

    print("\n=== domain-shift robustness ===")
    ds = run_domain_shift(args.data, out)
    _print_table([{"model": r["model"], "split": r["split"], "bal_acc": r["balanced_accuracy"],
                   "drop": r["drop_vs_clean"]} for r in ds],
                 ["model", "split", "bal_acc", "drop"], limit=40)

    if args.cues:
        print("\n=== cue ablation (regenerating proxy data per cue) ===")
        ca = run_cue_ablation(out, root_dir=os.path.join(os.path.dirname(args.data), "cue_ablation"))
        base = {(r["model"]): r["balanced_accuracy"] for r in ca if r["cue_removed"] == "none_removed"}
        summary = []
        for r in ca:
            if r["cue_removed"] == "none_removed":
                continue
            summary.append({"cue_removed": r["cue_removed"], "model": r["model"],
                            "bal_acc": r["balanced_accuracy"],
                            "delta": r["balanced_accuracy"] - base.get(r["model"], np.nan)})
        summary.sort(key=lambda r: r["delta"])
        _print_table(summary, ["cue_removed", "model", "bal_acc", "delta"], limit=30)
        evaluate.plot_bars([f"{r['cue_removed']}\n({r['model']})" for r in summary],
                           [max(0.0, r["delta"]) for r in summary],
                           os.path.join(out, "figures", "cue_ablation_drop.png"),
                           "Accuracy drop when a physical cue is removed (proxy data)",
                           ylabel="Δ balanced accuracy (negative = cue mattered)")

    # interpretability: importances of the best tree model
    et = fitted.get("extra_trees")
    if et is not None:
        ev_names = ev["names"]
        tops = top_discriminative_features(et, ev_names, top_k=22)
        if tops:
            evaluate.plot_feature_importance([t[0] for t in tops], [t[1] for t in tops],
                                             os.path.join(out, "figures", "top_features.png"),
                                             "ExtraTrees importances (proxy data)")

    with open(os.path.join(out, "run_summary.json"), "w") as fh:
        json.dump({
            "data_dir": os.path.abspath(args.data),
            "n_models": len({r["model"] for r in rows}),
            "best_model": best["model"],
            "best_test_balanced_accuracy": best["balanced_accuracy"],
            "ci95": [best["ba_ci_lo"], best["ba_ci_hi"]],
            "seconds": round(time.perf_counter() - t0, 1),
            "caveat": "synthetic proxy data - diagnostics only, see docs/03_datasets_and_benchmarks.md",
        }, fh, indent=2)
    print(f"\n[classical] done in {time.perf_counter() - t0:.1f}s -> {out}")
    return 0


def cmd_cues(args) -> int:
    """Cue ablation only: regenerate the proxy dataset with one physical cue disabled."""
    from .classical import run_cue_ablation
    from . import evaluate
    import numpy as _np
    os.makedirs(args.out, exist_ok=True)
    t0 = time.perf_counter()
    rows = run_cue_ablation(args.out, root_dir=args.root, n_per_class=args.n_per_class,
                            size=args.size)
    base = {r["model"]: r["balanced_accuracy"] for r in rows if r["cue_removed"] == "none_removed"}
    summary = []
    for r in rows:
        if r["cue_removed"] == "none_removed":
            continue
        summary.append({"cue_removed": r["cue_removed"].replace("without_", ""), "model": r["model"],
                        "bal_acc": r["balanced_accuracy"], "delta_vs_full_cues":
                        r["balanced_accuracy"] - base.get(r["model"], _np.nan)})
    summary.sort(key=lambda r: r["delta_vs_full_cues"])
    print("=== accuracy loss when a physical cue is removed from both classes ===")
    _print_table(summary, ["cue_removed", "model", "bal_acc", "delta_vs_full_cues"], limit=40)

    models = sorted({r["model"] for r in summary})
    cues = sorted({r["cue_removed"] for r in summary})
    mat = _np.full((len(models), len(cues)), _np.nan)
    for r in summary:
        mat[models.index(r["model"]), cues.index(r["cue_removed"])] = r["bal_acc"]
    evaluate.plot_matrix(models, cues, mat, os.path.join(args.out, "figures", "cue_ablation.png"),
                         "Proxy accuracy with each cue removed", vmin=0.4, vmax=1.0)
    print(f"\n[cues] done in {time.perf_counter() - t0:.1f}s")
    return 0


def cmd_shifts(args) -> int:
    """Domain-shift robustness only."""
    from .classical import run_domain_shift
    from . import evaluate
    import numpy as _np
    rows = run_domain_shift(args.data, args.out,
                            models=tuple(args.models.split(",")) if args.models else
                            ("svm_rbf", "extra_trees", "hist_gbdt"))
    cols = ["model", "split", "balanced_accuracy", "glass_as_plastic", "plastic_as_glass",
            "drop_vs_clean"]
    print("=== robustness under acquisition shift (trained on clean in-distribution data) ===")
    _print_table(rows, cols, limit=60)
    models = sorted({r["model"] for r in rows})
    splits = [s for s in ("test", *(f"shift_{k}" for k in __import__("gvp.synth", fromlist=["SHIFT_KINDS"]).SHIFT_KINDS))
              if any(r["split"] == s for r in rows)]
    mat = _np.full((len(models), len(splits)), _np.nan)
    for r in rows:
        if r["split"] in splits:
            mat[models.index(r["model"]), splits.index(r["split"])] = r["balanced_accuracy"]
    os.makedirs(args.out, exist_ok=True)
    evaluate.plot_matrix(models, [s.replace("shift_", "") for s in splits], mat,
                         os.path.join(args.out, "figures", "domain_shift_heatmap.png"),
                         "Robustness under acquisition shift (proxy data)", vmin=0.4, vmax=1.0)
    return 0


def cmd_deep(args) -> int:
    from .deep import DeepUnavailable, TrainConfig, train_and_eval
    cfg = TrainConfig(model=args.model, data_dir=args.data, out_dir=args.out, epochs=args.epochs,
                      batch_size=args.batch_size, lr=args.lr, img_size=args.img_size,
                      freeze_backbone=args.freeze, seed=args.seed,
                      test_splits=tuple(args.splits.split(",")) if args.splits else ("test",))
    try:
        res = train_and_eval(cfg)
    except DeepUnavailable as exc:
        print(f"[deep] {exc}", file=sys.stderr)
        return 2
    _print_table(res["rows"], ["model", "split", "balanced_accuracy", "glass_recall",
                               "plastic_recall", "roc_auc", "params_m", "macs_g",
                               "latency_ms_per_image"])
    return 0


def cmd_bench(args) -> int:
    from .deep import DeepUnavailable, RECOMMENDED_BACKBONES, benchmark_latency
    rows = []
    for name in (args.models.split(",") if args.models else RECOMMENDED_BACKBONES):
        try:
            rows.append(benchmark_latency(name, args.img_size, device=args.device))
        except DeepUnavailable as exc:
            print(f"[bench] {exc}", file=sys.stderr)
            return 2
        except Exception as exc:
            print(f"[bench] skipping {name}: {exc}", file=sys.stderr)
    _print_table(rows, ["model", "device", "params_m", "macs_g", "latency_ms_median", "fps_batch1"])
    from .classical import _write_csv
    _write_csv(rows, os.path.join(args.out, "deep_benchmark.csv"))
    return 0


def cmd_report(args) -> int:
    from .report import build_all_reports
    paths = build_all_reports(registry=args.registry, scores=args.scores, config=args.config,
                              results_dir=args.results, docs_dir=args.docs)
    for k, v in paths.items():
        print(f"[report] {k}: {v}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="gvp", description="Glass vs plastic: algorithms and model comparison")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="generate the synthetic proxy dataset")
    s.add_argument("--out", default="data/processed/proxy")
    s.add_argument("--n-per-class", type=int, default=300)
    s.add_argument("--size", type=int, default=128)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--shifts", default="blur,dark,noise,jpeg,lowcontrast,clutter,whitebg")
    s.add_argument("--n-shift", type=int, default=120)
    s.add_argument("--ambiguous", type=float, default=0.18)
    s.set_defaults(func=cmd_synth)

    s = sub.add_parser("inspect", help="dataset + feature statistics")
    s.add_argument("--data", default="data/processed/proxy")
    s.add_argument("--stats", action="store_true")
    s.set_defaults(func=cmd_inspect)

    s = sub.add_parser("classical", help="classical tier experiments")
    s.add_argument("--data", default="data/processed/proxy")
    s.add_argument("--out", default="results")
    s.add_argument("--cues", action="store_true", help="also run the cue ablation (slower)")
    s.set_defaults(func=cmd_classical)

    s = sub.add_parser("cues", help="cue ablation: what is the model looking at?")
    s.add_argument("--out", default="results")
    s.add_argument("--root", default="data/processed/cue_ablation")
    s.add_argument("--n-per-class", type=int, default=400,
                   help="400 gives ~120 test items per condition; anything less is underpowered")
    s.add_argument("--size", type=int, default=128)
    s.set_defaults(func=cmd_cues)

    s = sub.add_parser("shifts", help="domain-shift robustness sweep")
    s.add_argument("--data", default="data/processed/proxy")
    s.add_argument("--out", default="results")
    s.add_argument("--models", default="")
    s.set_defaults(func=cmd_shifts)

    s = sub.add_parser("deep", help="fine-tune one backbone (needs torch)")
    s.add_argument("--data", default="data/processed/proxy")
    s.add_argument("--out", default="results/deep")
    s.add_argument("--model", default="tv:resnet18")
    s.add_argument("--epochs", type=int, default=20)
    s.add_argument("--batch-size", type=int, default=32)
    s.add_argument("--lr", type=float, default=3e-4)
    s.add_argument("--img-size", type=int, default=224)
    s.add_argument("--freeze", action="store_true")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--splits", default="test,shift_blur,shift_clutter")
    s.set_defaults(func=cmd_deep)

    s = sub.add_parser("bench", help="latency benchmark for deep backbones (needs torch)")
    s.add_argument("--models", default="")
    s.add_argument("--img-size", type=int, default=224)
    s.add_argument("--device", default="auto")
    s.add_argument("--out", default="results")
    s.set_defaults(func=cmd_bench)

    s = sub.add_parser("report", help="render comparison tables (Markdown + HTML)")
    s.add_argument("--registry", default="data/model_registry.csv")
    s.add_argument("--scores", default="data/model_scores.csv")
    s.add_argument("--config", default="configs/scoring.yaml")
    s.add_argument("--results", default="results")
    s.add_argument("--docs", default="docs")
    s.set_defaults(func=cmd_report)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
