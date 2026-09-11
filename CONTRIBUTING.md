# Contributing

The most valuable contributions here are **evidence**, **corrections** and **weighting arguments**.

## Adding or correcting a model entry

1. Edit `MODELS` in `scripts/build_registry.py`. Every entry needs:
   * a `family`, cost facts (`params_m`, `macs_g`, `input_res`, `pretrain`, `imagenet_top1`);
   * an `evidence` field quoting **reported** numbers with `[Rn]` keys — no un-cited claims;
   * honest `glass_plastic_strengths` / `glass_plastic_weaknesses` (this is the point of the study);
   * a `tier`.
2. Add the source to `REFERENCES` (stable keys — never renumber an existing `Rn`).
3. Score the entry on the ten criteria in `SCORES` and justify the score in the PR description.
4. `python scripts/build_registry.py && make report` and commit the regenerated artefacts.

## Ground rules

* **Never mix cited and measured numbers.** Cited numbers describe published work on real data;
  measured numbers come from `results/*.csv` and must carry the proxy-data caveat.
* **No benchmark league tables without caveats.** If you add a "best model" claim, state the
  dataset, the split and the metric.
* **Controls are mandatory** for any new measurement: shuffled-label, background-only, per-class
  recall, and a bootstrap CI.
* **Report negative results.** The underpowered cue ablation in `results/MEASURED_RESULTS.md` §5 is
  the model to follow.
* **Don't renumber references**, and don't cite a source you have not read.
* Performance/robustness changes to `src/gvp/**` need a test in `tests/`.

## Running the checks

```bash
make test          # unit + end-to-end smoke tests
make all           # full data -> results -> report pipeline
```
