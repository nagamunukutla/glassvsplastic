# data/

```
model_registry.csv   source of truth for the comparison: 38 options, architecture facts,
                     deployment facts, cited evidence ($refs column -> [Rn] keys)
model_scores.csv     ten criteria scored 1-5 per option (see configs/scoring.yaml)
references.csv       the 45 sources, stable keys R1..R45
raw/                 download public datasets here (never committed)
processed/           generated data: the synthetic proxy set and cue-ablation variants
                     regenerate with `make proxy` / `make cues`
```

## Getting real data

This repository redistributes **no** third-party datasets. Download them yourself:

| Dataset | Where |
|---|---|
| TrashNet | https://github.com/garythung/trashnet |
| RealWaste | https://archive.ics.uci.edu/dataset/908/realwaste |
| ZeroWaste | https://github.com/Trash-AI/ZeroWaste |
| TACO | https://arxiv.org/abs/2003.05664 (annotations + links) |
| TrashBox | see the federated-learning paper (Sci Rep 14, 2024) |
| Catalogue of ~20 more | https://github.com/AgaMiko/waste-datasets-review |

To use them with this codebase, convert a dataset to the manifest schema written by
`gvp.synth.build_proxy_dataset` — a CSV per split with:

```
sample_id, path, label (1=glass, 0=plastic), material, object, background, size, split,
shift, ambiguous, cue_transparency, cue_specular, ...
```

Only `path` and `label` are required by the training/eval code; the remaining columns are used for
stratified analysis and for the cue ablation. `docs/09_evaluation_protocol.md` explains why the
object-level grouping and the `ambiguous` flag matter more than the rest.

## Regenerating everything

```bash
make proxy && make classical && make cues && make report
```
