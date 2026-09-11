# Glass vs Plastic study — reproducible pipeline
#
#   make all        # data -> classical results -> cue ablation -> report
#   make test       # unit + end-to-end smoke tests (includes the leakage controls)
#   make report     # regenerate docs/06 + results/model_comparison.html
#
# Everything runs CPU-only; the deep tier is opt-in (`make deep`) and needs requirements-deep.txt.

SHELL      := /bin/bash
PY         ?= python
export PYTHONPATH := src
DATA       ?= data/processed/proxy
OUT        ?= results
N_PER_CLASS ?= 300
SIZE       ?= 128

.PHONY: help all proxy inspect classical cues shifts report real real-deep real-compare test deep bench clean clean-data

help:
	@echo "make all         full pipeline (proxy -> classical -> cues -> report)"
	@echo "make proxy       render the synthetic proxy dataset (N_PER_CLASS=$(N_PER_CLASS), SIZE=$(SIZE))"
	@echo "make inspect     feature/dataset statistics"
	@echo "make classical   model comparison, rules, controls, shift sweep"
	@echo "make cues        cue ablation with confidence intervals"
	@echo "make shifts      robustness sweep only"
	@echo "make report      regenerate the comparison documents"
	@echo "make real        real-data study on TrashNet (needs data/raw/trashnet; see data/README.md)"
	@echo "make real-deep   fine-tune backbones on the real data (needs torch)"
	@echo "make real-compare fuse measured results into one model-comparison table"
	@echo "make test        run the test suite"
	@echo "make deep        example deep-tier fine-tune (requires torch)"
	@echo "make bench       deep-tier latency/params/MACs benchmark (requires torch)"
	@echo "make clean       remove caches; make clean-data also removes generated data"

all: proxy classical cues report

proxy:
	$(PY) -m gvp.cli synth --out $(DATA) --n-per-class $(N_PER_CLASS) --size $(SIZE)

inspect:
	$(PY) -m gvp.cli inspect --data $(DATA) --stats

classical: proxy
	$(PY) -m gvp.cli classical --data $(DATA) --out $(OUT)

cues:
	$(PY) -m gvp.cli cues --out $(OUT)

shifts:
	$(PY) -m gvp.cli shifts --data $(DATA) --out $(OUT)

report:
	$(PY) -m gvp.cli report --registry data/model_registry.csv --scores data/model_scores.csv \
		--config configs/scoring.yaml --results $(OUT) --docs docs

real:
	$(PY) scripts/merge_deep_results.py --out $(OUT)/real
	$(PY) scripts/real_data_study.py --data data/raw/trashnet --out $(OUT)/real
	$(PY) scripts/build_measured_comparison.py --real-dir $(OUT)/real

real-deep:
	$(PY) scripts/deep_real_study.py --data data/raw/trashnet --out $(OUT)/real \
		--models tv:resnet18,tv:mobilenet_v3_large --crop-models tv:resnet18 --epochs 8

real-compare:
	$(PY) scripts/merge_deep_results.py --out $(OUT)/real
	$(PY) scripts/build_measured_comparison.py --real-dir $(OUT)/real

test:
	$(PY) -m unittest discover -s tests -v

deep:
	$(PY) -m gvp.cli deep --data $(DATA) --out results/deep --model tv:resnet18 --epochs 20 \
		--splits test,shift_blur,shift_clutter

bench:
	$(PY) -m gvp.cli bench --out $(OUT)

clean:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} + ; \
	find . -name '.pytest_cache' -type d -prune -exec rm -rf {} + ; \
	rm -rf data/processed/*/.cache

real-prepare:
	$(PY) scripts/prepare_real_dataset.py --layout splitfiles --root data/raw/trashnet \
		--images dataset-resized --index-base 1 --out data/raw/trashnet

clean-data: clean
	rm -rf data/processed/*
	rm -rf data/raw/trashnet_cropped
