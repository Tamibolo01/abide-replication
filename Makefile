# Development commands. `make check` is the gate every change must pass.
# Uses the project venv when present, else the active `python` (the cluster's conda env).
PY ?= $(if $(wildcard venv/bin/python),venv/bin/python,python)

.PHONY: check lint format test test-fast smoke dev

check: lint test  ## the gate: lint, format check, tests

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:  ## fix what ruff can fix, then reformat
	$(PY) -m ruff check --fix .
	$(PY) -m ruff format .

test:
	$(PY) -m pytest

test-fast:  ## everything that does not import torch
	$(PY) -m pytest -m "not vlm"

smoke:  ## one-minute real-data run; needs data/ABIDE_pcp cached (laptop only); not part of check
	$(PY) src/connectivity_experiment.py --sites PITT OLIN --n-splits 5 --output-dir results/smoke

dev:
	$(PY) -m pip install -r requirements-dev.txt
