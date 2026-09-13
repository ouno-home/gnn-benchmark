# Agent Instructions

## Project Overview

This repository contains `gnnbench`, a Python package for benchmarking graph neural network models on transductive node classification tasks. It uses TensorFlow 2.x in `tf.compat.v1` mode (`disable_v2_behavior`), Sacred (with a SQLite results backend via `SqlObserver`), YAML experiment configs, and preprocessed graph datasets stored under `data/`.

## Repository Layout

- `gnnbench/`: package code for datasets, model definitions, training, metrics, and single-job execution.
- `gnnbench/models/`: model implementations. New models should follow `GNNModel` in `base_model.py` and expose a Sacred ingredient plus `build_model`.
- `gnnbench/data/`: dataset loading, preprocessing, and split helpers.
- `config/`: default training settings, experiment definitions, fixed model configs, optimized configs, and search spaces.
- `scripts/`: operational scripts for creating jobs, spawning workers, and aggregating results.
- `data/`: included Planetoid and `.npz` datasets. Treat these as benchmark fixtures; avoid rewriting them unless the task explicitly concerns dataset generation.

## Environment And Dependencies

- The project targets modern Python (3.11+) with TensorFlow 2.x used through `tf.compat.v1` with
  v2 behavior disabled; keep this pattern when editing model or training code.
- Install dependencies from `requirements.txt`; Sacred is pinned to `0.8.7`.
- Install the package in editable mode from the repository root:

```bash
pip install -e .
```

- Experiments are tracked in a local SQLite database file given by `db_path` in the experiment
  config. It holds both the job queue (`pending` table) and the Sacred results (`run`/`metric` tables).
- The job queue is accessed via `gnnbench.util.get_pending_connection` and `fetch_pending_job`
  (atomic claim using `BEGIN IMMEDIATE`); Sacred results use `SqlObserver` from `gnnbench/run_single_job.py`.
- Workers run on a GPU when available; `spawn_worker.py` also supports CPU execution when `--gpu` is
  omitted (the default on machines without CUDA).

## Common Commands

Create fixed-configuration jobs:

```bash
python scripts/create_jobs.py -c config/fixed_configs.conf.yaml --op fixed
```

Create hyperparameter-search jobs:

```bash
python scripts/create_jobs.py -c config/hyperparameter_search.conf.yaml --op search
```

Check, reset, or clear pending jobs:

```bash
python scripts/create_jobs.py -c config/fixed_configs.conf.yaml --op status
python scripts/create_jobs.py -c config/fixed_configs.conf.yaml --op reset
python scripts/create_jobs.py -c config/fixed_configs.conf.yaml --op clear
```

Run a worker on a selected GPU:

```bash
python scripts/spawn_worker.py -c config/fixed_configs.conf.yaml --gpu 0
```

Run a worker on CPU (requires the CPU TensorFlow build; `--gpu` is optional and defaults to CPU):

```bash
python scripts/spawn_worker.py -c config/fixed_configs.conf.yaml
```

Aggregate results:

```bash
python scripts/aggregate_results.py -c config/fixed_configs.conf.yaml -o results/
```

## Testing And Verification

- There is no committed test suite or lint configuration in this repository.
- For code-only changes, at minimum run targeted import or syntax checks where the legacy dependency stack allows it.
- For changes to job creation, worker behavior, aggregation, configs, or database access, verify against a small local SQLite-backed run when feasible.
- For model or training changes, verify with a minimal dataset/config run on an available GPU when feasible.
- If verification cannot be run because dependencies, MongoDB, or GPU support are unavailable, state that explicitly in the final response.

## Coding Conventions

- Keep changes compatible with the pinned legacy stack unless the task explicitly asks to modernize it.
- Follow the existing Python style: small module-level functions, Sacred `@capture` hooks, TensorFlow 1.x graph/session APIs, and f-strings where already used.
- Preserve existing config precedence: sampled search parameters override model-specific parameters, which override `config/train.conf.yaml`.
- Keep model-specific options in the relevant YAML files and model modules rather than hard-coding experiment behavior.
- Prefer explicit errors for unsupported models, metrics, datasets, or config modes.
- Avoid broad refactors when touching training, data loading, and database workflow code; these paths are coupled to experiment reproducibility.

## Data And Results Safety

- Do not modify files under `data/` unless asked.
- Be careful with commands that clear database state:
  - `scripts/create_jobs.py --op clear`
  - `scripts/aggregate_results.py --clear`
- Do not run destructive database cleanup commands unless the user explicitly asks for that operation.
- Avoid committing generated results, large artifacts, local database dumps, or environment-specific outputs.

## Documentation Updates

- Update `README.md` or this file when changing setup, experiment workflow, supported configs, or operational commands.
- When adding a model, dataset, metric, or config mode, document both the code entry point and the expected YAML shape.
