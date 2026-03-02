# Repository Guidelines

## Project Structure & Module Organization

This repository is a flat Python-first evaluation toolkit for HOI benchmarks. Core evaluators live at the repo root as `eval_*.py` files, grouped by dataset (`swig`, `hico`), task (`action_referring`, `ground`), and provider (`openai`, `gemini`, `claude`, `qwen3vl`, `batch`, `sft`). Shared utilities are also top-level: `eval_api_utils.py`, `batch_api_utils.py`, `checkpoint_manager.py`, and `response_cache.py`. Shell entrypoints such as `run_batch_eval.sh` and `run_*_eval_*.sh` wrap common runs. Generated outputs are written under `results/` or `results-sft/`; treat these as artifacts, not source.

## Build, Test, and Development Commands

Use Python 3.11+ with `uv`.

- `uv sync`: install dependencies from [pyproject.toml](/media/shaun/workspace/hoi/hoi-benchmarks/pyproject.toml).
- `bash run_batch_eval.sh --help`: list supported providers, tasks, models, and runtime flags.
- `MAX_SAMPLES=10 bash run_batch_eval.sh gemini swig_action`: run a quick smoke test on a small subset.
- `RESUME=1 bash run_batch_eval.sh openai hico_ground`: resume an interrupted batch job from checkpoints.
- `python calculate_bertscore.py --pred-file <results.json>`: compute post-run semantic scoring for action tasks.

## Coding Style & Naming Conventions

Follow the existing style in the evaluation scripts: 4-space indentation, snake_case for functions and files, and explicit type hints where practical. Keep modules task-specific and name new scripts using the established pattern: `eval_<dataset>_<task>_<provider>.py` or `run_<dataset>_<task>_eval_<provider>.sh`. Prefer small shared helpers in existing utility modules instead of duplicating API, metric, or image-processing logic.

## Testing Guidelines

There is no dedicated `tests/` directory yet. Validate changes with targeted smoke runs using `MAX_SAMPLES` (or `MAX_IMAGES` in older scripts) before launching full jobs. For changes affecting metrics, confirm that output JSON, `*_metrics.json`, and logs are written to the expected results directory. Keep test runs small enough to avoid unnecessary API cost.

## Commit & Pull Request Guidelines

Recent commits use short, imperative, lowercase subjects such as `hotfix the path` and `update for sft trained model evaluation`. Keep messages focused and specific to one change. In pull requests, include: the affected dataset/task, provider or model, exact command used for validation, any new environment variables, and representative output paths if results formats changed.

## Configuration & Secrets

Store API keys in a local `.env` file only. Common keys include `GEMINI_API_KEY`, `OPENAI_API_KEY` (or `OPEN_AI_API_KEY`), and `CLAUDE_API_KEY`. Never commit credentials or large generated result files unless the change explicitly requires benchmark artifacts.
