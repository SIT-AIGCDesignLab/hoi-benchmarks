# HOI Benchmark Evaluation

Evaluation scripts for Human-Object Interaction (HOI) benchmarks using Vision-Language Models.

## Supported Datasets

- **SWIG-HOI**: Situation With Humans for HOI detection
- **HICO-DET**: Human-Object Interaction Detection

## Quick Start

### Prerequisites

1. Create `.env` file with your API keys:
```bash
GEMINI_API_KEY=your-gemini-api-key
OPEN_AI_API_KEY=your-openai-api-key
CLAUDE_API_KEY=your-claude-api-key
```

2. Activate virtual environment:
```bash
source .venv/bin/activate
```

---

## Batch Processing (50% Cost Savings) - RECOMMENDED

For large-scale evaluations, use the **Batch API mode** which offers:
- **50% cost reduction** compared to real-time API calls
- **No rate limit issues** (separate quota pools)
- **Automatic checkpointing** for resumability
- **24-hour completion window** (usually much faster)

### Quick Start - Batch Mode

```bash
# Usage: bash run_batch_eval.sh <provider> <task> [model]
# Providers: claude, gemini, openai
# Tasks: swig_action, swig_ground, hico_action, hico_ground

# Show help and available models
bash run_batch_eval.sh --help

# SWIG Action Referring - Gemini (uses default: gemini-2.5-flash)
bash run_batch_eval.sh gemini swig_action

# SWIG Action Referring - Gemini 3 Pro (best accuracy)
bash run_batch_eval.sh gemini swig_action gemini-3-pro-preview

# SWIG Grounding - Claude Sonnet (default)
bash run_batch_eval.sh claude swig_ground

# SWIG Grounding - Claude Opus (highest accuracy)
bash run_batch_eval.sh claude swig_ground claude-opus-4-5

# HICO Action Referring - OpenAI GPT-4o (default)
bash run_batch_eval.sh openai hico_action

# HICO Action Referring - OpenAI GPT-5.2 (most capable)
bash run_batch_eval.sh openai hico_action gpt-5.2-2025-12-11
```

---

### Choosing Your Model

The batch script allows you to specify any model as the **third argument**:

```bash
# Syntax
bash run_batch_eval.sh <provider> <task> <model>
```

**View all available models:**
```bash
bash run_batch_eval.sh --help
```

#### Model Selection Examples

```bash
# Use default model (cost-effective)
bash run_batch_eval.sh gemini swig_action

# Specify a different Gemini model
bash run_batch_eval.sh gemini swig_action gemini-3-pro-preview
bash run_batch_eval.sh gemini swig_action gemini-2.5-pro

# Specify a different Claude model
bash run_batch_eval.sh claude swig_action claude-opus-4-5-20251101
bash run_batch_eval.sh claude swig_action claude-haiku-4-5-20250514

# Specify a different OpenAI model
bash run_batch_eval.sh openai swig_action gpt-5.2-2025-12-11
bash run_batch_eval.sh openai swig_action gpt-4o-mini
```

#### Available Models Reference

| Provider | Model Name | Type | Cost (Input/Output per MTok) |
|----------|------------|------|------------------------------|
| **Gemini** | `gemini-2.5-flash` | Default | $0.15 / $1.25 |
| | `gemini-2.5-pro` | Better | $0.625 / $5.00 |
| | `gemini-3-flash-preview` | Latest Fast | $0.15 / $1.25 |
| | `gemini-3-pro-preview` | **Best Accuracy** | $0.625 / $5.00 |
| **Claude** | `claude-haiku-4-5` | Fastest | $0.50 / $2.50 |
| | `claude-sonnet-4-5` | Default | $1.50 / $7.50 |
| | `claude-opus-4-5` | **Best Accuracy** | $7.50 / $37.50 |
| **OpenAI** | `gpt-4o-mini` | Cheapest | $0.075 / $0.30 |
| | `gpt-4o` | Default | $1.25 / $5.00 |
| | `gpt-5.2-2025-12-11` | **Best Accuracy** | ~$2.50 / $10.00 |

#### When to Use Which Model

| Use Case | Gemini | Claude | OpenAI |
|----------|--------|--------|--------|
| **Testing/Development** | `gemini-2.5-flash` | `claude-haiku-4-5` | `gpt-4o-mini` |
| **Production (balanced)** | `gemini-2.5-pro` | `claude-sonnet-4-5` | `gpt-4o` |
| **Best Accuracy** | `gemini-3-pro-preview` | `claude-opus-4-5` | `gpt-5.2-2025-12-11` |

---

### Resume Interrupted Batch

If a batch job is interrupted, resume it with:

```bash
RESUME=1 bash run_batch_eval.sh gemini swig_action
```

### Batch Mode with Limited Samples (Testing)

```bash
# Test with 10 samples
MAX_SAMPLES=10 bash run_batch_eval.sh gemini swig_action

# Test with 5 samples, skip BERTScore for speed
MAX_SAMPLES=5 SKIP_BERTSCORE=1 bash run_batch_eval.sh claude hico_action
```

### Batch vs Real-time Comparison

| Aspect | Real-time API | Batch API |
|--------|--------------|-----------|
| Cost | Full price | **50% off** |
| Rate limits | Shared pool (hits limits) | **Separate pool** |
| Latency | Immediate | Up to 24 hours (usually <1 hour) |
| Resumability | Manual caching | **Auto checkpoint** |
| Best for | Quick tests | **Full evaluations** |

### Batch Environment Variables

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `RESUME=1` | Resume from checkpoint | `0` | `RESUME=1` |
| `MAX_SAMPLES=N` | Limit to N samples | all | `MAX_SAMPLES=100` |
| `POLL_INTERVAL=N` | Seconds between status checks | `60` | `POLL_INTERVAL=30` |
| `OPTIMIZED_PROMPTS=1` | Use shorter prompts | `0` | `OPTIMIZED_PROMPTS=1` |
| `SKIP_BERTSCORE=1` | Skip BERTScore computation | `0` | `SKIP_BERTSCORE=1` |
| `WANDB=1` | Enable Weights & Biases logging | `0` | `WANDB=1` |
| `WANDB_PROJECT=name` | Custom W&B project name | `hoi-batch-eval` | `WANDB_PROJECT=my-project` |
| `VERBOSE=1` | Enable visualizations | `0` | `VERBOSE=1` |

### W&B and Visualization Examples

```bash
# Enable W&B logging
WANDB=1 bash run_batch_eval.sh gemini swig_action

# Enable visualizations (saves annotated images to results/*/visualizations/)
VERBOSE=1 bash run_batch_eval.sh gemini swig_action

# Both W&B and visualizations
WANDB=1 VERBOSE=1 bash run_batch_eval.sh gemini swig_action gemini-3-pro-preview

# Custom W&B project name
WANDB=1 WANDB_PROJECT=hoi-gemini-eval bash run_batch_eval.sh gemini swig_action
```

**Note**: Visualizations are limited to the first 100 samples to avoid excessive storage usage. Visualization files are saved as JPG in the `visualizations/` subdirectory of your results folder.

### Full Batch Evaluation Commands

```bash
# === SWIG Dataset - Cost-Effective Models ===

# SWIG Action Referring - All Providers (default models)
bash run_batch_eval.sh gemini swig_action
bash run_batch_eval.sh claude swig_action
bash run_batch_eval.sh openai swig_action

# SWIG Grounding - All Providers (default models)
bash run_batch_eval.sh gemini swig_ground
bash run_batch_eval.sh claude swig_ground
bash run_batch_eval.sh openai swig_ground

# === SWIG Dataset - Best Accuracy Models ===

# SWIG Action Referring - Top-tier models
bash run_batch_eval.sh gemini swig_action gemini-3-pro-preview
bash run_batch_eval.sh claude swig_action claude-opus-4-5
bash run_batch_eval.sh openai swig_action gpt-5.2-2025-12-11

# SWIG Grounding - Top-tier models
bash run_batch_eval.sh gemini swig_ground gemini-3-pro-preview
bash run_batch_eval.sh claude swig_ground claude-opus-4-5
bash run_batch_eval.sh openai swig_ground gpt-5.2-2025-12-11

# === HICO-DET Dataset - Cost-Effective Models ===

# HICO Action Referring - All Providers (default models)
bash run_batch_eval.sh gemini hico_action
bash run_batch_eval.sh claude hico_action
bash run_batch_eval.sh openai hico_action

# HICO Grounding - All Providers (default models)
bash run_batch_eval.sh gemini hico_ground
bash run_batch_eval.sh claude hico_ground
bash run_batch_eval.sh openai hico_ground

# === HICO-DET Dataset - Best Accuracy Models ===

# HICO Action Referring - Top-tier models
bash run_batch_eval.sh gemini hico_action gemini-3-pro-preview
bash run_batch_eval.sh claude hico_action claude-opus-4-5
bash run_batch_eval.sh openai hico_action gpt-5.2-2025-12-11

# HICO Grounding - Top-tier models
bash run_batch_eval.sh gemini hico_ground gemini-3-pro-preview
bash run_batch_eval.sh claude hico_ground claude-opus-4-5
bash run_batch_eval.sh openai hico_ground gpt-5.2-2025-12-11
```

---

## Real-time Evaluation Commands (Original)

> **Note:** For large-scale evaluations, use **Batch Processing** above. Real-time mode is better for quick tests or when you need immediate results.

## SWIG-HOI Evaluation Commands

### Grounding Task (SWIG)

Evaluates the model's ability to localize person-object pairs performing a specific action.

#### Gemini 3 Pro (Best Performance)
```bash
rm -rf results/swig_ground_gemini && \
VERBOSE=1 USE_CACHE=0 bash run_swig_ground_eval_gemini.sh gemini-3-pro-preview
```

#### OpenAI GPT-5.2
```bash
rm -rf results/swig_ground_openai && \
VERBOSE=1 USE_CACHE=0 bash run_swig_ground_eval_openai.sh gpt-5.2-2025-12-11
```

#### Claude Opus 4.5
```bash
rm -rf results/swig_ground_claude && \
VERBOSE=1 USE_CACHE=0 bash run_swig_ground_eval_claude.sh claude-opus-4-5-20251101
```

### Action Referring Task (SWIG)

Evaluates the model's ability to describe the action between highlighted person-object pairs.

#### Gemini 3 Pro
```bash
rm -rf results/swig_action_gemini && \
VERBOSE=1 USE_CACHE=0 bash run_swig_action_referring_eval_gemini.sh gemini-3-pro-preview
```

#### OpenAI GPT-5.2
```bash
rm -rf results/swig_action_openai && \
VERBOSE=1 USE_CACHE=0 bash run_swig_action_referring_eval_openai.sh gpt-5.2-2025-12-11
```

#### Claude Opus 4.5
```bash
rm -rf results/swig_action_claude && \
VERBOSE=1 USE_CACHE=0 bash run_swig_action_referring_eval_claude.sh claude-opus-4-5-20251101
```

---

## HICO-DET Evaluation Commands

> **Note:** HICO-DET scripts use **full detailed prompts** by default (not optimized prompts) for better accuracy. To enable optimized prompts, set `OPTIMIZED_PROMPTS=1`.

### Grounding Task (HICO)

Evaluates the model's ability to localize person-object pairs performing a specific action in HICO-DET images.

#### Gemini 3 Pro (Best Performance)
```bash
rm -rf results/hico_ground_gemini && \
VERBOSE=1 USE_CACHE=0 bash run_hico_ground_eval_gemini.sh gemini-3-pro-preview
```

#### OpenAI GPT-5.2 (Medium Reasoning)
```bash
rm -rf results/hico_ground_openai && \
VERBOSE=1 USE_CACHE=0 bash run_hico_ground_eval_openai.sh gpt-5.2-2025-12-11
```

#### Claude Opus 4.5
```bash
rm -rf results/hico_ground_claude && \
VERBOSE=1 USE_CACHE=0 bash run_hico_ground_eval_claude.sh claude-opus-4-5-20251101
```

### Action Referring Task (HICO)

Evaluates the model's ability to describe the action between highlighted person-object pairs.

#### Gemini 3 Pro (Best Performance)
```bash
rm -rf results/hico_action_gemini && \
VERBOSE=1 USE_CACHE=0 bash run_hico_action_referring_eval_gemini.sh gemini-3-pro-preview
```

#### OpenAI GPT-5.2
```bash
rm -rf results/hico_action_openai && \
VERBOSE=1 USE_CACHE=0 bash run_hico_action_referring_eval_openai.sh gpt-5.2-2025-12-11
```

#### Claude Opus 4.5
```bash
rm -rf results/hico_action_claude && \
VERBOSE=1 USE_CACHE=0 bash run_hico_action_referring_eval_claude.sh claude-opus-4-5-20251101
```

---

## Quick Testing (Limited Samples)

Add `MAX_IMAGES=N` to limit evaluation to first N images:

```bash
# SWIG - Test grounding with 5 images
rm -rf results/swig_ground_gemini && \
MAX_IMAGES=5 VERBOSE=1 USE_CACHE=0 bash run_swig_ground_eval_gemini.sh gemini-3-pro-preview

# SWIG - Test action referring with 5 images
rm -rf results/swig_action_gemini && \
MAX_IMAGES=5 VERBOSE=1 USE_CACHE=0 bash run_swig_action_referring_eval_gemini.sh gemini-3-pro-preview

# HICO - Test grounding with 5 images
rm -rf results/hico_ground_gemini && \
MAX_IMAGES=5 VERBOSE=1 USE_CACHE=0 bash run_hico_ground_eval_gemini.sh gemini-3-pro-preview

# HICO - Test action referring with 5 images
rm -rf results/hico_action_gemini && \
MAX_IMAGES=5 VERBOSE=1 USE_CACHE=0 bash run_hico_action_referring_eval_gemini.sh gemini-3-pro-preview
```

---

## Environment Variables

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `VERBOSE=1` | Enable detailed output and visualizations | `0` | `VERBOSE=1` |
| `USE_CACHE=0` | Disable caching (fresh evaluation) | `1` | `USE_CACHE=0` |
| `MAX_IMAGES=N` | Limit to first N images | all | `MAX_IMAGES=10` |
| `EXTENDED_THINKING=1` | Enable Claude's extended thinking mode | `0` | `EXTENDED_THINKING=1` |
| `OPTIMIZED_PROMPTS=0` | Use full detailed prompts (better accuracy) | SWIG:`1`, HICO:`0` | `OPTIMIZED_PROMPTS=0` |
| `WANDB=1` | Enable Weights & Biases logging | `0` | `WANDB=1` |
| `WANDB_PROJECT` | Custom W&B project name | auto | `WANDB_PROJECT=my-project` |
| `WANDB_RUN_NAME` | Custom W&B run name | auto | `WANDB_RUN_NAME=exp-001` |

---

## Weights & Biases (W&B) Logging

Enable experiment tracking with W&B by adding `WANDB=1` to your commands.

### Basic W&B Usage

```bash
# SWIG Grounding with W&B logging
rm -rf results/swig_ground_gemini && \
WANDB=1 VERBOSE=1 USE_CACHE=0 bash run_swig_ground_eval_gemini.sh gemini-3-pro-preview

# HICO Action Referring with W&B logging
rm -rf results/hico_action_openai && \
WANDB=1 VERBOSE=1 USE_CACHE=0 bash run_hico_action_referring_eval_openai.sh gpt-5.2-2025-12-11
```

### Custom W&B Project and Run Names

```bash
# Custom project name
rm -rf results/swig_ground_gemini && \
WANDB=1 WANDB_PROJECT=hoi-benchmark-v2 VERBOSE=1 USE_CACHE=0 \
bash run_swig_ground_eval_gemini.sh gemini-3-pro-preview

# Custom project and run name
rm -rf results/hico_ground_openai && \
WANDB=1 WANDB_PROJECT=hico-experiments WANDB_RUN_NAME=gpt5-high-reasoning \
VERBOSE=1 USE_CACHE=0 bash run_hico_ground_eval_openai.sh gpt-5.2-2025-12-11
```

### Default W&B Project Names

| Task | Dataset | Model | Default Project |
|------|---------|-------|-----------------|
| Grounding | SWIG | Gemini | `swig-grounding-gemini` |
| Grounding | SWIG | OpenAI | `swig-grounding-openai` |
| Grounding | SWIG | Claude | `swig-grounding-claude` |
| Action Referring | SWIG | Gemini | `swig-action-referring-gemini` |
| Action Referring | SWIG | OpenAI | `swig-action-referring-openai` |
| Action Referring | SWIG | Claude | `swig-action-referring-claude` |
| Grounding | HICO | Gemini | `hico-grounding-gemini` |
| Grounding | HICO | OpenAI | `hico-grounding-openai` |
| Grounding | HICO | Claude | `hico-grounding-claude` |
| Action Referring | HICO | Gemini | `hico-action-referring-gemini` |
| Action Referring | HICO | OpenAI | `hico-action-referring-openai` |
| Action Referring | HICO | Claude | `hico-action-referring-claude` |

---

## Thinking/Reasoning Effort Settings

Different tasks require different levels of reasoning effort for optimal performance.

### Grounding Task (Spatial Precision Required)

| Model | Parameter | SWIG Value | HICO Value | Notes |
|-------|-----------|------------|------------|-------|
| **Gemini 3 Pro** | `thinking_level` | **high** | **high** | Spatial reasoning needs deep thinking |
| **OpenAI GPT-5.2** | `reasoning_effort` | **high** | **medium** | HICO uses medium for speed/accuracy balance |
| **Claude Opus 4.5** | `extended_thinking` | **disabled** | **disabled** | Extended thinking hurts performance |

### Action Referring Task (Language Understanding)

| Model | Parameter | Value | Notes |
|-------|-----------|-------|-------|
| **Gemini 3 Pro** | `thinking_level` | **low** | Language tasks need less reasoning |
| **OpenAI GPT-5.2** | `reasoning_effort` | **medium** | Balanced for action understanding |
| **Claude Opus 4.5** | `extended_thinking` | **disabled** | Standard mode works best |

---

## Benchmark Results

### SWIG-HOI Results (5 samples)

#### Grounding Task (SWIG)

| Model | Thinking | AR@0.5 | AR@0.75 | AR |
|-------|----------|--------|---------|-----|
| **Gemini 3 Pro** | high | **66.67%** | **66.67%** | **60%** |
| **OpenAI GPT-5.2** | high | 60% | 0% | 26% |
| **Claude Opus 4.5** | disabled | 20% | 0% | 4% |

#### Action Referring Task (SWIG)

| Model | Thinking | Exact Match | BERTScore F1 |
|-------|----------|-------------|--------------|
| **Gemini 3 Pro** | low | **40%** | **81.79%** |
| OpenAI GPT-5.2 | medium | 20% | 80.54% |
| Claude Opus 4.5 | disabled | 20% | 78.68% |

### HICO-DET Results (5 samples)

#### Grounding Task (HICO)

| Model | Thinking | AR@0.5 | AR@0.75 | AR | Notes |
|-------|----------|--------|---------|-----|-------|
| **Gemini 3 Pro** | high | **66.67%** | **66.67%** | **72.5%** | Best spatial accuracy |
| **OpenAI GPT-5.2** | medium | 33.33% | 0% | 6.67% | Medium effort for speed |
| **Claude Opus 4.5** | disabled | 0% | 0% | 0% | Semantic understanding limitation |

> **Note on Claude HICO Grounding:** Claude's 0% AR is due to semantic understanding limitations - it sometimes detects the wrong person for actions (e.g., detecting a person walking beside a horse instead of the person riding it). This is a model capability issue, not a code bug.

#### Action Referring Task (HICO)

| Model | Thinking | Exact Match | BERTScore F1 |
|-------|----------|-------------|--------------|
| **Gemini 3 Pro** | low | **60%** | **93.01%** |
| OpenAI GPT-5.2 | medium | 60% | 93.01% |
| Claude Opus 4.5 | disabled | 60% | 93.01% |

---

## Output Files

Results are saved to `results/` directory:
- `*_results_*.json` - Raw predictions
- `*_metrics_*.json` - Computed metrics
- `*_visualizations/` - Visualization images (when VERBOSE=1)

For batch mode, additional files:
- `checkpoints/` - Checkpoint files for resumability
- `batch_requests.json` - Saved batch requests
- `batch_eval_*.log` - Batch evaluation logs

---

## Project Structure

```
hoi-benchmarks/
├── # Batch Processing (NEW - 50% cost savings)
├── run_batch_eval.sh              # Unified batch evaluation runner
├── batch_api_utils.py             # Batch API implementations (Claude, Gemini, OpenAI)
├── checkpoint_manager.py          # Checkpointing for resumability
├── eval_swig_action_referring_batch.py   # SWIG action batch evaluation
├── eval_swig_ground_batch.py             # SWIG grounding batch evaluation
├── eval_hico_action_referring_batch.py   # HICO action batch evaluation
├── eval_hico_ground_batch.py             # HICO grounding batch evaluation
│
├── # Real-time Evaluation (Original)
├── eval_swig_action_referring_*.py       # SWIG action real-time (claude/gemini/openai)
├── eval_swig_ground_*.py                 # SWIG grounding real-time
├── eval_hico_action_referring_*.py       # HICO action real-time
├── eval_hico_ground_*.py                 # HICO grounding real-time
├── run_*_eval_*.sh                       # Real-time evaluation shell scripts
│
├── # Shared Utilities
├── eval_api_utils.py              # Shared evaluation utilities
├── response_cache.py              # Response caching for real-time mode
├── calculate_bertscore.py         # BERTScore computation
│
├── # Configuration
├── .env                           # API keys (create this file)
├── pyproject.toml                 # Python dependencies
└── README.md                      # This file
```

---

## Troubleshooting

### Batch Job Stuck or Failed

1. Check the checkpoint status:
```bash
ls -la results/<task>_<provider>_batch/checkpoints/
```

2. Resume the job:
```bash
RESUME=1 bash run_batch_eval.sh <provider> <task>
```

3. If the batch expired (>24h), delete checkpoint and restart:
```bash
rm -rf results/<task>_<provider>_batch/checkpoints/
bash run_batch_eval.sh <provider> <task>
```

### Rate Limit Errors (Real-time Mode)

Switch to **Batch Mode** which uses separate rate limit pools:
```bash
bash run_batch_eval.sh <provider> <task>
```

### API Key Issues

Ensure your `.env` file has the correct keys:
```bash
GEMINI_API_KEY=your-gemini-key
ANTHROPIC_API_KEY=your-claude-key  # or CLAUDE_API_KEY
OPENAI_API_KEY=your-openai-key     # or OPEN_AI_API_KEY
```

### Claude Batch "Model Not Found" Errors

If Claude batch jobs complete but all requests fail with "model not found":

1. **Use model aliases**: Use short alias names like `claude-sonnet-4-5` instead of full versioned names
2. **Confirmed working models**:
   - `claude-sonnet-4-5` (recommended)
   - `claude-haiku-4-5` (cheapest)
   - `claude-opus-4-5` (best accuracy)

3. **Try real-time mode**: If batch fails, use the real-time evaluation scripts instead:
```bash
bash run_swig_action_referring_eval_claude.sh claude-sonnet-4-5
```

---

## References

- [Anthropic Batch API Documentation](https://platform.claude.com/docs/en/build-with-claude/batch-processing)
- [Google Gemini Batch API Documentation](https://ai.google.dev/gemini-api/docs/batch-api)
- [OpenAI Batch API Documentation](https://platform.openai.com/docs/guides/batch)
