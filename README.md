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
