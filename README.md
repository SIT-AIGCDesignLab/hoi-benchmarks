# HOI Benchmark Evaluation

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

## Evaluation Commands

### Grounding Task

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

#### Claude Opus 4.5 (with Extended Thinking)
```bash
rm -rf results/swig_ground_claude && \
VERBOSE=1 USE_CACHE=0 EXTENDED_THINKING=1 bash run_swig_ground_eval_claude.sh claude-opus-4-5-20251101
```

---

### Action Referring Task

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

## Quick Testing (Limited Samples)

Add `MAX_IMAGES=N` to limit evaluation to first N images:

```bash
# Test grounding with 5 images
rm -rf results/swig_ground_gemini && \
MAX_IMAGES=5 VERBOSE=1 USE_CACHE=0 bash run_swig_ground_eval_gemini.sh gemini-3-pro-preview

# Test action referring with 5 images
rm -rf results/swig_action_gemini && \
MAX_IMAGES=5 VERBOSE=1 USE_CACHE=0 bash run_swig_action_referring_eval_gemini.sh gemini-3-pro-preview
```

---

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `VERBOSE=1` | Enable detailed output and visualizations | `VERBOSE=1` |
| `USE_CACHE=0` | Disable caching (fresh evaluation) | `USE_CACHE=0` |
| `MAX_IMAGES=N` | Limit to first N images | `MAX_IMAGES=10` |
| `EXTENDED_THINKING=1` | Enable Claude's extended thinking mode | `EXTENDED_THINKING=1` |

---

## Thinking/Reasoning Effort Settings

Different tasks require different levels of reasoning effort for optimal performance.

### Grounding Task (Spatial Precision Required)

| Model | Parameter | Value | Notes |
|-------|-----------|-------|-------|
| **Gemini 3 Pro** | `thinking_level` | **high** | Spatial reasoning needs deep thinking |
| **OpenAI GPT-5.2** | `reasoning_effort` | **high** | Improves bounding box precision |
| **Claude Opus 4.5** | `extended_thinking` | **disabled** | Extended thinking hurts performance |

### Action Referring Task (Language Understanding)

| Model | Parameter | Value | Notes |
|-------|-----------|-------|-------|
| **Gemini 3 Pro** | `thinking_level` | **low** | Language tasks need less reasoning |
| **OpenAI GPT-5.2** | `reasoning_effort` | **medium** | Balanced for action understanding |
| **Claude Opus 4.5** | `extended_thinking` | **disabled** | Standard mode works best |

---

## Benchmark Results (5 samples)

### Grounding Task

| Model | Thinking | AR@0.5 | AR@0.75 | AR |
|-------|----------|--------|---------|-----|
| **Gemini 3 Pro** | high | **66.67%** | **66.67%** | **60%** |
| **OpenAI GPT-5.2** | high | 60% | 0% | 26% |
| **Claude Opus 4.5** | disabled | 20% | 0% | 4% |

### Action Referring Task

| Model | Thinking | Exact Match | BERTScore F1 |
|-------|----------|-------------|--------------|
| **Gemini 3 Pro** | low | **40%** | **81.79%** |
| OpenAI GPT-5.2 | medium | 20% | 80.54% |
| Claude Opus 4.5 | disabled | 20% | 78.68% |

---

## Output Files

Results are saved to `results/` directory:
- `*_results_*.json` - Raw predictions
- `*_metrics_*.json` - Computed metrics
- `*_visualizations/` - Visualization images (when VERBOSE=1)
