# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an evaluation framework for benchmarking Vision-Language Models on Human-Object Interaction (HOI) tasks using the **HICO-DET** and **SWIG-HOI** datasets. It evaluates two model families:

1. **Qwen3VL baseline** (Instruct/Thinking, 8B/32B) — direct inference via local GPU
2. **SFT-trained Qwen3VL tool-use agent** — served via vLLM, uses `zoom_in`/`zoom_out` tool calls for iterative visual inspection

Two complementary tasks:

1. **Action Referring**: Given person and object bounding boxes, predict the connecting action verb (e.g., "riding", "stirring")
2. **Grounding**: Given an action description and object category, detect all person-object pair bounding boxes in the image

The framework supports both **Instruct** and **Thinking** variants of Qwen3VL models (8B and 32B sizes).

## Common Commands

### Running Action Referring Evaluation

```bash
# Basic evaluation on GPU 0 with Thinking model (default)
bash run_swig_action_referring_eval_qwen3vl.sh 0

# Use different model
bash run_swig_action_referring_eval_qwen3vl.sh 0 "Qwen/Qwen3-VL-32B-Instruct"

# With verbose output (per-triplet results + visualizations)
VERBOSE=1 bash run_swig_action_referring_eval_qwen3vl.sh 0

# Quick test on first 10 triplets
MAX_IMAGES=10 bash run_swig_action_referring_eval_qwen3vl.sh 0

# Enable Weights & Biases logging
WANDB=1 bash run_swig_action_referring_eval_qwen3vl.sh 0

# Combine multiple flags
VERBOSE=1 MAX_IMAGES=10 WANDB=1 bash run_swig_action_referring_eval_qwen3vl.sh 1
```

### Running Grounding Evaluation

```bash
# Basic evaluation on GPU 0 with Thinking model (default)
bash run_swig_ground_eval_qwen3vl.sh 0

# Use different model
bash run_swig_ground_eval_qwen3vl.sh 0 "Qwen/Qwen3-VL-8B-Instruct"

# With verbose output (per-image results)
VERBOSE=1 bash run_swig_ground_eval_qwen3vl.sh 0

# Quick test on first 10 images
MAX_IMAGES=10 bash run_swig_ground_eval_qwen3vl.sh 0

# Enable Weights & Biases logging
WANDB=1 WANDB_PROJECT="my-project" bash run_swig_ground_eval_qwen3vl.sh 0
```

### LLM-as-a-Judge Evaluation (Open-Source, Free)

Replaces the expensive Gemini Vertex AI judge with a local NVIDIA Nemotron model.
Judge model: **[nvidia/Llama-3_3-Nemotron-Super-49B-v1_5-FP8](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5-FP8)**
- Non-Qwen (avoids self-preference bias), outperforms o1-mini on JudgeBench (arxiv:2505.00949)
- Requires FP8-capable GPU (H100 / A100-Ada / Blackwell), ~50GB VRAM

#### Step 1 — Start the vLLM server

```bash
python -m vllm.entrypoints.openai.api_server \
    --model nvidia/Llama-3_3-Nemotron-Super-49B-v1_5-FP8 \
    --quantization fp8 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 32768 \
    --trust-remote-code \
    --port 8000
```

Wait until you see `INFO: Application startup complete.` before proceeding.

#### Step 2 — Run the evaluation

```bash
# Evaluate all 14 result files
bash run_llm_judge_opensource_all.sh

# Filter to specific result folders
bash run_llm_judge_opensource_all.sh --folders swig_action_claude_batch
bash run_llm_judge_opensource_all.sh --folders swig_action_claude_batch,hico_action_openai_batch

# Quick test on first 10 samples with verbose output
MAX_IMAGES=10 VERBOSE=1 bash run_llm_judge_opensource_all.sh

# Enable W&B logging
WANDB=1 WANDB_PROJECT="my-project" bash run_llm_judge_opensource_all.sh

# Run a single file directly
python eval_llm_judge_opensource.py \
    --pred-file results/swig_referring_ours.json \
    --verbose

# Resume an interrupted run
python eval_llm_judge_opensource.py \
    --pred-file results/swig_referring_ours.json \
    --resume results/swig_referring_ours_checkpoint.json

# Override model or server URL
python eval_llm_judge_opensource.py \
    --pred-file results/swig_referring_ours.json \
    --model nvidia/Llama-3_3-Nemotron-Super-49B-v1_5-FP8 \
    --base-url http://localhost:8000/v1
```

#### Output files (per pred-file)

- `*_judge_opensource_{timestamp}.json` — full results with `judge_score` (1–10) + `judge_reason`
- `*_judge_opensource_{timestamp}_metrics.json` — `average_score`, `score_distribution`, model name

#### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VERBOSE=1` | off | Print per-sample score + reason |
| `MAX_IMAGES=N` | all | Limit to first N samples (quick test) |
| `WANDB=1` | off | Log metrics to Weights & Biases |
| `WANDB_PROJECT` | `llm-judge-opensource` | W&B project name |
| `WANDB_RUN_NAME` | auto | W&B run name |
| `BASE_URL` | `http://localhost:8000/v1` | Override vLLM server URL |

### Running SFT Tool-Use Agent Evaluation

Evaluates a SFT-trained Qwen3VL checkpoint that uses `zoom_in`/`zoom_out` tool calls.
Requires a vLLM server running with the SFT checkpoint.

```bash
# SWIG-HOI Action Referring (SFT agent)
bash run_swig_action_sft_eval.sh 0
bash run_swig_action_sft_eval.sh 0 http://localhost:8000

# SWIG-HOI Grounding (SFT agent)
bash run_swig_ground_sft_eval.sh 0

# HICO-DET Action Referring (SFT agent)
bash run_hico_action_sft_eval.sh 0

# HICO-DET Grounding (SFT agent)
bash run_hico_ground_sft_eval.sh 0

# Quick test with verbose
MAX_IMAGES=10 VERBOSE=1 bash run_swig_action_sft_eval.sh 0

# Run single image
IMAGE_ID=tattooing_86.jpg bash run_swig_action_sft_eval.sh 0

# Resume interrupted run
RESUME=1 bash run_swig_action_sft_eval.sh 0
```

**Key environment variables for SFT eval:**

| Variable | Default | Description |
|----------|---------|-------------|
| `VERBOSE=1` | off | Per-sample results |
| `MAX_IMAGES=N` | all | Limit samples |
| `MAX_TURNS=N` | 5 | Max tool-call turns |
| `CHECKPOINT_PATH` | local path | SFT checkpoint for auto-starting vLLM |
| `RESUME=1` | off | Resume from latest `.json.partial.jsonl` |
| `IMAGE_ID=<file>` | — | Run only samples matching this filename |
| `WANDB=1` | off | W&B logging |

**Default checkpoint paths:**
- Local: `/media/shaun/workspace/AdaTooler-V/checkpoints/qwen3VL-4B`
- GPU server (RTX 6000 Ada): `/mnt/d/Work/latest_checkpoints/sft-checkpoints/qwen3VL-4B`

**Output directory:** `results-sft/{swig,hico}_{action,ground}_sft/`

### Running HICO-DET Baseline Evaluation

```bash
# HICO-DET Action Referring (Qwen3VL baseline, via vLLM)
bash run_hico_action_referring_eval_qwen3vl.sh 0

# HICO-DET Grounding (Qwen3VL baseline, via vLLM)
bash run_hico_ground_eval_qwen3vl.sh 0

# Quick test
MAX_IMAGES=10 bash run_hico_action_referring_eval_qwen3vl.sh 0

# Resume from checkpoint
RESUME=1 bash run_hico_action_referring_eval_qwen3vl.sh 0

# Single image
IMAGE_ID=HICO_test2015_00003584.jpg bash run_hico_action_referring_eval_qwen3vl.sh 0
```

**Note:** HICO-DET baseline scripts require a running vLLM server (`VLLM_URL`, default `http://localhost:8000`).

### Computing BERTScore (Post-Evaluation)

After running action referring evaluation, compute semantic similarity:

```bash
python calculate_bertscore.py \
    --pred-file results/swig_action_qwen3vl_results_TIMESTAMP.json \
    --model roberta-large \
    --batch-size 32 \
    --device cuda:0
```

Supported BERT models: `roberta-large` (default), `microsoft/deberta-v2-xlarge`, `distilbert-base-uncased`

## Architecture

### Evaluation Pipeline Flow

**Baseline (direct inference):**
```
Shell Script (run_{swig,hico}_*_qwen3vl.sh)
    ↓
Environment Setup (GPU, paths, flags)
    ↓
Python Evaluation Script (eval_{swig,hico}_*_qwen3vl.py)
    ↓
Load Qwen3VL Model (with auto device mapping) OR connect to vLLM server
    ↓
For each image/triplet:
    - Prepare prompt with bounding boxes
    - Generate prediction
    - Extract thinking content (if Thinking model)
    - Parse response
    ↓
Compute Metrics:
    - Action Referring: METEOR, CIDEr, BLEU, ROUGE-L
    - Grounding: COCO-style Average Recall @ IoU thresholds
    ↓
Save Results (JSON with timestamps)
    ↓
Optional: Log to Weights & Biases
```

**SFT Tool-Use Agent:**
```
Shell Script (run_{swig,hico}_*_sft_eval.sh)
    ↓
Check/start vLLM server with SFT checkpoint
    ↓
Python Evaluation Script (eval_{swig,hico}_*_sft_qwen3vl.py)
    ↓
Multi-turn tool-call loop (up to MAX_TURNS):
    - Model calls zoom_in(bbox) or zoom_out() as needed
    - Each tool call returns a cropped image
    - Final turn: produce action/bbox prediction
    ↓
Same metrics as baseline
    ↓
Save Results + partial checkpoints (.json.partial.jsonl)
```

### Model Architecture Support

**Thinking Models** (e.g., Qwen3-VL-8B-Thinking):
- Generate reasoning process in `<think>` tags before final answer
- Special token ID **151668** used to extract `</think>` tag
- Thinking content saved separately in `*_thinking.jsonl` files
- Uses neutral prompts without "person"/"object" hints

**Instruct Models** (e.g., Qwen3-VL-8B-Instruct):
- Direct answer generation without explicit reasoning
- Standard prompt format with task-specific hints

Both model types:
- Convert bounding boxes to [0, 1000] range for model input
- Support multi-GPU with `CUDA_VISIBLE_DEVICES`
- Auto device mapping for large models

## Key Scripts

### Evaluation Scripts

**`eval_swig_action_referring_qwen3vl.py`** (916 lines)
- **Task**: Predict action given (person, object) bounding boxes
- **Input**: Image + 2 bounding boxes (person and object coordinates)
- **Output**: Action phrase in -ing form (e.g., "riding bicycle")
- **Metrics**: METEOR (semantic similarity), CIDEr (corpus consensus), BLEU, ROUGE-L
- **Key Functions**:
  - `load_qwen3vl_model()`: Loads model with auto device mapping
  - `prepare_action_referring_prompt()`: Creates prompt with bbox visualization
  - `extract_thinking_content()`: Extracts reasoning from thinking models (token ID 151668)
  - `compute_metrics()`: Uses pycocoevalcap for NLP metrics

**`eval_swig_ground_qwen3vl.py`** (1,170 lines)
- **Task**: Detect ALL person-object pairs performing specified action
- **Input**: Image + action description + object category
- **Output**: List of `{"person_bbox": [...], "object_bbox": [...]}` pairs
- **Metrics**: COCO Average Recall @ IoU thresholds (0.5, 0.75, 0.50:0.95)
- **Key Functions**:
  - `calculate_iou()`: Computes Intersection-over-Union for bbox matching
  - `parse_json_response()`: Extracts structured output from model text
  - `evaluate_with_coco()`: COCO-style evaluation for detection metrics
- **Special Feature**: Supports person-person interactions (not just person-object)

**`eval_swig_action_referring_sft_qwen3vl.py`**
- **Task**: Action Referring via SFT tool-use agent on SWIG-HOI
- **Input**: vLLM server + image + bounding boxes + object proposals
- **Tool calls**: `zoom_in(bbox_2d)` / `zoom_out()` in multi-turn loop
- **Proposals dir**: `../../hoi-dataset-curation/output/test_proposals`
- **Same metrics** as baseline (METEOR, CIDEr, BLEU, ROUGE-L)

**`eval_swig_ground_sft_qwen3vl.py`**
- **Task**: Grounding via SFT tool-use agent on SWIG-HOI
- **Same tool-call loop** as action referring SFT

**`eval_hico_action_referring_sft_qwen3vl.py`** / **`eval_hico_ground_sft_qwen3vl.py`**
- Same SFT agent pattern for HICO-DET dataset

**`eval_llm_judge_opensource.py`**
- **Purpose**: LLM-as-a-Judge scoring of action referring predictions
- **Judge model**: `nvidia/Llama-3_3-Nemotron-Super-49B-v1_5-FP8` via local vLLM
- **Input**: Prediction JSON + ground truth; **Output**: score 1–10 + reason per sample
- Supports `--resume` to continue interrupted runs

**`calculate_bertscore.py`** (458 lines)
- **Purpose**: Post-hoc semantic similarity evaluation for action predictions
- **Input**: Existing prediction JSON file (no model reloading needed)
- **Output**: Precision, Recall, F1 scores using BERT embeddings
- **Key Function**: `clean_text()` - Removes markdown formatting from model outputs

### Shell Scripts

**`run_swig_action_referring_eval_qwen3vl.sh`**
- Comprehensive wrapper with GPU handling, path validation, and output management
- Default model: `Qwen/Qwen3-VL-8B-Thinking`
- Default output: `results-redo/swig_action_qwen3vl_thinking/`

**`run_swig_ground_eval_qwen3vl.sh`**
- Similar structure for grounding task
- Default output: `results-redo/swig_ground_qwen3vl_thinking/`

**`run_{swig,hico}_{action,ground}_sft_eval.sh`**
- SFT tool-use agent wrappers; auto-start vLLM if not running
- Default checkpoint: `/media/shaun/workspace/AdaTooler-V/checkpoints/qwen3VL-4B`
- Output: `results-sft/{dataset}_{task}_sft/`

**`run_llm_judge_opensource_all.sh`**
- Batch wrapper to run LLM judge on all result files
- Filters by `--folders` flag for targeted evaluation

## Dataset Structure

### Expected Paths

```
../dataset/
├── benchmarks_simplified/                                 # Simplified test annotations
│   ├── hico_action_referring_test_simplified.json         # 33,405 samples
│   ├── hico_ground_test_simplified.json                   # 20,028 samples
│   ├── swig_action_referring_test_simplified.json         # 19,680 samples
│   └── swig_ground_test_simplified.json                   # 19,333 samples
├── hico_20160224_det/images/test2015/                     # HICO-DET test images
└── swig_hoi/images_512/                                   # SWIG images (512px)

../../hoi-dataset-curation/output/test_proposals/          # Object proposals for SFT agent
```

The SFT scripts use `../dataset/` relative paths. The baseline Qwen3VL scripts for HICO also use `../dataset/` (not `groma_data/`).

### Annotation Format

**Action Referring** (`swig_action_referring_test.json`):
```json
{
  "annotations": [
    {
      "image_id": "...",
      "person_bbox": [x1, y1, x2, y2],
      "object_bbox": [x1, y1, x2, y2],
      "action": "riding"  // Ground truth action in -ing form
    }
  ]
}
```

**Grounding** (`swig_ground_test.json`):
```json
{
  "samples": [
    {
      "image_id": "...",
      "action": "riding",
      "object": "bicycle",
      "pairs": [  // Ground truth pairs
        {
          "person_bbox": [x1, y1, x2, y2],
          "object_bbox": [x1, y1, x2, y2]
        }
      ]
    }
  ]
}
```

## Output File Conventions

All output files include timestamps in format `YYYYMMDD_HHMMSS`:

### Action Referring Outputs
- `swig_action_qwen3vl_results_{timestamp}.json` - Raw predictions with thinking content
- `swig_action_qwen3vl_results_{timestamp}_metrics.json` - METEOR/CIDEr/BLEU/ROUGE-L scores
- `swig_action_qwen3vl_results_{timestamp}_per_triplet.json` - Per-triplet analysis (VERBOSE mode)
- `swig_action_qwen3vl_results_{timestamp}_per_action.json` - Per-action breakdown (VERBOSE mode)
- `swig_action_qwen3vl_results_{timestamp}_thinking.jsonl` - Thinking process only (Thinking models)
- `swig_action_qwen3vl_evaluation_{timestamp}.log` - Full evaluation log

### Grounding Outputs
- `swig_ground_qwen3vl_results_{timestamp}.json` - Raw predictions with thinking content
- `swig_ground_qwen3vl_results_{timestamp}_metrics.json` - COCO AR metrics
- `swig_ground_qwen3vl_results_{timestamp}_action_stats.json` - Per-action statistics
- `swig_ground_qwen3vl_results_{timestamp}_thinking.jsonl` - Thinking process only (Thinking models)
- `swig_ground_qwen3vl_evaluation_{timestamp}.log` - Full evaluation log

## Key Metrics Explained

### Action Referring Metrics
- **METEOR**: Semantic similarity using stemming and synonyms (0-100%, higher is better)
- **CIDEr**: Corpus consensus, measures agreement with multiple references (0-200%+, higher is better)
- **BLEU**: N-gram overlap with reference (0-100%, higher is better)
- **ROUGE-L**: Longest common subsequence F-score (0-100%, higher is better)

### Grounding Metrics (COCO-style)
- **AR**: Average Recall @ IoU=0.50:0.95 (primary metric)
- **AR@0.5**: Average Recall @ IoU=0.50 (lenient threshold)
- **AR@0.75**: Average Recall @ IoU=0.75 (strict threshold)

IoU (Intersection-over-Union) measures bounding box overlap quality. Higher thresholds require more precise localization.

## Output File Conventions (SFT Agent)

Output files under `results-sft/{dataset}_{task}_sft/`:
- `*_sft_results_{timestamp}.json` — predictions with tool-call history
- `*_sft_results_{timestamp}_metrics.json` — aggregate metrics
- `*_sft_evaluation_{timestamp}.log` — full evaluation log
- `*.json.partial.jsonl` — checkpoint for resuming interrupted runs

## Important Notes

- **SWIG Actions**: All actions are in present participle form with -ing suffix (e.g., "stapling", not "staple")
- **Person-Person Interactions**: SWIG-HOI dataset includes cases where both bounding boxes refer to people
- **Bounding Box Format**: All boxes are converted to [0, 1000] range for Qwen3VL model input
- **GPU Memory**: Large models (32B) require significant GPU memory; use auto device mapping
- **Thinking Token**: For Thinking models, the special token ID **151668** marks the end of reasoning (`</think>` tag)
- **SFT Agent System Prompt**: The system prompt (including tool definitions) in eval scripts **must match SFT training exactly** or performance degrades
- **Object Proposals**: SFT agent uses pre-computed proposals from `hoi-dataset-curation/output/test_proposals/`; ensure this path exists before running SFT eval
