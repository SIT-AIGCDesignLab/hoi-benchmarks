# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an evaluation framework for benchmarking Vision-Language Models on Human-Object Interaction (HOI) tasks using the SWIG-HOI dataset. It evaluates the **Qwen3VL model family** on two complementary tasks:

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

```
Shell Script (run_swig_*_qwen3vl.sh)
    ↓
Environment Setup (GPU, paths, flags)
    ↓
Python Evaluation Script (eval_swig_*_qwen3vl.py)
    ↓
Load Qwen3VL Model (with auto device mapping)
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

## Dataset Structure

### Expected Paths

```
../data/swig_hoi/
└── images_512/          # SWIG images (512px resolution)
    └── *.jpg

groma_data/benchmarks/
├── swig_action_referring_test.json   # Action referring annotations
└── swig_ground_test.json             # Grounding annotations
```

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

## Important Notes

- **SWIG Actions**: All actions are in present participle form with -ing suffix (e.g., "stapling", not "staple")
- **Person-Person Interactions**: Dataset includes cases where both bounding boxes refer to people
- **Bounding Box Format**: All boxes are converted to [0, 1000] range for Qwen3VL model input
- **GPU Memory**: Large models (32B) require significant GPU memory; use auto device mapping
- **Thinking Token**: For Thinking models, the special token ID **151668** marks the end of reasoning (`</think>` tag)
