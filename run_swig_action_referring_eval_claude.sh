#!/bin/bash
################################################################################
# SWIG-HOI Action Referring Task Evaluation Script for Claude API
# Evaluates Claude models on action prediction using METEOR, CIDEr, and BERTScore
#
# Task: Given (person, object) bounding boxes, predict the connecting action
# Metrics: METEOR, CIDEr, BLEU, ROUGE-L, BERTScore
#
# Supports extended thinking mode for Claude models
#
# Usage:
#   bash run_swig_action_referring_eval_claude.sh [MODEL] [OUTPUT_DIR]
#
# Examples:
#   # Basic usage with Sonnet 4.5
#   ANTHROPIC_API_KEY=xxx bash run_swig_action_referring_eval_claude.sh
#
#   # With extended thinking
#   EXTENDED_THINKING=1 bash run_swig_action_referring_eval_claude.sh
#
#   # Quick test on first 10 triplets
#   MAX_IMAGES=10 bash run_swig_action_referring_eval_claude.sh
#
#   # With BERTScore computation
#   COMPUTE_BERTSCORE=1 bash run_swig_action_referring_eval_claude.sh
#
#   # Use different model
#   bash run_swig_action_referring_eval_claude.sh "claude-opus-4.5-20250514"
#
#   # Combine multiple flags
#   VERBOSE=1 MAX_IMAGES=10 EXTENDED_THINKING=1 COMPUTE_BERTSCORE=1 bash run_swig_action_referring_eval_claude.sh
#
# Environment Variables:
#   VERBOSE=1              Show per-triplet results
#   MAX_IMAGES=N          Limit to first N triplets (for quick testing)
#   WANDB=1               Enable Weights & Biases logging
#   EXTENDED_THINKING=1   Enable Claude extended thinking mode
#   COMPUTE_BERTSCORE=1   Compute BERTScore metric
#   WANDB_PROJECT         W&B project name (default: swig-action-referring-claude)
#   WANDB_RUN_NAME        W&B run name (default: auto-generated)
#   ANTHROPIC_API_KEY     Required: Your Anthropic API key
#
# Cost Optimization Variables (NEW):
#   USE_CACHE=1           Enable response caching (default: 1, 50-100% savings on reruns)
#   CACHE_DIR=./cache     Cache directory (default: ./cache)
#   OPTIMIZE_IMAGES=1     Enable image optimization (default: 0, 30-50% savings)
#   IMAGE_MAX_SIZE=448    Max image dimension in pixels (default: 448, conservative)
#   IMAGE_QUALITY=90      JPEG quality 1-100 (default: 90, high quality)
#   OPTIMIZED_PROMPTS=1   Use shorter prompts (default: 1, 20-40% savings)
#   CONCURRENT_REQUESTS=N Number of concurrent API requests (default: 1 sequential)
#
# Output files:
#   {output_dir}/swig_action_claude_results_{timestamp}.json          # Raw predictions
#   {output_dir}/swig_action_claude_results_{timestamp}_metrics.json  # Metrics
#   {output_dir}/swig_action_claude_results_{timestamp}_bertscore.json # BERTScore (if computed)
#   {output_dir}/swig_action_claude_results_{timestamp}_thinking.jsonl # Thinking content (if enabled)
#   {output_dir}/swig_action_claude_results_{timestamp}_per_sample.json # Per-sample details (VERBOSE)
#   {output_dir}/swig_action_claude_evaluation_{timestamp}.log        # Full log
################################################################################

set -e  # Exit on error

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    echo "Loading environment variables from .env file..."
    export $(grep -v '^#' .env | grep -v '^$' | xargs)
fi

# Activate virtual environment if it exists
if [ -d .venv ]; then
    source .venv/bin/activate
fi

# Configuration with defaults
MODEL_NAME="${1:-claude-opus-4-5-20251101}"
OUTPUT_DIR="${2:-results/swig_action_claude}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Timestamp for output files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/swig_action_claude_evaluation_${TIMESTAMP}.log"

# SWIG dataset paths
# Use environment variable if set, otherwise use default
SWIG_ROOT="${SWIG_ROOT:-data/swig_hoi}"
IMG_PREFIX="${IMG_PREFIX:-${SWIG_ROOT}/images_512}"
ANN_FILE="${ANN_FILE:-data/benchmarks_simplified/swig_action_referring_test_simplified.json}"
PRED_FILE="${OUTPUT_DIR}/swig_action_claude_results_${TIMESTAMP}.json"

echo "========================================================================"
echo "SWIG-HOI Action Referring Evaluation (Claude API)"
echo "========================================================================"
echo "Model:       $MODEL_NAME"
echo "Annotation:  $ANN_FILE"
echo "Images:      $IMG_PREFIX"
echo "Output:      $OUTPUT_DIR"
echo "Log file:    $LOG_FILE"
echo "Pred file:   $PRED_FILE"
echo "========================================================================"
echo ""

# API key will be loaded from .env by Python script (via python-dotenv)
echo ""

# Check if files exist
if [ ! -f "$ANN_FILE" ]; then
    echo "ERROR: Annotation file not found at $ANN_FILE"
    echo "Please ensure the benchmark file has been generated"
    exit 1
fi

if [ ! -d "$IMG_PREFIX" ]; then
    echo "ERROR: Images directory not found at $IMG_PREFIX"
    echo "Please check the path to SWIG images_512"
    exit 1
fi

# Count number of test images
NUM_IMAGES=$(ls -1 "$IMG_PREFIX"/*.jpg 2>/dev/null | wc -l)
echo "Found $NUM_IMAGES images in test set"
echo ""

echo "Starting evaluation..."
echo ""

# Parse optional flags from environment variables
VERBOSE_FLAG=""
MAX_IMAGES_FLAG=""
WANDB_FLAG=""
EXTENDED_THINKING_FLAG=""

# Cost optimization flags with defaults
USE_CACHE="${USE_CACHE:-1}"  # Default: enabled
CACHE_DIR="${CACHE_DIR:-./cache}"
OPTIMIZE_IMAGES="${OPTIMIZE_IMAGES:-0}"  # Default: disabled (preserve original results)
IMAGE_MAX_SIZE="${IMAGE_MAX_SIZE:-448}"
IMAGE_QUALITY="${IMAGE_QUALITY:-90}"
OPTIMIZED_PROMPTS="${OPTIMIZED_PROMPTS:-1}"  # Default: enabled (20-40% savings)
CONCURRENT_REQUESTS="${CONCURRENT_REQUESTS:-1}"  # Default: sequential

if [ ! -z "$VERBOSE" ]; then
    VERBOSE_FLAG="--verbose"
    echo "✓ Verbose mode enabled (per-triplet results)"
fi

if [ ! -z "$MAX_IMAGES" ]; then
    MAX_IMAGES_FLAG="--max-images $MAX_IMAGES"
    echo "✓ Limiting to first $MAX_IMAGES triplets"
fi

if [ ! -z "$WANDB" ]; then
    WANDB_FLAG="--wandb"
    echo "✓ Weights & Biases logging enabled"

    # Optional WandB project and run name
    if [ ! -z "$WANDB_PROJECT" ]; then
        WANDB_FLAG="$WANDB_FLAG --wandb-project $WANDB_PROJECT"
        echo "  WandB project: $WANDB_PROJECT"
    else
        echo "  WandB project: swig-action-referring-claude (default)"
    fi

    if [ ! -z "$WANDB_RUN_NAME" ]; then
        WANDB_FLAG="$WANDB_FLAG --wandb-run-name $WANDB_RUN_NAME"
        echo "  WandB run name: $WANDB_RUN_NAME"
    fi
fi

if [ ! -z "$EXTENDED_THINKING" ]; then
    EXTENDED_THINKING_FLAG="--extended-thinking"
    echo "✓ Extended thinking mode enabled"
fi

echo "✓ BERTScore computation enabled (always)"
echo "  BERTScore model: microsoft/deberta-v2-xxlarge-mnli"

# Display cost optimization settings
echo ""
echo "Cost Optimization Settings:"
if [ "$USE_CACHE" = "1" ]; then
    echo "✓ Response caching enabled (cache dir: $CACHE_DIR)"
    echo "  Savings: 50-100% on reruns"
else
    echo "  Response caching disabled"
fi

if [ "$OPTIMIZED_PROMPTS" = "1" ]; then
    echo "✓ Optimized prompts enabled (75% shorter)"
    echo "  Savings: ~90 tokens per request"
else
    echo "  Using verbose prompts"
fi

if [ "$OPTIMIZE_IMAGES" = "1" ]; then
    echo "✓ Image optimization enabled"
    echo "  Max size: ${IMAGE_MAX_SIZE}px, Quality: ${IMAGE_QUALITY}"
    echo "  Savings: 30-50% vision tokens"
else
    echo "  Image optimization disabled (using original 512x512)"
fi

if [ "$CONCURRENT_REQUESTS" -gt 1 ]; then
    echo "✓ Concurrent processing enabled: $CONCURRENT_REQUESTS workers"
    echo "  Rate limit: 50 requests/minute (Claude default)"
else
    echo "  Sequential processing (1 request at a time)"
fi

echo ""

# Build evaluation command
EVAL_CMD="python3 eval_swig_action_referring_claude.py \
    --model-name \"$MODEL_NAME\" \
    --ann-file $ANN_FILE \
    --img-prefix $IMG_PREFIX \
    --pred-file $PRED_FILE"

# Add optional flags if they exist
if [ ! -z "$VERBOSE_FLAG" ]; then
    EVAL_CMD="$EVAL_CMD $VERBOSE_FLAG"
fi

if [ ! -z "$MAX_IMAGES_FLAG" ]; then
    EVAL_CMD="$EVAL_CMD $MAX_IMAGES_FLAG"
fi

if [ ! -z "$WANDB_FLAG" ]; then
    EVAL_CMD="$EVAL_CMD $WANDB_FLAG"
fi

if [ ! -z "$EXTENDED_THINKING_FLAG" ]; then
    EVAL_CMD="$EVAL_CMD $EXTENDED_THINKING_FLAG"
fi

# Add cost optimization flags
if [ "$USE_CACHE" = "1" ]; then
    EVAL_CMD="$EVAL_CMD --use-cache --cache-dir $CACHE_DIR"
fi

if [ "$OPTIMIZE_IMAGES" = "1" ]; then
    EVAL_CMD="$EVAL_CMD --optimize-images --image-max-size $IMAGE_MAX_SIZE --image-quality $IMAGE_QUALITY"
fi

if [ "$OPTIMIZED_PROMPTS" = "1" ]; then
    EVAL_CMD="$EVAL_CMD --optimized-prompts"
fi

if [ "$CONCURRENT_REQUESTS" -gt 1 ]; then
    EVAL_CMD="$EVAL_CMD --concurrent-requests $CONCURRENT_REQUESTS"
fi

# Execute the command
eval "$EVAL_CMD" 2>&1 | tee "$LOG_FILE"

# Check if evaluation succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "Evaluation Complete!"
    echo "========================================================================"
    echo "Results saved to:"
    echo "  Predictions:  $PRED_FILE"
    echo "  Metrics:      ${PRED_FILE//.json/_metrics.json}"
    echo "  Log:          $LOG_FILE"
    echo ""

    # Check if extended thinking was enabled
    if [ ! -z "$EXTENDED_THINKING" ]; then
        THINKING_FILE="${PRED_FILE//.json/_thinking.jsonl}"
        if [ -f "$THINKING_FILE" ]; then
            echo "Extended thinking outputs:"
            echo "  Thinking:     $THINKING_FILE"
            echo ""
        fi
    fi

    # Check if BERTScore was computed
    BERTSCORE_FILE="${PRED_FILE//.json/_bertscore.json}"
    if [ -f "$BERTSCORE_FILE" ]; then
        echo "BERTScore outputs:"
        echo "  BERTScore:    $BERTSCORE_FILE"
        echo ""
    fi

    if [ ! -z "$VERBOSE" ]; then
        PER_SAMPLE_FILE="${PRED_FILE//.json/_per_sample.json}"
        if [ -f "$PER_SAMPLE_FILE" ]; then
            echo "Verbose outputs:"
            echo "  Per-sample:   $PER_SAMPLE_FILE"
            echo ""
        fi
    fi

    echo "Key metrics:"
    echo "  METEOR:      Semantic similarity (0-100%, higher=better)"
    echo "  CIDEr:       Corpus consensus (0-200%+, higher=better)"
    echo "  BLEU:        N-gram overlap (0-100%, higher=better)"
    echo "  ROUGE-L:     Longest common subsequence (0-100%, higher=better)"
    if [ ! -z "$COMPUTE_BERTSCORE" ]; then
        echo "  BERTScore:   Semantic similarity using BERT embeddings (0-100%, higher=better)"
    fi
    echo ""
    echo "Note: SWIG actions are in -ing form (e.g., 'stirring', 'stapling')"
    echo "      Dataset includes person-person interactions"
    echo "========================================================================"
else
    echo ""
    echo "========================================================================"
    echo "ERROR: Evaluation failed!"
    echo "========================================================================"
    echo "Check the log file for details:"
    echo "  $LOG_FILE"
    echo "========================================================================"
    exit 1
fi
