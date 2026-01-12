#!/bin/bash
################################################################################
# HICO-DET Grounding Task Evaluation Script for Claude API
# Evaluates Claude models on grounding performance with COCO-style AR metrics
#
# Task: Detect person-object pairs with bounding boxes
# Metrics: COCO-style AR (Average Recall) at multiple IoU thresholds
#
# Supports extended thinking mode for Claude models
#
# Usage:
#   bash run_hico_ground_eval_claude.sh [MODEL] [OUTPUT_DIR]
#
# Examples:
#   # Basic usage with Sonnet 4.5
#   ANTHROPIC_API_KEY=xxx bash run_hico_ground_eval_claude.sh
#
#   # With extended thinking
#   EXTENDED_THINKING=1 bash run_hico_ground_eval_claude.sh
#
#   # Quick test on first 10 images
#   MAX_IMAGES=10 bash run_hico_ground_eval_claude.sh
#
#   # Use different model
#   bash run_hico_ground_eval_claude.sh "claude-opus-4.5-20250514"
#
#   # Combine multiple flags
#   VERBOSE=1 MAX_IMAGES=10 EXTENDED_THINKING=1 bash run_hico_ground_eval_claude.sh
#
# Environment Variables:
#   VERBOSE=1              Show per-image results
#   MAX_IMAGES=N          Limit to first N images (for quick testing)
#   WANDB=1               Enable Weights & Biases logging
#   EXTENDED_THINKING=1   Enable Claude extended thinking mode
#   WANDB_PROJECT         W&B project name (default: hico-grounding-claude)
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
#   {output_dir}/hico_ground_claude_results_{timestamp}.json          # Raw predictions
#   {output_dir}/hico_ground_claude_results_{timestamp}_metrics.json  # AR metrics
#   {output_dir}/hico_ground_claude_results_{timestamp}_thinking.jsonl # Thinking content (if enabled)
#   {output_dir}/hico_ground_claude_evaluation_{timestamp}.log        # Full log
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
OUTPUT_DIR="${2:-results/hico_ground_claude}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Timestamp for output files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/hico_ground_claude_evaluation_${TIMESTAMP}.log"

# SWIG dataset paths
# Use environment variable if set, otherwise use default
HICO_ROOT="${HICO_ROOT:-data/hico_20160224_det}"
IMG_PREFIX="${IMG_PREFIX:-${HICO_ROOT}/images/test2015}"
ANN_FILE="${ANN_FILE:-data/benchmarks_simplified/hico_ground_test_simplified.json}"
RESULT_FILE="${OUTPUT_DIR}/hico_ground_claude_results_${TIMESTAMP}.json"

echo "========================================================================"
echo "HICO-DET Grounding Evaluation (Claude API)"
echo "========================================================================"
echo "Model:       $MODEL_NAME"
echo "Annotation:  $ANN_FILE"
echo "Images:      $IMG_PREFIX"
echo "Output:      $OUTPUT_DIR"
echo "Log file:    $LOG_FILE"
echo "Result file: $RESULT_FILE"
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
    echo "Please check the path to SWIG images/test2015"
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

if [ ! -z "$VERBOSE" ]; then
    VERBOSE_FLAG="--verbose"
    echo "✓ Verbose mode enabled (per-image results)"
fi

if [ ! -z "$MAX_IMAGES" ]; then
    MAX_IMAGES_FLAG="--max-images $MAX_IMAGES"
    echo "✓ Limiting to first $MAX_IMAGES images"
fi

if [ ! -z "$WANDB" ]; then
    WANDB_FLAG="--wandb"
    echo "✓ Weights & Biases logging enabled"

    # Optional WandB project and run name
    if [ ! -z "$WANDB_PROJECT" ]; then
        WANDB_FLAG="$WANDB_FLAG --wandb-project $WANDB_PROJECT"
        echo "  WandB project: $WANDB_PROJECT"
    else
        echo "  WandB project: hico-grounding-claude (default)"
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

# Cost Optimization Flags
USE_CACHE_FLAG=""
OPTIMIZE_IMAGES_FLAG=""
OPTIMIZED_PROMPTS_FLAG=""
CACHE_DIR_FLAG=""
IMAGE_MAX_SIZE_FLAG=""
IMAGE_QUALITY_FLAG=""
CONCURRENT_REQUESTS_FLAG=""

# Default to enabled for cache and optimized prompts
if [ "${USE_CACHE:-1}" = "1" ]; then
    USE_CACHE_FLAG="--use-cache"
    echo "✓ Response caching enabled (50-100% savings on reruns)"
fi

if [ ! -z "$CACHE_DIR" ]; then
    CACHE_DIR_FLAG="--cache-dir $CACHE_DIR"
    echo "  Cache directory: $CACHE_DIR"
fi

if [ ! -z "$OPTIMIZE_IMAGES" ]; then
    OPTIMIZE_IMAGES_FLAG="--optimize-images"
    echo "✓ Image optimization enabled (30-50% vision token savings)"
    
    if [ ! -z "$IMAGE_MAX_SIZE" ]; then
        IMAGE_MAX_SIZE_FLAG="--image-max-size $IMAGE_MAX_SIZE"
        echo "  Image max size: $IMAGE_MAX_SIZE"
    fi
    
    if [ ! -z "$IMAGE_QUALITY" ]; then
        IMAGE_QUALITY_FLAG="--image-quality $IMAGE_QUALITY"
        echo "  Image quality: $IMAGE_QUALITY"
    fi
fi

# HICO requires full detailed prompts for better accuracy (default: disabled)
if [ "${OPTIMIZED_PROMPTS:-0}" = "1" ]; then
    OPTIMIZED_PROMPTS_FLAG="--optimized-prompts"
    echo "✓ Optimized prompts enabled (20-40% token savings)"
fi

if [ ! -z "$CONCURRENT_REQUESTS" ]; then
    CONCURRENT_REQUESTS_FLAG="--concurrent-requests $CONCURRENT_REQUESTS"
    echo "✓ Concurrent requests: $CONCURRENT_REQUESTS"
fi

echo ""

# Build evaluation command
EVAL_CMD="python3 eval_hico_ground_claude.py \
    --model-name \"$MODEL_NAME\" \
    --ann-file $ANN_FILE \
    --img-prefix $IMG_PREFIX \
    --result-file $RESULT_FILE"

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
if [ ! -z "$USE_CACHE_FLAG" ]; then
    EVAL_CMD="$EVAL_CMD $USE_CACHE_FLAG"
fi

if [ ! -z "$CACHE_DIR_FLAG" ]; then
    EVAL_CMD="$EVAL_CMD $CACHE_DIR_FLAG"
fi

if [ ! -z "$OPTIMIZE_IMAGES_FLAG" ]; then
    EVAL_CMD="$EVAL_CMD $OPTIMIZE_IMAGES_FLAG"
fi

if [ ! -z "$IMAGE_MAX_SIZE_FLAG" ]; then
    EVAL_CMD="$EVAL_CMD $IMAGE_MAX_SIZE_FLAG"
fi

if [ ! -z "$IMAGE_QUALITY_FLAG" ]; then
    EVAL_CMD="$EVAL_CMD $IMAGE_QUALITY_FLAG"
fi

if [ ! -z "$OPTIMIZED_PROMPTS_FLAG" ]; then
    EVAL_CMD="$EVAL_CMD $OPTIMIZED_PROMPTS_FLAG"
fi

if [ ! -z "$CONCURRENT_REQUESTS_FLAG" ]; then
    EVAL_CMD="$EVAL_CMD $CONCURRENT_REQUESTS_FLAG"
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
    echo "  Predictions:    $RESULT_FILE"
    echo "  Metrics:        ${RESULT_FILE//.json/_metrics.json}"

    # Check if extended thinking was enabled
    THINKING_FILE="${RESULT_FILE//.json/_thinking.jsonl}"
    if [ -f "$THINKING_FILE" ]; then
        echo "  Thinking:       $THINKING_FILE  (extended thinking output)"
    fi

    echo "  Log:            $LOG_FILE"
    echo ""

    echo "Key metrics (from COCO evaluation):"
    echo "  AR:      Average Recall @ IoU=0.50:0.95"
    echo "  AR@0.5:  Average Recall @ IoU=0.50"
    echo "  AR@0.75: Average Recall @ IoU=0.75"
    echo ""
    echo "Note: This dataset includes person-person interactions"
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
