#!/bin/bash
################################################################################
# HICO-DET Grounding Task Evaluation Script for Gemini API
# Evaluates Google Gemini models on grounding performance
#
# Task: Detect person-object pairs with bounding boxes
# Metrics: COCO-style AR (Average Recall)
#
# Usage:
#   bash run_hico_ground_eval_gemini.sh [MODEL] [OUTPUT_DIR]
#
# Examples:
#   GOOGLE_API_KEY=xxx bash run_hico_ground_eval_gemini.sh
#   MAX_IMAGES=10 bash run_hico_ground_eval_gemini.sh
#
# Environment Variables:
#   VERBOSE=1              Show per-image results
#   MAX_IMAGES=N          Limit to first N images
#   WANDB=1               Enable Weights & Biases logging
#   GOOGLE_API_KEY        Required: Your Google API key
#
# Cost Optimization Variables:
#   USE_CACHE=1           Enable response caching (default: 1, 50-100% savings on reruns)
#   CACHE_DIR=./cache     Cache directory (default: ./cache)
#   OPTIMIZE_IMAGES=1     Enable image optimization (default: 0, 30-50% savings)
#   IMAGE_MAX_SIZE=448    Max image dimension in pixels (default: 448, conservative)
#   IMAGE_QUALITY=90      JPEG quality 1-100 (default: 90, high quality)
#   OPTIMIZED_PROMPTS=1   Use shorter prompts (default: 1, 20-40% savings)
#   CONCURRENT_REQUESTS=N Number of concurrent API requests (default: 1 sequential)
################################################################################

set -e

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    echo "Loading environment variables from .env file..."
    export $(grep -v '^#' .env | grep -v '^$' | xargs)
fi

# Activate virtual environment if it exists
if [ -d .venv ]; then
    source .venv/bin/activate
fi

MODEL_NAME="${1:-gemini-3-pro-preview}"
OUTPUT_DIR="${2:-results/hico_ground_gemini}"

mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/hico_ground_gemini_evaluation_${TIMESTAMP}.log"

# HICO-DET paths
HICO_ROOT="${HICO_ROOT:-data/hico_20160224_det}"
IMG_PREFIX="${IMG_PREFIX:-${HICO_ROOT}/images/test2015}"
ANN_FILE="${ANN_FILE:-data/benchmarks_simplified/hico_ground_test_simplified.json}"
RESULT_FILE="${OUTPUT_DIR}/hico_ground_gemini_results_${TIMESTAMP}.json"

echo "========================================================================"
echo "HICO-DET Grounding Evaluation (Gemini API)"
echo "========================================================================"
echo "Model:       $MODEL_NAME"
echo "Output:      $OUTPUT_DIR"
echo "========================================================================"

# API key will be loaded from .env by Python script (via python-dotenv)
echo ""

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

EVAL_CMD="python3 eval_hico_ground_gemini.py \
    --model-name \"$MODEL_NAME\" \
    --ann-file $ANN_FILE \
    --img-prefix $IMG_PREFIX \
    --result-file $RESULT_FILE"

[ ! -z "$VERBOSE" ] && EVAL_CMD="$EVAL_CMD --verbose"
[ ! -z "$MAX_IMAGES" ] && EVAL_CMD="$EVAL_CMD --max-images $MAX_IMAGES"
[ ! -z "$WANDB" ] && EVAL_CMD="$EVAL_CMD --wandb"

# Add cost optimization flags
[ ! -z "$USE_CACHE_FLAG" ] && EVAL_CMD="$EVAL_CMD $USE_CACHE_FLAG"
[ ! -z "$CACHE_DIR_FLAG" ] && EVAL_CMD="$EVAL_CMD $CACHE_DIR_FLAG"
[ ! -z "$OPTIMIZE_IMAGES_FLAG" ] && EVAL_CMD="$EVAL_CMD $OPTIMIZE_IMAGES_FLAG"
[ ! -z "$IMAGE_MAX_SIZE_FLAG" ] && EVAL_CMD="$EVAL_CMD $IMAGE_MAX_SIZE_FLAG"
[ ! -z "$IMAGE_QUALITY_FLAG" ] && EVAL_CMD="$EVAL_CMD $IMAGE_QUALITY_FLAG"
[ ! -z "$OPTIMIZED_PROMPTS_FLAG" ] && EVAL_CMD="$EVAL_CMD $OPTIMIZED_PROMPTS_FLAG"
[ ! -z "$CONCURRENT_REQUESTS_FLAG" ] && EVAL_CMD="$EVAL_CMD $CONCURRENT_REQUESTS_FLAG"

eval "$EVAL_CMD" 2>&1 | tee "$LOG_FILE"

if [ $? -eq 0 ]; then
    echo "✅ Evaluation Complete! Results: $RESULT_FILE"
else
    echo "ERROR: Evaluation failed! Check: $LOG_FILE"
    exit 1
fi
