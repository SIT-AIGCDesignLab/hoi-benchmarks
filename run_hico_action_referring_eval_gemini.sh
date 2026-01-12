#!/bin/bash
################################################################################
# HICO-DET Action Referring Task Evaluation Script for Gemini API
# Evaluates Google Gemini models on action prediction
#
# Task: Given (person, object) bounding boxes, predict the connecting action
# Metrics: METEOR, CIDEr, BLEU, ROUGE-L, BERTScore
#
# Usage:
#   bash run_hico_action_referring_eval_gemini.sh [MODEL] [OUTPUT_DIR]
#
# Examples:
#   # Basic usage with Gemini 2.0 Flash
#   GOOGLE_API_KEY=xxx bash run_hico_action_referring_eval_gemini.sh
#
#   # Quick test on first 10 triplets
#   MAX_IMAGES=10 bash run_hico_action_referring_eval_gemini.sh
#
#   # With BERTScore computation
#   COMPUTE_BERTSCORE=1 bash run_hico_action_referring_eval_gemini.sh
#
#   # Use different model
#   bash run_hico_action_referring_eval_gemini.sh "gemini-2.5-pro"
#
# Environment Variables:
#   VERBOSE=1              Show per-triplet results
#   MAX_IMAGES=N          Limit to first N triplets
#   WANDB=1               Enable Weights & Biases logging
#   COMPUTE_BERTSCORE=1   Compute BERTScore metric
#   GOOGLE_API_KEY        Required: Your Google API key
#
# Cost Optimization Variables (NEW):
#   USE_CACHE=1           Enable response caching (default: 1, 50-100% savings on reruns)
#   CACHE_DIR=./cache     Cache directory (default: ./cache)
#   OPTIMIZE_IMAGES=1     Enable image optimization (default: 0, 30-50% savings)
#   IMAGE_MAX_SIZE=448    Max image dimension in pixels (default: 448, conservative)
#   IMAGE_QUALITY=90      JPEG quality 1-100 (default: 90, high quality)
#   OPTIMIZED_PROMPTS=1   Use shorter prompts (default: 1, 20-40% savings)
#   CONCURRENT_REQUESTS=N Number of concurrent API requests (default: 1 sequential)
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
# Parse command-line arguments
MODEL_NAME="gemini-3-pro-preview"
OUTPUT_DIR="results/hico_action_gemini"

# Parse arguments to handle flags and positional parameters
while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose)
            VERBOSE=1
            shift
            ;;
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            # First non-flag argument is model name (for backward compatibility)
            if [ "$MODEL_NAME" = "gemini-2.0-flash-exp" ]; then
                MODEL_NAME="$1"
                shift
            elif [ "$OUTPUT_DIR" = "results/hico_action_gemini" ]; then
                OUTPUT_DIR="$1"
                shift
            else
                shift
            fi
            ;;
    esac
done

mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/hico_action_gemini_evaluation_${TIMESTAMP}.log"

# Use environment variable if set, otherwise use default
HICO_ROOT="${HICO_ROOT:-data/hico_20160224_det}"
IMG_PREFIX="${IMG_PREFIX:-${HICO_ROOT}/images/test2015}"
ANN_FILE="${ANN_FILE:-data/benchmarks_simplified/hico_action_referring_test_simplified.json}"
PRED_FILE="${OUTPUT_DIR}/hico_action_gemini_results_${TIMESTAMP}.json"

echo "========================================================================"
echo "HICO-DET Action Referring Evaluation (Gemini API)"
echo "========================================================================"
echo "Model:       $MODEL_NAME"
echo "Annotation:  $ANN_FILE"
echo "Images:      $IMG_PREFIX"
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

if [ "${OPTIMIZED_PROMPTS:-1}" = "1" ]; then
    OPTIMIZED_PROMPTS_FLAG="--optimized-prompts"
    echo "✓ Optimized prompts enabled (20-40% token savings)"
fi

if [ ! -z "$CONCURRENT_REQUESTS" ]; then
    CONCURRENT_REQUESTS_FLAG="--concurrent-requests $CONCURRENT_REQUESTS"
    echo "✓ Concurrent requests: $CONCURRENT_REQUESTS"
fi

echo ""

# Build evaluation command
EVAL_CMD="python3 eval_hico_action_referring_gemini.py \
    --model-name \"$MODEL_NAME\" \
    --ann-file $ANN_FILE \
    --img-prefix $IMG_PREFIX \
    --pred-file $PRED_FILE"

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
    echo "========================================================================"
    echo "✅ Evaluation Complete!"
    echo "Results: $PRED_FILE"
    echo "========================================================================"
else
    echo "ERROR: Evaluation failed! Check: $LOG_FILE"
    exit 1
fi
