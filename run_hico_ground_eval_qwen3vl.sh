#!/bin/bash
################################################################################
# HICO-DET Grounding Task Evaluation Script for Qwen3VL
# Evaluates Qwen3VL grounding performance on HICO-DET dataset
# Each sample = one (action, object) combination with multi-pair support
#
# Task: Detect person-object pairs with bounding boxes
# Metrics: COCO-style AR (Average Recall)
#
# Supports both Instruct and Thinking models:
#   - Instruct models (e.g., Qwen3-VL-8B-Instruct): Standard response format
#   - Thinking models (e.g., Qwen3-VL-8B-Thinking):
#     * Extracts and saves reasoning process from <think> tags
#     * Saves final answer after </think> token
#     * Model type is automatically detected from model name
#
# Usage:
#   bash run_hico_ground_eval_qwen3vl.sh [GPU] [MODEL] [OUTPUT_DIR]
#
# Examples:
#   bash run_hico_ground_eval_qwen3vl.sh 0                    # Use GPU 0
#   bash run_hico_ground_eval_qwen3vl.sh 1                    # Use GPU 1
#   VERBOSE=1 bash run_hico_ground_eval_qwen3vl.sh 0          # Show per-image results
#   MAX_IMAGES=10 bash run_hico_ground_eval_qwen3vl.sh 0      # Test on first 10 images
#   WANDB=1 bash run_hico_ground_eval_qwen3vl.sh 0            # Enable WandB
#
# Environment Variables:
#   VERBOSE=1         Show per-image results
#   MAX_IMAGES=N      Limit to first N images (for quick testing)
#   WANDB=1           Enable Weights & Biases logging
#   WANDB_PROJECT     W&B project name (default: hico-grounding-qwen3vl)
#   WANDB_RUN_NAME    W&B run name (default: auto-generated)
#
# Output files:
#   {output_dir}/hico_ground_qwen3vl_results_{timestamp}.json
#   {output_dir}/hico_ground_qwen3vl_results_{timestamp}_metrics.json
#   {output_dir}/hico_ground_qwen3vl_results_{timestamp}_action_stats.json
#   {output_dir}/hico_ground_qwen3vl_results_{timestamp}_thinking.jsonl
#   {output_dir}/hico_ground_qwen3vl_evaluation_{timestamp}.log
################################################################################

set -e  # Exit on error

# Configuration with defaults
GPU_ID="${1:-0}"
MODEL_NAME="${2:-Qwen/Qwen3-VL-8B-Thinking}"
OUTPUT_DIR="${3:-results-redo/hico_ground_qwen3vl_thinking}"

# Set GPU (handle both "0" and "cuda:0" formats)
if [[ "$GPU_ID" == cuda:* ]]; then
    DEVICE_ARG="$GPU_ID"
    GPU_NUM="${GPU_ID#cuda:}"
    export CUDA_VISIBLE_DEVICES="$GPU_NUM"
else
    DEVICE_ARG="cuda:$GPU_ID"
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Timestamp for output files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/hico_ground_qwen3vl_evaluation_${TIMESTAMP}.log"

# HICO dataset paths
ANN_FILE="../dataset/benchmarks_simplified/hico_ground_test_simplified.json"
IMG_PREFIX="../dataset/hico_20160224_det/images/test2015"
RESULT_FILE="${OUTPUT_DIR}/hico_ground_qwen3vl_results_${TIMESTAMP}.json"

# GPU availability check
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | nl -v 0
    echo ""
fi

echo "========================================================================"
echo "HICO-DET Grounding Evaluation (Qwen3VL)"
echo "========================================================================"
echo "GPU:         $GPU_ID (Device: $DEVICE_ARG)"
echo "Model:       $MODEL_NAME"
echo "Annotation:  $ANN_FILE"
echo "Images:      $IMG_PREFIX"
echo "Output:      $OUTPUT_DIR"
echo "Log file:    $LOG_FILE"
echo "Result file: $RESULT_FILE"
echo "========================================================================"
echo ""

# Check if files exist
if [ ! -f "$ANN_FILE" ]; then
    echo "ERROR: Annotation file not found at $ANN_FILE"
    echo "Please ensure the benchmark file has been generated"
    exit 1
fi

if [ ! -d "$IMG_PREFIX" ]; then
    echo "ERROR: Images directory not found at $IMG_PREFIX"
    echo "Please check the path to HICO test images"
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

    if [ ! -z "$WANDB_PROJECT" ]; then
        WANDB_FLAG="$WANDB_FLAG --wandb-project $WANDB_PROJECT"
        echo "  WandB project: $WANDB_PROJECT"
    else
        echo "  WandB project: hico-grounding-qwen3vl (default)"
    fi

    if [ ! -z "$WANDB_RUN_NAME" ]; then
        WANDB_FLAG="$WANDB_FLAG --wandb-run-name $WANDB_RUN_NAME"
        echo "  WandB run name: $WANDB_RUN_NAME"
    fi
fi

echo ""

# Build evaluation command
EVAL_CMD="python3 eval_hico_ground_qwen3vl.py \
    --model-name \"$MODEL_NAME\" \
    --device $DEVICE_ARG \
    --ann-file $ANN_FILE \
    --img-prefix $IMG_PREFIX \
    --result-file $RESULT_FILE"

if [ ! -z "$VERBOSE_FLAG" ]; then
    EVAL_CMD="$EVAL_CMD $VERBOSE_FLAG"
fi

if [ ! -z "$MAX_IMAGES_FLAG" ]; then
    EVAL_CMD="$EVAL_CMD $MAX_IMAGES_FLAG"
fi

if [ ! -z "$WANDB_FLAG" ]; then
    EVAL_CMD="$EVAL_CMD $WANDB_FLAG"
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
    echo "  Action stats:   ${RESULT_FILE//.json/_action_stats.json}"
    THINKING_FILE="${RESULT_FILE//.json/_thinking.jsonl}"
    if [ -f "$THINKING_FILE" ]; then
        echo "  Thinking:       $THINKING_FILE  (thinking model output)"
    fi
    echo "  Log:            $LOG_FILE"
    echo ""

    echo "Key metrics (from COCO evaluation):"
    echo "  AR:      Average Recall @ IoU=0.50:0.95"
    echo "  AR@0.5:  Average Recall @ IoU=0.50"
    echo "  AR@0.75: Average Recall @ IoU=0.75"
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
