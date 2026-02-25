#!/bin/bash
################################################################################
# SWIG-HOI Grounding Evaluation for SFT-trained Qwen3VL (Tool-Use Agent)
#
# Requires vLLM server running with the SFT checkpoint.
# Optionally starts vLLM server if not already running.
#
# Usage:
#   bash run_swig_ground_sft_eval.sh [GPU_ID] [VLLM_URL] [CHECKPOINT_PATH]
#
# Examples:
#   bash run_swig_ground_sft_eval.sh 0
#   bash run_swig_ground_sft_eval.sh 0 http://localhost:8000
#   MAX_IMAGES=10 VERBOSE=1 bash run_swig_ground_sft_eval.sh 0
#   WANDB=1 bash run_swig_ground_sft_eval.sh 0
#
# Environment Variables:
#   VERBOSE=1              Show per-sample results
#   MAX_IMAGES=N           Limit to first N samples
#   WANDB=1                Enable Weights & Biases logging
#   MAX_TURNS=N            Max tool-call turns (default: 5)
#   CHECKPOINT_PATH        Path to SFT checkpoint (for auto-starting vLLM)
################################################################################

set -e

GPU_ID="${1:-0}"
VLLM_URL="${2:-http://localhost:8000}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/media/shaun/workspace/AdaTooler-V/checkpoints/qwen3VL-4B}"
# GPU server (RTX 6000 Ada): CHECKPOINT_PATH="${CHECKPOINT_PATH:-/mnt/d/Work/latest_checkpoints/sft-checkpoints/qwen3VL-4B}"
OUTPUT_DIR="${OUTPUT_DIR:-results-redo/swig_ground_sft}"
MAX_TURNS="${MAX_TURNS:-5}"

if [[ "$GPU_ID" == cuda:* ]]; then
    DEVICE_ARG="$GPU_ID"
    GPU_NUM="${GPU_ID#cuda:}"
    export CUDA_VISIBLE_DEVICES="$GPU_NUM"
else
    DEVICE_ARG="cuda:$GPU_ID"
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/swig_ground_sft_evaluation_${TIMESTAMP}.log"
RESULT_FILE="$OUTPUT_DIR/swig_ground_sft_results_${TIMESTAMP}.json"

ANN_FILE="../dataset/benchmarks_simplified/swig_ground_test_simplified.json"
IMG_PREFIX="../dataset/swig_hoi/images_512"
PROPOSALS_DIR="../../hoi-dataset-curation/output/test_proposals"
# GPU server (RTX 6000 Ada):
# ANN_FILE="/mnt/d/Work/data/benchmarks_simplified/swig_ground_test_simplified.json"
# IMG_PREFIX="/mnt/d/Work/data/swig_hoi/images_512"
# PROPOSALS_DIR="/mnt/d/Work/data/test_proposals"

echo "========================================================================"
echo "SWIG-HOI Grounding Evaluation (SFT Qwen3VL)"
echo "========================================================================"
echo "GPU:           $GPU_ID"
echo "vLLM URL:      $VLLM_URL"
echo "Annotation:    $ANN_FILE"
echo "Images:        $IMG_PREFIX"
echo "Proposals:     $PROPOSALS_DIR"
echo "Output:        $OUTPUT_DIR"
echo "Max turns:     $MAX_TURNS"
echo "========================================================================"
echo ""

# Check if vLLM server is running
if ! curl -s "${VLLM_URL}/health" > /dev/null 2>&1; then
    echo "vLLM server not running at $VLLM_URL"
    if [ -n "$CHECKPOINT_PATH" ] && [ -d "$CHECKPOINT_PATH" ]; then
        echo "Starting vLLM server with checkpoint: $CHECKPOINT_PATH"
        echo "NOTE: Server startup takes ~30-60 seconds."
        CUDA_VISIBLE_DEVICES=$GPU_ID python -m vllm.entrypoints.openai.api_server \
            --model "$CHECKPOINT_PATH" \
            --port 8000 \
            --max-model-len 8192 \
            --gpu-memory-utilization 0.85 \
            --trust-remote-code &
        VLLM_PID=$!
        echo "vLLM PID: $VLLM_PID"
        echo "Waiting for server to start..."
        sleep 60
        if ! curl -s "${VLLM_URL}/health" > /dev/null 2>&1; then
            echo "ERROR: vLLM server failed to start. Check logs."
            kill $VLLM_PID 2>/dev/null || true
            exit 1
        fi
        echo "✓ vLLM server started"
    else
        echo "ERROR: No checkpoint path set and server not running."
        echo "Please start vLLM server manually:"
        echo "  vllm serve $CHECKPOINT_PATH --port 8000 --max-model-len 8192"
        exit 1
    fi
else
    echo "✓ vLLM server is running at $VLLM_URL"
fi

# Get model name from vLLM
MODEL_NAME=$(curl -s "${VLLM_URL}/v1/models" | python3 -c "import json,sys; models=json.load(sys.stdin)['data']; print(models[0]['id'])" 2>/dev/null || echo "qwen3VL-4B")
echo "Model name: $MODEL_NAME"
echo ""

if [ ! -f "$ANN_FILE" ]; then
    echo "ERROR: Annotation file not found: $ANN_FILE"
    exit 1
fi

if [ ! -d "$IMG_PREFIX" ]; then
    echo "ERROR: Images directory not found: $IMG_PREFIX"
    exit 1
fi

if [ ! -d "$PROPOSALS_DIR" ]; then
    echo "WARNING: Proposals directory not found: $PROPOSALS_DIR"
    echo "Run generate_test_proposals.sh first!"
fi

VERBOSE_FLAG=""
MAX_IMAGES_FLAG=""
WANDB_FLAG=""

[ ! -z "$VERBOSE" ] && VERBOSE_FLAG="--verbose" && echo "✓ Verbose mode enabled"
[ ! -z "$MAX_IMAGES" ] && MAX_IMAGES_FLAG="--max-images $MAX_IMAGES" && echo "✓ Limiting to $MAX_IMAGES images"
[ ! -z "$WANDB" ] && WANDB_FLAG="--wandb" && echo "✓ WandB logging enabled"

echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${SCRIPT_DIR}/.venv/bin/python"
[ ! -f "$PYTHON" ] && PYTHON="python3"

EVAL_CMD="$PYTHON eval_swig_ground_sft_qwen3vl.py \
    --vllm-url $VLLM_URL \
    --model-name \"$MODEL_NAME\" \
    --ann-file $ANN_FILE \
    --img-prefix $IMG_PREFIX \
    --proposals-dir $PROPOSALS_DIR \
    --result-file $RESULT_FILE \
    --max-turns $MAX_TURNS"

[ ! -z "$VERBOSE_FLAG" ] && EVAL_CMD="$EVAL_CMD $VERBOSE_FLAG"
[ ! -z "$MAX_IMAGES_FLAG" ] && EVAL_CMD="$EVAL_CMD $MAX_IMAGES_FLAG"
[ ! -z "$WANDB_FLAG" ] && EVAL_CMD="$EVAL_CMD $WANDB_FLAG"

eval "$EVAL_CMD" 2>&1 | tee "$LOG_FILE"

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "Evaluation Complete!"
    echo "========================================================================"
    echo "Results: $RESULT_FILE"
    echo "Metrics: ${RESULT_FILE//.json/_metrics.json}"
    echo "Log:     $LOG_FILE"
    echo "========================================================================"
else
    echo "ERROR: Evaluation failed. Check log: $LOG_FILE"
    exit 1
fi
