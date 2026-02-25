#!/bin/bash
################################################################################
# HICO-DET Action Referring Evaluation for SFT-trained Qwen3VL (Tool-Use Agent)
#
# Usage:
#   bash run_hico_action_sft_eval.sh [GPU_ID] [VLLM_URL] [CHECKPOINT_PATH]
#
# Environment Variables:
#   VERBOSE=1, MAX_IMAGES=N, WANDB=1, MAX_TURNS=N, CHECKPOINT_PATH
################################################################################

set -eo pipefail

GPU_ID="${1:-0}"
VLLM_URL="${2:-http://localhost:8000}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/media/shaun/workspace/AdaTooler-V/checkpoints/qwen3VL-4B}"
OUTPUT_DIR="${OUTPUT_DIR:-results-redo/hico_action_sft}"
MAX_TURNS="${MAX_TURNS:-5}"

if [[ "$GPU_ID" == cuda:* ]]; then
    GPU_NUM="${GPU_ID#cuda:}"
    export CUDA_VISIBLE_DEVICES="$GPU_NUM"
else
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/hico_action_sft_evaluation_${TIMESTAMP}.log"
PRED_FILE="$OUTPUT_DIR/hico_action_sft_results_${TIMESTAMP}.json"

ANN_FILE="../dataset/benchmarks_simplified/hico_action_referring_test_simplified.json"
IMG_PREFIX="../dataset/hico_20160224_det/images/test2015"
PROPOSALS_DIR="../../hoi-dataset-curation/output/test_proposals"

echo "========================================================================"
echo "HICO-DET Action Referring Evaluation (SFT Qwen3VL)"
echo "========================================================================"
echo "GPU:        $GPU_ID"
echo "vLLM URL:   $VLLM_URL"
echo "Output:     $OUTPUT_DIR"
echo "========================================================================"
echo ""

if ! curl -s "${VLLM_URL}/health" > /dev/null 2>&1; then
    echo "vLLM server not running at $VLLM_URL"
    if [ -n "$CHECKPOINT_PATH" ] && [ -d "$CHECKPOINT_PATH" ]; then
        echo "Starting vLLM server..."
        CUDA_VISIBLE_DEVICES=$GPU_ID python -m vllm.entrypoints.openai.api_server \
            --model "$CHECKPOINT_PATH" \
            --port 8000 \
            --max-model-len 8192 \
            --gpu-memory-utilization 0.85 \
            --trust-remote-code &
        VLLM_PID=$!
        sleep 60
        if ! curl -s "${VLLM_URL}/health" > /dev/null 2>&1; then
            echo "ERROR: vLLM server failed to start."
            kill $VLLM_PID 2>/dev/null || true
            exit 1
        fi
        echo "✓ vLLM server started"
    else
        echo "ERROR: Set CHECKPOINT_PATH or start vLLM manually."
        exit 1
    fi
else
    echo "✓ vLLM server running"
fi

MODEL_NAME=$(curl -s "${VLLM_URL}/v1/models" | python3 -c "import json,sys; models=json.load(sys.stdin)['data']; print(models[0]['id'])" 2>/dev/null || echo "qwen3VL-4B")
echo "Model: $MODEL_NAME"
echo ""

if [ ! -d "$PROPOSALS_DIR" ]; then
    echo "WARNING: Proposals directory not found: $PROPOSALS_DIR"
    echo "Run generate_test_proposals.sh first!"
fi

VERBOSE_FLAG=""
MAX_IMAGES_FLAG=""
WANDB_FLAG=""

[ ! -z "$VERBOSE" ] && VERBOSE_FLAG="--verbose"
[ ! -z "$MAX_IMAGES" ] && MAX_IMAGES_FLAG="--max-images $MAX_IMAGES"
[ ! -z "$WANDB" ] && WANDB_FLAG="--wandb"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${SCRIPT_DIR}/.venv/bin/python"
[ ! -f "$PYTHON" ] && PYTHON="python3"

EVAL_CMD="$PYTHON eval_hico_action_referring_sft_qwen3vl.py \
    --vllm-url $VLLM_URL \
    --model-name \"$MODEL_NAME\" \
    --ann-file $ANN_FILE \
    --img-prefix $IMG_PREFIX \
    --proposals-dir $PROPOSALS_DIR \
    --pred-file $PRED_FILE \
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
    echo "Predictions: $PRED_FILE"
    echo "Metrics:     ${PRED_FILE//.json/_metrics.json}"
    echo "Log:         $LOG_FILE"
    echo ""
    echo "BERTScore (run separately):"
    echo "  python calculate_bertscore.py --pred-file $PRED_FILE --model roberta-large"
    echo "========================================================================"
else
    echo "ERROR: Evaluation failed. Check log: $LOG_FILE"
    exit 1
fi
