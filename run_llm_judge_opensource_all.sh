#!/bin/bash
# Run LLM-as-a-Judge evaluation using local vLLM server (open-source, no API cost)
# Judge model: nvidia/Llama-3_3-Nemotron-Super-49B-v1_5-FP8
#   Non-Qwen, paper-credible (Llama-Nemotron, arxiv:2505.00949), outperforms o1-mini on JudgeBench
#   HuggingFace: https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5-FP8
#
# Usage:
#   bash run_llm_judge_opensource_all.sh [--folders FOLDER1,FOLDER2,...]
#
# Arguments:
#   --folders NAME,...  Comma-separated folder/file name substrings to filter
#                       (e.g. swig_action_claude_batch or swig_referring_ours.json)
#                       If omitted, evaluates all files
#
# Environment variables:
#   VERBOSE=1           Print per-sample score + reason during evaluation
#   MAX_IMAGES=N        Limit to first N samples (for quick testing)
#   WANDB=1             Enable Weights & Biases logging
#   WANDB_PROJECT       W&B project name (default: llm-judge-opensource)
#   WANDB_RUN_NAME      W&B run name (default: auto-generated)
#   BASE_URL            Override vLLM server URL (default: http://localhost:8000/v1)
#
# Examples:
#   bash run_llm_judge_opensource_all.sh
#   bash run_llm_judge_opensource_all.sh --folders swig_action_claude_batch
#   bash run_llm_judge_opensource_all.sh --folders swig_action_claude_batch,hico_action_openai_batch
#   MAX_IMAGES=10 VERBOSE=1 bash run_llm_judge_opensource_all.sh --folders swig_referring_ours.json
#   WANDB=1 WANDB_PROJECT="my-project" bash run_llm_judge_opensource_all.sh
#
# Prerequisites - Start vLLM server before running (requires FP8-capable GPU ~50GB VRAM):
#   python -m vllm.entrypoints.openai.api_server \
#       --model nvidia/Llama-3_3-Nemotron-Super-49B-v1_5-FP8 \
#       --quantization fp8 \
#       --gpu-memory-utilization 0.90 \
#       --max-model-len 32768 \
#       --trust-remote-code \
#       --port 8000

FOLDERS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --folders) FOLDERS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

MODEL="nvidia/Llama-3_3-Nemotron-Super-49B-v1_5-FP8"
BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
CONCURRENCY=16

VERBOSE_FLAG=""
MAX_IMAGES_FLAG=""
WANDB_FLAG=""

if [ ! -z "$VERBOSE" ]; then
    VERBOSE_FLAG="--verbose"
    echo "✓ Verbose mode enabled"
fi

if [ ! -z "$MAX_IMAGES" ]; then
    MAX_IMAGES_FLAG="--max-images $MAX_IMAGES"
    echo "✓ Limiting to first $MAX_IMAGES samples"
fi

if [ ! -z "$WANDB" ]; then
    WANDB_FLAG="--wandb"
    echo "✓ WandB logging enabled"
    if [ ! -z "$WANDB_PROJECT" ]; then
        WANDB_FLAG="$WANDB_FLAG --wandb-project $WANDB_PROJECT"
        echo "  WandB project: $WANDB_PROJECT"
    fi
    if [ ! -z "$WANDB_RUN_NAME" ]; then
        WANDB_FLAG="$WANDB_FLAG --wandb-run-name $WANDB_RUN_NAME"
        echo "  WandB run name: $WANDB_RUN_NAME"
    fi
fi

PRED_FILES=(
    # SWIG Action Referring
    # "results/swig_action_claude_batch/results_20260115_173500.json"  # Already evaluated
    "results/swig_action_openai_batch/results_20260116_122357.json"
    "results/swig_referring_ours.json"

    # HICO Action Referring
    "results/hico_action_claude_batch/results_20260115_183406.json"
    "results/hico_action_openai_batch/results_20260116_150309.json"
    "results/hico_referring_ours.json"

    # Groma baseline
    "results/groma_referring_result/swig_action_predictions_20251111_095839_per_triplet.json"
    "results/groma_referring_result/hico_action_predictions_20251111_095427_per_triplet.json"

    # Qwen3VL models
    "results/swig_action_qwen3vl_8B_results_20251031_103142_per_triplet.json"
    "results/swig_action_qwen3vl_32B_results_20251102_131217_per_triplet.json"
    "results/hico_action_qwen3vl_8B_results_20251031_101049_per_triplet.json"
    "results/hico_action_qwen3vl_32B_results_20251102_131213_per_triplet.json"

    # InternVL3 models
    "results/swig_action_internvl3_8B_results_20251104_040939_per_triplet.json"
    "results/swig_action_internvl3_38B_results_20251105_023748_per_triplet.json"
    "results/hico_action_internvl3_8B_results_20251104_040806_per_triplet.json"
    "results/hico_action_internvl3_38B_results_20251105_075300_per_triplet.json"
)

echo "============================================================"
echo "LLM-as-a-Judge Evaluation (Open Source - Local vLLM)"
echo "============================================================"
echo "Model:       ${MODEL}"
echo "Server:      ${BASE_URL}"
echo "Concurrency: ${CONCURRENCY}"
echo "Total files: ${#PRED_FILES[@]}"
if [ ! -z "$FOLDERS" ]; then
    echo "Filter:      ${FOLDERS}"
fi
echo "============================================================"
echo ""

echo "Checking vLLM server connectivity..."
if ! curl -sf "${BASE_URL}/models" > /dev/null 2>&1; then
    echo ""
    echo "ERROR: Cannot reach vLLM server at ${BASE_URL}"
    echo ""
    echo "Start the server first (requires FP8-capable GPU, e.g. H100/A100-Ada/Blackwell ~50GB):"
    echo "  python -m vllm.entrypoints.openai.api_server \\"
    echo "      --model nvidia/Llama-3_3-Nemotron-Super-49B-v1_5-FP8 \\"
    echo "      --quantization fp8 \\"
    echo "      --gpu-memory-utilization 0.90 \\"
    echo "      --max-model-len 32768 \\"
    echo "      --trust-remote-code \\"
    echo "      --port 8000"
    exit 1
fi
echo "✓ vLLM server is reachable"
echo ""

ACTIVE_FILES=()
for pred_file in "${PRED_FILES[@]}"; do
    if [[ -z "$FOLDERS" ]]; then
        ACTIVE_FILES+=("$pred_file")
    else
        IFS=',' read -ra FOLDER_LIST <<< "$FOLDERS"
        for folder in "${FOLDER_LIST[@]}"; do
            if [[ "$pred_file" == *"$folder"* ]]; then
                ACTIVE_FILES+=("$pred_file")
                break
            fi
        done
    fi
done

echo "Files to evaluate: ${#ACTIVE_FILES[@]} / ${#PRED_FILES[@]}"
if [[ ${#ACTIVE_FILES[@]} -eq 0 ]]; then
    echo "ERROR: No files matched the --folders filter: ${FOLDERS}"
    echo "Available files:"
    for f in "${PRED_FILES[@]}"; do echo "  $f"; done
    exit 1
fi
echo ""

SUCCESS_COUNT=0
FAIL_COUNT=0
FAILED_FILES=()

for pred_file in "${ACTIVE_FILES[@]}"; do
    echo ""
    echo "------------------------------------------------------------"
    echo "Processing: ${pred_file}"
    echo "------------------------------------------------------------"

    if [[ ! -f "${pred_file}" ]]; then
        echo "WARNING: File not found, skipping: ${pred_file}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_FILES+=("${pred_file} (not found)")
        continue
    fi

    python3 eval_llm_judge_opensource.py \
        --pred-file "${pred_file}" \
        --model "${MODEL}" \
        --base-url "${BASE_URL}" \
        --concurrency "${CONCURRENCY}" \
        ${VERBOSE_FLAG} \
        ${MAX_IMAGES_FLAG} \
        ${WANDB_FLAG}

    if [[ $? -eq 0 ]]; then
        echo "SUCCESS: ${pred_file}"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "FAILED: ${pred_file}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_FILES+=("${pred_file}")
    fi
done

echo ""
echo "============================================================"
echo "Summary"
echo "============================================================"
echo "Successful: ${SUCCESS_COUNT}"
echo "Failed:     ${FAIL_COUNT}"

if [[ ${FAIL_COUNT} -gt 0 ]]; then
    echo ""
    echo "Failed files:"
    for f in "${FAILED_FILES[@]}"; do
        echo "  - ${f}"
    done
fi

echo "============================================================"
