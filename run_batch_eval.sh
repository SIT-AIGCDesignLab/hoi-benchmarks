#!/bin/bash
################################################################################
# Unified Batch Evaluation Script for HOI Benchmarks
#
# Runs batch evaluation for SWIG-HOI and HICO-DET datasets using Claude, Gemini,
# or OpenAI Batch APIs. Offers 50% cost savings and avoids rate limit issues.
#
# Usage:
#   bash run_batch_eval.sh <provider> <task> [model]
#
# Arguments:
#   provider    API provider: claude, gemini, or openai
#   task        Evaluation task:
#               - swig_action   : SWIG Action Referring
#               - swig_ground   : SWIG Grounding
#               - hico_action   : HICO Action Referring
#               - hico_ground   : HICO Grounding
#   model       (Optional) Model name. Defaults to cost-effective model per provider.
#
# Available Models:
#   Gemini:
#     - gemini-2.5-flash       (default, cheapest)
#     - gemini-2.5-pro         (better accuracy)
#     - gemini-3-flash-preview (latest flash)
#     - gemini-3-pro-preview   (latest pro, best accuracy)
#
#   Claude:
#     - claude-sonnet-4-5     (default, good balance)
#     - claude-haiku-4-5      (fastest, cheapest)
#     - claude-opus-4-5       (highest accuracy)
#
#   OpenAI:
#     - gpt-4o          (default, good balance)
#     - gpt-4o-mini     (cheaper, faster)
#     - gpt-4.1         (latest)
#     - gpt-5.2-2025-12-11  (most capable)
#
# Environment Variables:
#   RESUME=1              Resume from checkpoint if interrupted
#   MAX_SAMPLES=N         Limit to first N samples (for testing)
#   POLL_INTERVAL=60      Seconds between status polls (default: 60)
#   OPTIMIZED_PROMPTS=1   Use shorter prompts (saves tokens, may affect accuracy)
#   SKIP_BERTSCORE=1      Skip BERTScore computation (faster for action tasks)
#
# Examples:
#   # SWIG Action Referring with Gemini Flash (default, cheapest)
#   bash run_batch_eval.sh gemini swig_action
#
#   # SWIG Action Referring with Gemini 3 Pro (best accuracy)
#   bash run_batch_eval.sh gemini swig_action gemini-3-pro-preview
#
#   # HICO Grounding with Claude Opus (highest accuracy)
#   bash run_batch_eval.sh claude hico_ground claude-opus-4-5-20251101
#
#   # HICO Action with OpenAI GPT-5.2
#   bash run_batch_eval.sh openai hico_action gpt-5.2-2025-12-11
#
#   # Resume interrupted job
#   RESUME=1 bash run_batch_eval.sh gemini swig_action gemini-3-pro-preview
#
#   # Test with 10 samples using a specific model
#   MAX_SAMPLES=10 bash run_batch_eval.sh claude swig_action claude-haiku-4-5-20250514
#
# Cost Comparison (per 1M tokens, batch pricing = 50% of regular):
#   Provider    Model                   Batch Input   Batch Output
#   ---------   ---------------------   -----------   ------------
#   Gemini      gemini-2.5-flash        $0.15         $1.25
#   Gemini      gemini-2.5-pro          $0.625        $5.00
#   Gemini      gemini-3-flash-preview  $0.15         $1.25
#   Gemini      gemini-3-pro-preview    $0.625        $5.00
#   Claude      claude-haiku-4-5        $0.50         $2.50
#   Claude      claude-sonnet-4-5       $1.50         $7.50
#   Claude      claude-opus-4-5         $7.50         $37.50
#   OpenAI      gpt-4o-mini             $0.075        $0.30
#   OpenAI      gpt-4o                  $1.25         $5.00
#
################################################################################

set -e  # Exit on error

# Help function
show_help() {
    echo ""
    echo "Usage: bash run_batch_eval.sh <provider> <task> [model]"
    echo ""
    echo "Arguments:"
    echo "  provider    API provider: claude, gemini, or openai"
    echo "  task        Evaluation task: swig_action, swig_ground, hico_action, hico_ground"
    echo "  model       (Optional) Model name - see available models below"
    echo ""
    echo "Available Models:"
    echo ""
    echo "  Gemini:"
    echo "    gemini-2.5-flash        (default, cheapest)"
    echo "    gemini-2.5-pro          (better accuracy)"
    echo "    gemini-3-flash-preview  (latest flash)"
    echo "    gemini-3-pro-preview    (latest pro, best accuracy)"
    echo ""
    echo "  Claude:"
    echo "    claude-sonnet-4-5     (default, good balance)"
    echo "    claude-haiku-4-5      (fastest, cheapest)"
    echo "    claude-opus-4-5       (highest accuracy)"
    echo ""
    echo "  OpenAI:"
    echo "    gpt-4o          (default, good balance)"
    echo "    gpt-4o-mini     (cheaper, faster)"
    echo "    gpt-4.1         (latest)"
    echo "    gpt-5.2-2025-12-11  (most capable)"
    echo ""
    echo "Examples:"
    echo "  # Use default model (cheapest)"
    echo "  bash run_batch_eval.sh gemini swig_action"
    echo ""
    echo "  # Specify Gemini 3 Pro for best accuracy"
    echo "  bash run_batch_eval.sh gemini swig_action gemini-3-pro-preview"
    echo ""
    echo "  # Use Claude Opus for highest accuracy"
    echo "  bash run_batch_eval.sh claude hico_ground claude-opus-4-5"
    echo ""
    echo "  # Use OpenAI GPT-5.2"
    echo "  bash run_batch_eval.sh openai hico_action gpt-5.2-2025-12-11"
    echo ""
    echo "  # Resume an interrupted job"
    echo "  RESUME=1 bash run_batch_eval.sh gemini swig_action gemini-3-pro-preview"
    echo ""
    echo "  # Test with limited samples"
    echo "  MAX_SAMPLES=10 bash run_batch_eval.sh claude swig_action"
    echo ""
    echo "Environment Variables:"
    echo "  RESUME=1              Resume from checkpoint"
    echo "  MAX_SAMPLES=N         Limit to N samples"
    echo "  POLL_INTERVAL=N       Seconds between status polls (default: 60)"
    echo "  OPTIMIZED_PROMPTS=1   Use shorter prompts"
    echo "  SKIP_BERTSCORE=1      Skip BERTScore computation"
    echo "  WANDB=1               Enable Weights & Biases logging"
    echo "  WANDB_PROJECT=name    Custom W&B project name"
    echo "  VERBOSE=1             Enable visualizations"
    echo ""
}

# Check for help flag
if [ "$1" = "-h" ] || [ "$1" = "--help" ] || [ "$1" = "help" ]; then
    show_help
    exit 0
fi

# Load environment variables from .env if exists
if [ -f .env ]; then
    echo "Loading environment variables from .env..."
    export $(grep -v '^#' .env | grep -v '^$' | xargs)
fi

# Activate virtual environment if exists
if [ -d .venv ]; then
    source .venv/bin/activate
fi

# Parse arguments
PROVIDER="${1:-gemini}"
TASK="${2:-swig_action}"
MODEL="${3:-}"

# Check if no arguments provided
if [ $# -eq 0 ]; then
    echo "No arguments provided. Use --help for usage information."
    echo ""
    echo "Quick start:"
    echo "  bash run_batch_eval.sh gemini swig_action"
    echo "  bash run_batch_eval.sh gemini swig_action gemini-3-pro-preview"
    echo ""
    exit 1
fi

# Validate provider
case "$PROVIDER" in
    claude|gemini|openai)
        ;;
    *)
        echo "ERROR: Invalid provider '$PROVIDER'"
        echo "Valid options: claude, gemini, openai"
        echo ""
        echo "Use --help for more information"
        exit 1
        ;;
esac

# Validate task
case "$TASK" in
    swig_action|swig_ground|hico_action|hico_ground)
        ;;
    *)
        echo "ERROR: Invalid task '$TASK'"
        echo "Valid options: swig_action, swig_ground, hico_action, hico_ground"
        echo ""
        echo "Use --help for more information"
        exit 1
        ;;
esac

# Set default models per provider (cost-effective choices for batch)
if [ -z "$MODEL" ]; then
    case "$PROVIDER" in
        gemini)
            MODEL="gemini-2.5-flash"
            echo "Using default Gemini model: $MODEL"
            echo "  (Use 'gemini-3-pro-preview' for best accuracy)"
            ;;
        claude)
            MODEL="claude-sonnet-4-5"
            echo "Using default Claude model: $MODEL"
            echo "  (Use 'claude-opus-4-5' for best accuracy)"
            ;;
        openai)
            MODEL="gpt-4o"
            echo "Using default OpenAI model: $MODEL"
            echo "  (Use 'gpt-5.2-2025-12-11' for best accuracy)"
            ;;
    esac
    echo ""
fi

# Set paths based on task
case "$TASK" in
    swig_action)
        SCRIPT="eval_swig_action_referring_batch.py"
        ANN_FILE="data/benchmarks_simplified/swig_action_referring_test_simplified.json"
        IMG_PREFIX="${SWIG_ROOT:-data/swig_hoi}/images_512"
        ;;
    swig_ground)
        SCRIPT="eval_swig_ground_batch.py"
        ANN_FILE="data/benchmarks_simplified/swig_ground_test_simplified.json"
        IMG_PREFIX="${SWIG_ROOT:-data/swig_hoi}/images_512"
        ;;
    hico_action)
        SCRIPT="eval_hico_action_referring_batch.py"
        ANN_FILE="data/benchmarks_simplified/hico_action_referring_test_simplified.json"
        IMG_PREFIX="${HICO_ROOT:-data/hico_20160224_det}/images/test2015"
        ;;
    hico_ground)
        SCRIPT="eval_hico_ground_batch.py"
        ANN_FILE="data/benchmarks_simplified/hico_ground_test_simplified.json"
        IMG_PREFIX="${HICO_ROOT:-data/hico_20160224_det}/images/test2015"
        ;;
esac

# Output directory
OUTPUT_DIR="results/${TASK}_${PROVIDER}_batch"

# Build command
CMD="python3 $SCRIPT \
    --provider $PROVIDER \
    --model \"$MODEL\" \
    --ann-file $ANN_FILE \
    --img-prefix $IMG_PREFIX \
    --output-dir $OUTPUT_DIR"

# Add optional flags
if [ "${RESUME:-0}" = "1" ]; then
    CMD="$CMD --resume"
fi

if [ ! -z "$MAX_SAMPLES" ]; then
    CMD="$CMD --max-samples $MAX_SAMPLES"
fi

if [ ! -z "$POLL_INTERVAL" ]; then
    CMD="$CMD --poll-interval $POLL_INTERVAL"
fi

if [ "${OPTIMIZED_PROMPTS:-0}" = "1" ]; then
    CMD="$CMD --optimized-prompts"
fi

if [ "${SKIP_BERTSCORE:-0}" = "1" ]; then
    CMD="$CMD --skip-bertscore"
fi

if [ "${WANDB:-0}" = "1" ]; then
    CMD="$CMD --wandb"
fi

if [ ! -z "$WANDB_PROJECT" ]; then
    CMD="$CMD --wandb-project $WANDB_PROJECT"
fi

if [ "${VERBOSE:-0}" = "1" ]; then
    CMD="$CMD --verbose"
fi

# Print configuration
echo ""
echo "========================================================================"
echo "HOI Batch Evaluation"
echo "========================================================================"
echo "Provider:        $PROVIDER"
echo "Task:            $TASK"
echo "Model:           $MODEL"
echo "Annotation:      $ANN_FILE"
echo "Images:          $IMG_PREFIX"
echo "Output:          $OUTPUT_DIR"
echo ""
echo "Options:"
echo "  Resume:        ${RESUME:-0}"
echo "  Max samples:   ${MAX_SAMPLES:-all}"
echo "  Poll interval: ${POLL_INTERVAL:-60}s"
echo "  Optimized:     ${OPTIMIZED_PROMPTS:-0}"
echo "  Skip BERT:     ${SKIP_BERTSCORE:-0}"
echo "  W&B logging:   ${WANDB:-0}"
echo "  Verbose/Viz:   ${VERBOSE:-0}"
echo "========================================================================"
echo ""

# Check if annotation file exists
if [ ! -f "$ANN_FILE" ]; then
    echo "ERROR: Annotation file not found: $ANN_FILE"
    echo "Please ensure the benchmark file has been generated."
    exit 1
fi

# Check if image directory exists
if [ ! -d "$IMG_PREFIX" ]; then
    echo "ERROR: Image directory not found: $IMG_PREFIX"
    echo "Please check the path to images."
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/batch_eval_${TIMESTAMP}.log"

echo "Starting batch evaluation..."
echo "Log file: $LOG_FILE"
echo ""

# Execute
eval "$CMD" 2>&1 | tee "$LOG_FILE"

# Check result
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "Batch Evaluation Complete!"
    echo "========================================================================"
    echo "Results saved to: $OUTPUT_DIR"
    echo "Log file: $LOG_FILE"
    echo ""
    echo "Batch API Benefits:"
    echo "  - 50% cost savings compared to real-time API"
    echo "  - No rate limit issues (separate quota pool)"
    echo "  - Automatic checkpointing for resumability"
    echo "========================================================================"
else
    echo ""
    echo "========================================================================"
    echo "ERROR: Batch evaluation failed!"
    echo "========================================================================"
    echo "Check the log file: $LOG_FILE"
    echo ""
    echo "To resume from checkpoint:"
    echo "  RESUME=1 bash run_batch_eval.sh $PROVIDER $TASK $MODEL"
    echo "========================================================================"
    exit 1
fi
