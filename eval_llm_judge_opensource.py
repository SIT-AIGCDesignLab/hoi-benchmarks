#!/usr/bin/env python3
"""
LLM-as-a-Judge evaluation using local vLLM server (OpenAI-compatible API).
Replaces the expensive Gemini Vertex AI batch evaluation.

Judge model: nvidia/Llama-3_3-Nemotron-Super-49B-v1_5-FP8 (FP8 via vLLM)
Non-Qwen judge to avoid self-preference bias (evaluated models are Qwen3-VL based).
Model: https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5-FP8
Paper: Llama-Nemotron (arxiv:2505.00949) — outperforms o1-mini on JudgeBench (2025).

Usage:
    # Start vLLM server first (requires FP8-capable GPU: H100/A100-Ada/Blackwell ~50GB):
    # python -m vllm.entrypoints.openai.api_server \\
    #     --model nvidia/Llama-3_3-Nemotron-Super-49B-v1_5-FP8 \\
    #     --quantization fp8 \\
    #     --gpu-memory-utilization 0.90 \\
    #     --max-model-len 32768 \\
    #     --trust-remote-code \\
    #     --port 8000

    python eval_llm_judge_opensource.py \\
        --pred-file results/swig_action_claude_batch/results_20260115_173500.json

    # With options:
    python eval_llm_judge_opensource.py \\
        --pred-file results/swig_referring_ours.json \\
        --max-images 100 --verbose

    # Resume interrupted run:
    python eval_llm_judge_opensource.py \\
        --pred-file results/swig_referring_ours.json \\
        --resume results/swig_referring_ours_checkpoint.json
"""

import os
import sys
import json
import argparse
import time
import re
import concurrent.futures
from datetime import datetime
from typing import List, Dict, Any, Optional

from openai import OpenAI
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# =============================================================================
# Functions copied verbatim from eval_llm_judge_batch.py
# =============================================================================

def detect_file_type(filename: str, data: List[Dict[str, Any]]) -> str:
    """
    Detect the type of result file based on filename and content.

    Returns:
        'referring' - Action referring results (text prediction vs ground truth)
        'grounding' - Grounding results (bbox prediction vs ground truth) - not for LLM judge
        'unknown' - Unknown format
    """
    filename_lower = filename.lower()

    # Check filename patterns
    if 'grounding' in filename_lower or 'ground' in filename_lower:
        # Check if it's actually grounding (has gt_pairs/pred_pairs)
        if data and ('gt_pairs' in data[0] or 'pred_pairs' in data[0]):
            return 'grounding'

    if 'referring' in filename_lower or 'action' in filename_lower:
        return 'referring'

    # Check content
    if data:
        sample = data[0]
        if 'gt_pairs' in sample or 'pred_pairs' in sample:
            return 'grounding'
        if 'prediction' in sample and 'ground_truth' in sample:
            return 'referring'

    return 'unknown'


def load_predictions(pred_file: str, max_images: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load predictions from JSON file.

    Supports multiple formats:
    - New "ours" format: {sample_id, file_name, ground_truth, prediction, ...}
    - Batch results format: {image_id, image_path, ground_truth, prediction, ...}
    - Per-triplet format: {triplet_id, file_name, ground_truth, prediction, ...}
    - Dict wrapper: {"results": [...]} or {"predictions": [...]}
    """
    print(f"Loading predictions from {pred_file}...")

    with open(pred_file, 'r') as f:
        data = json.load(f)

    # Handle different wrapper formats
    if isinstance(data, dict):
        if "results" in data:
            predictions = data["results"]
        elif "predictions" in data:
            predictions = data["predictions"]
        elif "annotations" in data:
            predictions = data["annotations"]
        else:
            # Assume the dict itself contains the data we need
            predictions = [data]
    elif isinstance(data, list):
        predictions = data
    else:
        raise ValueError("Unknown prediction file format. Expected list or dict.")

    # Detect file type
    file_type = detect_file_type(pred_file, predictions)
    print(f"Detected file type: {file_type}")

    if file_type == 'grounding':
        print("Warning: This appears to be a grounding result file (bbox predictions).")
        print("         LLM judge is designed for action referring (text predictions).")
        print("         Skipping this file or convert to referring format.")
        return []

    # Filter and normalize to items with both prediction and ground_truth
    valid_predictions = []
    for idx, item in enumerate(predictions):
        normalized = item.copy()

        # Handle various field name conventions
        # Ground truth field
        if "ground_truth" not in normalized:
            if "gt" in item:
                normalized["ground_truth"] = item["gt"]
            elif "label" in item:
                normalized["ground_truth"] = item["label"]
            elif "action" in item and "object_category" in item:
                # Grounding format - construct action phrase
                normalized["ground_truth"] = f"{item['action']} {item['object_category']}"

        # Prediction field
        if "prediction" not in normalized:
            if "pred" in item:
                normalized["prediction"] = item["pred"]
            elif "predicted_action" in item:
                normalized["prediction"] = item["predicted_action"]
            elif "raw_output" in item:
                normalized["prediction"] = item["raw_output"]

        # Sample ID for tracking
        if "sample_id" not in normalized:
            if "triplet_id" in item:
                normalized["sample_id"] = item["triplet_id"]
            elif "image_id" in item:
                normalized["sample_id"] = item["image_id"]
            else:
                normalized["sample_id"] = idx

        # File name for reference
        if "file_name" not in normalized:
            if "image_path" in item:
                normalized["file_name"] = item["image_path"]

        # Validate required fields
        if "prediction" in normalized and "ground_truth" in normalized:
            # Skip empty predictions
            if normalized["prediction"] and normalized["ground_truth"]:
                valid_predictions.append(normalized)

    if not valid_predictions:
        raise ValueError(
            "No valid predictions found. Items must have 'prediction' and 'ground_truth' fields.\n"
            "Supported formats:\n"
            "  - {prediction, ground_truth}\n"
            "  - {pred, gt}\n"
            "  - {raw_output, ground_truth}"
        )

    if max_images:
        valid_predictions = valid_predictions[:max_images]
        print(f"Limiting to first {max_images} samples.")

    print(f"Loaded {len(valid_predictions)} valid samples.")
    return valid_predictions


def construct_eval_prompt(ground_truth: str, prediction: str) -> str:
    """Construct the evaluation prompt for a single sample."""
    return f"""You are an impartial judge evaluating the quality of action descriptions for Human-Object Interaction (HOI).
Your task is to score a predicted action description by comparing it to the Ground Truth action.

Criteria:
1. Semantic Correctness: Does the prediction describe the same action as the ground truth? (e.g. "riding" vs "sitting on" might be close, but "standing" is wrong).
2. Interaction Accuracy: Is the interaction captured correctly?
3. Natural Language Quality: Is the phrase natural and grammatical?

Scale: 1 to 10
- 10: Perfect match or semantically identical synonym.
- 7-9: Minor differences but correct meaning.
- 4-6: Partially correct but misses key nuance.
- 1-3: Incorrect action.

Ground Truth: "{ground_truth}"
Prediction: "{prediction}"

Output the result in JSON format:
{{
  "score": <int>,
  "reason": "<string>"
}}"""


def parse_judge_response(response_text: str) -> Dict[str, Any]:
    """Parse JSON from LLM judge response."""
    try:
        # Try to find JSON block
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
        else:
            # Try parsing the whole text
            return json.loads(response_text)
    except json.JSONDecodeError:
        return {"score": None, "reason": "Failed to parse JSON response", "raw_response": response_text[:500]}


def sanitize_custom_id(custom_id: str, max_length: int = 64) -> str:
    """Sanitize custom_id to be compatible with batch APIs."""
    # Replace invalid chars with underscores
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', str(custom_id))
    # Remove consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    # Truncate to max length
    return sanitized[:max_length] if sanitized else "sample_0"


def make_request_key(item: dict, idx: int) -> str:
    """Generate a consistent request key for a prediction item."""
    return sanitize_custom_id(str(item.get("sample_id", idx)))


# =============================================================================
# New functions for local vLLM server
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LLM-as-a-Judge evaluation using local vLLM server (OpenAI-compatible API)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python eval_llm_judge_opensource.py \\
        --pred-file results/swig_action_claude_batch/results_20260115_173500.json

    # With options
    python eval_llm_judge_opensource.py \\
        --pred-file results/swig_referring_ours.json \\
        --max-images 100 --verbose

    # Resume interrupted run
    python eval_llm_judge_opensource.py \\
        --pred-file results/swig_referring_ours.json \\
        --resume results/swig_referring_ours_checkpoint.json

    # Custom model and server
    python eval_llm_judge_opensource.py \\
        --pred-file results/swig_referring_ours.json \\
        --model "mistralai/Mistral-7B-Instruct-v0.3" \\
        --base-url "http://localhost:8001/v1"
        """
    )

    # Input/Output
    parser.add_argument("--pred-file", type=str, required=True,
                        help="Path to prediction JSON file")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (defaults to same dir as pred-file)")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit number of samples to evaluate (for testing)")

    # Model Configuration
    parser.add_argument("--model", type=str,
                        default="nvidia/Llama-3_3-Nemotron-Super-49B-v1_5-FP8",
                        help="HF model ID served by vLLM (default: nvidia/Llama-3_3-Nemotron-Super-49B-v1_5-FP8)")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1",
                        help="vLLM server URL (default: http://localhost:8000/v1)")

    # Inference Configuration
    parser.add_argument("--concurrency", type=int, default=16,
                        help="Number of parallel API calls (default: 16)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (default: 0.0)")

    # Resume
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint file to resume from")

    # Display
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-sample score and reason to stdout")

    # W&B
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="llm-judge-opensource",
                        help="W&B project name (default: llm-judge-opensource)")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="W&B run name (default: auto-generated)")

    return parser.parse_args()


def create_client(base_url: str) -> OpenAI:
    """Create an OpenAI client pointed at a local vLLM server."""
    return OpenAI(base_url=base_url, api_key="EMPTY")


def judge_single(
    client: OpenAI,
    item: Dict[str, Any],
    model: str,
    temperature: float,
    idx: int
) -> Dict[str, Any]:
    """
    Judge a single prediction item.

    Returns a result dict with custom_id, status, and response or error.
    Retries once on failure after a 1-second delay.
    """
    key = make_request_key(item, idx)
    prompt = construct_eval_prompt(item["ground_truth"], item["prediction"])

    def _call() -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=256,
        )
        return response.choices[0].message.content

    try:
        response_text = _call()
        return {"custom_id": key, "status": "success", "response": response_text}
    except Exception as first_exc:
        time.sleep(1)
        try:
            response_text = _call()
            return {"custom_id": key, "status": "success", "response": response_text}
        except Exception as second_exc:
            return {"custom_id": key, "status": "error", "error": str(second_exc)}


def load_checkpoint(checkpoint_file: str) -> Dict[str, Any]:
    """Load checkpoint from file. Returns empty dict if file doesn't exist."""
    if not os.path.exists(checkpoint_file):
        return {}
    with open(checkpoint_file, 'r') as f:
        data = json.load(f)
    # data may be a list (older format) or dict keyed by custom_id
    if isinstance(data, list):
        return {r["custom_id"]: r for r in data}
    return data


def save_checkpoint(checkpoint_file: str, results: Dict[str, Any]) -> None:
    """Atomically save checkpoint to avoid corruption."""
    tmp_file = checkpoint_file + ".tmp"
    with open(tmp_file, 'w') as f:
        json.dump(results, f)
    os.replace(tmp_file, checkpoint_file)


def run_concurrent_judging(
    client: OpenAI,
    predictions: List[Dict[str, Any]],
    model: str,
    concurrency: int,
    temperature: float,
    checkpoint_file: Optional[str],
    verbose: bool,
) -> List[Dict[str, Any]]:
    """
    Run judging concurrently using a thread pool.

    Loads from checkpoint if provided, skips already-done samples, and saves
    a checkpoint every 100 new completions.

    Returns a list of all result dicts (checkpoint + new).
    """
    # Load existing checkpoint results
    checkpoint: Dict[str, Any] = {}
    if checkpoint_file:
        checkpoint = load_checkpoint(checkpoint_file)
        if checkpoint:
            print(f"Resuming from checkpoint: {len(checkpoint)} samples already done.")

    # Determine which samples still need processing
    pending = [
        (idx, item) for idx, item in enumerate(predictions)
        if make_request_key(item, idx) not in checkpoint
    ]
    total = len(predictions)
    already_done = total - len(pending)

    print(f"Total samples: {total} | Already done: {already_done} | Remaining: {len(pending)}")

    new_results: Dict[str, Any] = {}
    new_completions = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_idx = {
            executor.submit(judge_single, client, item, model, temperature, idx): (idx, item)
            for idx, item in pending
        }

        progress = tqdm(
            concurrent.futures.as_completed(future_to_idx),
            total=len(pending),
            desc="Judging",
            disable=verbose,
        )

        for future in progress:
            idx, item = future_to_idx[future]
            try:
                result = future.result()
            except Exception as exc:
                key = make_request_key(item, idx)
                result = {"custom_id": key, "status": "error", "error": f"Unexpected error: {exc}"}
            new_results[result["custom_id"]] = result
            new_completions += 1

            if verbose:
                parsed = parse_judge_response(result.get("response", ""))
                score = parsed.get("score", "N/A")
                reason = parsed.get("reason", "")
                gt = item.get("ground_truth", "")
                pred = item.get("prediction", "")
                print(
                    f"[{already_done + new_completions}/{total}] "
                    f"GT: {gt} | PRED: {pred} | SCORE: {score} | REASON: {reason}"
                )

            # Save checkpoint every 100 new completions
            if checkpoint_file and new_completions % 100 == 0:
                combined = {**checkpoint, **new_results}
                save_checkpoint(checkpoint_file, combined)

    # Final checkpoint save
    if checkpoint_file:
        combined = {**checkpoint, **new_results}
        save_checkpoint(checkpoint_file, combined)

    # Return all results as a list
    all_results = {**checkpoint, **new_results}
    return list(all_results.values())


def merge_and_save(
    predictions: List[Dict[str, Any]],
    results: List[Dict[str, Any]],
    request_keys: List[str],
    output_dir: str,
    input_filename: str,
    model: str,
) -> str:
    """
    Merge judging results with original predictions and save to disk.

    Returns:
        Path to saved results file.
    """
    # Create key -> result mapping
    result_map = {r["custom_id"]: r for r in results}

    # Merge results
    evaluated_results = []
    total_score = 0
    valid_scores = 0

    for idx, item in enumerate(predictions):
        result_item = item.copy()
        key = request_keys[idx] if idx < len(request_keys) else f"sample_{idx}"

        judge_result = result_map.get(key)

        if judge_result and judge_result["status"] == "success":
            eval_data = parse_judge_response(judge_result["response"])

            result_item["judge_reason"] = eval_data.get("reason")

            if eval_data.get("raw_response"):
                result_item["judge_raw_response"] = eval_data["raw_response"]

            score = eval_data.get("score")
            if isinstance(score, (int, float)) and score is not None:
                score = max(1, min(10, int(score)))
                result_item["judge_score"] = score
                total_score += score
                valid_scores += 1
            else:
                result_item["judge_score"] = score
        else:
            error_msg = (
                judge_result.get("error", "Result not found")
                if judge_result
                else "Result not found"
            )
            result_item["judge_error"] = error_msg

        evaluated_results.append(result_item)

    # Calculate average score
    avg_score = total_score / valid_scores if valid_scores > 0 else 0

    print(f"\nEvaluation Summary:")
    print(f"  Total samples: {len(predictions)}")
    print(f"  Scored samples: {valid_scores}")
    print(f"  Average score: {avg_score:.2f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(input_filename))[0]

    # Save detailed results
    results_filename = f"{base_name}_judge_opensource_{timestamp}.json"
    results_path = os.path.join(output_dir, results_filename)

    with open(results_path, 'w') as f:
        json.dump(evaluated_results, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    # Save metrics summary
    metrics = {
        "model": model,
        "average_score": avg_score,
        "valid_samples": valid_scores,
        "total_samples": len(predictions),
        "timestamp": timestamp,
        "input_file": input_filename,
    }

    metrics_filename = f"{base_name}_judge_opensource_{timestamp}_metrics.json"
    metrics_path = os.path.join(output_dir, metrics_filename)

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to: {metrics_path}")

    return results_path


def log_to_wandb(
    metrics: Dict[str, Any],
    args: argparse.Namespace,
    evaluated_results: List[Dict[str, Any]],
) -> None:
    """Log results to Weights & Biases."""
    if not WANDB_AVAILABLE:
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or f"judge_{timestamp}",
    )

    # Log scalar metrics
    wandb.log({
        "average_score": metrics["average_score"],
        "valid_samples": metrics["valid_samples"],
        "total_samples": metrics["total_samples"],
        "model": metrics["model"],
    })

    # Log score distribution as histogram
    scores = [
        r["judge_score"]
        for r in evaluated_results
        if isinstance(r.get("judge_score"), (int, float))
    ]
    if scores:
        wandb.log({"score_distribution": wandb.Histogram(scores)})

    # Log first 200 samples as a table
    table_rows = []
    for r in evaluated_results[:200]:
        table_rows.append([
            r.get("ground_truth", ""),
            r.get("prediction", ""),
            r.get("judge_score", None),
            r.get("judge_reason", ""),
        ])

    table = wandb.Table(
        columns=["ground_truth", "prediction", "judge_score", "judge_reason"],
        data=table_rows,
    )
    wandb.log({"samples": table})

    wandb.finish()


def main() -> None:
    args = parse_args()

    # Determine output directory
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.pred_file))
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("LLM-as-a-Judge Evaluation (Local vLLM)")
    print("=" * 60)
    print(f"Prediction file: {args.pred_file}")
    print(f"Model:           {args.model}")
    print(f"Server:          {args.base_url}")
    print(f"Concurrency:     {args.concurrency}")
    print(f"Temperature:     {args.temperature}")
    print("-" * 60)

    # Check vLLM server is reachable
    import urllib.request
    import urllib.error

    models_url = args.base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(models_url, timeout=10) as resp:
            resp.read()
        print(f"vLLM server reachable at {args.base_url}")
    except (urllib.error.URLError, ConnectionError, OSError) as e:
        print(f"\nError: Cannot reach vLLM server at {args.base_url}")
        print(f"  Reason: {e.reason if hasattr(e, 'reason') else e}")
        print("\nStart the server with:")
        print("  python -m vllm.entrypoints.openai.api_server \\")
        print(f"      --model {args.model} \\")
        print("      --port 8000")
        sys.exit(1)

    # Load predictions
    predictions = load_predictions(args.pred_file, args.max_images)
    if not predictions:
        print("No predictions to evaluate. Exiting.")
        sys.exit(0)

    # Auto-enable verbose for small datasets
    verbose = args.verbose or len(predictions) <= 20

    # Create OpenAI-compatible client
    client = create_client(args.base_url)

    # Build request keys (same logic as judge_single)
    request_keys = [
        make_request_key(item, idx)
        for idx, item in enumerate(predictions)
    ]

    # Determine checkpoint file path
    base_name = os.path.splitext(os.path.basename(args.pred_file))[0]
    checkpoint_file = os.path.join(output_dir, f"{base_name}_checkpoint.json")

    # Allow explicit --resume path to override
    if args.resume:
        checkpoint_file = args.resume

    print(f"Checkpoint file: {checkpoint_file}")

    # Run judging
    all_results = run_concurrent_judging(
        client=client,
        predictions=predictions,
        model=args.model,
        concurrency=args.concurrency,
        temperature=args.temperature,
        checkpoint_file=checkpoint_file,
        verbose=verbose,
    )

    # Merge and save
    results_path = merge_and_save(
        predictions=predictions,
        results=all_results,
        request_keys=request_keys,
        output_dir=output_dir,
        input_filename=args.pred_file,
        model=args.model,
    )

    # Load metrics for W&B logging
    # Find the matching metrics file
    timestamp_str = os.path.basename(results_path).split("_judge_opensource_")[1].replace(".json", "")
    metrics_file = os.path.join(
        output_dir,
        f"{base_name}_judge_opensource_{timestamp_str}_metrics.json"
    )

    metrics: Dict[str, Any] = {}
    if os.path.exists(metrics_file):
        with open(metrics_file) as f:
            metrics = json.load(f)

    # W&B logging
    if args.wandb:
        if not WANDB_AVAILABLE:
            print("Warning: wandb not installed. Skipping W&B logging.")
        else:
            # Re-load evaluated results for table logging
            with open(results_path) as f:
                evaluated_results = json.load(f)
            log_to_wandb(metrics, args, evaluated_results)

    # Print final summary
    print("\n" + "=" * 60)
    print("Done!")
    print(f"  Total samples:   {metrics.get('total_samples', len(predictions))}")
    print(f"  Scored samples:  {metrics.get('valid_samples', 'N/A')}")
    print(f"  Average score:   {metrics.get('average_score', 0):.2f} / 10")
    print(f"  Results:         {results_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
