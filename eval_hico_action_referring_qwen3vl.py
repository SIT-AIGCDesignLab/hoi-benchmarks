"""
HICO-DET Action Referring Evaluation Script for Qwen3VL

Evaluates Qwen3VL action prediction using METEOR and CIDEr metrics.

Task: Given (person, object) bounding boxes, predict the connecting action.

Key Differences from Groma Evaluation:
- Uses Qwen3VL model instead of Groma
- Different prompt format (no refer_box tokens)
- Direct bbox input to model
- Same metrics: METEOR, CIDEr, BLEU, ROUGE-L

Usage:
    python groma/eval/eval_hico_action_referring_qwen3vl.py \
        --model-name Qwen/Qwen3-VL-8B-Instruct \
        --img-prefix /path/to/hico/images/test2015 \
        --ann-file groma_data/benchmarks/hico_action_referring_test.json \
        --pred-file results/predictions.json \
        --verbose \
        --max-images 10
"""

import argparse
import base64
import io
import json
import os
import re
import tempfile
from collections import defaultdict
from datetime import datetime

from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def extract_response_from_text(response_text, is_thinking_model=False):
    """
    Extract response from text output, handling thinking and instruct models.

    Returns:
        (thinking_content, final_answer)
    """
    response_text = response_text.strip()
    prefixes_to_remove = ["output: ", "Output: ", "ANSWER: ", "Answer: "]

    if not is_thinking_model:
        return "", response_text

    if "</think>" in response_text:
        thinking_match = re.findall(r"<think>(.*?)</think>", response_text, re.DOTALL)
        thinking_content = thinking_match[-1].strip() if thinking_match else ""
        final_answer = response_text.rsplit("</think>", 1)[-1].strip()
    else:
        if "<think>" in response_text and "</think>" in response_text:
            parts = response_text.split("</think>")
            thinking_content = parts[0].split("<think>")[-1].strip()
            final_answer = parts[1].strip() if len(parts) > 1 else ""
        else:
            return "", response_text

    for prefix in prefixes_to_remove:
        if final_answer.startswith(prefix):
            final_answer = final_answer[len(prefix):].strip()
            break

    return thinking_content, final_answer


def build_action_referring_prompt(object_category=None, is_thinking_model=False):
    """
    Build prompt for action referring task.

    Args:
        object_category: Object category name
        is_thinking_model: Whether this is a thinking model (adds thinking instructions)

    Returns:
        List of message dicts for Qwen3VL
    """
    if is_thinking_model:
        prompt_text = (
            "Task: Describe the action/interaction between the two objects in the two bounding boxes.\n\n"
            "EXPLAIN in thinking:\n"
            "1. What visual cues indicate each object's identity, pose, and positioning\n"
            "2. How the spatial relationship between the two objects suggests an interaction\n"
            "3. What action verb best describes this interaction\n"
            "4. Why this action is more appropriate than alternatives\n\n"
            "IMPORTANT - Use this exact format:\n"
            "<think>\n"
            "Step 1 - Analyze the two objects:\n"
            "[Your detailed reasoning about object identities and poses]\n\n"
            "Step 2 - Analyze the interaction:\n"
            "[Your reasoning about spatial relationships and interaction]\n\n"
            "Step 3 - Determine the action:\n"
            "[Your reasoning about the specific action verb]\n"
            "</think>\n\n"
            "Then output ONLY a SHORT action phrase (2-4 words) in format: [action] [object]\n"
            "Examples: 'riding bicycle', 'holding cup', 'sitting on chair'\n\n"
            "Output Format:\n"
            "Provide ONLY the action phrase (no explanations, no punctuation)."
        )
    else:
        if object_category:
            prompt_text = (
                f"Task: Describe what action the person (in the first bounding box) is performing with the {object_category} (in the second bounding box).\n\n"
                f"Instructions:\n"
                f"1. Focus on the interaction between the person and the {object_category}\n"
                f"2. Provide a SHORT action phrase (2-4 words)\n"
                f"3. Use format: [action] [object] (e.g., 'riding bicycle', 'holding cup')\n\n"
                f"Output Format:\n"
                f"Provide ONLY the action phrase (no explanations, no punctuation)."
            )
        else:
            prompt_text = (
                "Task: Describe what action the person (in the first bounding box) is performing with the object (in the second bounding box).\n\n"
                "Instructions:\n"
                "1. Focus on the interaction between the person and the object\n"
                "2. Provide a SHORT action phrase (2-4 words)\n"
                "3. Include both the action verb and the object\n\n"
                "Output Format:\n"
                "Provide ONLY the action phrase (no explanations, no punctuation)."
            )

    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "image_placeholder"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]


def convert_bbox_to_qwen_format(bbox, img_size):
    """
    Convert bounding box to Qwen3VL format [0, 1000].
    """
    width, height = img_size
    x1, y1, x2, y2 = bbox
    return [
        int((x1 / width) * 1000),
        int((y1 / height) * 1000),
        int((x2 / width) * 1000),
        int((y2 / height) * 1000),
    ]


def image_to_base64(image):
    """Convert a PIL image to a JPEG data URL."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def get_response_text(response):
    """Normalize OpenAI client content into plain text."""
    content = response.choices[0].message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(text)
            else:
                text = getattr(item, "text", None)
                if text:
                    parts.append(text)
        return "".join(parts)
    return str(content)


def run_inference(
    client,
    model_name,
    image_path,
    person_bbox,
    object_bbox,
    object_category=None,
    is_thinking_model=False,
):
    """
    Run Qwen3VL inference for action referring through a vLLM endpoint.
    """
    image = Image.open(image_path).convert("RGB")
    img_width, img_height = image.size

    person_bbox_qwen = convert_bbox_to_qwen_format(person_bbox, (img_width, img_height))
    object_bbox_qwen = convert_bbox_to_qwen_format(object_bbox, (img_width, img_height))

    messages = build_action_referring_prompt(object_category, is_thinking_model)
    text_content = messages[0]["content"][1]["text"]

    if is_thinking_model:
        bbox_prompt = (
            f"\n\nFirst bounding box: {person_bbox_qwen}\n"
            f"Second bounding box: {object_bbox_qwen}\n\n"
            f"{text_content}"
        )
    else:
        bbox_prompt = (
            f"\n\nPerson bounding box: {person_bbox_qwen}\n"
            f"Object bounding box: {object_bbox_qwen}\n\n"
            f"{text_content}"
        )

    request_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_to_base64(image)}},
                {"type": "text", "text": bbox_prompt},
            ],
        }
    ]

    request_kwargs = {
        "model": model_name,
        "messages": request_messages,
    }
    if is_thinking_model:
        request_kwargs.update(
            {
                "max_tokens": 2048,
                "temperature": 0.2,
                "top_p": 0.9,
                "extra_body": {"top_k": 20},
            }
        )
    else:
        request_kwargs.update(
            {
                "max_tokens": 30,
                "temperature": 0.0,
            }
        )

    response = client.chat.completions.create(**request_kwargs)
    raw_text = get_response_text(response)
    thinking_content, output_text = extract_response_from_text(raw_text, is_thinking_model)

    return thinking_content, output_text, image


def clean_action_response(response_text):
    """Clean action response to extract action phrase."""
    response = response_text.strip()
    prefixes_to_remove = [
        "the person is ",
        "person is ",
        "they are ",
        "action: ",
        "answer: ",
    ]

    response_lower = response.lower()
    for prefix in prefixes_to_remove:
        if response_lower.startswith(prefix):
            response = response[len(prefix) :].strip()
            break

    response = response.rstrip(".!?,;:")
    return response.lower().strip()


def visualize_action_triplet(
    image_path, person_bbox, object_bbox, predicted_action, gt_action, object_category, output_path
):
    """Visualize action referring result."""
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except Exception:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()

    px1, py1, px2, py2 = person_bbox
    draw.rectangle([px1, py1, px2, py2], outline="red", width=4)
    draw.text((px1, py1 - 25), "Person", fill="red", font=font_small)

    ox1, oy1, ox2, oy2 = object_bbox
    draw.rectangle([ox1, oy1, ox2, oy2], outline="blue", width=4)
    draw.text((ox1, oy1 - 25), object_category.capitalize(), fill="blue", font=font_small)

    person_center = ((px1 + px2) / 2, (py1 + py2) / 2)
    object_center = ((ox1 + ox2) / 2, (oy1 + oy2) / 2)
    draw.line([person_center, object_center], fill="green", width=3)

    draw.text((10, 10), f"Predicted: {predicted_action}", fill="white", font=font)
    draw.text((10, 40), f"GT: {gt_action}", fill="yellow", font=font)

    match = predicted_action.lower().strip() == gt_action.lower().strip()
    match_text = "MATCH" if match else "MISMATCH"
    match_color = "green" if match else "red"
    draw.text((10, 70), match_text, fill=match_color, font=font)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)


def eval_model(args):
    """Main evaluation function."""
    if not args.vllm_url:
        raise ValueError("--vllm-url is required for vLLM endpoint inference")

    print("=" * 80)
    print("HICO-DET Action Referring Evaluation (Qwen3VL)")
    print("=" * 80)
    print(f"Model:       {args.model_name}")
    print(f"Device:      {args.device}")
    print(f"vLLM URL:    {args.vllm_url}")
    print(f"Images:      {args.img_prefix}")
    print(f"Annotations: {args.ann_file}")
    if args.image:
        print(f"Image filter: {args.image}")
    if args.max_images:
        print(f"Max images:  {args.max_images} (DEBUGGING MODE)")
    if args.verbose:
        print("Verbose:     ENABLED (per-triplet results + visualizations)")
    print("=" * 80)
    print()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    use_wandb = WANDB_AVAILABLE and args.wandb
    if use_wandb:
        print("Initializing Weights & Biases...")
        try:
            wandb.login()
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or f"hico_action_qwen3vl_{timestamp}",
                config={
                    "model": args.model_name,
                    "device": args.device,
                    "vllm_url": args.vllm_url,
                    "dataset": "HICO-DET-Action",
                    "task": "action_referring",
                    "max_images": args.max_images,
                    "timestamp": timestamp,
                },
                tags=["hico", "action-referring", "qwen3vl", "vllm"],
            )
            print("WandB initialized successfully")
            print(f"  Run URL: {wandb.run.url}")
            print(f"  Project: {wandb.run.project}")
            print(f"  Run name: {wandb.run.name}\n")
        except Exception as e:
            print(f"Warning: WandB initialization failed: {e}")
            print("Continuing evaluation without WandB logging...\n")
            use_wandb = False

    client = OpenAI(base_url=f"{args.vllm_url}/v1", api_key="placeholder")
    is_thinking_model = "Thinking" in args.model_name or "thinking" in args.model_name
    print(f"Model type: {'Thinking' if is_thinking_model else 'Instruct'}\n")

    print(f"Loading annotations from: {args.ann_file}")
    with open(args.ann_file, "r") as f:
        dataset_samples = json.load(f)

    print(f"Loaded {len(dataset_samples)} action referring triplets")

    if args.image:
        dataset_samples = [sample for sample in dataset_samples if args.image in sample["file_name"]]
        print(f"After image filter '{args.image}': {len(dataset_samples)} triplets")

    if args.max_images is not None and args.max_images < len(dataset_samples):
        print(f"\nLimiting evaluation to first {args.max_images} triplets")
        dataset_samples = dataset_samples[: args.max_images]

    print(f"\nDataset: {len(dataset_samples)} triplets")
    print("=" * 80)

    predictions = []
    per_triplet_results = []
    action_stats = defaultdict(
        lambda: {
            "total": 0,
            "exact_match": 0,
            "predictions": [],
            "ground_truths": [],
        }
    )

    partial_file = args.pred_file + ".partial.jsonl"
    processed_indices = set()
    if args.resume and os.path.exists(partial_file):
        print(f"\nResuming from partial checkpoint: {partial_file}")
        loaded = 0
        with open(partial_file, "r") as partial_in:
            for line in partial_in:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    idx = rec["idx"]
                    triplet_result = rec["triplet_result"]
                    prediction = rec["prediction"]
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"  Warning: skipping corrupt partial record: {e}")
                    continue

                processed_indices.add(idx)
                per_triplet_results.append(triplet_result)
                predictions.append(prediction)
                gt_action = triplet_result["ground_truth"]
                pred_action = triplet_result["prediction"]
                action_stats[gt_action]["total"] += 1
                action_stats[gt_action]["exact_match"] += 1 if triplet_result["exact_match"] else 0
                action_stats[gt_action]["predictions"].append(pred_action)
                action_stats[gt_action]["ground_truths"].append(gt_action)
                loaded += 1

        print(f"Loaded {loaded} completed samples, resuming from next unprocessed...")
    elif args.resume:
        print(f"No partial checkpoint found at {partial_file}, starting fresh")

    partial_f = open(partial_file, "a")

    viz_dir = None
    if args.verbose:
        viz_dir = args.pred_file.replace(".json", "_visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        print(f"Visualization directory: {viz_dir}\n")

    show_verbose = args.verbose or len(dataset_samples) <= 100

    print("\nStarting evaluation...")
    for idx, sample in enumerate(tqdm(dataset_samples, disable=show_verbose)):
        if idx in processed_indices:
            continue

        file_name = sample["file_name"]
        img_path = os.path.join(args.img_prefix, file_name)

        boxes = sample["boxes"]
        person_bbox = boxes[sample["person_box_idx"]]
        object_bbox = boxes[sample["object_box_idx"]]
        gt_action = sample["gt_action"]

        object_category = sample.get("object_category")
        if not object_category or object_category == "object":
            parts = gt_action.split()
            object_category = parts[-1] if len(parts) >= 2 else None

        if show_verbose:
            print(f"\n[Triplet {idx + 1}/{len(dataset_samples)}] {file_name}")
            print(f"  Person bbox: {person_bbox}")
            print(f"  Object bbox: {object_bbox}")
            print(f"  GT action: {gt_action}")

        thinking_content, output_text, _ = run_inference(
            client,
            args.model_name,
            img_path,
            person_bbox,
            object_bbox,
            object_category,
            is_thinking_model,
        )

        if show_verbose:
            if thinking_content:
                print(f"  Thinking: {thinking_content[:150]}...")
            print(f"  Raw output: {output_text[:100]}...")

        predicted_action = clean_action_response(output_text)

        if show_verbose:
            print(f"  Predicted: {predicted_action}")

        prediction = {"image_id": idx, "caption": predicted_action}
        predictions.append(prediction)

        exact_match = predicted_action.lower().strip() == gt_action.lower().strip()
        triplet_result = {
            "triplet_id": idx,
            "file_name": file_name,
            "person_bbox": person_bbox,
            "object_bbox": object_bbox,
            "ground_truth": gt_action,
            "prediction": predicted_action,
            "raw_output": output_text[:200],
            "exact_match": exact_match,
        }
        if thinking_content:
            triplet_result["thinking_content"] = thinking_content

        per_triplet_results.append(triplet_result)

        action_stats[gt_action]["total"] += 1
        action_stats[gt_action]["exact_match"] += 1 if exact_match else 0
        action_stats[gt_action]["predictions"].append(predicted_action)
        action_stats[gt_action]["ground_truths"].append(gt_action)

        partial_f.write(json.dumps({"idx": idx, "prediction": prediction, "triplet_result": triplet_result}) + "\n")
        partial_f.flush()

        if viz_dir is not None:
            viz_path = os.path.join(viz_dir, f"{idx:05d}_{file_name}")
            try:
                visualize_action_triplet(
                    img_path,
                    person_bbox,
                    object_bbox,
                    predicted_action,
                    gt_action,
                    object_category or "object",
                    viz_path,
                )

                if use_wandb:
                    wandb.log(
                        {
                            f"visualization/{idx:04d}": wandb.Image(
                                viz_path,
                                caption=f"{file_name} | Pred: {predicted_action} | GT: {gt_action}",
                            )
                        }
                    )
            except Exception as e:
                if show_verbose:
                    print(f"  Warning: Visualization failed: {e}")

        if use_wandb:
            wandb_metrics = {
                "sample_idx": idx,
                "exact_match": 1 if exact_match else 0,
            }
            if thinking_content:
                wandb_metrics["has_thinking"] = 1
            wandb.log(wandb_metrics)

    partial_f.close()

    print("\n" + "=" * 80)
    print("Computing METEOR and CIDEr metrics...")
    print("=" * 80)

    annotations = []
    images_info = []
    for idx, sample in enumerate(dataset_samples):
        images_info.append({"id": idx})
        annotations.append({"image_id": idx, "caption": sample["gt_action"], "id": idx})

    gt_coco_format = {
        "info": {
            "description": "HICO-DET Action Referring Ground Truth",
            "version": "1.0",
            "year": 2025,
        },
        "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
        "images": images_info,
        "annotations": annotations,
        "type": "captions",
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(gt_coco_format, f)
        gt_file = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(predictions, f)
        pred_file = f.name

    coco = COCO(gt_file)
    coco_result = coco.loadRes(pred_file)
    coco_eval = COCOEvalCap(coco, coco_result)
    try:
        coco_eval.evaluate()
    except Exception as e:
        print(f"\nWarning: SPICE metric failed: {type(e).__name__}")
        print("Continuing with BLEU, METEOR, ROUGE_L, CIDEr metrics.")

    os.unlink(gt_file)
    os.unlink(pred_file)

    print("\n" + "=" * 80)
    print("HICO-DET Action Referring Results (Qwen3VL)")
    print("=" * 80)
    print(f"{'Metric':<15} {'Score':>10}  {'Description':<50}")
    print("-" * 80)

    metrics = {}
    for metric, score in coco_eval.eval.items():
        metrics[metric] = score
        desc = {
            "BLEU_1": "BLEU-1 (unigram overlap)",
            "BLEU_2": "BLEU-2 (bigram overlap)",
            "BLEU_3": "BLEU-3 (trigram overlap)",
            "BLEU_4": "BLEU-4 (4-gram overlap)",
            "METEOR": "METEOR (semantic similarity)",
            "ROUGE_L": "ROUGE-L (longest common subsequence)",
            "CIDEr": "CIDEr (corpus consensus)",
            "SPICE": "SPICE (semantic propositional content)",
        }.get(metric, metric)
        print(f"{metric:<15} {score * 100:>9.2f}%  {desc:<50}")

    total_triplets = len(per_triplet_results)
    exact_matches = sum(1 for r in per_triplet_results if r["exact_match"])
    exact_match_acc = exact_matches / total_triplets if total_triplets > 0 else 0.0
    metrics["exact_match"] = exact_match_acc

    print("-" * 80)
    print(f"{'Exact Match':<15} {exact_match_acc * 100:>9.2f}%  {'Exact string match accuracy':<50}")
    print("=" * 80)

    if use_wandb:
        wandb.log(metrics)
        wandb.log({"total_triplets": total_triplets, "exact_matches": exact_matches})

    print("\n" + "=" * 80)
    print("Per-Action Statistics (Top 20 by frequency)")
    print("=" * 80)
    print(f"{'Action':<30} {'Count':>8} {'Exact Match':>12}")
    print("-" * 80)

    sorted_actions = sorted(action_stats.items(), key=lambda x: x[1]["total"], reverse=True)
    for action, stats in sorted_actions[:20]:
        accuracy = stats["exact_match"] / stats["total"] if stats["total"] > 0 else 0.0
        print(f"{action:<30} {stats['total']:>8} {accuracy * 100:>11.1f}%")

    print("=" * 80)

    os.makedirs(os.path.dirname(os.path.abspath(args.pred_file)), exist_ok=True)

    print(f"\nSaving predictions to: {args.pred_file}")
    with open(args.pred_file, "w") as f:
        json.dump(predictions, f, indent=2)

    metrics_file = args.pred_file.replace(".json", "_metrics.json")
    print(f"Saving metrics to: {metrics_file}")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    if args.verbose:
        per_triplet_file = args.pred_file.replace(".json", "_per_triplet.json")
        print(f"Saving per-triplet results to: {per_triplet_file}")
        with open(per_triplet_file, "w") as f:
            json.dump(per_triplet_results, f, indent=2)

        per_action_file = args.pred_file.replace(".json", "_per_action.json")
        print(f"Saving per-action stats to: {per_action_file}")
        action_stats_dict = {action: stats for action, stats in action_stats.items()}
        with open(per_action_file, "w") as f:
            json.dump(action_stats_dict, f, indent=2)

    if is_thinking_model:
        thinking_samples = [s for s in per_triplet_results if s.get("thinking_content")]
        if thinking_samples:
            thinking_file = args.pred_file.replace(".json", "_thinking.jsonl")
            print(f"Saving thinking content to: {thinking_file}")
            with open(thinking_file, "w") as f:
                for result in thinking_samples:
                    thinking_entry = {
                        "file_name": result["file_name"],
                        "triplet_id": result["triplet_id"],
                        "ground_truth": result["ground_truth"],
                        "prediction": result["prediction"],
                        "thinking_content": result["thinking_content"],
                        "raw_output": result["raw_output"],
                    }
                    f.write(json.dumps(thinking_entry) + "\n")
            print(f"  Saved {len(thinking_samples)} samples with thinking content")

    if os.path.exists(partial_file):
        os.remove(partial_file)
        print(f"Removed partial checkpoint: {partial_file}")

    if viz_dir is not None:
        viz_count = len([f for f in os.listdir(viz_dir) if f.endswith(".jpg")])
        print(f"\nVisualizations: {viz_dir}/")
        print(f"  Total images saved: {viz_count}")

    if use_wandb:
        wandb.save(args.pred_file)
        wandb.save(metrics_file)
        if args.verbose:
            wandb.save(per_triplet_file)
            wandb.save(per_action_file)
        if is_thinking_model and thinking_samples:
            wandb.save(thinking_file)
            wandb.log({"thinking_samples_count": len(thinking_samples)})

        action_table_data = []
        for action, stats in sorted_actions[:20]:
            accuracy = stats["exact_match"] / stats["total"] if stats["total"] > 0 else 0.0
            action_table_data.append([action, stats["total"], stats["exact_match"], f"{accuracy:.1%}"])

        wandb.log(
            {
                "action_performance_table": wandb.Table(
                    columns=["Action", "Total", "Exact Match", "Accuracy"],
                    data=action_table_data,
                )
            }
        )

        wandb.finish()
        print("WandB logging complete")

    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HICO-DET Action Referring Evaluation with Qwen3VL")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-VL-8B-Instruct", help="Qwen3VL model name")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (kept for compatibility; inference now uses the vLLM endpoint)",
    )
    parser.add_argument(
        "--ann-file",
        type=str,
        default="../dataset/benchmarks_simplified/hico_action_referring_test_simplified.json",
        help="Path to HICO action referring annotation file",
    )
    parser.add_argument(
        "--img-prefix",
        type=str,
        default="../dataset/hico_20160224_det/images/test2015",
        help="Path to HICO images directory",
    )
    parser.add_argument("--pred-file", type=str, required=True, help="Output file for predictions")
    parser.add_argument("--vllm-url", type=str, default=None, help="URL of running vLLM server")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Filter samples to those whose file_name contains this string",
    )
    parser.add_argument("--max-images", type=int, default=None, help="Limit evaluation to first N triplets")
    parser.add_argument("--verbose", action="store_true", help="Show detailed per-triplet results and visualizations")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="hico-action-referring-qwen3vl", help="W&B project")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name")
    parser.add_argument("--resume", action="store_true", help="Resume from partial checkpoint if available")

    eval_model(parser.parse_args())
