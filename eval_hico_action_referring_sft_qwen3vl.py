"""
HICO-DET Action Referring Evaluation Script for SFT-trained Qwen3VL with Tool Use

Evaluates a SFT-trained Qwen3VL checkpoint on HICO-DET action referring using
a vLLM server and a custom multi-turn tool-call loop.

Metrics: METEOR, CIDEr, BLEU, ROUGE-L (same as baseline, compatible with
calculate_bertscore.py for BERTScore post-processing).
"""

import os
import json
import re
import argparse
import base64
import io
import tempfile
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from openai import OpenAI
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# =============================================================================
# System Prompt (must match SFT training exactly)
# =============================================================================
SYSTEM_PROMPT = """\
You are an expert at analyzing human-object interactions in images. You have \
access to visual tools to help you inspect details when needed.

Your task is to analyze the spatial relationships and interactions between \
people and objects in an image. You will be provided with an image and object \
detection proposals (bounding boxes with class labels and confidence scores).

## Available Tools

You have access to two tools to help you analyze the image:

- **zoom_in(bbox_2d)**: Crop and zoom into a specific region of the image \
for closer inspection. The bbox_2d parameter should be a bounding box in the \
format [x1, y1, x2, y2] using 1000x1000 normalized coordinates, where \
(x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.

- **zoom_out()**: Return to the full original image view after zooming in. \
This takes no parameters.

## When to Use Tools

- Use zoom_in when the interaction between a person and object is ambiguous, \
when objects are small or far away, or when you need to verify fine-grained \
contact details (e.g., is the person actually holding the object, or just \
standing near it?).

- Do NOT use tools when the interaction is clearly visible from the full \
image and proposals alone.

## Response Format

Follow this process:
1. Reason about the image and proposals in a <think> block.
2. If needed, zoom in with the zoom_in tool. Reason again in a <think> block.
3. Provide your final answer in an <answer> block.

Do not include tool calls or thinking blocks in your final answer.\
"""


# =============================================================================
# Visualization
# =============================================================================

def visualize_action_triplet(image_path, person_bbox, object_bbox,
                             predicted_action, gt_action, object_category, output_path):
    """Visualize action referring result: bboxes + predicted/GT action text."""
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()

    px1, py1, px2, py2 = person_bbox
    draw.rectangle([px1, py1, px2, py2], outline="red", width=4)
    draw.text((px1, max(0, py1 - 20)), "Person", fill="red", font=font_small)

    ox1, oy1, ox2, oy2 = object_bbox
    draw.rectangle([ox1, oy1, ox2, oy2], outline="blue", width=4)
    draw.text((ox1, max(0, oy1 - 20)), object_category.capitalize(), fill="blue", font=font_small)

    pc = ((px1 + px2) / 2, (py1 + py2) / 2)
    oc = ((ox1 + ox2) / 2, (oy1 + oy2) / 2)
    draw.line([pc, oc], fill="green", width=3)

    draw.text((10, 10), f"Pred: {predicted_action}", fill="white", font=font)
    draw.text((10, 38), f"GT:   {gt_action}", fill="yellow", font=font)
    match = predicted_action.lower().strip() == gt_action.lower().strip()
    draw.text((10, 66), "✓ MATCH" if match else "✗ MISMATCH",
              fill="lime" if match else "red", font=font)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)


# =============================================================================
# Utility Functions
# =============================================================================

def format_proposals(proposals: list[dict]) -> str:
    items = []
    for idx, p in enumerate(proposals):
        items.append({
            "id": idx,
            "bbox_2d": p["bbox_1000"],
            "label": p["class_name"],
            "confidence": round(p["confidence"], 2),
        })
    return json.dumps(items, indent=2)


def load_proposals(image_stem: str, proposals_dir: str) -> list[dict]:
    proposal_path = os.path.join(proposals_dir, f"{image_stem}.json")
    if not os.path.exists(proposal_path):
        return []
    with open(proposal_path) as f:
        data = json.load(f)
    return data.get("proposals", [])


def convert_bbox_to_1000(bbox: list, width: int, height: int) -> list:
    """Convert pixel bbox to [0,1000] normalized format."""
    x1, y1, x2, y2 = bbox
    return [
        int(x1 / (width - 1) * 1000) if width > 1 else int(x1),
        int(y1 / (height - 1) * 1000) if height > 1 else int(y1),
        int(x2 / (width - 1) * 1000) if width > 1 else int(x2),
        int(y2 / (height - 1) * 1000) if height > 1 else int(y2),
    ]


def build_referring_prompt(person_bbox_1000: list, object_bbox_1000: list,
                           proposals: list[dict]) -> str:
    """Build user prompt for action referring task."""
    proposals_text = format_proposals(proposals)
    person_bbox_json = json.dumps({"bbox_2d": person_bbox_1000, "label": "person"})
    object_bbox_json = json.dumps({"bbox_2d": object_bbox_1000, "label": "object"})

    return (
        "You will be identifying and describing the action that a person "
        "is performing with a specific object in an image.\n\n"
        "Here are the candidate object proposals detected in the image:\n\n"
        f"<proposals>\n{proposals_text}\n</proposals>\n\n"
        f"The person you need to analyze is located at: **{person_bbox_json}**\n\n"
        f"The object they are interacting with is located at: **{object_bbox_json}**\n\n"
        "Your task is to describe the action the person is performing with "
        "this object. Analyze the spatial relationship between the person "
        "and object, their positioning, and any visible interaction patterns "
        "to determine what action is taking place.\n\n"
        "Before providing your final answer, reason about what you observe "
        "in a <think> block.\n\n"
        "Your response must follow these formatting rules:\n"
        '- Use the format: "{verb+ing} {object}" where the verb is in '
        "present participle (-ing) form\n"
        "- Do not include articles (a, an, the)\n"
        "- Be concise and specific\n"
        "- Use only the action phrase, nothing else\n\n"
        "Examples: 'riding bicycle', 'holding umbrella', 'sitting on bench'\n\n"
        "Write your final answer inside <answer> tags. Your answer should "
        "contain only the action phrase in the specified format."
    )


def image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=90)
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_b64}"


def execute_zoom_in(image: Image.Image, bbox_2d_1000: list) -> Image.Image:
    w, h = image.size
    x1 = max(0, int(bbox_2d_1000[0] * w / 1000))
    y1 = max(0, int(bbox_2d_1000[1] * h / 1000))
    x2 = min(w, int(bbox_2d_1000[2] * w / 1000))
    y2 = min(h, int(bbox_2d_1000[3] * h / 1000))
    if x2 <= x1 or y2 <= y1:
        return image
    return image.crop((x1, y1, x2, y2))


def parse_tool_call(text: str) -> dict | None:
    match = re.search(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def extract_thinking(text: str) -> str:
    matches = re.findall(r'<think>(.*?)</think>', text, re.DOTALL)
    return matches[-1].strip() if matches else ""


def extract_answer(text: str) -> str | None:
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    return match.group(1).strip() if match else None


def run_sft_agent_loop(client: OpenAI, model_name: str, messages: list,
                       original_image: Image.Image, max_turns: int = 5) -> tuple:
    """
    Multi-turn inference with zoom_in/zoom_out tool handling.

    Returns:
        (answer_text, tool_calls, thinking_text, zoom_crops)
        zoom_crops: list of (turn, bbox, PIL.Image) for each zoom_in call
    """
    current_image = original_image
    tool_calls_log = []
    all_thinking = []
    zoom_crops = []  # (turn, bbox, cropped_image)

    for turn in range(max_turns):
        max_tokens = 4096
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0,
            )
        except Exception as e:
            err_str = str(e)
            import re as _re
            m = _re.search(r'maximum context length is (\d+) tokens and your request has (\d+) input tokens', err_str)
            if m:
                available = int(m.group(1)) - int(m.group(2)) - 64
                if available > 64:
                    print(f"\n  ⚠️  Context overflow (input too long), retrying with max_tokens={available}")
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        max_tokens=available,
                        temperature=0.0,
                    )
                else:
                    print(f"\n  ⚠️  Context overflow, insufficient space for response. Skipping turn.")
                    break
            else:
                raise
        text = response.choices[0].message.content

        thinking = extract_thinking(text)
        if thinking:
            all_thinking.append(thinking)

        answer = extract_answer(text)
        if answer is not None:
            return answer, tool_calls_log, "\n\n".join(all_thinking), zoom_crops

        tool = parse_tool_call(text)
        if tool:
            tool_name = tool.get("name", "")
            tool_args = tool.get("arguments", {})

            if tool_name == "zoom_in":
                bbox = tool_args.get("bbox_2d", [0, 0, 1000, 1000])
                current_image = execute_zoom_in(original_image, bbox)
                tool_calls_log.append({"name": "zoom_in", "bbox": bbox, "turn": turn})
                zoom_crops.append((turn, bbox, current_image.copy()))
            elif tool_name == "zoom_out":
                current_image = original_image
                tool_calls_log.append({"name": "zoom_out", "turn": turn})

            messages.append({"role": "assistant", "content": text})
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_to_base64(current_image)}},
                    {"type": "text", "text": "Here is the zoomed view. Continue your analysis."}
                ]
            })
        else:
            break

    return None, tool_calls_log, "\n\n".join(all_thinking), zoom_crops


def clean_action_response(response_text: str) -> str:
    """Clean action response to extract action phrase."""
    if not response_text:
        return ""
    response = response_text.strip()
    prefixes_to_remove = ["the person is ", "person is ", "they are ", "action: ", "answer: "]
    response_lower = response.lower()
    for prefix in prefixes_to_remove:
        if response_lower.startswith(prefix):
            response = response[len(prefix):].strip()
            break
    response = response.rstrip('.!?,;:')
    return response.lower().strip()


# =============================================================================
# Main Evaluation
# =============================================================================

def eval_model(args):
    print("=" * 80)
    print("HICO-DET Action Referring Evaluation (SFT Qwen3VL + vLLM)")
    print("=" * 80)
    print(f"vLLM URL:    {args.vllm_url}")
    print(f"Model:       {args.model_name}")
    print(f"Annotation:  {args.ann_file}")
    print(f"Images:      {args.img_prefix}")
    print(f"Proposals:   {args.proposals_dir}")
    print(f"Output:      {args.pred_file}")
    if args.max_images:
        print(f"Max images:  {args.max_images} (DEBUGGING MODE)")
    print("=" * 80)
    print()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    client = OpenAI(base_url=f"{args.vllm_url}/v1", api_key="placeholder")

    use_wandb = WANDB_AVAILABLE and args.wandb
    if use_wandb:
        try:
            wandb.login()
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or f"hico_action_sft_{timestamp}",
                config={"model": args.model_name, "dataset": "HICO-DET-Action"},
                tags=["hico", "action-referring", "sft", "tool-use"]
            )
        except Exception as e:
            print(f"Warning: WandB init failed: {e}")
            use_wandb = False

    print(f"Loading annotations from: {args.ann_file}")
    with open(args.ann_file) as f:
        dataset_samples = json.load(f)
    print(f"Loaded {len(dataset_samples)} triplets")

    if args.max_images:
        dataset_samples = dataset_samples[:args.max_images]
    print(f"Evaluating {len(dataset_samples)} triplets")

    predictions = []
    per_triplet_results = []
    action_stats = defaultdict(lambda: {"total": 0, "exact_match": 0})
    thinking_records = []
    missing_proposals = 0

    show_verbose = args.verbose or len(dataset_samples) <= 100

    viz_dir = None
    if show_verbose:
        viz_dir = args.pred_file.replace(".json", "_visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        print(f"Visualization directory: {viz_dir}\n")

    print("\nStarting evaluation...")
    for idx, sample in enumerate(tqdm(dataset_samples, disable=show_verbose)):
        file_name = sample["file_name"]
        img_path = os.path.join(args.img_prefix, file_name)
        img_width = sample["width"]
        img_height = sample["height"]

        boxes = sample["boxes"]
        person_bbox = boxes[sample["person_box_idx"]]
        object_bbox = boxes[sample["object_box_idx"]]
        gt_action = sample["gt_action"]

        # Convert bboxes to 1000-format for prompt
        person_bbox_1000 = convert_bbox_to_1000(person_bbox, img_width, img_height)
        object_bbox_1000 = convert_bbox_to_1000(object_bbox, img_width, img_height)

        # Load proposals
        image_stem = os.path.splitext(file_name)[0]
        proposals = load_proposals(image_stem, args.proposals_dir)
        if not proposals:
            missing_proposals += 1

        if show_verbose:
            print(f"\n[{idx+1}/{len(dataset_samples)}] {file_name}")
            print(f"  GT action: {gt_action}")
            print(f"  Proposals: {len(proposals)}")

        image = Image.open(img_path).convert("RGB")

        user_prompt = build_referring_prompt(person_bbox_1000, object_bbox_1000, proposals)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_to_base64(image)}},
                    {"type": "text", "text": user_prompt},
                ]
            }
        ]

        answer_text, tool_calls, thinking, zoom_crops = run_sft_agent_loop(
            client, args.model_name, messages, image, max_turns=args.max_turns
        )

        predicted_action = clean_action_response(answer_text or "")

        if show_verbose:
            print(f"  Tool calls: {len(tool_calls)}")
            print(f"  Predicted: {predicted_action}")

        predictions.append({"image_id": idx, "caption": predicted_action})

        exact_match = predicted_action.lower().strip() == gt_action.lower().strip()
        triplet_result = {
            "triplet_id": idx,
            "file_name": file_name,
            "ground_truth": gt_action,
            "prediction": predicted_action,
            "tool_calls": tool_calls,
            "exact_match": exact_match,
        }
        if thinking:
            triplet_result["thinking_content"] = thinking
            thinking_records.append({
                "file_name": file_name,
                "gt_action": gt_action,
                "prediction": predicted_action,
                "thinking": thinking,
            })

        per_triplet_results.append(triplet_result)
        action_stats[gt_action]["total"] += 1
        action_stats[gt_action]["exact_match"] += (1 if exact_match else 0)

        # Visualization (verbose mode only)
        if viz_dir is not None:
            base_name = os.path.splitext(file_name)[0]
            prefix = f"{idx:05d}_{base_name}"
            try:
                viz_path = os.path.join(viz_dir, f"{prefix}_viz.jpg")
                visualize_action_triplet(
                    img_path, person_bbox, object_bbox,
                    predicted_action, gt_action,
                    sample.get("object_category", "object"),
                    viz_path
                )
                if show_verbose:
                    print(f"  Visualization: {prefix}_viz.jpg")
            except Exception as e:
                if show_verbose:
                    print(f"  Visualization failed: {e}")
            # Save zoom-in crops
            for crop_turn, crop_bbox, crop_img in zoom_crops:
                try:
                    crop_path = os.path.join(viz_dir, f"{prefix}_turn{crop_turn}_zoomin.jpg")
                    crop_img.save(crop_path, quality=90)
                    if show_verbose:
                        print(f"  Zoom crop turn{crop_turn}: bbox={crop_bbox}")
                except Exception as e:
                    if show_verbose:
                        print(f"  Zoom crop save failed (turn {crop_turn}): {e}")

    # COCO evaluation
    print("\n" + "=" * 80)
    print("Computing METEOR and CIDEr metrics...")

    images_info = [{"id": i} for i in range(len(dataset_samples))]
    annotations = [
        {"image_id": i, "caption": dataset_samples[i]["gt_action"], "id": i}
        for i in range(len(dataset_samples))
    ]
    gt_coco_format = {
        "info": {"description": "HICO-DET Action Referring SFT Ground Truth"},
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
        pred_file_tmp = f.name

    coco = COCO(gt_file)
    coco_result = coco.loadRes(pred_file_tmp)
    coco_eval = COCOEvalCap(coco, coco_result)
    try:
        coco_eval.evaluate()
    except Exception as e:
        # SPICE metric fails on ARM64 (AMD64-only Java library). BLEU/METEOR/ROUGE_L/CIDEr
        # are computed and stored in coco_eval.eval before SPICE runs, so they are valid.
        print(f"\n⚠️  SPICE metric failed (likely ARM64 incompatibility): {type(e).__name__}")
        print("   Continuing with BLEU, METEOR, ROUGE_L, CIDEr metrics.")
    os.unlink(gt_file)
    os.unlink(pred_file_tmp)

    metrics = {}
    print("\n" + "=" * 80)
    print("HICO-DET Action Referring Results (SFT)")
    print("=" * 80)
    for metric, score in coco_eval.eval.items():
        metrics[metric] = score
        print(f"{metric:<15} {score*100:>9.2f}%")

    total = len(per_triplet_results)
    exact_matches = sum(1 for r in per_triplet_results if r["exact_match"])
    exact_match_acc = exact_matches / total if total > 0 else 0.0
    metrics["exact_match"] = exact_match_acc
    metrics["missing_proposals"] = missing_proposals
    print(f"{'Exact Match':<15} {exact_match_acc*100:>9.2f}%")
    print(f"\nMissing proposals: {missing_proposals}/{len(dataset_samples)}")

    # Save results
    os.makedirs(os.path.dirname(os.path.abspath(args.pred_file)), exist_ok=True)

    print(f"\nSaving predictions to: {args.pred_file}")
    with open(args.pred_file, "w") as f:
        json.dump(predictions, f, indent=2)

    metrics_file = args.pred_file.replace(".json", "_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    if args.verbose:
        per_triplet_file = args.pred_file.replace(".json", "_per_triplet.json")
        with open(per_triplet_file, "w") as f:
            json.dump(per_triplet_results, f, indent=2)

    if thinking_records:
        thinking_file = args.pred_file.replace(".json", "_thinking.jsonl")
        with open(thinking_file, "w") as f:
            for r in thinking_records:
                f.write(json.dumps(r) + "\n")
        print(f"Thinking saved: {thinking_file} ({len(thinking_records)} samples)")

    if use_wandb:
        wandb.log(metrics)
        wandb.finish()

    print("\nEvaluation complete!")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HICO-DET SFT Action Referring Evaluation")
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000")
    parser.add_argument("--model-name", type=str, default="qwen3VL-4B")
    parser.add_argument("--ann-file", type=str,
                        default="../dataset/benchmarks_simplified/hico_action_referring_test_simplified.json")
    parser.add_argument("--img-prefix", type=str,
                        default="../dataset/hico_20160224_det/images/test2015")
    parser.add_argument("--proposals-dir", type=str,
                        default="../../hoi-dataset-curation/output/test_proposals")
    parser.add_argument("--pred-file", type=str, required=True)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="hico-action-referring-sft")
    parser.add_argument("--wandb-run-name", type=str, default=None)

    args = parser.parse_args()
    eval_model(args)
