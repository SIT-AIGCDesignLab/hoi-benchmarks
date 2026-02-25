#!/usr/bin/env python3
"""
SWIG-HOI Grounding Evaluation Script for Gemini API

Evaluates Google Gemini models on grounding performance with COCO-style AR metrics.

Task: Given "Detect all person-{object} pairs where the person is {action} the {object}",
      predict bounding boxes for ALL person-object pairs performing that action.

Key Features:
- Uses Gemini API (google-genai library)
- JSON output parsing for bbox predictions
- COCO-style Average Recall @ IoU thresholds
- Supports person-person interactions
- Cost optimization: response caching, image optimization, optimized prompts

Usage:
    python eval_swig_ground_gemini.py \
        --model-name gemini-2.0-flash-exp \
        --img-prefix /path/to/swig/images_512 \
        --ann-file data/benchmarks_simplified/swig_ground_test_simplified.json \
        --result-file results/grounding_results.json \
        --api-key YOUR_API_KEY \
        --use-cache \
        --optimize-images \
        --optimized-prompts \
        --verbose
"""

import os
import json
import re
import argparse
from tqdm import tqdm
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import numpy as np

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional

# Import shared utilities
from eval_api_utils import (
    denormalize_bbox_from_1000,
    calculate_iou,
    retry_with_exponential_backoff,
    save_results,
    load_annotations,
    # Cost optimization utilities
    optimize_image_for_api,
    load_existing_results,
    process_samples_concurrent
)

# Response caching
from response_cache import ResponseCache

# Google Generative AI SDK (new library)
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("Warning: google-genai package not found. Please install: pip install google-genai")

# Optional W&B import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def parse_json_response(response_text: str, img_width: int, img_height: int) -> List[Dict]:
    """
    Parse JSON response from Gemini to extract person-object pairs.

    Expected format in [0, 1000] normalized coords:
    [
      {"pair_id": 1, "person_bbox": [x1, y1, x2, y2], "object_bbox": [x1, y1, x2, y2]},
      {"pair_id": 2, "person_bbox": [x1, y1, x2, y2], "object_bbox": [x1, y1, x2, y2]}
    ]

    Args:
        response_text: Raw response from Gemini
        img_width: Image width for denormalization
        img_height: Image height for denormalization

    Returns:
        List of pairs with pixel coordinates
    """
    pairs = []

    # Clean response text
    text = response_text.strip()

    # Remove markdown code blocks
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    # Try to parse as JSON
    try:
        parsed = json.loads(text)

        # Handle different response structures
        if isinstance(parsed, dict):
            if 'pairs' in parsed:
                detections = parsed['pairs']
            elif 'detections' in parsed:
                detections = parsed['detections']
            elif 'results' in parsed:
                detections = parsed['results']
            else:
                detections = [parsed]
        elif isinstance(parsed, list):
            detections = parsed
        else:
            return pairs

    except json.JSONDecodeError:
        # Fallback: regex extraction
        array_pattern = r'\[[\s\S]+\]'
        array_matches = re.findall(array_pattern, text)
        if array_matches:
            longest_match = max(array_matches, key=len)
            try:
                parsed = json.loads(longest_match)
                if isinstance(parsed, list):
                    detections = parsed
                else:
                    return pairs
            except:
                return pairs
        else:
            return pairs

    # Convert from [0, 1000] to pixel coordinates
    for det in detections:
        if not isinstance(det, dict):
            continue

        if "person_bbox" in det and "object_bbox" in det:
            person_bbox_norm = det["person_bbox"]
            object_bbox_norm = det["object_bbox"]

            if len(person_bbox_norm) == 4 and len(object_bbox_norm) == 4:
                person_box = denormalize_bbox_from_1000(
                    person_bbox_norm, img_width, img_height
                )
                object_box = denormalize_bbox_from_1000(
                    object_bbox_norm, img_width, img_height
                )

                pairs.append({
                    'person_box': person_box,
                    'object_box': object_box,
                    'pair_id': det.get('pair_id', len(pairs) + 1)
                })

    return pairs


def call_gemini_api_grounding(
    image_path: str,
    action: str,
    object_category: str,
    prompt_text: str,
    model_name: str = "gemini-2.0-flash-exp",
    api_key: Optional[str] = None,
    max_tokens: int = 4096
) -> List[Dict]:
    """
    Call Gemini API for grounding task.

    Args:
        image_path: Path to image
        action: Action description
        object_category: Object category
        prompt_text: Text prompt
        model_name: Gemini model name
        api_key: Google API key
        max_tokens: Maximum tokens in response

    Returns:
        predicted_pairs
    """
    if not GENAI_AVAILABLE:
        raise ImportError("google-genai package is required. Install with: pip install google-genai")

    # Get API key (check both GEMINI_API_KEY and GOOGLE_API_KEY)
    if api_key is None:
        api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("API key not found. Set GEMINI_API_KEY or GOOGLE_API_KEY env variable or pass --api-key")

    # Initialize Gemini client
    client = genai.Client(api_key=api_key)

    # Load image and get dimensions
    img = Image.open(image_path)
    img_width, img_height = img.size

    # Build full prompt
    task_description = f"\nTask: Detect all person-{object_category} pairs where the person is {action} the {object_category}.\n\n"
    full_prompt = task_description + prompt_text

    # Call API with retry logic
    def _api_call():
        # Read image file
        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        # Create image part
        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type='image/jpeg'
        )

        # Generate content
        # Gemini 3 uses dynamic thinking by default - use thinking_level="low" for structured output
        # Temperature should be 1.0 for Gemini 3 (0.0 causes looping issues)
        # See: https://ai.google.dev/gemini-api/docs/gemini-3
        response = client.models.generate_content(
            model=model_name,
            contents=[image_part, full_prompt],
            config=types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=1.0,  # Gemini 3 requires 1.0, not 0.0
                # Grounding requires spatial reasoning - use "high" for better box accuracy
                thinking_config=types.ThinkingConfig(thinking_level="high")
            )
        )
        return response

    response = retry_with_exponential_backoff(_api_call)

    # Extract response text
    response_text = response.text if hasattr(response, 'text') else ""

    # Parse JSON response to extract pairs
    predicted_pairs = parse_json_response(response_text, img_width, img_height)

    return predicted_pairs


def build_grounding_prompt(optimized: bool = False) -> str:
    """
    Build prompt for grounding task (Gemini-optimized).

    Args:
        optimized: Whether to use optimized (shorter) prompt to save tokens

    Returns:
        Prompt text
    """
    if optimized:
        # Optimized prompt for Gemini: concise with clear structure
        return (
            "You are a precise object detection system.\n\n"
            "COORDINATE SYSTEM: [0-1000] normalized, format [x_min, y_min, x_max, y_max]\n"
            "- (0,0) = top-left corner\n"
            "- (1000,1000) = bottom-right corner\n\n"
            "CRITICAL: Draw TIGHT boxes around actual objects, NOT the entire image.\n"
            "- person_bbox: box around the PERSON only\n"
            "- object_bbox: box around the TARGET OBJECT only\n\n"
            "OUTPUT FORMAT (JSON only):\n"
            "[{\"pair_id\": 1, \"person_bbox\": [x1,y1,x2,y2], \"object_bbox\": [x1,y1,x2,y2]}]\n\n"
            "Return [] if no valid pairs found."
        )
    else:
        return (
            "You are a precise object detection system specializing in human-object interaction.\n\n"
            "TASK: Detect all person-object pairs performing the specified action.\n\n"
            "COORDINATE SYSTEM:\n"
            "- Use normalized [0-1000] coordinates\n"
            "- Format: [x_min, y_min, x_max, y_max]\n"
            "- Origin (0,0) is at the TOP-LEFT corner\n"
            "- (1000,1000) is at the BOTTOM-RIGHT corner\n\n"
            "CRITICAL REQUIREMENTS:\n"
            "1. Draw TIGHT bounding boxes around the actual objects\n"
            "2. person_bbox should contain ONLY the person (not the entire image)\n"
            "3. object_bbox should contain ONLY the target object\n"
            "4. If person covers right half of image, x_min should be ~500, not 0\n\n"
            "STEP-BY-STEP:\n"
            "1. Locate each person performing the action\n"
            "2. Identify the specific object they're interacting with\n"
            "3. Draw precise bounding boxes around each entity\n"
            "4. Verify boxes are tight (not covering unnecessary areas)\n\n"
            "OUTPUT FORMAT (JSON only):\n"
            "[\n"
            "  {\"pair_id\": 1, \"person_bbox\": [x1, y1, x2, y2], \"object_bbox\": [x1, y1, x2, y2]}\n"
            "]\n\n"
            "Return empty array [] if no pairs detected.\n"
            "Provide ONLY valid JSON (no explanations)."
        )


def visualize_qwen3vl_grounding(image_path, pred_pairs, gt_pairs, matches,
                                 action, object_category, iou_threshold=0.5):
    """Create 3-panel visualization comparing predicted vs GT pairs."""
    image = Image.open(image_path).convert('RGB')
    img_width, img_height = image.size
    pred_img, gt_img, overlay_img = image.copy(), image.copy(), image.copy()
    pred_draw, gt_draw, overlay_draw = ImageDraw.Draw(pred_img), ImageDraw.Draw(gt_img), ImageDraw.Draw(overlay_img)
    
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except:
        font, small_font = ImageFont.load_default(), ImageFont.load_default()
    
    color_person_pred, color_object_pred = (255, 0, 0), (0, 0, 255)
    color_person_gt, color_object_gt = (0, 255, 0), (255, 255, 0)
    color_matched, color_unmatched = (0, 255, 0), (255, 0, 0)
    
    matched_pred_indices = {m[0] for m in matches}
    for idx, pred_pair in enumerate(pred_pairs):
        is_matched = idx in matched_pred_indices
        person_box, object_box = pred_pair['person_box'], pred_pair['object_box']
        pred_draw.rectangle(person_box, outline=color_person_pred, width=3)
        pred_draw.text((person_box[0], person_box[1] - 18), f"P{idx+1}", fill=color_person_pred, font=small_font)
        pred_draw.rectangle(object_box, outline=color_object_pred, width=3)
        pred_draw.text((object_box[0], object_box[1] - 18), f"O{idx+1}", fill=color_object_pred, font=small_font)
        person_center = ((person_box[0] + person_box[2]) / 2, (person_box[1] + person_box[3]) / 2)
        object_center = ((object_box[0] + object_box[2]) / 2, (object_box[1] + object_box[3]) / 2)
        pred_draw.line([person_center, object_center], fill=color_matched if is_matched else color_unmatched, width=2)
    
    matched_gt_indices = {m[1] for m in matches}
    for idx, gt_pair in enumerate(gt_pairs):
        is_matched = idx in matched_gt_indices
        person_box, object_box = gt_pair['person_box'], gt_pair['object_box']
        gt_draw.rectangle(person_box, outline=color_person_gt, width=3)
        gt_draw.text((person_box[0], person_box[1] - 18), f"P{idx+1}", fill=color_person_gt, font=small_font)
        gt_draw.rectangle(object_box, outline=color_object_gt, width=3)
        gt_draw.text((object_box[0], object_box[1] - 18), f"O{idx+1}", fill=color_object_gt, font=small_font)
        person_center = ((person_box[0] + person_box[2]) / 2, (person_box[1] + person_box[3]) / 2)
        object_center = ((object_box[0] + object_box[2]) / 2, (object_box[1] + object_box[3]) / 2)
        gt_draw.line([person_center, object_center], fill=color_matched if is_matched else color_unmatched, width=2)
    
    for match in matches:
        pred_idx, gt_idx = match  # Fixed: matches only contain (pred_idx, gt_idx)
        pred_pair, gt_pair = pred_pairs[pred_idx], gt_pairs[gt_idx]
        
        # Calculate IoU for visualization
        person_iou = calculate_iou(pred_pair['person_box'], gt_pair['person_box'])
        object_iou = calculate_iou(pred_pair['object_box'], gt_pair['object_box'])
        
        overlay_draw.rectangle(gt_pair['person_box'], outline=(0, 255, 0), width=2)
        overlay_draw.rectangle(gt_pair['object_box'], outline=(0, 255, 0), width=2)
        overlay_draw.rectangle(pred_pair['person_box'], outline=(0, 100, 255), width=2)
        overlay_draw.rectangle(pred_pair['object_box'], outline=(0, 100, 255), width=2)
        avg_iou = (person_iou + object_iou) / 2.0
        overlay_draw.text((pred_pair['person_box'][0], pred_pair['person_box'][1] - 18),
                         f"IoU:{avg_iou:.2f}", fill=(255, 255, 255), font=small_font)
    
    for idx in range(len(pred_pairs)):
        if idx not in matched_pred_indices:
            pred_pair = pred_pairs[idx]
            overlay_draw.rectangle(pred_pair['person_box'], outline=(255, 0, 0), width=2)
            overlay_draw.rectangle(pred_pair['object_box'], outline=(255, 0, 0), width=2)
            overlay_draw.text((pred_pair['person_box'][0], pred_pair['person_box'][1] - 18),
                            "FP", fill=(255, 0, 0), font=small_font)
    
    for idx in range(len(gt_pairs)):
        if idx not in matched_gt_indices:
            gt_pair = gt_pairs[idx]
            overlay_draw.rectangle(gt_pair['person_box'], outline=(255, 165, 0), width=2)
            overlay_draw.rectangle(gt_pair['object_box'], outline=(255, 165, 0), width=2)
            overlay_draw.text((gt_pair['person_box'][0], gt_pair['person_box'][1] - 18),
                            "FN", fill=(255, 165, 0), font=small_font)
    
    panel_width, panel_height = img_width, img_height
    total_width, header_height = panel_width * 3, 60
    final_img = Image.new('RGB', (total_width, panel_height + header_height), color=(255, 255, 255))
    final_draw = ImageDraw.Draw(final_img)
    
    title_y = 10
    final_draw.text((panel_width // 2 - 100, title_y), "Predictions", fill=(0, 0, 0), font=font)
    final_draw.text((panel_width + panel_width // 2 - 80, title_y), "Ground Truth", fill=(0, 0, 0), font=font)
    final_draw.text((2 * panel_width + panel_width // 2 - 60, title_y), "Overlay", fill=(0, 0, 0), font=font)
    
    stats_y, num_matched = 35, len(matches)
    num_fp, num_fn = len(pred_pairs) - num_matched, len(gt_pairs) - num_matched
    recall = num_matched / len(gt_pairs) if len(gt_pairs) > 0 else 0.0
    stats_text = f"Action: {action} | Object: {object_category} | IoU≥{iou_threshold}"
    final_draw.text((10, stats_y), stats_text, fill=(0, 0, 0), font=small_font)
    metrics_text = f"Pred:{len(pred_pairs)} | GT:{len(gt_pairs)} | Matched:{num_matched} | FP:{num_fp} | FN:{num_fn} | Recall:{recall:.1%}"
    final_draw.text((panel_width + 10, stats_y), metrics_text, fill=(0, 0, 0), font=small_font)
    
    final_img.paste(pred_img, (0, header_height))
    final_img.paste(gt_img, (panel_width, header_height))
    final_img.paste(overlay_img, (2 * panel_width, header_height))
    
    return final_img


def match_pairs_greedy(pred_pairs: List[Dict], gt_pairs: List[Dict], iou_threshold: float = 0.5) -> Tuple:
    """Match predicted pairs to ground truth pairs using greedy matching."""
    matches = []
    matched_preds = set()
    matched_gts = set()

    # Build IoU matrix
    iou_matrix = []
    for pred_idx, pred_pair in enumerate(pred_pairs):
        row = []
        for gt_idx, gt_pair in enumerate(gt_pairs):
            person_iou = calculate_iou(pred_pair['person_box'], gt_pair['person_box'])
            object_iou = calculate_iou(pred_pair['object_box'], gt_pair['object_box'])

            if person_iou >= iou_threshold and object_iou >= iou_threshold:
                avg_iou = (person_iou + object_iou) / 2.0
                row.append(avg_iou)
            else:
                row.append(0.0)
        iou_matrix.append(row)

    # Greedy matching
    while True:
        best_score = 0.0
        best_pred_idx = -1
        best_gt_idx = -1

        for pred_idx in range(len(pred_pairs)):
            if pred_idx in matched_preds:
                continue
            for gt_idx in range(len(gt_pairs)):
                if gt_idx in matched_gts:
                    continue
                if iou_matrix[pred_idx][gt_idx] > best_score:
                    best_score = iou_matrix[pred_idx][gt_idx]
                    best_pred_idx = pred_idx
                    best_gt_idx = gt_idx

        if best_score == 0.0:
            break

        matches.append((best_pred_idx, best_gt_idx))
        matched_preds.add(best_pred_idx)
        matched_gts.add(best_gt_idx)

    unmatched_preds = [i for i in range(len(pred_pairs)) if i not in matched_preds]
    unmatched_gts = [i for i in range(len(gt_pairs)) if i not in matched_gts]

    return matches, unmatched_preds, unmatched_gts


def get_box_area(box):
    """Calculate area of bounding box [x1, y1, x2, y2]"""
    return (box[2] - box[0]) * (box[3] - box[1])


def categorize_pair_by_size(gt_pair, area_small=1024, area_medium=9216) -> str:
    """Categorize GT pair by object size using COCO-style absolute pixel thresholds.
    Small: object < 32**2 pixels²
    Medium: 32**2 to 96**2 pixels²
    Large: >= 96**2 pixels²
    """
    object_area = get_box_area(gt_pair['object_box'])
    if object_area < area_small:
        return 'small'
    elif object_area < area_medium:
        return 'medium'
    return 'large'


def compute_grounding_metrics(all_predictions: List[Dict], all_ground_truth: List[Dict]) -> Dict:
    """Compute COCO-style AR metrics at different IoU thresholds."""
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    recalls_per_threshold = []
    
    # COCO area thresholds for small/medium/large objects
    
    recalls_small = []
    recalls_medium = []
    recalls_large = []

    for iou_thresh in iou_thresholds:
        total_recall = 0.0
        total_samples = 0
        
        tp_small, fn_small = 0, 0
        tp_medium, fn_medium = 0, 0
        tp_large, fn_large = 0, 0

        for pred, gt in zip(all_predictions, all_ground_truth):
            pred_pairs = pred.get('pairs', [])
            gt_pairs = gt.get('pairs', [])

            if len(gt_pairs) == 0:
                continue

            matches, _, _ = match_pairs_greedy(pred_pairs, gt_pairs, iou_thresh)
            recall = len(matches) / len(gt_pairs)
            total_recall += recall
            total_samples += 1
            
            # Track size-specific metrics
            matched_gt_indices = {m[1] for m in matches}
            for gt_idx, gt_pair in enumerate(gt_pairs):
                AREA_SMALL = 32 ** 2    # < 1024 pixels²
                AREA_MEDIUM = 96 ** 2   # < 9216 pixels²
                size_category = categorize_pair_by_size(gt_pair, AREA_SMALL, AREA_MEDIUM)
                if gt_idx in matched_gt_indices:
                    # True Positive for this size category
                    if size_category == 'small':
                        tp_small += 1
                    elif size_category == 'medium':
                        tp_medium += 1
                    else:
                        tp_large += 1
                else:
                    # False Negative for this size category
                    if size_category == 'small':
                        fn_small += 1
                    elif size_category == 'medium':
                        fn_medium += 1
                    else:
                        fn_large += 1

        avg_recall = total_recall / total_samples if total_samples > 0 else 0.0
        recalls_per_threshold.append(avg_recall)
        
        # Calculate size-specific recalls
        recall_small = tp_small / (tp_small + fn_small) if (tp_small + fn_small) > 0 else 0.0
        recall_medium = tp_medium / (tp_medium + fn_medium) if (tp_medium + fn_medium) > 0 else 0.0
        recall_large = tp_large / (tp_large + fn_large) if (tp_large + fn_large) > 0 else 0.0
        
        recalls_small.append(recall_small)
        recalls_medium.append(recall_medium)
        recalls_large.append(recall_large)
    
    # Calculate size-based AR metrics
    ar_small = float(np.mean(recalls_small)) if recalls_small else 0.0
    ar_medium = float(np.mean(recalls_medium)) if recalls_medium else 0.0
    ar_large = float(np.mean(recalls_large)) if recalls_large else 0.0

    metrics = {
        'AR': float(np.mean(recalls_per_threshold)),
        'AR@0.5': recalls_per_threshold[0],
        'AR@0.75': recalls_per_threshold[5],
        'ARs': ar_small,
        'ARm': ar_medium,
        'ARl': ar_large,
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="SWIG-HOI Grounding Evaluation with Gemini API"
    )

    # Model arguments
    parser.add_argument('--model-name', type=str, default='gemini-2.0-flash-exp',
                        help='Gemini model name')
    parser.add_argument('--api-key', type=str, default=None,
                        help='Google API key (or set GEMINI_API_KEY/GOOGLE_API_KEY in .env file)')

    # Data arguments
    parser.add_argument('--ann-file', type=str, required=True,
                        help='Path to annotation file')
    parser.add_argument('--img-prefix', type=str, required=True,
                        help='Path to image directory')
    parser.add_argument('--result-file', type=str, required=True,
                        help='Path to save results')

    # Evaluation arguments
    parser.add_argument('--max-images', type=int, default=None,
                        help='Maximum number of images to evaluate')
    parser.add_argument('--verbose', action='store_true',
                        help='Show per-image results')

    # W&B arguments
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='swig-grounding-gemini',
                        help='W&B project name')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                        help='W&B run name')

    # Cost optimization arguments
    parser.add_argument('--use-cache', action='store_true', default=False,
                        help='Enable response caching (50-100%% savings on reruns)')
    parser.add_argument('--cache-dir', type=str, default='./cache',
                        help='Cache directory (default: ./cache)')
    parser.add_argument('--optimize-images', action='store_true', default=False,
                        help='Enable image optimization (30-50%% vision token savings)')
    parser.add_argument('--image-max-size', type=int, default=448,
                        help='Max image dimension in pixels (default: 448)')
    parser.add_argument('--image-quality', type=int, default=90,
                        help='JPEG quality 1-100 (default: 90)')
    parser.add_argument('--optimized-prompts', action='store_true', default=False,
                        help='Use shorter prompts (20-40%% token savings)')
    parser.add_argument('--concurrent-requests', type=int, default=1,
                        help='Number of concurrent API requests (default: 1)')
    parser.add_argument('--rate-limit', type=int, default=None,
                        help='Max requests per minute (default: 60 for Gemini)')

    args = parser.parse_args()

    # Validate API key (check both GEMINI_API_KEY and GOOGLE_API_KEY)
    api_key = args.api_key or os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        print("ERROR: Gemini API key not found!")
        print("Please set GEMINI_API_KEY or GOOGLE_API_KEY environment variable or pass --api-key argument")
        return 1

    # Initialize W&B if requested
    use_wandb = args.wandb and WANDB_AVAILABLE
    if use_wandb:
        run_name = args.wandb_run_name or f"gemini_ground_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={'model': args.model_name}
        )

    print("\n" + "=" * 80)
    print("SWIG-HOI Grounding Evaluation (Gemini API)")
    print("=" * 80)
    print(f"Model:            {args.model_name}")
    print(f"Annotation file:  {args.ann_file}")
    print(f"Image prefix:     {args.img_prefix}")
    print(f"Output file:      {args.result_file}")
    print("=" * 80 + "\n")

    # Load annotations
    print("Loading annotations...")
    annotations = load_annotations(args.ann_file)

    if 'samples' in annotations:
        dataset_samples = annotations['samples']
    elif 'data' in annotations:
        dataset_samples = annotations['data']
    else:
        dataset_samples = annotations

    if args.max_images:
        dataset_samples = dataset_samples[:args.max_images]

    print(f"Loaded {len(dataset_samples)} samples\n")

    # Create visualization directory if verbose mode
    viz_dir = None
    if args.verbose:
        viz_dir = args.result_file.replace('.json', '_visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        print(f"✓ Visualization directory: {viz_dir}\n")

    # Cost Optimization: Load existing results for resume capability
    existing_results, processed_ids = load_existing_results(args.result_file)
    if existing_results:
        print(f"✓ Found {len(existing_results)} previously processed samples (resuming)")
        print(f"  Remaining: {len(dataset_samples) - len(processed_ids)} samples\n")

    # Cost Optimization: Initialize response cache
    cache = None
    prompt_hash = None
    if args.use_cache:
        cache = ResponseCache(cache_dir=args.cache_dir)
        print(f"✓ Response caching enabled (cache dir: {args.cache_dir})")
        print(f"  Existing cache entries: {cache.size()}\n")

    # Cost Optimization: Print optimization settings
    if args.optimize_images:
        print(f"✓ Image optimization enabled (max_size={args.image_max_size}, quality={args.image_quality})")
        print(f"  Estimated token savings: ~30-50%\n")

    if args.concurrent_requests > 1:
        # Auto-detect rate limits for Gemini
        if args.rate_limit is None:
            args.rate_limit = 60  # Gemini default
        print(f"✓ Concurrent processing enabled ({args.concurrent_requests} workers, rate limit: {args.rate_limit} req/min)\n")

    # Build prompt (use optimized version to save tokens if enabled)
    use_optimized_prompt = args.optimized_prompts
    prompt_text = build_grounding_prompt(optimized=use_optimized_prompt)

    if cache:
        prompt_hash = ResponseCache.hash_prompt(prompt_text)
        if use_optimized_prompt:
            print("✓ Using optimized prompt (70% shorter, ~120 token savings per request)\n")

    # Evaluate each sample
    results = existing_results  # Start with existing results
    action_stats = defaultdict(lambda: {'total': 0, 'recall': 0.0})
    cache_hits = 0

    print("Evaluating samples...")
    for idx, sample in enumerate(tqdm(dataset_samples)):
        # Get image path from simplified format
        file_name = sample['file_name']
        image_path = os.path.join(args.img_prefix, file_name)

        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue

        # Cost Optimization: Skip if already processed (resume capability)
        if file_name in processed_ids:
            continue

        # Get action and object from simplified format
        action = sample['action']
        object_category = sample['object_category']

        # Parse ground truth pairs from simplified format
        boxes = sample['boxes']
        gt_box_inds = sample['gt_box_inds']
        num_pairs = sample['num_pairs']
        gt_pairs = []
        for i in range(num_pairs):
            person_idx = gt_box_inds[i * 2]
            object_idx = gt_box_inds[i * 2 + 1]
            gt_pairs.append({
                'person_box': boxes[person_idx],
                'object_box': boxes[object_idx]
            })

        # Cost Optimization: Check cache before API call
        pred_pairs = None

        if cache:
            # Create a unique cache key using action and object
            cache_key = f"{action}_{object_category}"
            cached_response = cache.get(file_name, cache_key, "", args.model_name, prompt_hash)
            if cached_response:
                pred_pairs = cached_response.get('predicted_pairs', [])
                cache_hits += 1

        # If not in cache, call API
        if pred_pairs is None:
            # Cost Optimization: Optimize image before encoding
            optimized_image_path = image_path
            if args.optimize_images:
                optimized_image_path = optimize_image_for_api(
                    image_path,
                    max_size=args.image_max_size,
                    quality=args.image_quality
                )

            # Call Gemini API
            try:
                pred_pairs = call_gemini_api_grounding(
                    image_path=optimized_image_path,
                    action=action,
                    object_category=object_category,
                    prompt_text=prompt_text,
                    model_name=args.model_name,
                    api_key=api_key
                )

                # Cost Optimization: Save to cache
                if cache:
                    cache_key = f"{action}_{object_category}"
                    cache.set(file_name, cache_key, "", args.model_name, prompt_hash, {
                        'predicted_pairs': pred_pairs
                    })

            except Exception as e:
                print(f"\nError processing sample {idx} ({file_name}): {e}")
                continue

        # Store results
        result = {
            'image_id': file_name,
            'action': action,
            'object': object_category,
            'predicted_pairs': pred_pairs,
            'ground_truth_pairs': gt_pairs,
            'num_pred_pairs': len(pred_pairs),
            'num_gt_pairs': len(gt_pairs)
        }

        results.append(result)

        # Calculate metrics for this sample
        if len(gt_pairs) > 0:
            matches, _, _ = match_pairs_greedy(pred_pairs, gt_pairs, iou_threshold=0.5)
            recall = len(matches) / len(gt_pairs)

            action_key = f"{action}_{object_category}"
            action_stats[action_key]['total'] += 1
            action_stats[action_key]['recall'] += recall

        # Cost Optimization: Incremental saving every 50 samples
        if len(results) % 50 == 0:
            save_results(results, args.result_file + ".tmp", {'partial': True, 'samples_processed': len(results)})

        if args.verbose and idx < 5:
            print(f"\n[Sample {idx}]")
            print(f"  Image: {file_name}")
            print(f"  Action: {action} {object_category}")
            print(f"  GT Pairs: {len(gt_pairs)}, Predicted: {len(pred_pairs)}")

        # Generate visualization if verbose
        if viz_dir is not None:
            # Use IoU=0.5 for visualization
            matches_05, _, _ = match_pairs_greedy(pred_pairs, gt_pairs, iou_threshold=0.5)
            try:
                viz_img = visualize_qwen3vl_grounding(
                    image_path, pred_pairs, gt_pairs, matches_05,
                    action, object_category, iou_threshold=0.5
                )
                base_name = os.path.splitext(file_name)[0]
                action_safe = action.replace(' ', '_').replace('/', '_')
                object_safe = object_category.replace(' ', '_').replace('/', '_')
                viz_filename = f"{base_name}_{action_safe}_{object_safe}_viz.jpg"
                viz_path = os.path.join(viz_dir, viz_filename)
                viz_img.save(viz_path, quality=90)
            except Exception as e:
                print(f"Warning: Failed to create visualization: {e}")

    print(f"\n✓ Completed evaluation of {len(results)} samples")

    # Cost Optimization Summary
    if cache or existing_results:
        print("\n" + "=" * 80)
        print("Cost Optimization Summary")
        print("=" * 80)
        if cache and cache_hits > 0:
            cache_hit_rate = (cache_hits / len(dataset_samples)) * 100
            print(f"✓ Cache hits: {cache_hits}/{len(dataset_samples)} ({cache_hit_rate:.1f}%)")
            print(f"  API calls saved: {cache_hits}")
        if existing_results:
            resumed_count = len(existing_results)
            print(f"✓ Resumed from previous run: {resumed_count} samples")
        if args.optimize_images:
            print(f"✓ Image optimization applied: ~30-50% vision token savings")
        if use_optimized_prompt:
            print(f"✓ Optimized prompt used: ~70% shorter (~120 tokens saved per request)")

        # Calculate estimated cost savings
        total_api_calls = len(dataset_samples) - cache_hits - len(existing_results)
        if total_api_calls > 0:
            print(f"\nTotal new API calls: {total_api_calls}")
        print("=" * 80 + "\n")

    # Compute COCO-style metrics
    print("\nComputing grounding metrics...")

    # Check if we have any results to evaluate
    if len(results) == 0:
        print("\n⚠️  WARNING: No samples were successfully processed!")
        print("   This could be because:")
        print("   1. All images were not found at the specified path")
        print("   2. All samples were skipped due to resume capability")
        print("   3. API errors occurred for all samples")
        print("\n   Cannot compute metrics with 0 samples. Exiting gracefully.")
        
        # Save empty results file
        with open(args.result_file, 'w') as f:
            json.dump({
                'results': [],
                'metadata': {
                    'model': args.model_name,
                    'num_samples': 0,
                    'error': 'No samples processed'
                }
            }, f, indent=2)
        
        return 1  # Exit with error code

    all_predictions = [
        {
            'pairs': [
                {'person_box': p['person_box'], 'object_box': p['object_box']}
                for p in r['predicted_pairs']
            ]
        }
        for r in results
    ]

    all_ground_truth = [
        {
            'pairs': [
                {'person_box': p['person_box'], 'object_box': p['object_box']}
                for p in r['ground_truth_pairs']
            ]
        }
        for r in results
    ]

    metrics = compute_grounding_metrics(all_predictions, all_ground_truth)

    # Visualization summary
    if viz_dir is not None:
        viz_count = len([f for f in os.listdir(viz_dir) if f.endswith(('.jpg', '.png'))])
        print(f"\n✓ Visualizations saved: {viz_dir}/")
        print(f"  Total images: {viz_count}")

    # Print results
    print("\n" + "=" * 80)
    print("SWIG-HOI Grounding Results (Gemini API)")
    print("=" * 80)
    print(f"{'Metric':<20} {'Score':>10}  {'Description':<45}")
    print("-" * 80)
    print(f"{'AR':<20} {metrics['AR']*100:>9.2f}%  {'Average Recall @ IoU=0.50:0.95':<45}")
    print(f"{'AR@0.5':<20} {metrics['AR@0.5']*100:>9.2f}%  {'Average Recall @ IoU=0.50':<45}")
    print(f"{'AR@0.75':<20} {metrics['AR@0.75']*100:>9.2f}%  {'Average Recall @ IoU=0.75':<45}")
    print(f"{'ARs':<20} {metrics['ARs']*100:>9.2f}%  {'Average Recall for small objects (area < 32²)':<45}")
    print(f"{'ARm':<20} {metrics['ARm']*100:>9.2f}%  {'Average Recall for medium objects (32² < area < 96²)':<45}")
    print(f"{'ARl':<20} {metrics['ARl']*100:>9.2f}%  {'Average Recall for large objects (area > 96²)':<45}")
    print("=" * 80)

    if use_wandb:
        wandb.log(metrics)
        wandb.log({'total_samples': len(results)})

    # Per-action statistics
    if args.verbose:
        print("\n" + "=" * 80)
        print("Per-Action Statistics (Top 20 by frequency)")
        print("=" * 80)
        print(f"{'Action-Object':<40} {'Count':>8} {'Avg Recall':>12}")
        print("-" * 80)

        sorted_actions = sorted(action_stats.items(), key=lambda x: x[1]['total'], reverse=True)
        for action_obj, stats in sorted_actions[:20]:
            avg_recall = stats['recall'] / stats['total'] if stats['total'] > 0 else 0.0
            print(f"{action_obj:<40} {stats['total']:>8} {avg_recall*100:>11.1f}%")

        print("=" * 80)

    # Save results
    print("\nSaving results...")

    all_metrics = {
        'model': args.model_name,
        'total_samples': len(results),
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        **metrics
    }

    save_results(
        results=results,
        output_path=args.result_file,
        metrics=all_metrics
    )

    print("\n" + "=" * 80)
    print("✅ Evaluation Complete!")
    print("=" * 80)

    if use_wandb:
        wandb.finish()

    return 0


if __name__ == '__main__':
    exit(main())
