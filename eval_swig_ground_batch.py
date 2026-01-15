#!/usr/bin/env python3
"""
SWIG-HOI Grounding Batch Evaluation Script

Unified batch evaluation script supporting Claude, Gemini, and OpenAI Batch APIs.
Offers 50% cost savings and avoids rate limit issues.

Task: Given "Detect all person-{object} pairs where the person is {action} the {object}",
      predict bounding boxes for ALL person-object pairs performing that action.

Usage:
    python eval_swig_ground_batch.py \
        --provider claude \
        --model claude-sonnet-4-5-20250514 \
        --ann-file data/benchmarks_simplified/swig_ground_test_simplified.json \
        --img-prefix data/swig_hoi/images_512 \
        --output-dir results/swig_ground_claude_batch \
        --resume
"""

import os
import sys
import json
import re
import argparse
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Optional W&B import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Import batch utilities
from batch_api_utils import (
    BatchProcessor,
    BatchJob,
    BatchStatus,
    wait_for_batch_completion,
    sanitize_custom_id
)
from checkpoint_manager import (
    CheckpointManager,
    EvaluationCheckpoint
)

# Import shared evaluation utilities
from eval_api_utils import (
    load_annotations,
    save_results,
    calculate_iou,
    denormalize_bbox_from_1000
)


def visualize_grounding_result(image_path, pred_pairs, gt_pairs, action, object_category, output_path):
    """
    Visualize grounding result with predicted and ground truth bounding boxes.
    
    Args:
        image_path: Path to image
        pred_pairs: List of predicted pairs with person_bbox and object_bbox
        gt_pairs: List of ground truth pairs
        action: Action being detected
        object_category: Object category
        output_path: Output path for visualization
    """
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except:
            font = ImageFont.load_default()
            font_small = ImageFont.load_default()
    
    # Draw ground truth pairs (green)
    for idx, gt_pair in enumerate(gt_pairs):
        person_bbox = gt_pair.get('person_bbox', [0, 0, 0, 0])
        object_bbox = gt_pair.get('object_bbox', [0, 0, 0, 0])
        draw.rectangle(person_bbox, outline="green", width=3)
        draw.rectangle(object_bbox, outline="green", width=3)
        draw.text((person_bbox[0], max(0, person_bbox[1] - 18)), f"GT-P{idx+1}", fill="green", font=font_small)
        draw.text((object_bbox[0], max(0, object_bbox[1] - 18)), f"GT-O{idx+1}", fill="green", font=font_small)
    
    # Draw predicted pairs (red for person, blue for object)
    for idx, pred_pair in enumerate(pred_pairs):
        person_bbox = pred_pair.get('person_bbox', [0, 0, 0, 0])
        object_bbox = pred_pair.get('object_bbox', [0, 0, 0, 0])
        draw.rectangle(person_bbox, outline="red", width=2)
        draw.rectangle(object_bbox, outline="blue", width=2)
        draw.text((person_bbox[0], max(0, person_bbox[1] - 35)), f"P{idx+1}", fill="red", font=font_small)
        draw.text((object_bbox[0], max(0, object_bbox[1] - 35)), f"O{idx+1}", fill="blue", font=font_small)
    
    # Add header text
    header = f"Action: {action} {object_category} | GT:{len(gt_pairs)} Pred:{len(pred_pairs)}"
    draw.text((10, 10), header, fill="white", font=font)
    
    # Legend
    draw.text((10, 35), "Green=GT, Red=Pred-Person, Blue=Pred-Object", fill="white", font=font_small)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)


def build_grounding_prompt(provider: str, optimized: bool = True) -> str:
    """
    Build prompt for grounding task - provider-specific to match real-time scripts.
    
    Args:
        provider: API provider ('claude', 'gemini', 'openai')
        optimized: Use optimized (shorter) prompt
        
    Returns:
        Prompt text
    """
    if provider == 'claude':
        if optimized:
            return (
                "<role>Expert object detection system</role>\n\n"
                "<coordinate_system>[0-1000] normalized, format: [x_min, y_min, x_max, y_max], origin at top-left</coordinate_system>\n\n"
                "<critical>Draw TIGHT boxes around actual objects only, NOT the entire image.</critical>\n\n"
                "<output_format>\n"
                "[{\"pair_id\": 1, \"person_bbox\": [x1,y1,x2,y2], \"object_bbox\": [x1,y1,x2,y2]}]\n"
                "</output_format>\n\n"
                "Return [] if no valid pairs. Output JSON only."
            )
        else:
            return (
                "<role>You are an expert computer vision analyst specializing in precise human-object interaction detection.</role>\n\n"
                "<task>Detect all person-object pairs performing the specified action. Return PRECISE bounding boxes.</task>\n\n"
                "<coordinate_system>\n"
                "- Normalized coordinates: [0-1000] range\n"
                "- Format: [x_min, y_min, x_max, y_max]\n"
                "- Origin (0,0) at TOP-LEFT, (1000,1000) at BOTTOM-RIGHT\n"
                "</coordinate_system>\n\n"
                "<precision_requirements>\n"
                "1. Boxes must be TIGHT around actual objects (not loose approximations)\n"
                "2. person_bbox: Should cover ONLY the person's body, not background\n"
                "3. object_bbox: Should cover ONLY the target object, sized to match it\n"
                "4. WRONG: [0, 0, 1000, 1000] for person (covers entire image)\n"
                "5. CORRECT: [350, 100, 650, 800] for a person in center-right (precise bounds)\n"
                "</precision_requirements>\n\n"
                "<example>\n"
                "For 'person riding bicycle' where person is in right half, bicycle in center:\n"
                "[{\"pair_id\": 1, \"person_bbox\": [450, 50, 750, 850], \"object_bbox\": [300, 400, 600, 700]}]\n"
                "</example>\n\n"
                "<output_format>\n"
                "[{\"pair_id\": 1, \"person_bbox\": [x1, y1, x2, y2], \"object_bbox\": [x1, y1, x2, y2]}]\n"
                "</output_format>\n\n"
                "Return [] if no pairs. Output ONLY valid JSON."
            )
    
    elif provider == 'gemini':
        if optimized:
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
    
    else:  # openai
        if optimized:
            return (
                "You are an object detection assistant.\n\n"
                "Task: Find person-object pairs performing the specified action.\n\n"
                "Coordinates: [0-1000] normalized, format [x_min, y_min, x_max, y_max]\n"
                "- (0,0) = top-left, (1000,1000) = bottom-right\n\n"
                "IMPORTANT: Draw TIGHT boxes around actual objects, NOT the entire image.\n\n"
                "Example:\n"
                "[{\"pair_id\": 1, \"person_bbox\": [100, 150, 400, 800], \"object_bbox\": [300, 200, 450, 350]}]\n\n"
                "Return [] if none found. Output JSON only."
            )
        else:
            return (
                "You are a PRECISE object detection assistant. Your boxes must be accurate.\n\n"
                "Task: Locate person-object pairs performing the specified action with PRECISE bounding boxes.\n\n"
                "Coordinate System:\n"
                "- Normalized [0-1000] coordinates\n"
                "- Format: [x_min, y_min, x_max, y_max]\n"
                "- (0,0) = top-left, (1000,1000) = bottom-right\n\n"
                "PRECISION Requirements (IoU >= 0.75 needed):\n"
                "- Boxes must be TIGHT - not loose approximations\n"
                "- person_bbox: Cover ONLY the person's body (shoulders to feet)\n"
                "- object_bbox: Cover ONLY the object being interacted with\n"
                "- WRONG: [0, 0, 1000, 1000] - this is the entire image!\n"
                "- CORRECT: [350, 80, 650, 900] - precise bounds around person\n\n"
                "Example - person in right half riding bicycle in center:\n"
                "[{\"pair_id\": 1, \"person_bbox\": [450, 50, 750, 850], \"object_bbox\": [300, 400, 600, 700]}]\n\n"
                "Return [] if no pairs. Output ONLY JSON."
            )


def parse_json_response(response_text: str, img_width: int, img_height: int) -> List[Dict]:
    """
    Parse JSON response to extract person-object pairs.

    Args:
        response_text: Raw response from model
        img_width: Image width for denormalization
        img_height: Image height for denormalization

    Returns:
        List of pairs with pixel coordinates
    """
    pairs = []
    text = response_text.strip()

    # Remove markdown code blocks
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    # Try to parse as JSON
    try:
        parsed = json.loads(text)

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

        person_bbox = det.get('person_bbox')
        object_bbox = det.get('object_bbox')

        if person_bbox is None or object_bbox is None:
            continue

        try:
            person_bbox = [float(x) for x in person_bbox]
            object_bbox = [float(x) for x in object_bbox]

            # Denormalize from [0, 1000] to pixel coordinates
            person_bbox_px = denormalize_bbox_from_1000(person_bbox, img_width, img_height)
            object_bbox_px = denormalize_bbox_from_1000(object_bbox, img_width, img_height)

            pairs.append({
                'person_bbox': person_bbox_px,
                'object_bbox': object_bbox_px
            })
        except (ValueError, TypeError):
            continue

    return pairs


def prepare_batch_requests(
    samples: List[Dict],
    img_prefix: str,
    processor: BatchProcessor,
    model: str,
    base_prompt: str,
    processed_ids: set = None,
    max_samples: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Prepare batch requests from samples.
    """
    if processed_ids is None:
        processed_ids = set()

    requests = []
    skipped = 0

    for idx, sample in enumerate(tqdm(samples, desc="Preparing requests")):
        if max_samples and len(requests) >= max_samples:
            break

        file_name = sample['file_name']
        action = sample.get('action', sample.get('gt_action', 'unknown'))
        object_category = sample.get('object_category', 'object')
        
        custom_id = sanitize_custom_id(f"{idx}_{file_name}_{action}")

        # Skip already processed
        if custom_id in processed_ids:
            skipped += 1
            continue

        image_path = os.path.join(img_prefix, file_name)
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue

        # Build task-specific prompt
        task_prompt = f"Task: Detect all person-{object_category} pairs where the person is {action} the {object_category}.\n\n"
        full_prompt = task_prompt + base_prompt

        # Prepare request (no bounding boxes to draw for grounding task)
        request = processor.prepare_request(
            custom_id=custom_id,
            image_path=image_path,
            prompt=full_prompt,
            model=model,
            max_tokens=1024
        )

        # Extract ground truth pairs from boxes and gt_box_inds
        boxes = sample.get('boxes', [])
        gt_box_inds = sample.get('gt_box_inds', [])
        gt_pairs = []
        if len(boxes) >= 2 and len(gt_box_inds) >= 2:
            person_bbox = boxes[gt_box_inds[0]]
            object_bbox = boxes[gt_box_inds[1]]
            gt_pairs = [{"person_bbox": person_bbox, "object_bbox": object_bbox}]
        elif sample.get('gt_pairs'):
            gt_pairs = sample.get('gt_pairs')
        elif sample.get('pairs'):
            gt_pairs = sample.get('pairs')
        
        # Store metadata for result matching
        request["_metadata"] = {
            "sample_idx": idx,
            "file_name": file_name,
            "action": action,
            "object_category": object_category,
            "gt_pairs": gt_pairs,
            "width": sample.get('width', 512),
            "height": sample.get('height', 512)
        }

        requests.append(request)

    if skipped > 0:
        print(f"Skipped {skipped} already processed samples")

    return requests


def get_box_area(box: List[float]) -> float:
    """Calculate box area from [x1, y1, x2, y2] format."""
    return (box[2] - box[0]) * (box[3] - box[1])


def categorize_pair_by_size(gt_pair: Dict, area_small: float = 1024, area_medium: float = 9216) -> str:
    """
    Categorize a ground truth pair by object size.
    Uses the object box area for categorization (COCO standard).
    - small: area < 32² = 1024
    - medium: 32² <= area < 96² = 9216
    - large: area >= 96²
    """
    object_area = get_box_area(gt_pair['object_bbox'])
    
    if object_area < area_small:
        return 'small'
    elif object_area < area_medium:
        return 'medium'
    else:
        return 'large'


def compute_grounding_metrics(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_thresholds: List[float] = [0.5, 0.75]
) -> Dict[str, float]:
    """
    Compute COCO-style Average Recall metrics including size-based AR.
    """
    metrics = {}
    
    # COCO area thresholds for small/medium/large objects
    AREA_SMALL = 32 ** 2    # < 1024 pixels²
    AREA_MEDIUM = 96 ** 2   # < 9216 pixels²

    # Compute AR at specific thresholds
    for iou_thresh in iou_thresholds:
        total_recall = 0.0
        total_samples = 0

        for pred, gt in zip(predictions, ground_truths):
            pred_pairs = pred.get('pairs', [])
            gt_pairs = gt.get('pairs', [])

            if len(gt_pairs) == 0:
                continue

            matched_gt = set()
            for pred_pair in pred_pairs:
                pred_person = pred_pair['person_bbox']
                pred_object = pred_pair['object_bbox']

                best_match_idx = -1
                best_match_score = 0.0

                for gt_idx, gt_pair in enumerate(gt_pairs):
                    if gt_idx in matched_gt:
                        continue

                    gt_person = gt_pair['person_bbox']
                    gt_object = gt_pair['object_bbox']

                    person_iou = calculate_iou(pred_person, gt_person)
                    object_iou = calculate_iou(pred_object, gt_object)
                    pair_score = (person_iou + object_iou) / 2.0

                    if pair_score >= iou_thresh and pair_score > best_match_score:
                        best_match_score = pair_score
                        best_match_idx = gt_idx

                if best_match_idx >= 0:
                    matched_gt.add(best_match_idx)

            recall = len(matched_gt) / len(gt_pairs)
            total_recall += recall
            total_samples += 1

        avg_recall = total_recall / total_samples if total_samples > 0 else 0.0
        metrics[f'AR@{iou_thresh}'] = avg_recall

    # Compute AR over range [0.5, 0.95] with size-based tracking
    iou_range = np.arange(0.5, 1.0, 0.05)
    ar_scores = []
    
    # Size-based recall tracking
    recalls_small = []
    recalls_medium = []
    recalls_large = []

    for iou_thresh in iou_range:
        total_recall = 0.0
        total_samples = 0
        
        # Size-specific counters for this threshold
        tp_small, fn_small = 0, 0
        tp_medium, fn_medium = 0, 0
        tp_large, fn_large = 0, 0

        for pred, gt in zip(predictions, ground_truths):
            pred_pairs = pred.get('pairs', [])
            gt_pairs = gt.get('pairs', [])

            if len(gt_pairs) == 0:
                continue

            matched_gt = set()
            for pred_pair in pred_pairs:
                pred_person = pred_pair['person_bbox']
                pred_object = pred_pair['object_bbox']

                best_match_idx = -1
                best_match_score = 0.0

                for gt_idx, gt_pair in enumerate(gt_pairs):
                    if gt_idx in matched_gt:
                        continue

                    gt_person = gt_pair['person_bbox']
                    gt_object = gt_pair['object_bbox']

                    person_iou = calculate_iou(pred_person, gt_person)
                    object_iou = calculate_iou(pred_object, gt_object)
                    pair_score = (person_iou + object_iou) / 2.0

                    if pair_score >= iou_thresh and pair_score > best_match_score:
                        best_match_score = pair_score
                        best_match_idx = gt_idx

                if best_match_idx >= 0:
                    matched_gt.add(best_match_idx)

            recall = len(matched_gt) / len(gt_pairs)
            total_recall += recall
            total_samples += 1
            
            # Track size-specific metrics
            for gt_idx, gt_pair in enumerate(gt_pairs):
                size_category = categorize_pair_by_size(gt_pair, AREA_SMALL, AREA_MEDIUM)
                if gt_idx in matched_gt:
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
        ar_scores.append(avg_recall)
        
        # Calculate size-specific recalls for this threshold
        recall_small = tp_small / (tp_small + fn_small) if (tp_small + fn_small) > 0 else 0.0
        recall_medium = tp_medium / (tp_medium + fn_medium) if (tp_medium + fn_medium) > 0 else 0.0
        recall_large = tp_large / (tp_large + fn_large) if (tp_large + fn_large) > 0 else 0.0
        
        recalls_small.append(recall_small)
        recalls_medium.append(recall_medium)
        recalls_large.append(recall_large)

    metrics['AR'] = float(np.mean(ar_scores))
    
    # Calculate size-based AR metrics (average across thresholds)
    metrics['ARs'] = float(np.mean(recalls_small)) if recalls_small else 0.0
    metrics['ARm'] = float(np.mean(recalls_medium)) if recalls_medium else 0.0
    metrics['ARl'] = float(np.mean(recalls_large)) if recalls_large else 0.0

    return metrics


def process_batch_results(
    results: List[Dict[str, Any]],
    requests: List[Dict[str, Any]],
    img_prefix: str
) -> tuple:
    """
    Process batch results and compute metrics.
    """
    from PIL import Image

    # Create lookup from custom_id to metadata
    metadata_lookup = {}
    for req in requests:
        custom_id = req.get("custom_id") or req.get("key")
        if "_metadata" in req:
            metadata_lookup[custom_id] = req["_metadata"]

    predictions = []
    ground_truths = []
    processed_results = []

    for result in results:
        custom_id = result.get("custom_id", "unknown")
        metadata = metadata_lookup.get(custom_id, {})

        file_name = metadata.get("file_name", "")
        image_path = os.path.join(img_prefix, file_name) if file_name else ""

        # Get image dimensions
        img_width, img_height = 512, 512
        if image_path and os.path.exists(image_path):
            try:
                with Image.open(image_path) as img:
                    img_width, img_height = img.size
            except:
                pass

        if result.get("status") == "success":
            response_text = result.get("response", "")
            pred_pairs = parse_json_response(response_text, img_width, img_height)
        else:
            pred_pairs = []
            print(f"Warning: Request {custom_id} failed: {result.get('error')}")

        gt_pairs = metadata.get("gt_pairs", [])

        predictions.append({"pairs": pred_pairs})
        ground_truths.append({"pairs": gt_pairs})

        processed_results.append({
            "image_id": metadata.get("sample_idx", 0),
            "image_path": file_name,
            "action": metadata.get("action", ""),
            "object_category": metadata.get("object_category", ""),
            "predicted_pairs": pred_pairs,
            "gt_pairs": gt_pairs,
            "custom_id": custom_id,
            "status": result.get("status", "unknown")
        })

    return processed_results, predictions, ground_truths


def main():
    parser = argparse.ArgumentParser(
        description="SWIG-HOI Grounding Batch Evaluation"
    )

    # Provider and model
    parser.add_argument('--provider', type=str, required=True,
                        choices=['claude', 'gemini', 'openai'],
                        help='API provider')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name')

    # Data paths
    parser.add_argument('--ann-file', type=str, required=True,
                        help='Path to annotation file')
    parser.add_argument('--img-prefix', type=str, required=True,
                        help='Path to image directory')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for results')

    # Batch options
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum samples to evaluate')
    parser.add_argument('--poll-interval', type=int, default=60,
                        help='Seconds between status polls')

    # Prompt options
    parser.add_argument('--optimized-prompts', action='store_true', default=True,
                        help='Use optimized (shorter) prompts')
    
    # W&B options
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='hoi-batch-eval',
                        help='W&B project name')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                        help='W&B run name')
    
    # Visualization options
    parser.add_argument('--verbose', action='store_true',
                        help='Enable visualizations')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize W&B if requested
    use_wandb = args.wandb and WANDB_AVAILABLE
    if args.wandb and not WANDB_AVAILABLE:
        print("Warning: W&B requested but not installed. Install with: pip install wandb")
    
    if use_wandb:
        run_name = args.wandb_run_name or f"{args.provider}_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                'provider': args.provider,
                'model': args.model,
                'task': 'swig_grounding',
                'ann_file': args.ann_file,
                'optimized_prompts': args.optimized_prompts,
                'mode': 'batch'
            }
        )
        print(f"W&B initialized: {args.wandb_project}/{run_name}")
    
    # Create visualization directory if verbose
    viz_dir = None
    if args.verbose:
        viz_dir = os.path.join(args.output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        print(f"Visualization directory: {viz_dir}")

    # Initialize
    print("\n" + "=" * 80)
    print("SWIG-HOI Grounding Batch Evaluation")
    print("=" * 80)
    print(f"Provider:    {args.provider}")
    print(f"Model:       {args.model}")
    print(f"Annotation:  {args.ann_file}")
    print(f"Images:      {args.img_prefix}")
    print(f"Output:      {args.output_dir}")
    print(f"Resume:      {args.resume}")
    print(f"W&B:         {use_wandb}")
    print(f"Verbose:     {args.verbose}")
    print("=" * 80 + "\n")

    # Initialize processor and checkpoint manager
    processor = BatchProcessor(provider=args.provider)
    checkpoint_mgr = CheckpointManager(checkpoint_dir=os.path.join(args.output_dir, "checkpoints"))

    # Check for existing checkpoint
    job = None
    requests = None

    if args.resume:
        incomplete = checkpoint_mgr.find_incomplete(
            provider=args.provider,
            task="swig_ground"
        )
        if incomplete:
            checkpoint = incomplete[0]
            print(f"Found incomplete job: {checkpoint.job_id}")

            job = BatchJob(
                id=checkpoint.job_id,
                provider=args.provider,
                model=args.model,
                status=BatchStatus(checkpoint.status),
                created_at=datetime.fromisoformat(checkpoint.submitted_at),
                total_requests=checkpoint.total_requests,
                metadata=checkpoint.metadata
            )

            requests_file = os.path.join(args.output_dir, "batch_requests.json")
            if os.path.exists(requests_file):
                with open(requests_file, 'r') as f:
                    requests = json.load(f)

    if job is None:
        # Load annotations
        print("Loading annotations...")
        annotations = load_annotations(args.ann_file)

        if 'annotations' in annotations:
            samples = annotations['annotations']
        elif 'data' in annotations:
            samples = annotations['data']
        else:
            samples = annotations

        print(f"Loaded {len(samples)} samples")

        # Get processed IDs if resuming
        processed_ids = set()
        if args.resume:
            latest = checkpoint_mgr.find_latest(provider=args.provider, task="swig_ground")
            if latest:
                processed_ids = latest.get_processed_set()

        # Build prompt (provider-specific to match real-time scripts)
        base_prompt = build_grounding_prompt(provider=args.provider, optimized=args.optimized_prompts)

        # Prepare requests
        print("\nPreparing batch requests...")
        requests = prepare_batch_requests(
            samples=samples,
            img_prefix=args.img_prefix,
            processor=processor,
            model=args.model,
            base_prompt=base_prompt,
            processed_ids=processed_ids,
            max_samples=args.max_samples
        )

        if not requests:
            print("No requests to process.")
            return 0

        print(f"\nPrepared {len(requests)} requests", flush=True)

        # Save request metadata for potential resume (not full requests with base64 images)
        requests_file = os.path.join(args.output_dir, "batch_requests.json")
        print(f"Saving request metadata to {requests_file}...", flush=True)
        with open(requests_file, 'w') as f:
            requests_to_save = []
            for req in requests:
                # Only save metadata, not full request with large base64 images
                req_copy = {'_metadata': req.get('_metadata', {}), 'custom_id': req.get('custom_id', req.get('key', ''))}
                requests_to_save.append(req_copy)
            json.dump(requests_to_save, f, indent=2)
        print("Request metadata saved.", flush=True)

        # Submit batch
        print("\nSubmitting batch job...", flush=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        display_name = f"swig_ground_{args.provider}_{timestamp}"

        batch_requests = []
        for req in requests:
            req_copy = {k: v for k, v in req.items() if not k.startswith('_')}
            batch_requests.append(req_copy)

        job = processor.submit_batch(
            requests=batch_requests,
            model=args.model,
            display_name=display_name
        )

        print(f"Batch job submitted: {job.id}")

        # Save checkpoint
        checkpoint = EvaluationCheckpoint(
            job_id=job.id,
            provider=args.provider,
            model=args.model,
            task="swig_ground",
            status=job.status.value,
            submitted_at=datetime.now().isoformat(),
            total_requests=len(requests),
            metadata={"display_name": display_name}
        )
        checkpoint_mgr.save(checkpoint)

    # Wait for completion
    print("\nWaiting for batch completion...")

    def on_status_update(updated_job: BatchJob):
        checkpoint = checkpoint_mgr.load(job.id)
        if checkpoint:
            checkpoint.status = updated_job.status.value
            checkpoint_mgr.save(checkpoint)

    try:
        job = wait_for_batch_completion(
            processor=processor,
            job=job,
            poll_interval=args.poll_interval,
            on_status_update=on_status_update
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted! Use --resume to continue.")
        return 1

    if job.status != BatchStatus.COMPLETED:
        print(f"\nBatch job ended with status: {job.status.value}")
        return 1

    # Download and process results
    print("\nDownloading results...")
    results = processor.download_results(job)
    print(f"Downloaded {len(results)} results")

    print("\nProcessing results...")
    processed_results, predictions, ground_truths = process_batch_results(
        results, requests, args.img_prefix
    )
    
    # Generate visualizations if verbose
    if viz_dir and processed_results:
        print("\nGenerating visualizations...")
        viz_count = 0
        for idx, result in enumerate(tqdm(processed_results[:100], desc="Visualizations")):  # Limit to first 100
            try:
                image_path = os.path.join(args.img_prefix, result.get('image_path', ''))
                if not os.path.exists(image_path):
                    continue
                
                pred_pairs = result.get('pred_pairs', [])
                gt_pairs = result.get('gt_pairs', [])
                action = result.get('action', '')
                object_category = result.get('object_category', '')
                
                # Create safe filename
                base_name = os.path.splitext(os.path.basename(result.get('image_path', 'img')))[0]
                viz_filename = f"{idx:05d}_{base_name}_{action}_viz.jpg"
                viz_path = os.path.join(viz_dir, viz_filename)
                
                visualize_grounding_result(
                    image_path=image_path,
                    pred_pairs=pred_pairs,
                    gt_pairs=gt_pairs,
                    action=action,
                    object_category=object_category,
                    output_path=viz_path
                )
                viz_count += 1
            except Exception as e:
                pass  # Skip failed visualizations
        print(f"Generated {viz_count} visualizations in {viz_dir}")

    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_grounding_metrics(predictions, ground_truths)

    # Print results
    print("\n" + "=" * 80)
    print("SWIG-HOI Grounding Results (Batch)")
    print("=" * 80)
    print(f"{'Metric':<20} {'Score':>10}")
    print("-" * 40)

    for metric, score in metrics.items():
        print(f"{metric:<20} {score*100:>9.2f}%")

    print("=" * 80)

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(args.output_dir, f"results_{timestamp}.json")

    save_results(
        results=processed_results,
        output_path=output_file,
        metrics={
            'provider': args.provider,
            'model': args.model,
            'task': 'swig_grounding',
            'total_samples': len(processed_results),
            'timestamp': timestamp,
            **metrics
        }
    )

    # Update checkpoint
    checkpoint = checkpoint_mgr.load(job.id)
    if checkpoint:
        checkpoint.status = "completed"
        checkpoint.results = processed_results
        # Populate processed_ids from results for resume functionality
        checkpoint.processed_ids = [r.get("custom_id", f"{r.get('image_id', '')}_{r.get('image_path', '')}") 
                                    for r in processed_results]
        checkpoint_mgr.save(checkpoint)

    # Log to W&B
    if use_wandb:
        wandb.log(metrics)
        wandb.log({
            'total_samples': len(processed_results),
            'provider': args.provider,
            'model': args.model
        })
        
        # Log sample predictions table
        if processed_results:
            columns = ["image", "action", "object", "num_pred", "num_gt", "ar"]
            data = []
            for r in processed_results[:50]:  # Log first 50 samples
                num_pred = len(r.get('pred_pairs', []))
                num_gt = len(r.get('gt_pairs', []))
                ar = r.get('ar', 0.0)
                data.append([
                    r.get('image_path', ''),
                    r.get('action', ''),
                    r.get('object_category', ''),
                    num_pred,
                    num_gt,
                    ar
                ])
            wandb.log({"predictions": wandb.Table(columns=columns, data=data)})
        
        wandb.finish()
        print("W&B run completed")

    print(f"\nResults: {output_file}")
    if viz_dir:
        print(f"Visualizations: {viz_dir}")
    print("Evaluation Complete!")

    return 0


if __name__ == '__main__':
    sys.exit(main())
