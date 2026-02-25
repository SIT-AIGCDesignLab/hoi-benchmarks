#!/usr/bin/env python3
"""
Shared Utility Functions for API-Based HOI Evaluation

This module provides common utilities for evaluating vision-language models
via APIs (Claude, Gemini, OpenAI) on HOI benchmarks.

Functions:
- Image encoding and manipulation
- Bounding box operations
- Metrics computation (NLP and COCO-style)
- Result saving and loading
- BERTScore integration
"""

import os
import io
import json
import base64
import subprocess
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
from collections import defaultdict
import time

# COCO evaluation imports
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap


def encode_image_to_base64(image_path: str, max_size: Optional[Tuple[int, int]] = None) -> str:
    """
    Encode image to base64 string for API requests.

    Args:
        image_path: Path to image file
        max_size: Optional (width, height) to resize image

    Returns:
        Base64 encoded string
    """
    img = Image.open(image_path)

    # Resize if needed
    if max_size:
        img.thumbnail(max_size, Image.Resampling.LANCZOS)

    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Encode to base64
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=95)
    img_bytes = buffer.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')

    return img_base64


def draw_bounding_boxes_on_image(
    image_path: str,
    bboxes: List[List[float]],
    labels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    output_path: Optional[str] = None
) -> Image.Image:
    """
    Draw bounding boxes on image for visualization.

    Args:
        image_path: Path to image file
        bboxes: List of [x1, y1, x2, y2] bounding boxes
        labels: Optional labels for each bbox
        colors: Optional colors for each bbox (default: red, blue, green, ...)
        output_path: Optional path to save annotated image

    Returns:
        PIL Image with bounding boxes drawn
    """
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)

    default_colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
    if colors is None:
        colors = default_colors

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()

    for idx, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        color = colors[idx % len(colors)]

        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Draw label if provided
        if labels and idx < len(labels):
            label = labels[idx]
            # Draw label background
            text_bbox = draw.textbbox((x1, y1 - 25), label, font=font)
            draw.rectangle(text_bbox, fill=color)
            draw.text((x1, y1 - 25), label, fill='white', font=font)

    if output_path:
        img.save(output_path)

    return img


def normalize_bbox_to_1000(bbox: List[float], img_width: int, img_height: int) -> List[int]:
    """
    Normalize bounding box to [0, 1000] range.

    Args:
        bbox: [x1, y1, x2, y2] in original coordinates
        img_width: Image width
        img_height: Image height

    Returns:
        [x1, y1, x2, y2] in [0, 1000] range
    """
    x1, y1, x2, y2 = bbox
    x1_norm = int(x1 / img_width * 1000)
    y1_norm = int(y1 / img_height * 1000)
    x2_norm = int(x2 / img_width * 1000)
    y2_norm = int(y2 / img_height * 1000)
    return [x1_norm, y1_norm, x2_norm, y2_norm]


def denormalize_bbox_from_1000(bbox: List[int], img_width: int, img_height: int) -> List[float]:
    """
    Denormalize bounding box from [0, 1000] range to original coordinates.

    Args:
        bbox: [x1, y1, x2, y2] in [0, 1000] range
        img_width: Image width
        img_height: Image height

    Returns:
        [x1, y1, x2, y2] in original coordinates
    """
    x1, y1, x2, y2 = bbox
    x1_denorm = x1 / 1000 * img_width
    y1_denorm = y1 / 1000 * img_height
    x2_denorm = x2 / 1000 * img_width
    y2_denorm = y2 / 1000 * img_height
    return [x1_denorm, y1_denorm, x2_denorm, y2_denorm]


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]

    Returns:
        IoU score (0.0 to 1.0)
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Calculate intersection
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def compute_bertscore(
    predictions: List[str],
    references: List[str],
    model_name: str = "microsoft/deberta-v2-xxlarge-mnli",
    batch_size: int = 32,
    device: str = "cuda:0",
    verbose: bool = True
) -> Dict[str, float]:
    """
    Compute BERTScore for predictions.

    Args:
        predictions: List of predicted texts
        references: List of reference texts
        model_name: BERTScore model
        batch_size: Batch size for computation
        device: Device to use
        verbose: Show progress

    Returns:
        Dictionary with precision, recall, f1 (mean values)
    """
    try:
        from bert_score import score as bert_score
    except ImportError:
        print("Warning: bert_score package not found. Skipping BERTScore computation.")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    print(f"\nComputing BERTScore with {model_name}...")

    try:
        # Try with use_fast_tokenizer=False and nthreads=1 to avoid tokenizer issues
        P, R, F1 = bert_score(
            predictions,
            references,
            model_type=model_name,
            lang='en',
            batch_size=batch_size,
            rescale_with_baseline=True,
            verbose=verbose,
            device=device,
            use_fast_tokenizer=False,  # Force slow tokenizer to avoid fast tokenizer bugs
            nthreads=1  # Single thread to avoid race conditions
        )
        
    except (TypeError, AttributeError, ValueError) as e:
        # If DeBERTa-v2 fails with tokenizer issues, fall back to deberta-xlarge (non-v2)
        # This handles known compatibility issues in transformers 4.45+ with DeBERTa-v2 tokenizers
        if 'deberta-v2' in model_name.lower():
            fallback_model = 'microsoft/deberta-xlarge-mnli'
            print(f"\n⚠️  DeBERTa-v2 tokenizer failed ({type(e).__name__}), falling back to {fallback_model}")
            
            P, R, F1 = bert_score(
                predictions,
                references,
                model_type=fallback_model,
                lang='en',
                batch_size=batch_size,
                rescale_with_baseline=True,
                verbose=verbose,
                device=device,
                use_fast_tokenizer=False,
                nthreads=1
            )
        else:
            raise

    return {
        "precision": float(P.mean()),
        "recall": float(R.mean()),
        "f1": float(F1.mean()),
        "precision_list": P.tolist(),
        "recall_list": R.tolist(),
        "f1_list": F1.tolist()
    }


def compute_nlp_metrics(predictions: List[Dict], references: List[Dict]) -> Dict[str, float]:
    """
    Compute NLP metrics (METEOR, CIDEr, BLEU, ROUGE-L) using COCO evaluation.

    Args:
        predictions: List of prediction dicts with 'image_id' and 'caption'
        references: List of reference dicts with 'image_id' and 'caption'

    Returns:
        Dictionary with metric scores
    """
    import tempfile

    # Create temporary files for COCO evaluation
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        ref_file = f.name
        # Format references for COCO
        ref_data = {
            "images": [{"id": r["image_id"]} for r in references],
            "annotations": [
                {
                    "image_id": r["image_id"],
                    "id": i,
                    "caption": r["caption"]
                }
                for i, r in enumerate(references)
            ]
        }
        json.dump(ref_data, f)

    try:
        # Initialize COCO
        coco = COCO(ref_file)
        coco_result = coco.loadRes(predictions)

        # Run evaluation
        coco_eval = COCOEvalCap(coco, coco_result)
        
        try:
            coco_eval.evaluate()
        except (BrokenPipeError, OSError, subprocess.CalledProcessError) as e:
            # Some metrics (METEOR, SPICE) require Java/native libraries
            # If evaluation partially completed before failure, use those metrics
            print(f"\n⚠️  Warning: SPICE metric failed (requires native libraries): {type(e).__name__}")
            print("   Note: SPICE will be excluded, but other metrics may have been computed successfully")
            
            # Check if any metrics were computed before the failure
            # If coco_eval.eval exists and has metrics, use them
            if hasattr(coco_eval, 'eval') and coco_eval.eval:
                print(f"   ✓ Using successfully computed metrics: {list(coco_eval.eval.keys())}")
                pass  # Continue to return the already-computed metrics below
            else:
                # If no metrics were computed, set all to 0
                print("   ⚠️  No metrics were computed successfully")
                metrics = {
                    "METEOR": 0.0,
                    "CIDEr": 0.0,
                    "BLEU_1": 0.0,
                    "BLEU_2": 0.0,
                    "BLEU_3": 0.0,
                    "BLEU_4": 0.0,
                    "ROUGE_L": 0.0
                }
                return metrics

        metrics = {
            "METEOR": coco_eval.eval.get("METEOR", 0.0),
            "CIDEr": coco_eval.eval.get("CIDEr", 0.0),
            "BLEU_1": coco_eval.eval.get("Bleu_1", 0.0),
            "BLEU_2": coco_eval.eval.get("Bleu_2", 0.0),
            "BLEU_3": coco_eval.eval.get("Bleu_3", 0.0),
            "BLEU_4": coco_eval.eval.get("Bleu_4", 0.0),
            "ROUGE_L": coco_eval.eval.get("ROUGE_L", 0.0)
        }

        return metrics
    finally:
        # Clean up temp file
        if os.path.exists(ref_file):
            os.remove(ref_file)


def compute_coco_grounding_metrics(
    predictions: List[Dict],
    ground_truth: List[Dict],
    iou_thresholds: List[float] = [0.5, 0.75, 0.5]
) -> Dict[str, float]:
    """
    Compute COCO-style grounding metrics (Average Recall).

    Args:
        predictions: List of prediction dicts with 'pairs' (person_bbox, object_bbox)
        ground_truth: List of ground truth dicts with 'pairs'
        iou_thresholds: IoU thresholds for evaluation

    Returns:
        Dictionary with AR metrics at different IoU thresholds
    """
    metrics = {}

    # For each IoU threshold
    for iou_thresh in iou_thresholds:
        total_recall = 0.0
        total_samples = 0

        for pred, gt in zip(predictions, ground_truth):
            pred_pairs = pred.get('pairs', [])
            gt_pairs = gt.get('pairs', [])

            if len(gt_pairs) == 0:
                continue

            # Match predicted pairs to ground truth pairs
            matched_gt = set()
            for pred_pair in pred_pairs:
                pred_person = pred_pair['person_bbox']
                pred_object = pred_pair['object_bbox']

                # Find best matching GT pair
                best_match_score = 0.0
                best_match_idx = -1

                for gt_idx, gt_pair in enumerate(gt_pairs):
                    if gt_idx in matched_gt:
                        continue

                    gt_person = gt_pair['person_bbox']
                    gt_object = gt_pair['object_bbox']

                    # Calculate IoU for both person and object
                    person_iou = calculate_iou(pred_person, gt_person)
                    object_iou = calculate_iou(pred_object, gt_object)

                    # Pair match score (average IoU)
                    pair_score = (person_iou + object_iou) / 2.0

                    if pair_score >= iou_thresh and pair_score > best_match_score:
                        best_match_score = pair_score
                        best_match_idx = gt_idx

                if best_match_idx >= 0:
                    matched_gt.add(best_match_idx)

            # Calculate recall for this sample
            recall = len(matched_gt) / len(gt_pairs)
            total_recall += recall
            total_samples += 1

        # Average recall across all samples
        avg_recall = total_recall / total_samples if total_samples > 0 else 0.0

        if iou_thresh == 0.5:
            metrics['AR@0.5'] = avg_recall
        elif iou_thresh == 0.75:
            metrics['AR@0.75'] = avg_recall
        else:
            metrics[f'AR@{iou_thresh:.2f}'] = avg_recall

    # Compute AR (average over IoU 0.5:0.95 with step 0.05)
    iou_range = np.arange(0.5, 1.0, 0.05)
    ar_scores = []
    for iou_thresh in iou_range:
        total_recall = 0.0
        total_samples = 0

        for pred, gt in zip(predictions, ground_truth):
            pred_pairs = pred.get('pairs', [])
            gt_pairs = gt.get('pairs', [])

            if len(gt_pairs) == 0:
                continue

            matched_gt = set()
            for pred_pair in pred_pairs:
                pred_person = pred_pair['person_bbox']
                pred_object = pred_pair['object_bbox']

                for gt_idx, gt_pair in enumerate(gt_pairs):
                    if gt_idx in matched_gt:
                        continue

                    gt_person = gt_pair['person_bbox']
                    gt_object = gt_pair['object_bbox']

                    person_iou = calculate_iou(pred_person, gt_person)
                    object_iou = calculate_iou(pred_object, gt_object)
                    pair_score = (person_iou + object_iou) / 2.0

                    if pair_score >= iou_thresh:
                        matched_gt.add(gt_idx)
                        break

            recall = len(matched_gt) / len(gt_pairs)
            total_recall += recall
            total_samples += 1

        avg_recall = total_recall / total_samples if total_samples > 0 else 0.0
        ar_scores.append(avg_recall)

    metrics['AR'] = float(np.mean(ar_scores))

    return metrics


def retry_with_exponential_backoff(
    func,
    max_retries: int = 5,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0
):
    """
    Retry a function with exponential backoff for rate limiting.

    Args:
        func: Function to retry
        max_retries: Maximum number of retries
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff

    Returns:
        Function result or raises exception
    """
    delay = initial_delay

    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise

            # Check if it's a rate limit error
            error_str = str(e).lower()
            if 'rate' in error_str or 'limit' in error_str or '429' in error_str:
                print(f"Rate limit hit. Retrying in {delay:.1f}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                delay = min(delay * exponential_base, max_delay)
            else:
                raise


def save_results(
    results: List[Dict],
    output_path: str,
    metrics: Optional[Dict] = None,
    thinking_content: Optional[List[Dict]] = None,
    per_sample_details: Optional[List[Dict]] = None
):
    """
    Save evaluation results to JSON files.

    Args:
        results: Main results list
        output_path: Base output path
        metrics: Optional metrics dictionary
        thinking_content: Optional thinking content for thinking models
        per_sample_details: Optional per-sample detailed results
    """
    # Save main results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved results to: {output_path}")

    # Save metrics
    if metrics:
        metrics_path = output_path.replace('.json', '_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"✓ Saved metrics to: {metrics_path}")

    # Save thinking content
    if thinking_content:
        thinking_path = output_path.replace('.json', '_thinking.jsonl')
        with open(thinking_path, 'w') as f:
            for item in thinking_content:
                f.write(json.dumps(item) + '\n')
        print(f"✓ Saved thinking content to: {thinking_path}")

    # Save per-sample details
    if per_sample_details:
        details_path = output_path.replace('.json', '_per_sample.json')
        with open(details_path, 'w') as f:
            json.dump(per_sample_details, f, indent=2)
        print(f"✓ Saved per-sample details to: {details_path}")


def load_annotations(ann_file: str) -> Dict:
    """
    Load annotation file.

    Args:
        ann_file: Path to annotation JSON file

    Returns:
        Annotation data dictionary
    """
    with open(ann_file, 'r') as f:
        data = json.load(f)
    return data


def clean_prediction_text(text: str) -> str:
    """
    Clean prediction text by removing common prefixes and formatting.

    Args:
        text: Raw prediction text

    Returns:
        Cleaned text
    """
    text = text.strip()

    # Remove common prefixes
    prefixes = [
        "output:", "Output:", "ANSWER:", "Answer:", "ACTION:", "Action:",
        "Prediction:", "prediction:", "Result:", "result:"
    ]
    for prefix in prefixes:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()

    # Remove markdown formatting
    text = text.replace('**', '').replace('*', '').replace('`', '')

    # Remove quotes
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    if text.startswith("'") and text.endswith("'"):
        text = text[1:-1]

    return text.strip()


# ============================================================================
# Cost Optimization Utilities
# ============================================================================

def optimize_image_for_api(
    image_path: str,
    max_size: int = 448,
    quality: int = 90,
    output_path: Optional[str] = None
) -> str:
    """
    Resize and compress image to reduce vision token costs.

    CONSERVATIVE DEFAULTS:
    - max_size=448 (vs 512 original) → 87.5% of original size, minimal quality loss
    - quality=90 (high quality) → perceptually lossless

    Args:
        image_path: Input image path
        max_size: Maximum dimension (default: 448px, conservative)
        quality: JPEG quality (default: 90, high quality)
        output_path: Where to save optimized image (temp file if None)

    Returns:
        Path to optimized image
    """
    import tempfile

    img = Image.open(image_path)

    # Only resize if larger than max_size
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

    # Save with high-quality compression
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix='.jpg')
        os.close(fd)

    img.convert('RGB').save(output_path, 'JPEG', quality=quality, optimize=True)
    return output_path


def load_existing_results(result_file: str) -> Tuple[List[Dict], set]:
    """
    Load previously saved results for resume capability.

    Args:
        result_file: Path to results JSON file

    Returns:
        (existing_results, processed_ids): List of results and set of processed image IDs
    """
    if os.path.exists(result_file):
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
                results = data.get('results', [])
                # Extract image IDs from results
                processed_ids = set()
                for r in results:
                    if 'image_id' in r:
                        processed_ids.add(r['image_id'])
                return results, processed_ids
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Failed to load existing results from {result_file}: {e}")
            return [], set()
    return [], set()


# ============================================================================
# Rate Limiting and Concurrent Processing
# ============================================================================

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


class RateLimiter:
    """Simple rate limiter for API calls."""

    def __init__(self, max_requests_per_minute: int):
        """
        Initialize rate limiter.

        Args:
            max_requests_per_minute: Maximum API requests allowed per minute
        """
        self.max_requests = max_requests_per_minute
        self.requests = []
        self.lock = threading.Lock()

    def wait_if_needed(self) -> None:
        """Block until it's safe to make another request."""
        with self.lock:
            now = time.time()

            # Remove requests older than 1 minute
            self.requests = [req_time for req_time in self.requests if now - req_time < 60]

            if len(self.requests) >= self.max_requests:
                # Wait until oldest request is > 1 minute old
                sleep_time = 60 - (now - self.requests[0]) + 0.1
                print(f"Rate limit reached, waiting {sleep_time:.1f}s...")
                time.sleep(sleep_time)
                # Remove the oldest request after waiting
                self.requests = self.requests[1:]

            self.requests.append(time.time())


def process_samples_concurrent(
    samples: List[Any],
    api_call_func: callable,
    max_workers: int = 5,
    max_requests_per_minute: int = 50,
    desc: str = "Processing"
) -> List[Any]:
    """
    Process samples concurrently with rate limiting.

    Args:
        samples: List of samples to process
        api_call_func: Function that takes (sample, idx) and returns result
        max_workers: Number of concurrent threads
        max_requests_per_minute: Rate limit
        desc: Progress bar description

    Returns:
        List of results in original order
    """
    from tqdm import tqdm

    rate_limiter = RateLimiter(max_requests_per_minute)
    results = [None] * len(samples)
    errors = []

    def wrapped_call(sample: Any, idx: int) -> Tuple[int, Any]:
        """Wrapper that adds rate limiting."""
        rate_limiter.wait_if_needed()
        try:
            result = api_call_func(sample, idx)
            return idx, result
        except Exception as e:
            errors.append({'idx': idx, 'error': str(e)})
            return idx, None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(wrapped_call, sample, idx): idx
            for idx, sample in enumerate(samples)
        }

        with tqdm(total=len(samples), desc=desc) as pbar:
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result
                pbar.update(1)

    if errors:
        print(f"\nWarning: {len(errors)} samples failed:")
        for err in errors[:5]:  # Show first 5 errors
            print(f"  Sample {err['idx']}: {err['error']}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")

    return results
