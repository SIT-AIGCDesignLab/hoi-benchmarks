#!/usr/bin/env python3
"""
Recalculate grounding metrics from existing results files.

This script computes COCO-style AR metrics including size-based metrics
(ARs, ARm, ARl) from previously saved result files without needing to
re-run the full evaluation.

Usage:
    python recalculate_ground_metrics.py results/swig_ground_claude_batch/results_20260115_200342.json
"""

import json
import argparse
import numpy as np
from typing import List, Dict
from pathlib import Path


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate IoU between two boxes in [x1, y1, x2, y2] format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def get_box_area(box: List[float]) -> float:
    """Calculate box area from [x1, y1, x2, y2] format."""
    return (box[2] - box[0]) * (box[3] - box[1])


def categorize_pair_by_size(gt_pair: Dict, area_small: float = 1024, area_medium: float = 9216) -> str:
    """
    Categorize a ground truth pair by object size.
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


def compute_grounding_metrics(results: List[Dict]) -> Dict[str, float]:
    """
    Compute COCO-style Average Recall metrics including size-based AR.
    """
    metrics = {}
    
    # COCO area thresholds
    AREA_SMALL = 32 ** 2
    AREA_MEDIUM = 96 ** 2
    
    # Extract predictions and ground truths from results
    predictions = []
    ground_truths = []
    
    for r in results:
        # Handle different key names
        pred_pairs = r.get('predicted_pairs', r.get('pairs', []))
        gt_pairs = r.get('gt_pairs', [])
        
        predictions.append({'pairs': pred_pairs})
        ground_truths.append({'pairs': gt_pairs})
    
    # Compute AR at specific thresholds
    for iou_thresh in [0.5, 0.75]:
        total_recall = 0.0
        total_samples = 0

        for pred, gt in zip(predictions, ground_truths):
            pred_pairs = pred.get('pairs', [])
            gt_pairs = gt.get('pairs', [])

            if len(gt_pairs) == 0:
                continue

            matched_gt = set()
            for pred_pair in pred_pairs:
                pred_person = pred_pair.get('person_bbox', pred_pair.get('person_box', []))
                pred_object = pred_pair.get('object_bbox', pred_pair.get('object_box', []))
                
                if not pred_person or not pred_object:
                    continue

                best_match_idx = -1
                best_match_score = 0.0

                for gt_idx, gt_pair in enumerate(gt_pairs):
                    if gt_idx in matched_gt:
                        continue

                    gt_person = gt_pair.get('person_bbox', gt_pair.get('person_box', []))
                    gt_object = gt_pair.get('object_bbox', gt_pair.get('object_box', []))
                    
                    if not gt_person or not gt_object:
                        continue

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
    
    recalls_small = []
    recalls_medium = []
    recalls_large = []

    for iou_thresh in iou_range:
        total_recall = 0.0
        total_samples = 0
        
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
                pred_person = pred_pair.get('person_bbox', pred_pair.get('person_box', []))
                pred_object = pred_pair.get('object_bbox', pred_pair.get('object_box', []))
                
                if not pred_person or not pred_object:
                    continue

                best_match_idx = -1
                best_match_score = 0.0

                for gt_idx, gt_pair in enumerate(gt_pairs):
                    if gt_idx in matched_gt:
                        continue

                    gt_person = gt_pair.get('person_bbox', gt_pair.get('person_box', []))
                    gt_object = gt_pair.get('object_bbox', gt_pair.get('object_box', []))
                    
                    if not gt_person or not gt_object:
                        continue

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
                gt_object = gt_pair.get('object_bbox', gt_pair.get('object_box', []))
                if not gt_object:
                    continue
                    
                size_category = categorize_pair_by_size({'object_bbox': gt_object}, AREA_SMALL, AREA_MEDIUM)
                if gt_idx in matched_gt:
                    if size_category == 'small':
                        tp_small += 1
                    elif size_category == 'medium':
                        tp_medium += 1
                    else:
                        tp_large += 1
                else:
                    if size_category == 'small':
                        fn_small += 1
                    elif size_category == 'medium':
                        fn_medium += 1
                    else:
                        fn_large += 1

        avg_recall = total_recall / total_samples if total_samples > 0 else 0.0
        ar_scores.append(avg_recall)
        
        recall_small = tp_small / (tp_small + fn_small) if (tp_small + fn_small) > 0 else 0.0
        recall_medium = tp_medium / (tp_medium + fn_medium) if (tp_medium + fn_medium) > 0 else 0.0
        recall_large = tp_large / (tp_large + fn_large) if (tp_large + fn_large) > 0 else 0.0
        
        recalls_small.append(recall_small)
        recalls_medium.append(recall_medium)
        recalls_large.append(recall_large)

    metrics['AR'] = float(np.mean(ar_scores))
    metrics['ARs'] = float(np.mean(recalls_small)) if recalls_small else 0.0
    metrics['ARm'] = float(np.mean(recalls_medium)) if recalls_medium else 0.0
    metrics['ARl'] = float(np.mean(recalls_large)) if recalls_large else 0.0
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Recalculate grounding metrics from results file")
    parser.add_argument("input", help="Path to results JSON file")
    parser.add_argument("--output", "-o", help="Path to save updated metrics (optional)")
    args = parser.parse_args()
    
    print(f"\nLoading results from: {args.input}")
    with open(args.input, 'r') as f:
        results = json.load(f)
    
    print(f"Loaded {len(results)} samples")
    
    print("\nComputing metrics...")
    metrics = compute_grounding_metrics(results)
    
    print("\n" + "=" * 80)
    print("Grounding Metrics (Recalculated)")
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
    
    if args.output:
        output_path = args.output
    else:
        # Update the existing metrics file
        input_path = Path(args.input)
        metrics_path = input_path.parent / f"{input_path.stem}_metrics_updated.json"
        output_path = str(metrics_path)
    
    print(f"\nSaving updated metrics to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("Done!")
    return 0


if __name__ == "__main__":
    exit(main())
