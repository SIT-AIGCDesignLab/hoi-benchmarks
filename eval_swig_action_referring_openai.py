#!/usr/bin/env python3
"""
SWIG-HOI Action Referring Evaluation Script for OpenAI API

Evaluates OpenAI models (GPT-4o, GPT-4-turbo, o1-preview) on action prediction.

Task: Given (person, object) bounding boxes, predict the connecting action.

Key Features:
- Uses OpenAI API (openai library)
- Supports GPT-4o, GPT-4-turbo, and o1 reasoning models
- O1 models have built-in reasoning (no separate thinking mode)
- Adds BERTScore metric with deberta-v2-xxlarge-mnli

Usage:
    python eval_swig_action_referring_openai.py \
        --model-name gpt-4o \
        --img-prefix /path/to/swig/images_512 \
        --ann-file data/benchmarks_simplified/swig_action_referring_test_simplified.json \
        --pred-file results/predictions.json \
        --api-key YOUR_API_KEY
"""

import os
import json
import argparse
import base64
from tqdm import tqdm
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional

# Import shared utilities
from eval_api_utils import (
    encode_image_to_base64,
    draw_bounding_boxes_on_image,
    normalize_bbox_to_1000,
    compute_bertscore,
    compute_nlp_metrics,
    retry_with_exponential_backoff,
    save_results,
    load_annotations,
    clean_prediction_text,
    # Cost optimization utilities
    optimize_image_for_api,
    load_existing_results,
    process_samples_concurrent
)

# Response caching
from response_cache import ResponseCache

# OpenAI SDK
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai package not found. Please install: pip install openai")

# Optional W&B import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def call_openai_api(
    image_path: str,
    person_bbox: List[float],
    object_bbox: List[float],
    prompt_text: str,
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    max_tokens: int = 1024
) -> Tuple[str, str]:
    """
    Call OpenAI API with image and bounding boxes.

    Args:
        image_path: Path to image file
        person_bbox: [x1, y1, x2, y2] in original coordinates
        object_bbox: [x1, y1, x2, y2] in original coordinates
        prompt_text: Text prompt
        model: OpenAI model name
        api_key: OpenAI API key (from env if None)
        max_tokens: Maximum tokens in response

    Returns:
        (response_text, reasoning_content)
        For o1 models, reasoning_content contains the internal reasoning
        For GPT-4o/GPT-4-turbo, reasoning_content is empty
    """
    if not OPENAI_AVAILABLE:
        raise ImportError("openai package is required. Install with: pip install openai")

    # Get API key (check both OPEN_AI_API_KEY and OPENAI_API_KEY)
    if api_key is None:
        api_key = os.environ.get('OPEN_AI_API_KEY') or os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("API key not found. Set OPEN_AI_API_KEY or OPENAI_API_KEY env variable or pass --api-key")

    # Initialize client
    client = OpenAI(api_key=api_key)

    # Load image and get dimensions
    img = Image.open(image_path)
    img_width, img_height = img.size

    # Draw bounding boxes on image for visual context
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        tmp_path = tmp.name

    annotated_img = draw_bounding_boxes_on_image(
        image_path,
        [person_bbox, object_bbox],
        labels=['Person/Object 1', 'Person/Object 2'],
        colors=['red', 'blue'],
        output_path=tmp_path
    )

    # Encode image to base64
    img_base64 = encode_image_to_base64(tmp_path)

    # Clean up temp file
    os.unlink(tmp_path)

    # Normalize bboxes to [0, 1000] range for text description
    person_bbox_norm = normalize_bbox_to_1000(person_bbox, img_width, img_height)
    object_bbox_norm = normalize_bbox_to_1000(object_bbox, img_width, img_height)

    # Build prompt with bbox information
    bbox_info = f"\nBounding boxes (in [0-1000] normalized coordinates):\n"
    bbox_info += f"- Red box (Person/Object 1): {person_bbox_norm}\n"
    bbox_info += f"- Blue box (Person/Object 2): {object_bbox_norm}\n\n"

    full_prompt = bbox_info + prompt_text

    # Build messages
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}"
                    }
                },
                {
                    "type": "text",
                    "text": full_prompt
                }
            ]
        }
    ]

    # Call API with retry logic
    def _api_call():
        # O1/O3 models don't support max_tokens, temperature, or system messages
        if "o1" in model.lower() or "o3" in model.lower():
            response = client.chat.completions.create(
                model=model,
                messages=messages
            )
        # GPT-5.x models use max_completion_tokens and support reasoning_effort
        elif "gpt-5" in model.lower():
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=max_tokens,
                temperature=1.0,
                reasoning_effort="medium"  # Enable reasoning for action understanding
            )
        else:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0
            )
        return response

    response = retry_with_exponential_backoff(_api_call)

    # Extract response and reasoning content
    reasoning_content = ""
    response_text = ""

    # O1 models may have reasoning in the response
    if hasattr(response.choices[0].message, 'reasoning'):
        reasoning_content = response.choices[0].message.reasoning or ""

    response_text = response.choices[0].message.content or ""

    return response_text, reasoning_content


def visualize_action_triplet(image_path, person_bbox, object_bbox, predicted_action,
                            gt_action, object_category, output_path):
    """
    Visualize action referring result.

    Args:
        image_path: Path to image
        person_bbox: [x1, y1, x2, y2] for person
        object_bbox: [x1, y1, x2, y2] for object
        predicted_action: Predicted action
        gt_action: Ground truth action
        object_category: Object category
        output_path: Output path
    """
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)

    # Try to load font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Draw person bbox (red)
    px1, py1, px2, py2 = person_bbox
    draw.rectangle([px1, py1, px2, py2], outline="red", width=4)
    draw.text((px1, py1 - 25), "Person", fill="red", font=font_small)

    # Draw object bbox (blue)
    ox1, oy1, ox2, oy2 = object_bbox
    draw.rectangle([ox1, oy1, ox2, oy2], outline="blue", width=4)
    draw.text((ox1, oy1 - 25), object_category.capitalize(), fill="blue", font=font_small)

    # Draw connecting line (green)
    person_center = ((px1 + px2) / 2, (py1 + py2) / 2)
    object_center = ((ox1 + ox2) / 2, (oy1 + oy2) / 2)
    draw.line([person_center, object_center], fill="green", width=3)

    # Add predicted action
    draw.text((10, 10), f"Predicted: {predicted_action}", fill="white", font=font)

    # Add ground truth
    draw.text((10, 40), f"GT: {gt_action}", fill="yellow", font=font)

    # Match indicator
    match = predicted_action.lower().strip() == gt_action.lower().strip()
    match_text = "✓ MATCH" if match else "✗ MISMATCH"
    match_color = "green" if match else "red"
    draw.text((10, 70), match_text, fill=match_color, font=font)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)


def build_action_referring_prompt(is_o1_model: bool = False, optimized: bool = False) -> str:
    """
    Build prompt for action referring task (OpenAI-optimized with examples).

    Args:
        is_o1_model: Whether this is for o1 reasoning model (simplified prompt)
        optimized: Whether to use optimized (shorter) prompt to save tokens

    Returns:
        Prompt text
    """
    if optimized:
        # Optimized prompt for OpenAI: concise with examples
        return (
            "You are an action recognition assistant.\n\n"
            "Task: What action connects the RED box (person) to the BLUE box (object)?\n\n"
            "Format: [verb-ing] [object] (2-4 words, lowercase)\n"
            "Examples: riding bicycle, holding cup, sitting on bench, tattooing needle\n\n"
            "Output: action phrase ONLY"
        )
    elif is_o1_model:
        return (
            "Task: Describe the action between the person (RED box) and object (BLUE box).\n\n"
            "Format: [action-ing] [object], 2-4 words, lowercase\n"
            "Examples: riding bicycle, holding cup, tattooing needle, boarding bus\n\n"
            "Output ONLY the action phrase (no explanations)."
        )
    else:
        return (
            "You are an action recognition assistant specializing in human-object interactions.\n\n"
            "Task: Describe the action connecting the person (RED box) to the object (BLUE box).\n\n"
            "Visual Guide:\n"
            "- RED box: Contains the person/agent performing the action\n"
            "- BLUE box: Contains the object being acted upon\n\n"
            "Output Requirements:\n"
            "1. Use present participle form (verb ending in -ing)\n"
            "2. Include the object: [action-ing] [object]\n"
            "3. Keep it SHORT: 2-4 words maximum\n"
            "4. Use lowercase\n\n"
            "Examples:\n"
            "- riding bicycle\n"
            "- holding cup\n"
            "- sitting on bench\n"
            "- tattooing needle\n"
            "- boarding bus\n"
            "- putting envelope\n\n"
            "Output: Provide ONLY the action phrase (no quotes, no explanation, no punctuation)."
        )


def main():
    parser = argparse.ArgumentParser(
        description="SWIG-HOI Action Referring Evaluation with OpenAI API"
    )

    # Model arguments
    parser.add_argument('--model-name', type=str, default='gpt-4o',
                        help='OpenAI model name (gpt-4o, gpt-4-turbo, o1-preview, etc.)')
    parser.add_argument('--api-key', type=str, default=None,
                        help='OpenAI API key (or set OPEN_AI_API_KEY/OPENAI_API_KEY in .env file)')

    # Data arguments
    parser.add_argument('--ann-file', type=str, required=True,
                        help='Path to annotation file')
    parser.add_argument('--img-prefix', type=str, required=True,
                        help='Path to image directory')
    parser.add_argument('--pred-file', type=str, required=True,
                        help='Path to save predictions')

    # Evaluation arguments
    parser.add_argument('--max-images', type=int, default=None,
                        help='Maximum number of images to evaluate')
    parser.add_argument('--verbose', action='store_true',
                        help='Show per-triplet results')

    # BERTScore arguments
    parser.add_argument('--bertscore-model', type=str, default='microsoft/deberta-v2-xxlarge-mnli',
                        help='BERTScore model name')
    parser.add_argument('--bertscore-batch-size', type=int, default=32,
                        help='Batch size for BERTScore')
    parser.add_argument('--bertscore-device', type=str, default='cpu',
                        help='Device for BERTScore computation')

    # W&B arguments
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='swig-action-referring-openai',
                        help='W&B project name')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                        help='W&B run name')

    # Cost optimization arguments
    parser.add_argument('--use-cache', action='store_true',
                        help='Enable response caching to avoid redundant API calls')
    parser.add_argument('--cache-dir', type=str, default='./cache',
                        help='Directory to store cached responses')
    parser.add_argument('--optimize-images', action='store_true',
                        help='Enable image optimization to reduce vision token costs')
    parser.add_argument('--image-max-size', type=int, default=448,
                        help='Maximum image dimension for optimization (default: 448)')
    parser.add_argument('--image-quality', type=int, default=90,
                        help='JPEG quality for optimized images (default: 90)')
    parser.add_argument('--optimized-prompts', action='store_true',
                        help='Use optimized (shorter) prompts to save tokens')
    parser.add_argument('--concurrent-requests', type=int, default=1,
                        help='Number of concurrent API requests (default: 1 for sequential)')
    parser.add_argument('--rate-limit', type=int, default=None,
                        help='Max requests per minute (auto-detected if not specified)')

    args = parser.parse_args()

    # Validate API key (check both OPEN_AI_API_KEY and OPENAI_API_KEY)
    api_key = args.api_key or os.environ.get('OPEN_AI_API_KEY') or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("ERROR: OpenAI API key not found!")
        print("Please set OPEN_AI_API_KEY or OPENAI_API_KEY environment variable or pass --api-key argument")
        return 1

    # Check if this is an o1 model
    is_o1_model = "o1" in args.model_name.lower()

    # Initialize W&B if requested
    use_wandb = args.wandb and WANDB_AVAILABLE
    if use_wandb:
        run_name = args.wandb_run_name or f"openai_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                'model': args.model_name,
                'is_o1_model': is_o1_model,
                'bertscore_model': args.bertscore_model
            }
        )

    print("\n" + "=" * 80)
    print("SWIG-HOI Action Referring Evaluation (OpenAI API)")
    print("=" * 80)
    print(f"Model:            {args.model_name}")
    print(f"O1 Reasoning:     {is_o1_model}")
    print(f"Annotation file:  {args.ann_file}")
    print(f"Image prefix:     {args.img_prefix}")
    print(f"Output file:      {args.pred_file}")
    print(f"BERTScore model:  {args.bertscore_model}")
    print("=" * 80 + "\n")

    # Load annotations
    print("Loading annotations...")
    annotations = load_annotations(args.ann_file)

    if 'annotations' in annotations:
        dataset_samples = annotations['annotations']
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
        viz_dir = args.pred_file.replace('.json', '_visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        print(f"✓ Visualization directory: {viz_dir}\n")

    # Cost Optimization: Load existing results for resume capability
    existing_results, processed_ids = load_existing_results(args.pred_file)
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
        # Auto-detect rate limits for OpenAI models
        if args.rate_limit is None:
            if "o1" in args.model_name.lower():
                args.rate_limit = 20  # O1 models have lower rate limits
            else:
                args.rate_limit = 500  # GPT-4o/GPT-4-turbo: 500 RPM
        print(f"✓ Concurrent processing enabled ({args.concurrent_requests} workers, rate limit: {args.rate_limit} req/min)\n")

    # Build prompt (use optimized version if enabled and not using o1 models)
    use_optimized_prompt = args.optimized_prompts and not is_o1_model
    prompt_text = build_action_referring_prompt(is_o1_model, optimized=use_optimized_prompt)

    if cache:
        prompt_hash = ResponseCache.hash_prompt(prompt_text)
        if use_optimized_prompt:
            print("✓ Using optimized prompt (75% shorter, ~90 token savings per request)\n")

    # Evaluate each sample
    results = existing_results  # Start with existing results
    per_triplet_results = []
    reasoning_contents = []
    action_stats = defaultdict(lambda: {'total': 0, 'exact_match': 0})
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

        # Get bounding boxes from simplified format
        boxes = sample['boxes']
        person_bbox = boxes[sample['person_box_idx']]
        object_bbox = boxes[sample['object_box_idx']]

        # Get ground truth action
        gt_action = sample['gt_action']
        
        # Get object category (if available)
        object_category = sample.get('object_category', 'object')

        # Cost Optimization: Check cache before API call
        pred_text = None
        reasoning_content = None

        if cache:
            cached_response = cache.get(file_name, gt_action, "", args.model_name, prompt_hash)
            if cached_response:
                pred_text = cached_response.get('prediction', '')
                reasoning_content = cached_response.get('reasoning', '')
                cache_hits += 1

        # If not in cache, call API
        if pred_text is None:
            # Cost Optimization: Optimize image before encoding
            optimized_image_path = image_path
            if args.optimize_images:
                optimized_image_path = optimize_image_for_api(
                    image_path,
                    max_size=args.image_max_size,
                    quality=args.image_quality
                )

            # Call OpenAI API
            try:
                pred_text, reasoning_content = call_openai_api(
                    image_path=optimized_image_path,
                    person_bbox=person_bbox,
                    object_bbox=object_bbox,
                    prompt_text=prompt_text,
                    model=args.model_name,
                    api_key=api_key
                )

                # Cost Optimization: Save to cache
                if cache:
                    cache.set(file_name, gt_action, "", args.model_name, prompt_hash, {
                        'prediction': pred_text,
                        'reasoning': reasoning_content
                    })

            except Exception as e:
                print(f"\nError processing sample {idx} ({file_name}): {e}")
                continue

        # Clean prediction
        if pred_text:
            pred_action = clean_prediction_text(pred_text)
        else:
            pred_action = ""
            print(f"\nWarning: Empty prediction for sample {idx} ({file_name})")

        # Store results
        result = {
            'image_id': idx,
            'image_path': file_name,
            'prediction': pred_action,
            'ground_truth': gt_action,
            'person_bbox': person_bbox,
            'object_bbox': object_bbox
        }

        if is_o1_model and reasoning_content:
            result['reasoning_content'] = reasoning_content
            reasoning_contents.append({
                'image_id': idx,
                'reasoning': reasoning_content,
                'prediction': pred_action,
                'ground_truth': gt_action
            })

        results.append(result)

        # Per-triplet metrics
        exact_match = (pred_action.lower() == gt_action.lower())
        per_triplet_result = {
            'image_id': idx,
            'image_path': file_name,
            'prediction': pred_action,
            'ground_truth': gt_action,
            'exact_match': exact_match,
            'person_bbox': person_bbox,
            'object_bbox': object_bbox
        }
        per_triplet_results.append(per_triplet_result)

        # Update action statistics
        action_stats[gt_action]['total'] += 1
        if exact_match:
            action_stats[gt_action]['exact_match'] += 1

        # Cost Optimization: Incremental saving every 50 samples
        if len(results) % 50 == 0:
            save_results(results, args.pred_file + ".tmp", {'partial': True, 'samples_processed': len(results)})

        if args.verbose and idx < 5:
            print(f"\n[Sample {idx}]")
            print(f"  Image: {file_name}")
            print(f"  Ground Truth: {gt_action}")
            print(f"  Prediction:   {pred_action}")
            print(f"  Exact Match:  {exact_match}")
            if is_o1_model and reasoning_content:
                print(f"  Reasoning: {reasoning_content[:200]}...")

        # Generate visualization if verbose
        if viz_dir is not None:
            viz_path = os.path.join(viz_dir, f"{idx:05d}_{file_name}")
            try:
                visualize_action_triplet(
                    image_path, person_bbox, object_bbox,
                    pred_action, gt_action, object_category,
                    viz_path
                )
            except Exception as e:
                print(f"Warning: Failed to create visualization for {file_name}: {e}")

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
            print(f"✓ Optimized prompt used: ~75% shorter (~90 tokens saved per request)")

        # Calculate estimated cost savings
        total_api_calls = len(dataset_samples) - cache_hits - len(existing_results)
        if total_api_calls > 0:
            print(f"\nTotal new API calls: {total_api_calls}")
        print("=" * 80 + "\n")

    # Compute NLP metrics
    print("\nComputing NLP metrics (METEOR, CIDEr, BLEU, ROUGE-L)...")

    # Check if we have any results to evaluate
    if len(results) == 0:
        print("\n⚠️  WARNING: No samples were successfully processed!")
        print("   This could be because:")
        print("   1. All images were not found at the specified path")
        print("   2. All samples were skipped due to resume capability")
        print("   3. API errors occurred for all samples")
        print("\n   Cannot compute metrics with 0 samples. Exiting gracefully.")
        
        # Save empty results file
        with open(args.pred_file, 'w') as f:
            json.dump({
                'results': [],
                'metadata': {
                    'model': args.model_name,
                    'num_samples': 0,
                    'error': 'No samples processed'
                }
            }, f, indent=2)
        
        return 1  # Exit with error code

    predictions_coco = [{'image_id': r['image_id'], 'caption': r['prediction']} for r in results]
    references_coco = [{'image_id': r['image_id'], 'caption': r['ground_truth']} for r in results]

    nlp_metrics = compute_nlp_metrics(predictions_coco, references_coco)

    total_triplets = len(per_triplet_results)
    exact_matches = sum(1 for r in per_triplet_results if r['exact_match'])
    exact_match_acc = exact_matches / total_triplets if total_triplets > 0 else 0.0
    nlp_metrics['exact_match'] = exact_match_acc

    # Compute BERTScore (always)
    print(f"\nComputing BERTScore with {args.bertscore_model}...")
    predictions_text = [r['prediction'] for r in results]
    references_text = [r['ground_truth'] for r in results]

    bertscore_metrics = compute_bertscore(
        predictions=predictions_text,
        references=references_text,
        model_name=args.bertscore_model,
        batch_size=args.bertscore_batch_size,
        device=args.bertscore_device,
        verbose=True
    )

    nlp_metrics['BERTScore_P'] = bertscore_metrics['precision']
    nlp_metrics['BERTScore_R'] = bertscore_metrics['recall']
    nlp_metrics['BERTScore_F1'] = bertscore_metrics['f1']

    # Visualization summary
    if viz_dir is not None:
        viz_count = len([f for f in os.listdir(viz_dir) if f.endswith(('.jpg', '.png'))])
        print(f"\n✓ Visualizations saved: {viz_dir}/")
        print(f"  Total images: {viz_count}")

    # Print results
    print("\n" + "=" * 80)
    print("SWIG-HOI Action Referring Results (OpenAI API)")
    print("=" * 80)
    print(f"{'Metric':<20} {'Score':>10}  {'Description':<45}")
    print("-" * 80)

    metric_descriptions = {
        'METEOR': 'METEOR (semantic similarity)',
        'CIDEr': 'CIDEr (corpus consensus)',
        'BLEU_1': 'BLEU-1 (unigram overlap)',
        'BLEU_2': 'BLEU-2 (bigram overlap)',
        'BLEU_3': 'BLEU-3 (trigram overlap)',
        'BLEU_4': 'BLEU-4 (4-gram overlap)',
        'ROUGE_L': 'ROUGE-L (longest common subsequence)',
        'exact_match': 'Exact string match accuracy',
        'BERTScore_P': 'BERTScore Precision',
        'BERTScore_R': 'BERTScore Recall',
        'BERTScore_F1': 'BERTScore F1'
    }

    for metric, score in nlp_metrics.items():
        desc = metric_descriptions.get(metric, metric)
        print(f"{metric:<20} {score*100:>9.2f}%  {desc:<45}")

    print("=" * 80)

    if use_wandb:
        wandb.log(nlp_metrics)

    # Save results
    print("\nSaving results...")

    all_metrics = {
        'model': args.model_name,
        'is_o1_model': is_o1_model,
        'total_samples': len(results),
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        **nlp_metrics
    }

    save_results(
        results=results,
        output_path=args.pred_file,
        metrics=all_metrics,
        thinking_content=reasoning_contents if reasoning_contents else None,
        per_sample_details=per_triplet_results if args.verbose else None
    )

    if bertscore_metrics:
        bertscore_path = args.pred_file.replace('.json', '_bertscore.json')
        bertscore_details = {
            'model': args.bertscore_model,
            'metrics': {
                'precision_mean': bertscore_metrics['precision'],
                'recall_mean': bertscore_metrics['recall'],
                'f1_mean': bertscore_metrics['f1']
            }
        }
        with open(bertscore_path, 'w') as f:
            json.dump(bertscore_details, f, indent=2)
        print(f"✓ Saved BERTScore details to: {bertscore_path}")

    print("\n✅ Evaluation Complete!")

    if use_wandb:
        wandb.finish()

    return 0


if __name__ == '__main__':
    exit(main())
