#!/usr/bin/env python3
"""
SWIG-HOI Action Referring Batch Evaluation Script

Unified batch evaluation script supporting Claude, Gemini, and OpenAI Batch APIs.
Offers 50% cost savings and avoids rate limit issues.

Task: Given (person, object) bounding boxes, predict the connecting action.

Usage:
    python eval_swig_action_referring_batch.py \
        --provider claude \
        --model claude-sonnet-4-5-20250514 \
        --ann-file data/benchmarks_simplified/swig_action_referring_test_simplified.json \
        --img-prefix data/swig_hoi/images_512 \
        --output-dir results/swig_action_claude_batch \
        --resume
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from tqdm import tqdm
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
    EvaluationCheckpoint,
    IncrementalResultSaver
)

# Import shared evaluation utilities
from eval_api_utils import (
    load_annotations,
    clean_prediction_text,
    compute_nlp_metrics,
    compute_bertscore,
    save_results,
    normalize_bbox_to_1000
)


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
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except:
            font = ImageFont.load_default()
            font_small = ImageFont.load_default()

    # Draw person bbox (red)
    px1, py1, px2, py2 = person_bbox
    draw.rectangle([px1, py1, px2, py2], outline="red", width=4)
    draw.text((px1, max(0, py1 - 25)), "Person", fill="red", font=font_small)

    # Draw object bbox (blue)
    ox1, oy1, ox2, oy2 = object_bbox
    draw.rectangle([ox1, oy1, ox2, oy2], outline="blue", width=4)
    draw.text((ox1, max(0, oy1 - 25)), object_category.capitalize(), fill="blue", font=font_small)

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
    match_text = "MATCH" if match else "MISMATCH"
    match_color = "green" if match else "red"
    draw.text((10, 70), match_text, fill=match_color, font=font)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)


def build_action_referring_prompt(provider: str, optimized: bool = True) -> str:
    """
    Build prompt for action referring task - provider-specific to match real-time scripts.
    
    Args:
        provider: API provider ('claude', 'gemini', 'openai')
        optimized: Use optimized (shorter) prompt
        
    Returns:
        Prompt text
    """
    if provider == 'claude':
        if optimized:
            return (
                "<role>Expert action recognition system</role>\n\n"
                "<task>Identify the action connecting RED box (person) to BLUE box (object).</task>\n\n"
                "<format>[verb-ing] [object], 2-4 words, lowercase</format>\n\n"
                "<examples>riding bicycle, holding cup, sitting on bench, tattooing needle</examples>\n\n"
                "Output: action phrase ONLY (no quotes, no explanation)"
            )
        else:
            return (
                "<role>You are an expert in human-object interaction recognition.</role>\n\n"
                "<task>Describe the action connecting the person (RED box) to the object (BLUE box).</task>\n\n"
                "<visual_guide>\n"
                "- RED box: Contains the person/agent performing the action\n"
                "- BLUE box: Contains the object being acted upon\n"
                "</visual_guide>\n\n"
                "<output_requirements>\n"
                "1. Use present participle form (verb ending in -ing)\n"
                "2. Include the object name: [action-ing] [object]\n"
                "3. Keep it SHORT: 2-4 words maximum\n"
                "4. Use lowercase\n"
                "</output_requirements>\n\n"
                "<examples>\n"
                "- riding bicycle\n"
                "- holding cup\n"
                "- sitting on bench\n"
                "- tattooing needle\n"
                "- boarding bus\n"
                "- putting envelope\n"
                "</examples>\n\n"
                "Output: Provide ONLY the action phrase (no quotes, no explanation, no punctuation)."
            )
    
    elif provider == 'gemini':
        if optimized:
            return (
                "You are an action recognition expert.\n\n"
                "TASK: Identify what action connects the RED box (person) to the BLUE box (object).\n\n"
                "FORMAT: [verb-ing] [object] (2-4 words)\n"
                "Examples: riding bicycle, holding cup, sitting on bench, tattooing needle\n\n"
                "Output: action phrase ONLY (lowercase, no quotes, no explanation)"
            )
        else:
            return (
                "You are an expert in human-object interaction recognition.\n\n"
                "TASK: Describe the action connecting the person (RED box) to the object (BLUE box).\n\n"
                "VISUAL GUIDE:\n"
                "- RED box: Contains the person/agent performing the action\n"
                "- BLUE box: Contains the object being acted upon\n\n"
                "OUTPUT REQUIREMENTS:\n"
                "1. Use present participle form (verb ending in -ing)\n"
                "2. Include the object name: [action-ing] [object]\n"
                "3. Keep it SHORT: 2-4 words maximum\n"
                "4. Use lowercase\n\n"
                "EXAMPLES:\n"
                "- riding bicycle\n"
                "- holding cup\n"
                "- sitting on bench\n"
                "- tattooing needle\n"
                "- boarding bus\n"
                "- putting envelope\n\n"
                "Output: Provide ONLY the action phrase (no quotes, no explanation, no punctuation)."
            )
    
    else:  # openai
        if optimized:
            return (
                "You are an action recognition assistant.\n\n"
                "Task: What action connects the RED box (person) to the BLUE box (object)?\n\n"
                "Format: [verb-ing] [object] (2-4 words, lowercase)\n"
                "Examples: riding bicycle, holding cup, sitting on bench, tattooing needle\n\n"
                "Output: action phrase ONLY"
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


def prepare_batch_requests(
    samples: List[Dict],
    img_prefix: str,
    processor: BatchProcessor,
    model: str,
    prompt: str,
    processed_ids: set = None,
    max_samples: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Prepare batch requests from samples.
    
    Args:
        samples: List of sample dictionaries
        img_prefix: Path prefix to images
        processor: BatchProcessor instance
        model: Model name
        prompt: Prompt text
        processed_ids: Set of already processed IDs to skip
        max_samples: Optional limit on number of samples
        
    Returns:
        List of prepared requests
    """
    if processed_ids is None:
        processed_ids = set()
    
    requests = []
    skipped = 0
    
    for idx, sample in enumerate(tqdm(samples, desc="Preparing requests")):
        if max_samples and len(requests) >= max_samples:
            break
        
        file_name = sample['file_name']
        custom_id = sanitize_custom_id(f"{idx}_{file_name}")
        
        # Skip already processed
        if custom_id in processed_ids:
            skipped += 1
            continue
        
        image_path = os.path.join(img_prefix, file_name)
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        
        # Get bounding boxes
        boxes = sample['boxes']
        person_bbox = boxes[sample['person_box_idx']]
        object_bbox = boxes[sample['object_box_idx']]
        
        # Build full prompt with bbox info
        from PIL import Image
        with Image.open(image_path) as img:
            img_width, img_height = img.size
        
        person_bbox_norm = normalize_bbox_to_1000(person_bbox, img_width, img_height)
        object_bbox_norm = normalize_bbox_to_1000(object_bbox, img_width, img_height)
        
        bbox_info = f"\nBounding boxes (in [0-1000] normalized coordinates):\n"
        bbox_info += f"- Red box (Person/Object 1): {person_bbox_norm}\n"
        bbox_info += f"- Blue box (Person/Object 2): {object_bbox_norm}\n\n"
        
        full_prompt = bbox_info + prompt
        
        # Prepare request
        request = processor.prepare_request(
            custom_id=custom_id,
            image_path=image_path,
            prompt=full_prompt,
            model=model,
            max_tokens=256,
            bboxes=[person_bbox, object_bbox],
            bbox_labels=['Person/Object 1', 'Person/Object 2']
        )
        
        # Store ground truth for later matching
        request["_metadata"] = {
            "sample_idx": idx,
            "file_name": file_name,
            "gt_action": sample['gt_action'],
            "person_bbox": person_bbox,
            "object_bbox": object_bbox
        }
        
        requests.append(request)
    
    if skipped > 0:
        print(f"Skipped {skipped} already processed samples")
    
    return requests


def process_batch_results(
    results: List[Dict[str, Any]],
    requests: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Process batch results and match with ground truth.
    
    Args:
        results: List of batch results
        requests: Original requests with metadata
        
    Returns:
        List of processed results with predictions and ground truth
    """
    # Create lookup from custom_id to metadata
    metadata_lookup = {}
    for req in requests:
        custom_id = req.get("custom_id") or req.get("key")
        if "_metadata" in req:
            metadata_lookup[custom_id] = req["_metadata"]
    
    processed = []
    for result in results:
        custom_id = result.get("custom_id", "unknown")
        metadata = metadata_lookup.get(custom_id, {})
        
        if result.get("status") == "success":
            prediction = clean_prediction_text(result.get("response", ""))
        else:
            prediction = ""
            print(f"Warning: Request {custom_id} failed: {result.get('error')}")
        
        processed.append({
            "image_id": metadata.get("sample_idx", 0),
            "image_path": metadata.get("file_name", ""),
            "prediction": prediction,
            "ground_truth": metadata.get("gt_action", ""),
            "person_bbox": metadata.get("person_bbox", []),
            "object_bbox": metadata.get("object_bbox", []),
            "custom_id": custom_id,
            "status": result.get("status", "unknown")
        })
    
    return processed


def compute_metrics(results: List[Dict[str, Any]], bertscore_model: str = None) -> Dict[str, Any]:
    """
    Compute evaluation metrics.
    
    Args:
        results: List of processed results
        bertscore_model: Optional BERTScore model name
        
    Returns:
        Dictionary of metrics
    """
    # Filter successful results
    valid_results = [r for r in results if r.get("status") == "success" or r.get("prediction")]
    
    if not valid_results:
        return {"error": "No valid results to evaluate"}
    
    # Prepare for COCO evaluation
    predictions_coco = [
        {'image_id': r['image_id'], 'caption': r['prediction']}
        for r in valid_results
    ]
    references_coco = [
        {'image_id': r['image_id'], 'caption': r['ground_truth']}
        for r in valid_results
    ]
    
    # Compute NLP metrics
    print("\nComputing NLP metrics...")
    nlp_metrics = compute_nlp_metrics(predictions_coco, references_coco)
    
    # Compute exact match
    exact_matches = sum(1 for r in valid_results 
                       if r['prediction'].lower().strip() == r['ground_truth'].lower().strip())
    nlp_metrics['exact_match'] = exact_matches / len(valid_results) if valid_results else 0
    
    # Compute BERTScore if model specified
    if bertscore_model:
        print(f"\nComputing BERTScore with {bertscore_model}...")
        predictions = [r['prediction'] for r in valid_results]
        references = [r['ground_truth'] for r in valid_results]
        
        bertscore = compute_bertscore(
            predictions=predictions,
            references=references,
            model_name=bertscore_model,
            batch_size=32,
            device='cpu',
            verbose=True
        )
        
        nlp_metrics['BERTScore_P'] = bertscore['precision']
        nlp_metrics['BERTScore_R'] = bertscore['recall']
        nlp_metrics['BERTScore_F1'] = bertscore['f1']
    
    return nlp_metrics


def main():
    parser = argparse.ArgumentParser(
        description="SWIG-HOI Action Referring Batch Evaluation"
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
    
    # Metrics options
    parser.add_argument('--bertscore-model', type=str, 
                        default='microsoft/deberta-v2-xxlarge-mnli',
                        help='BERTScore model')
    parser.add_argument('--skip-bertscore', action='store_true',
                        help='Skip BERTScore computation')
    
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
                'task': 'swig_action_referring',
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
    print("SWIG-HOI Action Referring Batch Evaluation")
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
            task="swig_action"
        )
        if incomplete:
            checkpoint = incomplete[0]
            print(f"Found incomplete job: {checkpoint.job_id}")
            print(f"  Progress: {len(checkpoint.processed_ids)}/{checkpoint.total_requests}")
            
            # Resume the batch job
            job = BatchJob(
                id=checkpoint.job_id,
                provider=args.provider,
                model=args.model,
                status=BatchStatus(checkpoint.status),
                created_at=datetime.fromisoformat(checkpoint.submitted_at),
                total_requests=checkpoint.total_requests,
                metadata=checkpoint.metadata
            )
            
            # Load cached requests if available
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
            latest = checkpoint_mgr.find_latest(provider=args.provider, task="swig_action")
            if latest:
                processed_ids = latest.get_processed_set()
                print(f"Found {len(processed_ids)} already processed samples")
        
        # Build prompt (provider-specific to match real-time scripts)
        prompt = build_action_referring_prompt(provider=args.provider, optimized=args.optimized_prompts)
        
        # Prepare requests
        print("\nPreparing batch requests...")
        requests = prepare_batch_requests(
            samples=samples,
            img_prefix=args.img_prefix,
            processor=processor,
            model=args.model,
            prompt=prompt,
            processed_ids=processed_ids,
            max_samples=args.max_samples
        )
        
        if not requests:
            print("No requests to process. All samples may already be complete.")
            return 0
        
        print(f"\nPrepared {len(requests)} requests for batch processing", flush=True)
        
        # Save requests for potential resume (metadata only, skip large base64 data)
        requests_file = os.path.join(args.output_dir, "batch_requests.json")
        print(f"Saving request metadata to {requests_file}...", flush=True)
        with open(requests_file, 'w') as f:
            # Only save metadata for resume, not the full request with base64 images
            requests_to_save = []
            for req in requests:
                # Save only metadata, not the full request (which contains large base64 images)
                req_copy = {'_metadata': req.get('_metadata', {}), 'custom_id': req.get('custom_id', req.get('key', ''))}
                requests_to_save.append(req_copy)
            json.dump(requests_to_save, f, indent=2)
        print("Request metadata saved.", flush=True)
        
        # Submit batch
        print("\nSubmitting batch job...", flush=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        display_name = f"swig_action_{args.provider}_{timestamp}"
        
        # Extract provider-specific requests (without metadata)
        batch_requests = []
        for req in requests:
            req_copy = {k: v for k, v in req.items() if not k.startswith('_')}
            batch_requests.append(req_copy)
        
        print(f"\nSubmitting batch with {len(batch_requests)} requests...", flush=True)
        
        job = processor.submit_batch(
            requests=batch_requests,
            model=args.model,
            display_name=display_name
        )
        
        print(f"Batch job submitted: {job.id}", flush=True)
        
        # Save initial checkpoint
        checkpoint = EvaluationCheckpoint(
            job_id=job.id,
            provider=args.provider,
            model=args.model,
            task="swig_action",
            status=job.status.value,
            submitted_at=datetime.now().isoformat(),
            total_requests=len(requests),
            metadata={"display_name": display_name}
        )
        checkpoint_mgr.save(checkpoint)
    
    # Wait for completion
    print("\nWaiting for batch completion...")
    print(f"(Polling every {args.poll_interval} seconds)")
    
    def on_status_update(updated_job: BatchJob):
        # Update checkpoint
        checkpoint = checkpoint_mgr.load(job.id)
        if checkpoint:
            checkpoint.status = updated_job.status.value
            checkpoint.last_updated = datetime.now().isoformat()
            checkpoint_mgr.save(checkpoint)
    
    try:
        job = wait_for_batch_completion(
            processor=processor,
            job=job,
            poll_interval=args.poll_interval,
            on_status_update=on_status_update
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted! Checkpoint saved. Use --resume to continue.")
        return 1
    
    # Check final status
    if job.status != BatchStatus.COMPLETED:
        print(f"\nBatch job ended with status: {job.status.value}")
        if job.error_message:
            print(f"Error: {job.error_message}")
        
        # Update checkpoint
        checkpoint = checkpoint_mgr.load(job.id)
        if checkpoint:
            checkpoint.status = job.status.value
            checkpoint.error_message = job.error_message
            checkpoint_mgr.save(checkpoint)
        
        return 1
    
    # Download results
    print("\nDownloading results...")
    results = processor.download_results(job)
    print(f"Downloaded {len(results)} results")
    
    # Process results
    print("\nProcessing results...")
    processed_results = process_batch_results(results, requests)
    
    # Generate visualizations if verbose
    if viz_dir and processed_results:
        print("\nGenerating visualizations...")
        viz_count = 0
        for result in tqdm(processed_results[:100], desc="Visualizations"):  # Limit to first 100
            try:
                image_path = os.path.join(args.img_prefix, result.get('image_path', ''))
                if not os.path.exists(image_path):
                    continue
                
                person_bbox = result.get('person_bbox', [0, 0, 100, 100])
                object_bbox = result.get('object_bbox', [0, 0, 100, 100])
                predicted = result.get('prediction', '')
                gt = result.get('ground_truth', '')
                
                # Create safe filename
                base_name = os.path.splitext(os.path.basename(result.get('image_path', 'img')))[0]
                viz_filename = f"{result.get('image_id', 0):05d}_{base_name}_viz.jpg"
                viz_path = os.path.join(viz_dir, viz_filename)
                
                visualize_action_triplet(
                    image_path=image_path,
                    person_bbox=person_bbox,
                    object_bbox=object_bbox,
                    predicted_action=predicted,
                    gt_action=gt,
                    object_category="object",
                    output_path=viz_path
                )
                viz_count += 1
            except Exception as e:
                pass  # Skip failed visualizations
        print(f"Generated {viz_count} visualizations in {viz_dir}")
    
    # Compute metrics
    bertscore_model = None if args.skip_bertscore else args.bertscore_model
    metrics = compute_metrics(processed_results, bertscore_model)
    
    # Print results
    print("\n" + "=" * 80)
    print("SWIG-HOI Action Referring Results (Batch)")
    print("=" * 80)
    print(f"{'Metric':<20} {'Score':>10}")
    print("-" * 40)
    
    for metric, score in metrics.items():
        if isinstance(score, float):
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
            'task': 'swig_action_referring',
            'total_samples': len(processed_results),
            'timestamp': timestamp,
            **metrics
        }
    )
    
    # Update checkpoint to completed
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
            columns = ["image", "prediction", "ground_truth", "match"]
            data = []
            for r in processed_results[:50]:  # Log first 50 samples
                match = r.get('prediction', '').lower().strip() == r.get('ground_truth', '').lower().strip()
                data.append([
                    r.get('image_path', ''),
                    r.get('prediction', ''),
                    r.get('ground_truth', ''),
                    match
                ])
            wandb.log({"predictions": wandb.Table(columns=columns, data=data)})
        
        wandb.finish()
        print("W&B run completed")
    
    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)
    print(f"Results: {output_file}")
    print(f"Metrics: {output_file.replace('.json', '_metrics.json')}")
    if viz_dir:
        print(f"Visualizations: {viz_dir}")
    print("=" * 80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
