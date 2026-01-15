#!/usr/bin/env python3
"""
Batch API Utilities for HOI Evaluation

This module provides unified batch processing interfaces for Claude, Gemini, and OpenAI APIs.
Batch processing offers 50% cost savings and uses separate rate limit pools.

Supported Providers:
- Claude (Anthropic): Message Batches API
- Gemini (Google): Batch API
- OpenAI: Batch API

Usage:
    from batch_api_utils import BatchProcessor
    
    processor = BatchProcessor(provider='claude')
    job = processor.submit_batch(requests, model='claude-sonnet-4-5-20250514')
    
    while not processor.is_complete(job):
        time.sleep(60)
        job = processor.poll_status(job)
    
    results = processor.download_results(job)
"""

import os
import io
import json
import time
import base64
import tempfile
from enum import Enum
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple, Callable
from PIL import Image

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


import re


def sanitize_custom_id(custom_id: str, max_length: int = 64) -> str:
    """
    Sanitize custom_id to be compatible with all batch APIs.
    
    Claude requires: ^[a-zA-Z0-9_-]{1,64}$
    This function removes invalid characters and truncates to max length.
    
    Args:
        custom_id: Original custom ID string
        max_length: Maximum length (default 64 for Claude)
        
    Returns:
        Sanitized custom ID string
    """
    # Replace dots and other invalid chars with underscores
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', custom_id)
    # Remove consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    # Truncate to max length
    return sanitized[:max_length]


class BatchStatus(Enum):
    """Unified batch job status across all providers."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


@dataclass
class BatchJob:
    """Represents a batch job with status tracking."""
    id: str
    provider: str  # 'claude', 'gemini', 'openai'
    model: str
    status: BatchStatus
    created_at: datetime
    total_requests: int
    completed_count: int = 0
    failed_count: int = 0
    results_url: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "provider": self.provider,
            "model": self.model,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "total_requests": self.total_requests,
            "completed_count": self.completed_count,
            "failed_count": self.failed_count,
            "results_url": self.results_url,
            "error_message": self.error_message,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BatchJob':
        """Create from dictionary."""
        return cls(
            id=data["id"],
            provider=data["provider"],
            model=data["model"],
            status=BatchStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            total_requests=data["total_requests"],
            completed_count=data.get("completed_count", 0),
            failed_count=data.get("failed_count", 0),
            results_url=data.get("results_url"),
            error_message=data.get("error_message"),
            metadata=data.get("metadata", {})
        )


class BatchProcessor:
    """
    Unified batch processing interface for Claude, Gemini, and OpenAI.
    
    Provides a consistent API for submitting, monitoring, and retrieving
    batch job results across all three providers.
    """
    
    def __init__(self, provider: str, api_key: Optional[str] = None):
        """
        Initialize BatchProcessor.
        
        Args:
            provider: One of 'claude', 'gemini', 'openai'
            api_key: Optional API key (uses environment variable if not provided)
        """
        self.provider = provider.lower()
        if self.provider not in ['claude', 'gemini', 'openai']:
            raise ValueError(f"Unsupported provider: {provider}")
        
        self.api_key = api_key or self._get_api_key()
        self._init_client()
    
    def _get_api_key(self) -> str:
        """Get API key from environment."""
        key_map = {
            'claude': ['ANTHROPIC_API_KEY', 'CLAUDE_API_KEY'],
            'gemini': ['GEMINI_API_KEY', 'GOOGLE_API_KEY'],
            'openai': ['OPENAI_API_KEY', 'OPEN_AI_API_KEY']
        }
        
        for key_name in key_map[self.provider]:
            key = os.environ.get(key_name)
            if key:
                return key
        
        raise ValueError(f"No API key found for {self.provider}. "
                        f"Set one of: {key_map[self.provider]}")
    
    def _init_client(self):
        """Initialize the appropriate client for the provider."""
        if self.provider == 'claude':
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic package required. Install with: pip install anthropic")
        
        elif self.provider == 'gemini':
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
            except ImportError:
                raise ImportError("google-genai package required. Install with: pip install google-genai")
        
        elif self.provider == 'openai':
            try:
                import openai
                self.client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai package required. Install with: pip install openai")
    
    def prepare_request(
        self,
        custom_id: str,
        image_path: str,
        prompt: str,
        model: str,
        max_tokens: int = 1024,
        bboxes: Optional[List[List[float]]] = None,
        bbox_labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Prepare a single request for batch submission.
        
        Args:
            custom_id: Unique identifier for this request
            image_path: Path to image file
            prompt: Text prompt
            model: Model name
            max_tokens: Maximum tokens in response
            bboxes: Optional list of bounding boxes to draw
            bbox_labels: Optional labels for bounding boxes
            
        Returns:
            Request dictionary in provider-specific format
        """
        # Encode image to base64
        img_base64 = self._encode_image(image_path, bboxes, bbox_labels)
        
        if self.provider == 'claude':
            return self._prepare_claude_request(custom_id, img_base64, prompt, model, max_tokens)
        elif self.provider == 'gemini':
            return self._prepare_gemini_request(custom_id, img_base64, prompt)
        else:  # openai
            return self._prepare_openai_request(custom_id, img_base64, prompt, model, max_tokens)
    
    def _encode_image(
        self,
        image_path: str,
        bboxes: Optional[List[List[float]]] = None,
        bbox_labels: Optional[List[str]] = None,
        max_size: int = 512
    ) -> str:
        """Encode image to base64, optionally drawing bounding boxes."""
        from PIL import Image, ImageDraw, ImageFont
        
        img = Image.open(image_path).convert('RGB')
        
        # Draw bounding boxes if provided
        if bboxes:
            draw = ImageDraw.Draw(img)
            colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
            
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
            except:
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
                except:
                    font = ImageFont.load_default()
            
            for idx, bbox in enumerate(bboxes):
                color = colors[idx % len(colors)]
                x1, y1, x2, y2 = bbox
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                
                if bbox_labels and idx < len(bbox_labels):
                    label = bbox_labels[idx]
                    draw.text((x1, max(0, y1 - 20)), label, fill=color, font=font)
        
        # Resize if needed
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Encode to base64
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=90)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def _prepare_claude_request(
        self,
        custom_id: str,
        img_base64: str,
        prompt: str,
        model: str,
        max_tokens: int
    ) -> Dict[str, Any]:
        """Prepare request for Claude Message Batches API."""
        return {
            "custom_id": custom_id,
            "params": {
                "model": model,
                "max_tokens": max_tokens,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": img_base64
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            }
        }
    
    def _prepare_gemini_request(
        self,
        custom_id: str,
        img_base64: str,
        prompt: str
    ) -> Dict[str, Any]:
        """Prepare request for Gemini Batch API."""
        return {
            "key": custom_id,
            "request": {
                "contents": [
                    {
                        "parts": [
                            {
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": img_base64
                                }
                            },
                            {
                                "text": prompt
                            }
                        ]
                    }
                ]
            }
        }
    
    def _prepare_openai_request(
        self,
        custom_id: str,
        img_base64: str,
        prompt: str,
        model: str,
        max_tokens: int
    ) -> Dict[str, Any]:
        """Prepare request for OpenAI Batch API."""
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "max_tokens": max_tokens,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_base64}",
                                    "detail": "high"
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            }
        }
    
    def submit_batch(
        self,
        requests: List[Dict[str, Any]],
        model: str,
        display_name: Optional[str] = None
    ) -> BatchJob:
        """
        Submit a batch of requests for processing.
        
        Args:
            requests: List of prepared request dictionaries
            model: Model name
            display_name: Optional name for the batch job
            
        Returns:
            BatchJob object with job ID and initial status
        """
        if self.provider == 'claude':
            return self._submit_claude_batch(requests, model, display_name)
        elif self.provider == 'gemini':
            return self._submit_gemini_batch(requests, model, display_name)
        else:  # openai
            return self._submit_openai_batch(requests, model, display_name)
    
    def _submit_claude_batch(
        self,
        requests: List[Dict[str, Any]],
        model: str,
        display_name: Optional[str] = None
    ) -> BatchJob:
        """Submit batch to Claude Message Batches API."""
        print(f"Submitting {len(requests)} requests to Claude Batch API...")
        
        # Claude has limits on batch size - split into chunks if needed
        # Max ~1000 requests per batch to avoid 502 errors with large image payloads
        MAX_REQUESTS_PER_BATCH = 1000
        
        if len(requests) <= MAX_REQUESTS_PER_BATCH:
            # Single batch submission
            batch = self.client.messages.batches.create(requests=requests)
            
            return BatchJob(
                id=batch.id,
                provider='claude',
                model=model,
                status=self._map_claude_status(batch.processing_status),
                created_at=datetime.now(),
                total_requests=len(requests),
                completed_count=batch.request_counts.succeeded,
                failed_count=batch.request_counts.errored,
                metadata={"display_name": display_name}
            )
        
        # Multi-batch submission for large datasets
        print(f"Large batch detected. Splitting into chunks of {MAX_REQUESTS_PER_BATCH}...")
        batch_ids = []
        total_chunks = (len(requests) + MAX_REQUESTS_PER_BATCH - 1) // MAX_REQUESTS_PER_BATCH
        
        for i in range(0, len(requests), MAX_REQUESTS_PER_BATCH):
            chunk = requests[i:i + MAX_REQUESTS_PER_BATCH]
            chunk_num = i // MAX_REQUESTS_PER_BATCH + 1
            print(f"  Submitting chunk {chunk_num}/{total_chunks} ({len(chunk)} requests)...", flush=True)
            
            try:
                batch = self.client.messages.batches.create(requests=chunk)
                batch_ids.append(batch.id)
                print(f"    Batch submitted: {batch.id}", flush=True)
            except Exception as e:
                print(f"    Error submitting chunk {chunk_num}: {e}")
                # If we have some successful batches, return what we have
                if batch_ids:
                    break
                raise
        
        print(f"Submitted {len(batch_ids)} batch jobs", flush=True)
        
        # Return a BatchJob that tracks all sub-batches
        return BatchJob(
            id=batch_ids[0],  # Primary batch ID
            provider='claude',
            model=model,
            status=BatchStatus.IN_PROGRESS,
            created_at=datetime.now(),
            total_requests=len(requests),
            completed_count=0,
            failed_count=0,
            metadata={
                "display_name": display_name,
                "batch_ids": batch_ids,  # Track all batch IDs
                "is_multi_batch": True,
                "total_batches": len(batch_ids)
            }
        )
    
    def _submit_gemini_batch(
        self,
        requests: List[Dict[str, Any]],
        model: str,
        display_name: Optional[str] = None
    ) -> BatchJob:
        """Submit batch to Gemini Batch API."""
        print(f"Submitting {len(requests)} requests to Gemini Batch API...")
        
        # For large batches (>200 requests), skip expensive size calculation and use file upload directly
        # Each request with base64 image is ~50-100KB, so 200 requests ≈ 10-20MB
        INLINE_REQUEST_THRESHOLD = 200
        
        if len(requests) > INLINE_REQUEST_THRESHOLD:
            print(f"Large batch ({len(requests)} requests > {INLINE_REQUEST_THRESHOLD}), using file upload...")
            return self._submit_gemini_batch_file(requests, model, display_name)
        
        # For small batches, calculate size to decide inline vs file upload
        print("Calculating total request size...")
        total_size = sum(len(json.dumps(r)) for r in requests)
        print(f"Total request size: {total_size / (1024*1024):.2f} MB")
        
        if total_size > 15_000_000:  # ~15MB, leave margin
            # Use file upload for large batches
            print(f"Using file upload for large batch ({total_size / (1024*1024):.2f} MB > 15 MB)")
            return self._submit_gemini_batch_file(requests, model, display_name)
        
        # Use inline requests
        inline_requests = [r["request"] for r in requests]
        
        batch_job = self.client.batches.create(
            model=f"models/{model}",
            src=inline_requests,
            config={
                "display_name": display_name or f"hoi-eval-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            }
        )
        
        return BatchJob(
            id=batch_job.name,
            provider='gemini',
            model=model,
            status=self._map_gemini_status(batch_job.state.name if hasattr(batch_job.state, 'name') else str(batch_job.state)),
            created_at=datetime.now(),
            total_requests=len(requests),
            metadata={
                "display_name": display_name,
                "request_keys": [r["key"] for r in requests]
            }
        )
    
    def _submit_gemini_batch_file(
        self,
        requests: List[Dict[str, Any]],
        model: str,
        display_name: Optional[str] = None
    ) -> BatchJob:
        """Submit large batch to Gemini using file upload."""
        from google.genai import types
        
        # Create JSONL file
        print(f"Creating JSONL file with {len(requests)} requests...")
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for idx, req in enumerate(requests):
                f.write(json.dumps(req) + "\n")
                if idx % 1000 == 0:
                    print(f"  Writing request {idx}/{len(requests)}...", flush=True)
            jsonl_path = f.name
        print(f"JSONL file created: {jsonl_path} ({os.path.getsize(jsonl_path) / (1024*1024):.2f} MB)", flush=True)
        
        try:
            # Upload file
            print("Uploading JSONL file to Google...", flush=True)
            uploaded_file = self.client.files.upload(
                file=jsonl_path,
                config=types.UploadFileConfig(
                    display_name=display_name or 'batch-requests',
                    mime_type='jsonl'
                )
            )
            print(f"File uploaded: {uploaded_file.name}", flush=True)
            
            # Create batch job with file
            print("Creating batch job...", flush=True)
            batch_job = self.client.batches.create(
                model=f"models/{model}",
                src=uploaded_file.name,
                config={
                    "display_name": display_name or f"hoi-eval-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                }
            )
            print(f"Batch job created: {batch_job.name}", flush=True)
            
            # Don't store all request keys in metadata for large batches (too much memory)
            request_keys_sample = [r["key"] for r in requests[:100]] if len(requests) > 100 else [r["key"] for r in requests]
            
            return BatchJob(
                id=batch_job.name,
                provider='gemini',
                model=model,
                status=self._map_gemini_status(batch_job.state.name if hasattr(batch_job.state, 'name') else str(batch_job.state)),
                created_at=datetime.now(),
                total_requests=len(requests),
                metadata={
                    "display_name": display_name,
                    "input_file": uploaded_file.name,
                    "request_keys_sample": request_keys_sample  # Only store sample, not all 19K keys
                }
            )
        finally:
            os.unlink(jsonl_path)
    
    def _submit_openai_batch(
        self,
        requests: List[Dict[str, Any]],
        model: str,
        display_name: Optional[str] = None
    ) -> BatchJob:
        """Submit batch to OpenAI Batch API."""
        print(f"Submitting {len(requests)} requests to OpenAI Batch API...")
        
        # Create JSONL file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for req in requests:
                f.write(json.dumps(req) + "\n")
            jsonl_path = f.name
        
        try:
            # Upload file
            with open(jsonl_path, 'rb') as f:
                batch_file = self.client.files.create(file=f, purpose="batch")
            
            # Create batch
            batch = self.client.batches.create(
                input_file_id=batch_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={"display_name": display_name or "hoi-eval"}
            )
            
            return BatchJob(
                id=batch.id,
                provider='openai',
                model=model,
                status=self._map_openai_status(batch.status),
                created_at=datetime.now(),
                total_requests=len(requests),
                metadata={
                    "display_name": display_name,
                    "input_file_id": batch_file.id
                }
            )
        finally:
            os.unlink(jsonl_path)
    
    def poll_status(self, job: BatchJob) -> BatchJob:
        """
        Poll the status of a batch job.
        
        Args:
            job: BatchJob to check
            
        Returns:
            Updated BatchJob with current status
        """
        if self.provider == 'claude':
            return self._poll_claude_status(job)
        elif self.provider == 'gemini':
            return self._poll_gemini_status(job)
        else:  # openai
            return self._poll_openai_status(job)
    
    def _poll_claude_status(self, job: BatchJob) -> BatchJob:
        """Poll Claude batch status."""
        # Handle multi-batch jobs
        if job.metadata and job.metadata.get("is_multi_batch"):
            batch_ids = job.metadata.get("batch_ids", [job.id])
            total_completed = 0
            total_failed = 0
            all_completed = True
            any_failed = False
            results_urls = []
            
            for batch_id in batch_ids:
                batch = self.client.messages.batches.retrieve(batch_id)
                status = self._map_claude_status(batch.processing_status)
                
                total_completed += batch.request_counts.succeeded
                total_failed += batch.request_counts.errored
                
                if status == BatchStatus.FAILED:
                    any_failed = True
                if status not in [BatchStatus.COMPLETED, BatchStatus.FAILED]:
                    all_completed = False
                if batch.results_url:
                    results_urls.append(batch.results_url)
            
            job.completed_count = total_completed
            job.failed_count = total_failed
            job.metadata["results_urls"] = results_urls
            
            if any_failed:
                job.status = BatchStatus.FAILED
            elif all_completed:
                job.status = BatchStatus.COMPLETED
            else:
                job.status = BatchStatus.IN_PROGRESS
            
            return job
        
        # Single batch
        batch = self.client.messages.batches.retrieve(job.id)
        
        job.status = self._map_claude_status(batch.processing_status)
        job.completed_count = batch.request_counts.succeeded
        job.failed_count = batch.request_counts.errored
        job.results_url = batch.results_url
        
        return job
    
    def _poll_gemini_status(self, job: BatchJob) -> BatchJob:
        """Poll Gemini batch status."""
        batch = self.client.batches.get(name=job.id)
        
        state_name = batch.state.name if hasattr(batch.state, 'name') else str(batch.state)
        job.status = self._map_gemini_status(state_name)
        
        # Update counts if available
        if hasattr(batch, 'batch_stats') and batch.batch_stats:
            job.completed_count = getattr(batch.batch_stats, 'successful_request_count', 0) or 0
            job.failed_count = getattr(batch.batch_stats, 'failed_request_count', 0) or 0
        
        return job
    
    def _poll_openai_status(self, job: BatchJob) -> BatchJob:
        """Poll OpenAI batch status."""
        batch = self.client.batches.retrieve(job.id)
        
        job.status = self._map_openai_status(batch.status)
        job.completed_count = batch.request_counts.completed if batch.request_counts else 0
        job.failed_count = batch.request_counts.failed if batch.request_counts else 0
        
        if batch.output_file_id:
            job.metadata["output_file_id"] = batch.output_file_id
        if batch.error_file_id:
            job.metadata["error_file_id"] = batch.error_file_id
        
        return job
    
    def is_complete(self, job: BatchJob) -> bool:
        """Check if batch job has reached a terminal state."""
        return job.status in [
            BatchStatus.COMPLETED,
            BatchStatus.FAILED,
            BatchStatus.CANCELLED,
            BatchStatus.EXPIRED
        ]
    
    def download_results(self, job: BatchJob) -> List[Dict[str, Any]]:
        """
        Download results from a completed batch job.
        
        Args:
            job: Completed BatchJob
            
        Returns:
            List of result dictionaries with custom_id and response
        """
        if job.status != BatchStatus.COMPLETED:
            raise ValueError(f"Cannot download results for job with status: {job.status}")
        
        if self.provider == 'claude':
            return self._download_claude_results(job)
        elif self.provider == 'gemini':
            return self._download_gemini_results(job)
        else:  # openai
            return self._download_openai_results(job)
    
    def _download_claude_results(self, job: BatchJob) -> List[Dict[str, Any]]:
        """Download Claude batch results."""
        results = []
        
        # Handle multi-batch jobs
        if job.metadata and job.metadata.get("is_multi_batch"):
            batch_ids = job.metadata.get("batch_ids", [job.id])
            print(f"Downloading results from {len(batch_ids)} batches...", flush=True)
            
            for idx, batch_id in enumerate(batch_ids):
                print(f"  Downloading batch {idx + 1}/{len(batch_ids)}: {batch_id}...", flush=True)
                for result in self.client.messages.batches.results(batch_id):
                    if result.result.type == "succeeded":
                        text = ""
                        for block in result.result.message.content:
                            if hasattr(block, 'text'):
                                text += block.text
                        
                        results.append({
                            "custom_id": result.custom_id,
                            "status": "success",
                            "response": text,
                            "usage": {
                                "input_tokens": result.result.message.usage.input_tokens,
                                "output_tokens": result.result.message.usage.output_tokens
                            }
                        })
                    else:
                        results.append({
                            "custom_id": result.custom_id,
                            "status": "error",
                            "error": str(result.result.error) if hasattr(result.result, 'error') else result.result.type
                        })
            
            return results
        
        # Single batch
        for result in self.client.messages.batches.results(job.id):
            if result.result.type == "succeeded":
                # Extract text from response
                text = ""
                for block in result.result.message.content:
                    if hasattr(block, 'text'):
                        text += block.text
                
                results.append({
                    "custom_id": result.custom_id,
                    "status": "success",
                    "response": text,
                    "usage": {
                        "input_tokens": result.result.message.usage.input_tokens,
                        "output_tokens": result.result.message.usage.output_tokens
                    }
                })
            else:
                results.append({
                    "custom_id": result.custom_id,
                    "status": "error",
                    "error": str(result.result.error) if hasattr(result.result, 'error') else result.result.type
                })
        
        return results
    
    def _download_gemini_results(self, job: BatchJob) -> List[Dict[str, Any]]:
        """Download Gemini batch results."""
        results = []
        batch = self.client.batches.get(name=job.id)
        
        # Check for inline responses
        if batch.dest and hasattr(batch.dest, 'inlined_responses') and batch.dest.inlined_responses:
            for i, inline_response in enumerate(batch.dest.inlined_responses):
                key = job.metadata.get("request_keys", [f"request-{i}"])[i] if i < len(job.metadata.get("request_keys", [])) else f"request-{i}"
                
                if inline_response.response:
                    try:
                        text = inline_response.response.text
                    except:
                        text = str(inline_response.response)
                    
                    results.append({
                        "custom_id": key,
                        "status": "success",
                        "response": text
                    })
                elif hasattr(inline_response, 'error') and inline_response.error:
                    results.append({
                        "custom_id": key,
                        "status": "error",
                        "error": str(inline_response.error)
                    })
        
        # Check for file-based results
        elif batch.dest and hasattr(batch.dest, 'file_name') and batch.dest.file_name:
            file_content = self.client.files.download(file=batch.dest.file_name)
            content = file_content.decode('utf-8')
            
            for line in content.splitlines():
                if line.strip():
                    parsed = json.loads(line)
                    key = parsed.get("key", "unknown")
                    
                    if "response" in parsed and parsed["response"]:
                        try:
                            text = parsed["response"]["candidates"][0]["content"]["parts"][0]["text"]
                        except (KeyError, IndexError):
                            text = str(parsed["response"])
                        
                        results.append({
                            "custom_id": key,
                            "status": "success",
                            "response": text
                        })
                    elif "error" in parsed:
                        results.append({
                            "custom_id": key,
                            "status": "error",
                            "error": str(parsed["error"])
                        })
        
        return results
    
    def _download_openai_results(self, job: BatchJob) -> List[Dict[str, Any]]:
        """Download OpenAI batch results."""
        results = []
        
        output_file_id = job.metadata.get("output_file_id")
        if not output_file_id:
            # Re-poll to get output file ID
            batch = self.client.batches.retrieve(job.id)
            output_file_id = batch.output_file_id
        
        if output_file_id:
            content = self.client.files.content(output_file_id)
            
            for line in content.text.splitlines():
                if line.strip():
                    parsed = json.loads(line)
                    custom_id = parsed.get("custom_id", "unknown")
                    
                    if parsed.get("response") and parsed["response"].get("body"):
                        body = parsed["response"]["body"]
                        if body.get("choices"):
                            text = body["choices"][0]["message"]["content"]
                            results.append({
                                "custom_id": custom_id,
                                "status": "success",
                                "response": text,
                                "usage": body.get("usage", {})
                            })
                        else:
                            results.append({
                                "custom_id": custom_id,
                                "status": "error",
                                "error": "No choices in response"
                            })
                    elif parsed.get("error"):
                        results.append({
                            "custom_id": custom_id,
                            "status": "error",
                            "error": str(parsed["error"])
                        })
        
        return results
    
    # Status mapping helpers
    def _map_claude_status(self, status: str) -> BatchStatus:
        """Map Claude status to unified BatchStatus."""
        mapping = {
            "in_progress": BatchStatus.IN_PROGRESS,
            "canceling": BatchStatus.IN_PROGRESS,
            "ended": BatchStatus.COMPLETED,
        }
        return mapping.get(status, BatchStatus.PENDING)
    
    def _map_gemini_status(self, status: str) -> BatchStatus:
        """Map Gemini status to unified BatchStatus."""
        mapping = {
            "JOB_STATE_PENDING": BatchStatus.PENDING,
            "JOB_STATE_RUNNING": BatchStatus.IN_PROGRESS,
            "JOB_STATE_SUCCEEDED": BatchStatus.COMPLETED,
            "JOB_STATE_FAILED": BatchStatus.FAILED,
            "JOB_STATE_CANCELLED": BatchStatus.CANCELLED,
            "JOB_STATE_EXPIRED": BatchStatus.EXPIRED,
            "BATCH_STATE_PENDING": BatchStatus.PENDING,
            "BATCH_STATE_RUNNING": BatchStatus.IN_PROGRESS,
        }
        return mapping.get(status, BatchStatus.PENDING)
    
    def _map_openai_status(self, status: str) -> BatchStatus:
        """Map OpenAI status to unified BatchStatus."""
        mapping = {
            "validating": BatchStatus.PENDING,
            "in_progress": BatchStatus.IN_PROGRESS,
            "finalizing": BatchStatus.IN_PROGRESS,
            "completed": BatchStatus.COMPLETED,
            "failed": BatchStatus.FAILED,
            "cancelled": BatchStatus.CANCELLED,
            "expired": BatchStatus.EXPIRED,
        }
        return mapping.get(status, BatchStatus.PENDING)
    
    def cancel_batch(self, job: BatchJob) -> BatchJob:
        """
        Cancel a batch job.
        
        Args:
            job: BatchJob to cancel
            
        Returns:
            Updated BatchJob with cancelled status
        """
        if self.provider == 'claude':
            self.client.messages.batches.cancel(job.id)
        elif self.provider == 'gemini':
            self.client.batches.cancel(name=job.id)
        elif self.provider == 'openai':
            self.client.batches.cancel(job.id)
        
        job.status = BatchStatus.CANCELLED
        return job


def wait_for_batch_completion(
    processor: BatchProcessor,
    job: BatchJob,
    poll_interval: int = 60,
    timeout: Optional[int] = None,
    on_status_update: Optional[Callable[[BatchJob], None]] = None
) -> BatchJob:
    """
    Wait for a batch job to complete with optional callback.
    
    Args:
        processor: BatchProcessor instance
        job: BatchJob to monitor
        poll_interval: Seconds between status checks
        timeout: Optional timeout in seconds
        on_status_update: Optional callback function called on each status update
        
    Returns:
        Completed BatchJob
    """
    start_time = time.time()
    
    while not processor.is_complete(job):
        if timeout and (time.time() - start_time) > timeout:
            raise TimeoutError(f"Batch job {job.id} did not complete within {timeout} seconds")
        
        time.sleep(poll_interval)
        job = processor.poll_status(job)
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Status: {job.status.value}, "
              f"Progress: {job.completed_count}/{job.total_requests}")
        
        if on_status_update:
            on_status_update(job)
    
    return job


if __name__ == '__main__':
    # Example usage
    print("Batch API Utils - Example")
    print("=" * 50)
    
    # This is just for testing the module can be imported
    print("Available providers: claude, gemini, openai")
    print("BatchStatus values:", [s.value for s in BatchStatus])
    print("\nTo use:")
    print("  processor = BatchProcessor(provider='claude')")
    print("  job = processor.submit_batch(requests, model='claude-sonnet-4-5-20250514')")
