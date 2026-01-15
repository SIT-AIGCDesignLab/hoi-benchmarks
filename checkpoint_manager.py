#!/usr/bin/env python3
"""
Checkpoint Manager for Batch Evaluation

Provides robust checkpointing and resume capability for batch evaluation runs.
Supports atomic writes to prevent data corruption on interruption.

Usage:
    from checkpoint_manager import CheckpointManager
    
    checkpoint_mgr = CheckpointManager(checkpoint_dir="./checkpoints")
    
    # Save checkpoint
    checkpoint_mgr.save(job_id, {
        "provider": "claude",
        "model": "claude-sonnet-4-5",
        "status": "in_progress",
        "total_requests": 1000,
        "processed_ids": ["id1", "id2", ...]
    })
    
    # Load checkpoint
    checkpoint = checkpoint_mgr.load(job_id)
    
    # Find incomplete jobs
    incomplete = checkpoint_mgr.find_incomplete(provider="claude")
"""

import os
import json
import glob
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field, asdict


@dataclass
class EvaluationCheckpoint:
    """Represents a checkpoint for a batch evaluation run."""
    job_id: str
    provider: str
    model: str
    task: str  # e.g., 'swig_action', 'swig_ground', 'hico_action', 'hico_ground'
    status: str
    submitted_at: str
    total_requests: int
    processed_ids: List[str] = field(default_factory=list)
    results: List[Dict[str, Any]] = field(default_factory=list)
    last_updated: str = ""
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationCheckpoint':
        """Create from dictionary."""
        return cls(
            job_id=data["job_id"],
            provider=data["provider"],
            model=data["model"],
            task=data.get("task", "unknown"),
            status=data["status"],
            submitted_at=data["submitted_at"],
            total_requests=data["total_requests"],
            processed_ids=data.get("processed_ids", []),
            results=data.get("results", []),
            last_updated=data.get("last_updated", ""),
            error_message=data.get("error_message"),
            metadata=data.get("metadata", {})
        )
    
    def get_processed_set(self) -> Set[str]:
        """Get set of processed IDs for fast lookup."""
        return set(self.processed_ids)
    
    def add_result(self, custom_id: str, result: Dict[str, Any]):
        """Add a result and mark as processed."""
        if custom_id not in self.processed_ids:
            self.processed_ids.append(custom_id)
        self.results.append(result)
        self.last_updated = datetime.now().isoformat()


class CheckpointManager:
    """
    Manages checkpoints for batch evaluation runs.
    
    Features:
    - Atomic writes (prevents corruption on crash)
    - Resume capability
    - Find incomplete jobs by provider/task
    """
    
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        """
        Initialize CheckpointManager.
        
        Args:
            checkpoint_dir: Directory to store checkpoint files
        """
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def _get_checkpoint_path(self, job_id: str) -> str:
        """Get path to checkpoint file for a job."""
        # Sanitize job_id for filesystem
        safe_id = job_id.replace("/", "_").replace("\\", "_")
        return os.path.join(self.checkpoint_dir, f"{safe_id}.json")
    
    def _get_temp_path(self, job_id: str) -> str:
        """Get path to temporary checkpoint file."""
        safe_id = job_id.replace("/", "_").replace("\\", "_")
        return os.path.join(self.checkpoint_dir, f"{safe_id}.tmp")
    
    def save(self, checkpoint: EvaluationCheckpoint) -> None:
        """
        Atomically save checkpoint to disk.
        
        Uses write-to-temp + rename pattern to prevent corruption.
        
        Args:
            checkpoint: EvaluationCheckpoint to save
        """
        checkpoint.last_updated = datetime.now().isoformat()
        
        tmp_path = self._get_temp_path(checkpoint.job_id)
        final_path = self._get_checkpoint_path(checkpoint.job_id)
        
        try:
            # Write to temporary file
            with open(tmp_path, 'w') as f:
                json.dump(checkpoint.to_dict(), f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            
            # Atomic rename
            os.replace(tmp_path, final_path)
            
        except Exception as e:
            # Clean up temp file on failure
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except:
                    pass
            raise RuntimeError(f"Failed to save checkpoint for {checkpoint.job_id}: {e}")
    
    def save_dict(self, job_id: str, state: Dict[str, Any]) -> None:
        """
        Save checkpoint from dictionary.
        
        Args:
            job_id: Job identifier
            state: State dictionary
        """
        state["job_id"] = job_id
        state.setdefault("task", "unknown")
        state.setdefault("submitted_at", datetime.now().isoformat())
        state.setdefault("processed_ids", [])
        state.setdefault("results", [])
        
        checkpoint = EvaluationCheckpoint.from_dict(state)
        self.save(checkpoint)
    
    def load(self, job_id: str) -> Optional[EvaluationCheckpoint]:
        """
        Load checkpoint from disk.
        
        Args:
            job_id: Job identifier
            
        Returns:
            EvaluationCheckpoint if exists, None otherwise
        """
        path = self._get_checkpoint_path(job_id)
        
        if not os.path.exists(path):
            return None
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            return EvaluationCheckpoint.from_dict(data)
        except (json.JSONDecodeError, IOError, KeyError) as e:
            print(f"Warning: Failed to load checkpoint {job_id}: {e}")
            return None
    
    def load_dict(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint as dictionary.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Checkpoint dictionary if exists, None otherwise
        """
        checkpoint = self.load(job_id)
        return checkpoint.to_dict() if checkpoint else None
    
    def delete(self, job_id: str) -> bool:
        """
        Delete a checkpoint.
        
        Args:
            job_id: Job identifier
            
        Returns:
            True if deleted, False if not found
        """
        path = self._get_checkpoint_path(job_id)
        
        if os.path.exists(path):
            try:
                os.remove(path)
                return True
            except IOError:
                return False
        return False
    
    def exists(self, job_id: str) -> bool:
        """Check if a checkpoint exists."""
        return os.path.exists(self._get_checkpoint_path(job_id))
    
    def find_incomplete(
        self,
        provider: Optional[str] = None,
        task: Optional[str] = None
    ) -> List[EvaluationCheckpoint]:
        """
        Find all incomplete batch jobs.
        
        Args:
            provider: Optional filter by provider ('claude', 'gemini', 'openai')
            task: Optional filter by task ('swig_action', 'swig_ground', etc.)
            
        Returns:
            List of incomplete EvaluationCheckpoint objects
        """
        incomplete = []
        terminal_statuses = {"completed", "failed", "cancelled", "expired"}
        
        for filepath in glob.glob(os.path.join(self.checkpoint_dir, "*.json")):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                status = data.get("status", "").lower()
                if status in terminal_statuses:
                    continue
                
                # Apply filters
                if provider and data.get("provider") != provider:
                    continue
                if task and data.get("task") != task:
                    continue
                
                incomplete.append(EvaluationCheckpoint.from_dict(data))
                
            except (json.JSONDecodeError, IOError, KeyError):
                continue
        
        # Sort by submission time (oldest first)
        incomplete.sort(key=lambda c: c.submitted_at)
        
        return incomplete
    
    def find_latest(
        self,
        provider: Optional[str] = None,
        task: Optional[str] = None
    ) -> Optional[EvaluationCheckpoint]:
        """
        Find the most recent checkpoint.
        
        Args:
            provider: Optional filter by provider
            task: Optional filter by task
            
        Returns:
            Most recent EvaluationCheckpoint or None
        """
        all_checkpoints = []
        
        for filepath in glob.glob(os.path.join(self.checkpoint_dir, "*.json")):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                # Apply filters
                if provider and data.get("provider") != provider:
                    continue
                if task and data.get("task") != task:
                    continue
                
                all_checkpoints.append(EvaluationCheckpoint.from_dict(data))
                
            except (json.JSONDecodeError, IOError, KeyError):
                continue
        
        if not all_checkpoints:
            return None
        
        # Sort by last_updated (newest first)
        all_checkpoints.sort(key=lambda c: c.last_updated or c.submitted_at, reverse=True)
        
        return all_checkpoints[0]
    
    def list_all(self) -> List[Dict[str, Any]]:
        """
        List all checkpoints with summary info.
        
        Returns:
            List of checkpoint summaries
        """
        summaries = []
        
        for filepath in glob.glob(os.path.join(self.checkpoint_dir, "*.json")):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                summaries.append({
                    "job_id": data.get("job_id"),
                    "provider": data.get("provider"),
                    "model": data.get("model"),
                    "task": data.get("task"),
                    "status": data.get("status"),
                    "progress": f"{len(data.get('processed_ids', []))}/{data.get('total_requests', 0)}",
                    "submitted_at": data.get("submitted_at"),
                    "last_updated": data.get("last_updated")
                })
                
            except (json.JSONDecodeError, IOError):
                continue
        
        # Sort by last_updated
        summaries.sort(key=lambda s: s.get("last_updated", ""), reverse=True)
        
        return summaries
    
    def cleanup_completed(self, days_old: int = 7) -> int:
        """
        Remove completed checkpoints older than specified days.
        
        Args:
            days_old: Remove completed checkpoints older than this many days
            
        Returns:
            Number of checkpoints removed
        """
        from datetime import timedelta
        
        removed = 0
        cutoff = datetime.now() - timedelta(days=days_old)
        
        for filepath in glob.glob(os.path.join(self.checkpoint_dir, "*.json")):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                if data.get("status") != "completed":
                    continue
                
                last_updated = data.get("last_updated", data.get("submitted_at", ""))
                if last_updated:
                    checkpoint_time = datetime.fromisoformat(last_updated)
                    if checkpoint_time < cutoff:
                        os.remove(filepath)
                        removed += 1
                        
            except (json.JSONDecodeError, IOError, ValueError):
                continue
        
        return removed
    
    def create_backup(self, backup_dir: Optional[str] = None) -> str:
        """
        Create a backup of all checkpoints.
        
        Args:
            backup_dir: Optional backup directory (default: checkpoints_backup_TIMESTAMP)
            
        Returns:
            Path to backup directory
        """
        if backup_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = f"{self.checkpoint_dir}_backup_{timestamp}"
        
        shutil.copytree(self.checkpoint_dir, backup_dir)
        return backup_dir


class IncrementalResultSaver:
    """
    Saves results incrementally during evaluation.
    
    Useful for real-time processing where results come in one at a time.
    """
    
    def __init__(
        self,
        checkpoint_mgr: CheckpointManager,
        job_id: str,
        provider: str,
        model: str,
        task: str,
        total_requests: int,
        save_interval: int = 10
    ):
        """
        Initialize IncrementalResultSaver.
        
        Args:
            checkpoint_mgr: CheckpointManager instance
            job_id: Job identifier
            provider: Provider name
            model: Model name
            task: Task name
            total_requests: Total number of requests
            save_interval: Save checkpoint every N results
        """
        self.checkpoint_mgr = checkpoint_mgr
        self.job_id = job_id
        self.save_interval = save_interval
        self.results_since_save = 0
        
        # Load existing checkpoint or create new one
        existing = checkpoint_mgr.load(job_id)
        if existing:
            self.checkpoint = existing
        else:
            self.checkpoint = EvaluationCheckpoint(
                job_id=job_id,
                provider=provider,
                model=model,
                task=task,
                status="in_progress",
                submitted_at=datetime.now().isoformat(),
                total_requests=total_requests
            )
            checkpoint_mgr.save(self.checkpoint)
    
    def add_result(self, custom_id: str, result: Dict[str, Any], force_save: bool = False):
        """
        Add a result and optionally save checkpoint.
        
        Args:
            custom_id: Request identifier
            result: Result dictionary
            force_save: Force save even if interval not reached
        """
        self.checkpoint.add_result(custom_id, result)
        self.results_since_save += 1
        
        if force_save or self.results_since_save >= self.save_interval:
            self.save()
            self.results_since_save = 0
    
    def save(self):
        """Save current checkpoint."""
        self.checkpoint_mgr.save(self.checkpoint)
    
    def complete(self, status: str = "completed"):
        """Mark evaluation as complete and save final checkpoint."""
        self.checkpoint.status = status
        self.checkpoint.last_updated = datetime.now().isoformat()
        self.checkpoint_mgr.save(self.checkpoint)
    
    def get_processed_ids(self) -> Set[str]:
        """Get set of already processed IDs."""
        return self.checkpoint.get_processed_set()
    
    def get_results(self) -> List[Dict[str, Any]]:
        """Get all results collected so far."""
        return self.checkpoint.results


if __name__ == '__main__':
    # Example usage
    print("Checkpoint Manager - Example")
    print("=" * 50)
    
    # Initialize
    mgr = CheckpointManager(checkpoint_dir="./test_checkpoints")
    
    # Create a test checkpoint
    checkpoint = EvaluationCheckpoint(
        job_id="test-job-001",
        provider="claude",
        model="claude-sonnet-4-5",
        task="swig_action",
        status="in_progress",
        submitted_at=datetime.now().isoformat(),
        total_requests=100
    )
    
    # Add some results
    checkpoint.add_result("req-1", {"custom_id": "req-1", "response": "test response 1"})
    checkpoint.add_result("req-2", {"custom_id": "req-2", "response": "test response 2"})
    
    # Save
    mgr.save(checkpoint)
    print(f"Saved checkpoint: {checkpoint.job_id}")
    
    # Load
    loaded = mgr.load("test-job-001")
    print(f"Loaded checkpoint: {loaded.job_id}, processed: {len(loaded.processed_ids)}")
    
    # List all
    print(f"\nAll checkpoints: {mgr.list_all()}")
    
    # Find incomplete
    incomplete = mgr.find_incomplete(provider="claude")
    print(f"Incomplete Claude jobs: {[c.job_id for c in incomplete]}")
    
    # Cleanup
    mgr.delete("test-job-001")
    print("\nTest checkpoint deleted")
    
    # Remove test directory
    import shutil
    shutil.rmtree("./test_checkpoints", ignore_errors=True)
