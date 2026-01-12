#!/usr/bin/env python3
"""
Response Cache System for API-Based Evaluations

Caches API responses to avoid redundant API calls and reduce costs by 50-100% on reruns.

Usage:
    from response_cache import ResponseCache

    cache = ResponseCache(cache_dir="./cache")

    # Check cache before API call
    cached = cache.get(image_id, action, object_name, model, prompt_hash)
    if cached:
        use cached response
    else:
        response = call_api(...)
        cache.set(image_id, action, object_name, model, prompt_hash, response)
"""

import hashlib
import json
import os
from typing import Optional, Dict, Any


class ResponseCache:
    """Cache API responses to avoid redundant API calls."""

    def __init__(self, cache_dir="./cache"):
        """
        Initialize response cache.

        Args:
            cache_dir: Directory to store cached responses (default: ./cache)
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _generate_key(self, image_id: str, action: str, object_name: str,
                      model: str, prompt_hash: str) -> str:
        """
        Generate unique cache key from inputs.

        Args:
            image_id: Image identifier
            action: Action being performed (or "" for grounding)
            object_name: Object category (or "" for action referring)
            model: Model name (e.g., "claude-sonnet-4.5")
            prompt_hash: Hash of the prompt text

        Returns:
            MD5 hash string as cache key
        """
        # Normalize empty strings
        action = action or ""
        object_name = object_name or ""

        # Create deterministic key from all inputs
        data = f"{image_id}_{action}_{object_name}_{model}_{prompt_hash}"
        return hashlib.md5(data.encode()).hexdigest()

    def get(self, image_id: str, action: str, object_name: str,
            model: str, prompt_hash: str) -> Optional[Dict[str, Any]]:
        """
        Get cached response if it exists.

        Args:
            image_id: Image identifier
            action: Action being performed (or "" for grounding)
            object_name: Object category (or "" for action referring)
            model: Model name
            prompt_hash: Hash of the prompt text

        Returns:
            Cached response dict if exists, None otherwise
        """
        key = self._generate_key(image_id, action, object_name, model, prompt_hash)
        cache_file = os.path.join(self.cache_dir, f"{key}.json")

        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                # Corrupted cache file, return None
                print(f"Warning: Corrupted cache file {cache_file}: {e}")
                return None

        return None

    def set(self, image_id: str, action: str, object_name: str,
            model: str, prompt_hash: str, response: Dict[str, Any]) -> None:
        """
        Cache API response.

        Args:
            image_id: Image identifier
            action: Action being performed (or "" for grounding)
            object_name: Object category (or "" for action referring)
            model: Model name
            prompt_hash: Hash of the prompt text
            response: Response dict to cache
        """
        key = self._generate_key(image_id, action, object_name, model, prompt_hash)
        cache_file = os.path.join(self.cache_dir, f"{key}.json")

        try:
            with open(cache_file, 'w') as f:
                json.dump(response, f, indent=2)
        except IOError as e:
            print(f"Warning: Failed to write cache file {cache_file}: {e}")

    def clear(self) -> int:
        """
        Clear all cached responses.

        Returns:
            Number of cache files deleted
        """
        count = 0
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.json'):
                try:
                    os.remove(os.path.join(self.cache_dir, filename))
                    count += 1
                except IOError:
                    pass
        return count

    def size(self) -> int:
        """
        Get number of cached responses.

        Returns:
            Number of cache files
        """
        return len([f for f in os.listdir(self.cache_dir) if f.endswith('.json')])

    @staticmethod
    def hash_prompt(prompt: str) -> str:
        """
        Generate hash of prompt text for cache key.

        Args:
            prompt: Prompt text string

        Returns:
            MD5 hash of prompt
        """
        return hashlib.md5(prompt.encode()).hexdigest()[:8]  # First 8 chars sufficient


if __name__ == '__main__':
    # Example usage
    cache = ResponseCache(cache_dir="./test_cache")

    # Test caching
    prompt = "Test prompt"
    prompt_hash = ResponseCache.hash_prompt(prompt)

    # Store response
    test_response = {
        'prediction': 'holding phone',
        'confidence': 0.95
    }
    cache.set("img_001", "holding", "phone", "claude-sonnet-4.5", prompt_hash, test_response)

    # Retrieve response
    cached = cache.get("img_001", "holding", "phone", "claude-sonnet-4.5", prompt_hash)
    print(f"Cached response: {cached}")

    # Check cache size
    print(f"Cache size: {cache.size()} files")

    # Clear cache
    deleted = cache.clear()
    print(f"Cleared {deleted} cache files")
