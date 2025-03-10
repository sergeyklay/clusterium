"""Embeddings module for QA dataset clustering.

This module provides functionality for working with embeddings models,
including a factory function to get the appropriate embeddings model
and utilities for caching and managing embeddings.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

logger = logging.getLogger(__name__)


class EmbeddingsCache:
    """Cache for embeddings to avoid recomputing them."""

    def __init__(self, output_dir: str = "./output"):
        """Initialize the embeddings cache.

        Args:
            output_dir: Directory to save cached embeddings
        """
        self.memory_cache: Dict[str, List[np.ndarray]] = {}
        self.output_dir = output_dir
        self.cache_dir = os.path.join(output_dir, "embedding_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    def get(self, cache_key: str) -> Optional[List[np.ndarray]]:
        """Get embeddings from cache.

        Args:
            cache_key: Key to identify the cached embeddings

        Returns:
            List of embeddings if found in cache, None otherwise
        """
        # Check in-memory cache first
        if cache_key in self.memory_cache:
            logger.info(f"Using in-memory cache for embeddings (key: {cache_key})")
            return self.memory_cache[cache_key]

        # Check disk cache
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.npy")
        if os.path.exists(cache_file):
            try:
                logger.debug(f"Loading embeddings from cache file: {cache_file}")
                embeddings_array = np.load(cache_file, allow_pickle=True)
                embeddings = [np.array(emb) for emb in embeddings_array]

                # Store in memory cache for faster access next time
                self.memory_cache[cache_key] = embeddings
                return embeddings
            except Exception as e:
                logger.warning(f"Failed to load embeddings from cache: {e}")

        return None

    def set(self, cache_key: str, embeddings: List[np.ndarray]) -> None:
        """Store embeddings in cache.

        Args:
            cache_key: Key to identify the cached embeddings
            embeddings: List of embeddings to cache
        """
        # Save to disk cache
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.npy")
        try:
            logger.debug(f"Saving embeddings to cache file: {cache_file}")
            np.save(cache_file, embeddings, allow_pickle=True)

            # Store in memory cache
            self.memory_cache[cache_key] = embeddings
        except Exception as e:
            logger.warning(f"Failed to save embeddings to cache: {e}")


class EmbeddingsProvider:
    """Provider for embeddings functionality."""

    def __init__(
        self,
        model: Any,
        cache: Optional[EmbeddingsCache] = None,
        output_dir: str = "./output",
    ):
        """Initialize the embeddings provider.

        Args:
            model: The embeddings model to use
            cache: Optional cache for embeddings
            output_dir: Directory to save cached embeddings if cache is not provided
        """
        self.model = model
        self.cache = cache or EmbeddingsCache(output_dir=output_dir)

    def get_embeddings(
        self, texts: List[str], cache_key: Optional[str] = None
    ) -> List[np.ndarray]:
        """Get embeddings for a list of texts, using cache when available.

        This method checks if embeddings are already cached (either in memory or
        on disk) before computing them, which can significantly improve performance
        across runs.

        Args:
            texts: List of texts to embed
            cache_key: Optional key to use for caching (e.g., a hash of the dataset)
                       If None, caching will not be used.

        Returns:
            List of numpy arrays containing the embeddings
        """
        if not texts:
            return []

        # If no cache key provided, compute embeddings without caching
        if cache_key is None:
            return self._embed_texts(texts)

        # Try to get from cache
        cached_embeddings = self.cache.get(cache_key)
        if cached_embeddings is not None:
            return cached_embeddings

        # Compute embeddings if not in cache
        logger.info(f"Computing embeddings for {len(texts)} texts")
        embeddings = self._embed_texts(texts)

        # Save to cache
        self.cache.set(cache_key, embeddings)

        return embeddings

    def _embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Embed texts using the embedding model.

        Args:
            texts: List of texts to embed

        Returns:
            List of numpy arrays containing the embeddings
        """
        embeddings_list = self.model.embed_documents(texts)
        return [np.array(emb) for emb in embeddings_list]

    def calculate_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity between the vectors (between -1 and 1)
        """
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))


def get_embeddings_model(
    model_name: str, api_key: Optional[str] = None, **kwargs
) -> Any:
    """Get an embeddings model based on the model name.

    Args:
        model_name: Name of the embeddings model to use
        api_key: Optional API key for the embeddings model
        **kwargs: Additional keyword arguments to pass to the model constructor

    Returns:
        An instance of the embeddings model

    Example:
        >>> model = get_embeddings_model("text-embedding-3-large")
        >>> embeddings = model.embed_documents(["Hello, world!"])
    """
    # Convert string api_key to SecretStr if provided
    if api_key is not None:
        api_key_secret = SecretStr(api_key)
    else:
        api_key_secret = None

    # Currently only supports OpenAI embeddings
    return OpenAIEmbeddings(model=model_name, api_key=api_key_secret, **kwargs)
