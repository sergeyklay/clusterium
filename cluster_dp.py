#!/usr/bin/env python3
"""
Simple script for text clustering using Dirichlet Process and Pitman-Yor Process.
This script takes a CSV file as input and performs clustering on the text data.
"""

import argparse
import csv
import hashlib
import json
import os
import pickle
from collections import Counter
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from torch import Tensor
from tqdm import tqdm


class EmbeddingCache:
    """Provides caching functionality for embeddings with class-specific cache files."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir
        self.embedding_cache: dict[str, Tensor] = {}

    def get_cache_path(self, base_name: str):
        """Generate cache file path based on class name."""
        if not self.cache_dir:
            return None

        base_name = base_name.lower().translate(str.maketrans(" ./", "___"))
        return os.path.join(self.cache_dir, f"{base_name}.pkl")

    def load_cache(self, base_name: Optional[str] = "embeddings") -> dict[str, Tensor]:
        """Load cache for the specified class."""
        if not self.cache_dir:
            return {}

        base_name = base_name or "embeddings"
        cache_path = self.get_cache_path(base_name)
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    self.embedding_cache = pickle.load(f)
                print(f"Loaded {len(self.embedding_cache)} cached embeddings")
            except Exception as e:
                print(f"Error loading embedding cache: {e}")
                self.embedding_cache = {}
        return self.embedding_cache

    def save_cache(self, base_name: Optional[str] = "embeddings"):
        """Save cache for the specified class."""
        if not self.cache_dir or not self.embedding_cache:
            return

        base_name = base_name or "embeddings"
        cache_path = self.get_cache_path(base_name)
        if cache_path:
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(self.embedding_cache, f)
            print(f"Saved {len(self.embedding_cache)} embeddings to cache")

    def _hash_key(self, key) -> str:
        """Calculate a stable hash for an items list."""
        combined = str(key).encode("utf-8")

        # SHA-256 produces 64-character hex digest
        return hashlib.sha256(combined).hexdigest()

    def get(self, key: str) -> Tensor | None:
        """Get item from cache."""
        return self.embedding_cache.get(self._hash_key(key))

    def set(self, key: str, value: Tensor):
        """Set item in cache."""
        self.embedding_cache[self._hash_key(key)] = value

    def __contains__(self, key: str) -> bool:
        """Check if key exists in cache."""
        return self._hash_key(key) in self.embedding_cache


class DirichletProcess:
    """Dirichlet Process clustering implementation."""

    def __init__(
        self,
        alpha: float,
        base_measure: Optional[Tensor] = None,
        similarity_metric: Optional[Callable[[str, Tensor], float]] = None,
        cache: Optional[EmbeddingCache] = None,
    ):
        self.alpha = alpha
        self.base_measure = base_measure
        self.clusters: list[int] = []
        self.cluster_params: list[Tensor] = []
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.similarity_metric = (
            similarity_metric if similarity_metric else self.bert_similarity
        )

        self.cache = cache
        if self.cache:
            self.cache.load_cache()

    def get_embedding(self, text: str) -> Tensor:
        # Try to get from cache first
        if self.cache and text in self.cache:
            embedding = self.cache.get(text)
            if embedding is not None:
                return embedding

        # Generate new embedding
        embedding = self.model.encode(text)

        # Store in cache if provider available
        if self.cache:
            self.cache.set(text, embedding)

        return embedding

    def save_embedding_cache(self):
        if self.cache:
            self.cache.save_cache()

    def bert_similarity(self, text, cluster_param):
        text_embedding = self.get_embedding(text)
        cluster_embedding = cluster_param
        similarity = 1 - cosine(text_embedding, cluster_embedding)
        return max(0.0, similarity)

    def sample_new_cluster(self, text):
        return self.get_embedding(text)

    def assign_cluster(self, text):
        probs = []
        total_points = len(self.clusters)

        for i, params in enumerate(self.cluster_params):
            cluster_size = self.clusters.count(i)
            prob = (
                cluster_size / (self.alpha + total_points)
            ) * self.similarity_metric(text, params)
            probs.append(max(0.0, prob))

        new_cluster_prob = (self.alpha / (self.alpha + total_points)) * 1.0
        probs.append(new_cluster_prob)

        probs = np.array(probs)
        if probs.sum() <= 0:
            probs = np.ones(len(probs)) / len(probs)
        else:
            probs /= probs.sum()

        choice = np.random.choice(len(probs), p=probs)
        if choice == len(self.cluster_params):
            new_params = self.sample_new_cluster(text)
            self.cluster_params.append(new_params)
            self.clusters.append(len(self.cluster_params) - 1)
        else:
            self.clusters.append(choice)

    def fit(self, texts):
        print(f"Processing {len(texts)} texts...")
        for text in tqdm(texts, desc="Clustering"):
            self.assign_cluster(text)

        self.save_embedding_cache()

        return self.clusters, self.cluster_params


class PitmanYorProcess(DirichletProcess):
    """Pitman-Yor Process clustering implementation."""

    def __init__(self, alpha, sigma, base_measure, similarity_metric=None, cache=None):
        super().__init__(alpha, base_measure, similarity_metric, cache)
        self.sigma = sigma
        # Keep track of cluster sizes for faster access
        self.cluster_sizes = {}

    def assign_cluster(self, text):
        """Uses Pitman-Yor process probability calculations."""
        probs = []
        total_points = len(self.clusters)

        # Pre-compute the embedding once
        text_embedding = self.get_embedding(text)

        # Update cluster sizes dictionary
        if not hasattr(self, "cluster_sizes") or self.cluster_sizes is None:
            self.cluster_sizes = {}
            for i in range(len(self.cluster_params)):
                self.cluster_sizes[i] = self.clusters.count(i)

        for i, params in enumerate(self.cluster_params):
            # Use the cached cluster size instead of counting each time
            cluster_size = self.cluster_sizes.get(i, 0)
            adjusted_size = max(self.sigma, cluster_size)

            # Calculate similarity directly with embeddings for speed
            similarity = 1 - cosine(text_embedding, params)
            similarity = max(0.0, similarity)

            prob = (
                (adjusted_size - self.sigma) / (self.alpha + total_points) * similarity
            )
            probs.append(max(0.0, prob))

        new_cluster_prob = (
            (self.alpha + self.sigma * len(self.cluster_params))
            / (self.alpha + total_points)
        ) * 1.0
        probs.append(new_cluster_prob)

        probs = np.array(probs)
        if probs.sum() <= 0:
            probs = np.ones(len(probs)) / len(probs)
        else:
            probs /= probs.sum()

        choice = np.random.choice(len(probs), p=probs)
        if choice == len(self.cluster_params):
            # Use the already computed embedding
            self.cluster_params.append(text_embedding)
            self.clusters.append(len(self.cluster_params) - 1)
            # Update cluster sizes
            self.cluster_sizes[len(self.cluster_params) - 1] = 1
        else:
            self.clusters.append(choice)
            # Update cluster sizes
            self.cluster_sizes[choice] = self.cluster_sizes.get(choice, 0) + 1

    def fit(self, texts):
        """Optimized version of fit for PitmanYorProcess."""
        print(f"Processing {len(texts)} texts with optimized PitmanYorProcess...")

        # Initialize cluster sizes dictionary
        self.cluster_sizes = {}

        # Process texts in batches for better progress reporting
        batch_size = 100
        total_batches = (len(texts) - 1) // batch_size + 1
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_num = i // batch_size + 1
            for text in tqdm(
                batch, desc=f"Clustering batch {batch_num}/{total_batches}"
            ):
                self.assign_cluster(text)

        self.save_embedding_cache()
        return self.clusters, self.cluster_params


def plot_cluster_distribution(cluster_assignments, title, color):
    cluster_counts = Counter(cluster_assignments)
    sizes = sorted(cluster_counts.values(), reverse=True)
    plt.bar(range(1, len(sizes) + 1), sizes, color=color, alpha=0.6)
    plt.xlabel("Cluster Rank")
    plt.ylabel("Cluster Size")
    plt.title(title)


def load_data_from_csv(csv_file, column="question", answer_column="answer"):
    texts = []
    data = []  # Store full data including answers

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if column in row and row[column].strip():
                texts.append(row[column])
                # Store the full row data
                data.append(row)

    return texts, data


def save_clusters_to_csv(output_file, texts, clusters, model_name):
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Text", f"Cluster_{model_name}"])
        for text, cluster in zip(texts, clusters):
            writer.writerow([text, cluster])
    print(f"Clustering results saved to {output_file}")


def save_clusters_to_json(
    output_file, texts, clusters, model_name, data=None, answer_column="answer"
):
    cluster_groups = {}
    data_map = {}

    # Create a mapping from question to data row if data is provided
    if data:
        for row in data:
            if "question" in row:
                data_map[row["question"]] = row

    # Group texts by cluster
    for text, cluster_id in zip(texts, clusters):
        if cluster_id not in cluster_groups:
            cluster_groups[cluster_id] = []
        cluster_groups[cluster_id].append(text)

    clusters_json = {"clusters": []}

    for i, (cluster_id, cluster_texts) in enumerate(cluster_groups.items()):
        representative_text = cluster_texts[0]

        # Get the answer for the representative text
        representative_answer = "No answer available"
        if (
            data_map
            and representative_text in data_map
            and answer_column in data_map[representative_text]
        ):
            representative_answer = data_map[representative_text][answer_column]
        else:
            representative_answer = f"Answer for cluster {i+1} using {model_name}"

        # Create the cluster object
        cluster_obj = {
            "id": i + 1,
            "representative": [
                {
                    "question": representative_text,
                    "answer": representative_answer,
                }
            ],
            "source": [],
        }

        # Add sources with their real answers if available
        for text in cluster_texts:
            answer = f"Answer for question in cluster {i+1}"
            if data_map and text in data_map and answer_column in data_map[text]:
                answer = data_map[text][answer_column]

            cluster_obj["source"].append({"question": text, "answer": answer})

        clusters_json["clusters"].append(cluster_obj)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(clusters_json, f, indent=2, ensure_ascii=False)

    print(f"JSON clusters saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Text clustering using Dirichlet Process and Pitman-Yor Process"
    )
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument(
        "--column",
        default="question",
        help="Column name to use for clustering (default: question)",
    )
    parser.add_argument(
        "--output", default="clusters_output.csv", help="Output CSV file path"
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory to save output files (default: output)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Concentration parameter (default: 1.0)",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.5,
        help="Discount parameter for Pitman-Yor (default: 0.5)",
    )
    parser.add_argument(
        "--plot", action="store_true", help="Generate cluster distribution plot"
    )
    parser.add_argument(
        "--cache-dir",
        default=".cache",
        help="Directory to cache embeddings (default: .cache)",
    )
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading data from {args.input}, using column '{args.column}'...")
    texts, data = load_data_from_csv(args.input, args.column)
    if not texts:
        print(f"No data found in column '{args.column}'. Please check your CSV file.")
        return
    print(f"Loaded {len(texts)} texts for clustering")

    # Create cache provider
    cache_provider = EmbeddingCache(cache_dir=args.cache_dir)

    print("Performing Dirichlet Process clustering...")
    dp = DirichletProcess(alpha=args.alpha, base_measure=None, cache=cache_provider)
    clusters_dp, params_dp = dp.fit(texts)
    print(f"DP clustering complete. Found {len(set(clusters_dp))} clusters")

    print("Performing Pitman-Yor Process clustering...")
    pyp = PitmanYorProcess(
        alpha=args.alpha,
        sigma=args.sigma,
        base_measure=None,
        cache=cache_provider,
    )
    clusters_pyp, params_pyp = pyp.fit(texts)
    print(f"PYP clustering complete. Found {len(set(clusters_pyp))} clusters")

    output_basename = os.path.basename(args.output)
    dp_output = os.path.join(
        args.output_dir, output_basename.replace(".csv", "_dp.csv")
    )
    pyp_output = os.path.join(
        args.output_dir, output_basename.replace(".csv", "_pyp.csv")
    )
    save_clusters_to_csv(dp_output, texts, clusters_dp, "DP")
    save_clusters_to_csv(pyp_output, texts, clusters_pyp, "PYP")

    dp_json = os.path.join(args.output_dir, output_basename.replace(".csv", "_dp.json"))
    pyp_json = os.path.join(
        args.output_dir, output_basename.replace(".csv", "_pyp.json")
    )
    save_clusters_to_json(dp_json, texts, clusters_dp, "DP", data)
    save_clusters_to_json(pyp_json, texts, clusters_pyp, "PYP", data)

    qa_clusters_path = os.path.join(args.output_dir, "qa_clusters.json")
    save_clusters_to_json(qa_clusters_path, texts, clusters_dp, "Combined", data)
    print(f"Combined clusters saved to {qa_clusters_path}")

    if args.plot:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plot_cluster_distribution(
            clusters_dp, "Dirichlet Process Cluster Sizes", "blue"
        )
        plt.subplot(1, 2, 2)
        plot_cluster_distribution(
            clusters_pyp, "Pitman-Yor Process Cluster Sizes", "red"
        )
        plt.tight_layout()
        plot_path = os.path.join(
            args.output_dir, output_basename.replace(".csv", "_clusters.png")
        )
        plt.savefig(plot_path)
        print(f"Cluster distribution plot saved to {plot_path}")
        plt.show()


if __name__ == "__main__":
    main()
