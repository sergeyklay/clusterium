import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

logger = logging.getLogger(__name__)


class ClusterBenchmarker:
    """Class for benchmarking the quality of clustering results."""

    def __init__(
        self,
        embedding_model_name: Optional[str] = None,
        llm_model_name: Optional[str] = None,
        output_dir: str = "./output",
    ):
        """Initialize the benchmarker.

        Args:
            embedding_model_name: Name of the embedding model to use
            llm_model_name: Name of the LLM model to use for topic labeling
        """
        # Initialize embedding model if name is provided
        self.embeddings_model = None
        if embedding_model_name:
            try:
                self.embeddings_model = OpenAIEmbeddings(model=embedding_model_name)
                logger.info(f"Initialized embeddings model: {embedding_model_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize embeddings model: {e}")

        # Initialize LLM if model name is provided
        self.llm = None
        if llm_model_name:
            try:
                self.llm = ChatOpenAI(model=llm_model_name, temperature=0.0)
                logger.info(f"Initialized LLM with model: {llm_model_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM: {e}")

        self.output_dir = output_dir

    def load_clusters(self, json_path: str) -> Dict[str, Any]:
        """Load clusters from a JSON file.

        Args:
            json_path: Path to the JSON file containing clustering results

        Returns:
            Dict containing clustering results
        """
        with open(json_path, "r") as f:
            return json.load(f)

    def load_qa_pairs(self, csv_path: str) -> List[Tuple[str, str]]:
        """Load question-answer pairs from a CSV file.

        Args:
            csv_path: Path to the CSV file containing question-answer pairs

        Returns:
            List of (question, answer) tuples
        """
        qa_pairs = []
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            # Skip header
            next(reader)
            for row in reader:
                if len(row) >= 2:
                    qa_pairs.append((row[0], row[1]))
        return qa_pairs

    def extract_embeddings_from_qa_pairs(
        self, qa_pairs: List[Tuple[str, str]]
    ) -> np.ndarray:
        """Extract embeddings from QA pairs using the embeddings model.

        Args:
            qa_pairs: List of (question, answer) tuples

        Returns:
            Array of embeddings
        """
        if self.embeddings_model is None:
            raise ValueError("Embeddings model not provided")

        questions = [q for q, _ in qa_pairs]
        return np.array(self.embeddings_model.embed_documents(questions))

    def prepare_cluster_data(
        self, clusters: Dict[str, Any], embeddings: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for cluster quality evaluation.

        Args:
            clusters: Dict containing clustering results
            embeddings: Array of embeddings

        Returns:
            Tuple of (embeddings_array, labels_array)
        """
        # Create a mapping from question to index
        question_to_idx = {}
        for i, (q, _) in enumerate(self.qa_pairs):
            question_to_idx[q] = i

        # Extract labels for each question
        labels = np.full(len(self.qa_pairs), -1)  # Default to noise

        for cluster_idx, cluster in enumerate(clusters["clusters"]):
            for qa_pair in cluster["source"]:
                question = qa_pair["question"]
                if question in question_to_idx:
                    labels[question_to_idx[question]] = cluster_idx

        return embeddings, labels

    def calculate_metrics(
        self, embeddings: np.ndarray, labels: np.ndarray
    ) -> Dict[str, float]:
        """Calculate cluster quality metrics.

        Args:
            embeddings: Array of embeddings
            labels: Array of cluster labels

        Returns:
            Dict containing metrics
        """
        # Filter out noise points (label -1)
        non_noise_mask = labels != -1
        non_noise_embeddings = embeddings[non_noise_mask]
        non_noise_labels = labels[non_noise_mask]

        metrics = {}

        # Calculate noise ratio
        metrics["noise_ratio"] = 1.0 - (np.sum(non_noise_mask) / len(labels))

        # Skip other metrics if there's only one cluster or all points are noise
        if len(np.unique(non_noise_labels)) <= 1 or len(non_noise_embeddings) == 0:
            metrics["davies_bouldin_score"] = float("nan")
            metrics["calinski_harabasz_score"] = float("nan")
            metrics["silhouette_score"] = float("nan")
            return metrics

        # Calculate Davies-Bouldin Index (lower is better)
        metrics["davies_bouldin_score"] = davies_bouldin_score(
            non_noise_embeddings, non_noise_labels
        )

        # Calculate Calinski-Harabasz Index (higher is better)
        metrics["calinski_harabasz_score"] = calinski_harabasz_score(
            non_noise_embeddings, non_noise_labels
        )

        # Calculate Silhouette Score (higher is better)
        metrics["silhouette_score"] = silhouette_score(
            non_noise_embeddings, non_noise_labels
        )

        return metrics

    def calculate_cluster_coherence(self, cluster_questions: List[str]) -> float:
        """Calculate coherence score for a cluster based on cosine similarity.

        Args:
            cluster_questions: List of questions in the cluster

        Returns:
            Coherence score (average pairwise similarity)
        """
        if len(cluster_questions) <= 1:
            return 1.0  # Perfect coherence for single-item clusters

        # Get embeddings for questions
        embeddings = np.array(self.embeddings_model.embed_documents(cluster_questions))

        # Calculate pairwise cosine similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                similarity = self._cosine_similarity(embeddings[i], embeddings[j])
                similarities.append(similarity)

        # Return average similarity
        return np.mean(similarities)

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity
        """
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def _generate_llm_topic_label(self, questions: List[str]) -> str:
        """Generate a topic label using an LLM.

        Args:
            questions: List of questions to generate a topic label for

        Returns:
            Topic label
        """
        if not self.llm:
            return "No LLM Available"

        # Format the questions as a numbered list
        formatted_questions = "\n".join(
            [f"{i+1}. {q}" for i, q in enumerate(questions)]
        )

        # Create a prompt template with more specific instructions
        prompt_template = PromptTemplate(
            input_variables=["questions"],
            template="""
            You are an expert taxonomist specializing in creating precise, distinctive category labels.

            Below is a list of questions that belong to the same cluster:

            {questions}

            Your task is to create a HIGHLY SPECIFIC topic label (2-4 words) that:
            1. Precisely captures what makes these questions UNIQUE compared to other topics
            2. Uses concrete, specific terminology rather than generic terms
            3. Avoids using the product name "SignNow" in the label
            4. Focuses on the distinctive FUNCTION or CONCEPT these questions address
            5. Is concise and memorable

            BAD EXAMPLES (too generic):
            - "Document Management"
            - "User Features"
            - "Account Settings"
            - "SignNow Features"

            GOOD EXAMPLES (specific and distinctive):
            - "Offline Deployment Options"
            - "Team Permission Hierarchy"
            - "Template Reusability"
            - "Audit Trail Functionality"
            - "CRM Integration Methods"

            Respond ONLY with the final topic label, nothing else.
            """,
        )

        # Create a runnable sequence
        if self.llm:
            chain = prompt_template | self.llm

            # Generate the topic label
            try:
                response = chain.invoke({"questions": formatted_questions})

                # Clean up the response
                if hasattr(response, "content"):
                    # For ChatOpenAI which returns a message with content
                    topic_label = response.content.strip().strip('"').strip("'")
                else:
                    # For other LLMs that return a string directly
                    topic_label = str(response).strip().strip('"').strip("'")

                # Limit to 50 characters
                if len(topic_label) > 50:
                    topic_label = topic_label[:47] + "..."

                return topic_label
            except Exception as e:
                logger.warning(f"Error generating topic label: {e}")
                return "LLM Error"
        else:
            return "No LLM Available"

    def extract_topic_labels(
        self,
        clusters: Dict[str, Any],
        n_topics: int = 1,
        n_top_words: int = 3,
        use_llm: bool = True,
        max_questions_per_cluster: int = 10,
    ) -> Dict[int, str]:
        """Extract topic labels for each cluster.

        Args:
            clusters: Dict containing clustering results
            n_topics: Number of topics to extract per cluster (for TF-IDF/NMF method)
            n_top_words: Number of top words to include in the label
                (for TF-IDF/NMF method)
            use_llm: Whether to use an LLM for generating topic labels
            max_questions_per_cluster: Maximum number of questions to use for
                LLM-based labeling

        Returns:
            Dict mapping cluster IDs to topic labels
        """
        topic_labels = {}

        if use_llm and self.llm is None:
            logger.warning("LLM not provided, falling back to TF-IDF/NMF method")
            use_llm = False

        # First pass: collect all questions for context
        all_cluster_questions = {}
        for cluster in clusters["clusters"]:
            cluster_id = cluster["id"]
            questions = [qa["question"] for qa in cluster["source"]]
            if questions:
                all_cluster_questions[cluster_id] = questions[
                    :max_questions_per_cluster
                ]

        # Second pass: generate labels with knowledge of other clusters
        for cluster in clusters["clusters"]:
            cluster_id = cluster["id"]
            questions = all_cluster_questions.get(cluster_id, [])

            if not questions:
                topic_labels[cluster_id] = "Empty Cluster"
                continue

            if use_llm:
                try:
                    # Use LLM to generate a topic label
                    topic_labels[cluster_id] = self._generate_llm_topic_label(questions)
                    continue
                except Exception as e:
                    logger.warning(
                        f"Error generating LLM topic label for cluster {cluster_id}: {e}"
                    )
                    # Fall back to TF-IDF/NMF method

            # TF-IDF/NMF method (fallback)
            try:
                # Use TF-IDF to extract important words
                vectorizer = TfidfVectorizer(
                    max_features=100, stop_words="english", ngram_range=(1, 2)
                )

                tfidf_matrix = vectorizer.fit_transform(questions)
                feature_names = vectorizer.get_feature_names_out()

                # If we have enough documents, use NMF to extract topics
                if len(questions) >= 3:
                    nmf_model = NMF(
                        n_components=min(n_topics, len(questions)), random_state=42
                    )
                    # Fit the model but don't need to store features
                    nmf_model.fit_transform(tfidf_matrix)

                    # Get the top words for the first topic
                    topic_idx = 0
                    top_word_indices = np.argsort(nmf_model.components_[topic_idx])[
                        ::-1
                    ][:n_top_words]
                    top_words = [str(feature_names[i]) for i in top_word_indices]

                    topic_labels[cluster_id] = " ".join(top_words).title()
                else:
                    # For small clusters, just use the top TF-IDF terms
                    tfidf_sum = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
                    top_indices = tfidf_sum.argsort()[::-1][:n_top_words]
                    top_words = [str(feature_names[i]) for i in top_indices]

                    topic_labels[cluster_id] = " ".join(top_words).title()
            except Exception as e:
                # Fallback to using the first question as the topic
                logger.warning(f"Error extracting topic for cluster {cluster_id}: {e}")
                topic_labels[cluster_id] = questions[0][:50] + "..."

        # Post-process to ensure uniqueness
        self._ensure_unique_labels(topic_labels)

        return topic_labels

    def _ensure_unique_labels(self, topic_labels: Dict[int, str]) -> None:
        """Ensure all topic labels are unique by adding suffixes if needed.

        Args:
            topic_labels: Dict mapping cluster IDs to topic labels
        """
        seen_labels = {}

        for cluster_id, label in sorted(topic_labels.items()):
            if label in seen_labels:
                # Add a suffix to make it unique
                count = seen_labels[label] + 1
                seen_labels[label] = count
                topic_labels[cluster_id] = f"{label} ({count})"
            else:
                seen_labels[label] = 1

    def generate_cluster_report(
        self,
        clusters_json_path: str,
        qa_csv_path: str,
        use_llm_for_topics: bool = True,
    ) -> pd.DataFrame:
        """Generate a comprehensive cluster quality report.

        Args:
            clusters_json_path: Path to the JSON file containing clustering results
            qa_csv_path: Path to the CSV file containing question-answer pairs
            use_llm_for_topics: Whether to use an LLM for generating topic labels

        Returns:
            DataFrame containing the cluster report
        """
        # Load data
        clusters = self.load_clusters(clusters_json_path)
        self.qa_pairs = self.load_qa_pairs(qa_csv_path)

        # Generate embeddings
        embeddings = self.extract_embeddings_from_qa_pairs(self.qa_pairs)

        # Prepare data for metrics calculation
        embeddings_array, labels_array = self.prepare_cluster_data(clusters, embeddings)

        # Calculate global metrics
        global_metrics = self.calculate_metrics(embeddings_array, labels_array)

        # Extract topic labels
        topic_labels = self.extract_topic_labels(clusters, use_llm=use_llm_for_topics)

        # Prepare report data
        report_data = []

        for cluster in clusters["clusters"]:
            cluster_id = cluster["id"]
            questions = [qa["question"] for qa in cluster["source"]]

            # Calculate cluster-specific metrics
            coherence_score = self.calculate_cluster_coherence(questions)

            report_data.append(
                {
                    "Cluster_ID": cluster_id,
                    "Num_QA_Pairs": len(questions),
                    "Avg_Similarity": coherence_score,
                    "Coherence_Score": coherence_score,
                    "Topic_Label": topic_labels.get(cluster_id, "Unknown"),
                }
            )

        # Create DataFrame
        report_df = pd.DataFrame(report_data)

        # Add global metrics as a summary row
        summary_metrics = (
            f"Noise Ratio: {global_metrics['noise_ratio']:.2f}, "
            f"DB: {global_metrics.get('davies_bouldin_score', np.nan):.2f}, "
            f"CH: {global_metrics.get('calinski_harabasz_score', np.nan):.2f}"
        )

        summary_row = pd.DataFrame(
            [
                {
                    "Cluster_ID": "SUMMARY",
                    "Num_QA_Pairs": len(self.qa_pairs),
                    "Avg_Similarity": np.nan,
                    "Coherence_Score": np.nan,
                    "Topic_Label": summary_metrics,
                }
            ]
        )

        report_df = pd.concat([report_df, summary_row], ignore_index=True)

        # Save to CSV
        report_df.to_csv(
            Path(self.output_dir) / "cluster_quality_report.csv", index=False
        )

        return report_df
