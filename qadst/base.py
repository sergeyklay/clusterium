import csv
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

logger = logging.getLogger(__name__)


class BaseClusterer(ABC):

    def __init__(
        self,
        embedding_model_name: str,
        output_dir: str = "./output",
        llm_model_name: Optional[str] = None,
        filter_enabled: bool = True,
    ):
        """Initialize the clusterer.

        Args:
            embedding_model_name: Name of the embedding model to use
            output_dir: Directory to save output files
            llm_model_name: Optional name of the LLM model to use for filtering
               and labeling
            filter_enabled: Whether to enable filtering of engineering questions
        """
        self.output_dir = output_dir
        self.embedding_model_name = embedding_model_name
        self.embeddings_model = OpenAIEmbeddings(model=embedding_model_name)
        self.filter_enabled = filter_enabled
        self.filter_cache = {}  # Cache for filter results to avoid redundant LLM calls

        # Initialize LLM if model name is provided
        self.llm = None
        if llm_model_name and filter_enabled:
            try:
                self.llm = ChatOpenAI(model=llm_model_name, temperature=0.0)
                logger.info(f"Initialized LLM with model: {llm_model_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM: {e}")

        os.makedirs(output_dir, exist_ok=True)

    def load_qa_pairs(self, csv_path: str) -> List[Tuple[str, str]]:
        """Load question-answer pairs from a CSV file.

        Args:
            csv_path: Path to the CSV file containing question-answer pairs

        Returns:
            List of (question, answer) tuples
        """
        qa_pairs = []

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)

            if (
                len(header) < 2
                or "question" not in header[0].lower()
                or "answer" not in header[1].lower()
            ):
                raise ValueError(
                    "CSV file must have 'question' and 'answer' columns. "
                    f"Found: {header}"
                )

            for row in reader:
                if len(row) >= 2 and row[0] and row[1]:
                    qa_pairs.append((row[0], row[1]))

        return qa_pairs

    def calculate_cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            float: Cosine similarity score between 0 and 1
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)

        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0

        return dot_product / (norm_vec1 * norm_vec2)

    def deduplicate_questions(
        self, qa_pairs: List[Tuple[str, str]]
    ) -> List[Tuple[str, str]]:
        """Remove semantically duplicate questions using embedding similarity.

        Args:
            qa_pairs: List of (question, answer) tuples

        Returns:
            List of deduplicated (question, answer) tuples
        """
        if not qa_pairs:
            return []

        questions = [q for q, _ in qa_pairs]
        question_embeddings = self.embeddings_model.embed_documents(questions)

        similarity_threshold = 0.85
        duplicate_groups = {}
        duplicate_indices = set()
        total_questions = len(questions)
        processed_count = 0
        duplicate_count = 0

        for i in range(len(questions)):
            processed_count += 1
            if i in duplicate_indices:
                continue

            duplicate_groups[i] = [i]

            for j in range(i + 1, len(questions)):
                if j in duplicate_indices:
                    continue

                similarity = self.calculate_cosine_similarity(
                    question_embeddings[i], question_embeddings[j]
                )

                if similarity > similarity_threshold:
                    duplicate_indices.add(j)
                    duplicate_groups[i].append(j)
                    duplicate_count += 1
                    logger.info(
                        f"[{processed_count}/{total_questions}] Found duplicate: "
                        f"'{questions[j]}' similar to '{questions[i]}'"
                    )

        deduplicated_pairs = []

        for canonical_idx, group in duplicate_groups.items():
            if len(group) == 1:
                deduplicated_pairs.append(qa_pairs[canonical_idx])
            else:
                # Take the first pair as representative
                deduplicated_pairs.append(qa_pairs[canonical_idx])

        logger.info(
            f"Found {duplicate_count} duplicates out of {total_questions} questions"
        )
        return deduplicated_pairs

    def filter_questions(
        self,
        qa_pairs: List[Tuple[str, str]],
        batch_size: int = 20,
        use_llm: bool = True,
        cache_file: Optional[str] = None,
    ) -> List[Tuple[str, str]]:
        """Filter out questions intended for engineering teams rather than end clients.

        Args:
            qa_pairs: List of (question, answer) tuples
            batch_size: Number of questions to process in each batch for efficiency
            use_llm: Whether to use an LLM for filtering
            cache_file: Optional path to a cache file to persist filter results

        Returns:
            List of filtered (question, answer) tuples
        """
        if not use_llm or not self.llm:
            logger.warning(
                "LLM not provided or filtering disabled, skipping engineering filter"
            )
            return qa_pairs

        # Load cache from file if provided
        if cache_file and os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    self.filter_cache = json.load(f)
                logger.info(f"Loaded {len(self.filter_cache)} cached filter results")
            except Exception as e:
                logger.warning(f"Error loading filter cache: {e}")

        filtered_pairs = []
        engineering_pairs = []
        questions_to_process = []

        # First pass: check cache and collect questions that need processing
        for q, a in qa_pairs:
            if q in self.filter_cache:
                # Use cached result
                if not self.filter_cache[
                    q
                ]:  # False means it's not an engineering question
                    filtered_pairs.append((q, a))
                else:
                    engineering_pairs.append((q, a))
            else:
                questions_to_process.append((q, a))

        if not questions_to_process:
            logger.info("All questions found in cache, no LLM calls needed")
            return filtered_pairs

        # Process questions in batches for efficiency
        logger.info(
            f"Processing {len(questions_to_process)} uncached questions in batches"
        )

        batches = [
            questions_to_process[i : i + batch_size]
            for i in range(0, len(questions_to_process), batch_size)
        ]

        total_processed = 0
        start_time = time.time()

        for batch in batches:
            batch_results = self._classify_questions_batch([q for q, _ in batch])

            for (q, a), is_engineering in zip(batch, batch_results):
                # Update cache
                self.filter_cache[q] = is_engineering

                if not is_engineering:
                    filtered_pairs.append((q, a))
                else:
                    engineering_pairs.append((q, a))

            total_processed += len(batch)
            elapsed = time.time() - start_time
            rate = total_processed / elapsed if elapsed > 0 else 0
            logger.info(
                f"Processed {total_processed}/{len(questions_to_process)} questions, "
                f"found {len(engineering_pairs)} engineering questions"
            )

            # Small delay to avoid rate limiting
            time.sleep(0.1)

        # Save cache to file if provided
        if cache_file:
            try:
                with open(cache_file, "w") as f:
                    json.dump(self.filter_cache, f)
                logger.info(f"Saved {len(self.filter_cache)} filter results to cache")
            except Exception as e:
                logger.warning(f"Error saving filter cache: {e}")

        logger.info(
            f"Filtered out {len(engineering_pairs)} engineering questions "
            f"({len(engineering_pairs)/len(qa_pairs)*100:.1f}%)"
        )

        # Optionally save engineering questions to a separate file for review
        engineering_file = os.path.join(self.output_dir, "engineering_questions.csv")
        with open(engineering_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["question", "answer"])
            for q, a in engineering_pairs:
                writer.writerow([q, a])
        logger.info(f"Saved engineering questions to {engineering_file}")

        return filtered_pairs

    def _classify_questions_batch(self, questions: List[str]) -> List[bool]:
        """Classify a batch of questions as engineering-focused or client-focused.

        Args:
            questions: List of questions to classify

        Returns:
            List of booleans where True means the question is engineering-focused
        """
        if not self.llm:
            return [False] * len(questions)

        # Format the questions as a numbered list
        formatted_questions = "\n".join(
            [f"{i+1}. {q}" for i, q in enumerate(questions)]
        )

        # Create a prompt template
        prompt_template = PromptTemplate(
            input_variables=["questions"],
            template="""
            You are an expert at classifying content for the right audience.

            Below is a list of questions about a software product:

            {questions}

            For each question, determine if it is intended for ENGINEERING TEAMS
            (developers, testers, DevOps) or for END CLIENTS (users of the product).

            Engineering questions typically involve:
            - Development processes
            - Deployment procedures
            - Testing methodologies
            - Technical infrastructure
            - Internal systems
            - Code or API details

            Client questions typically involve:
            - Product features and usage
            - User interface
            - Account management
            - Pricing and subscriptions
            - Common workflows
            - Troubleshooting from a user perspective

            Respond with ONLY a JSON array of boolean values (true/false), where:
            - true = question is for ENGINEERING TEAMS
            - false = question is for END CLIENTS

            Example response format:
            [false, true, false, false, true]
            """,
        )

        # Create a runnable sequence
        chain = prompt_template | self.llm

        try:
            # Generate the classification
            response = chain.invoke({"questions": formatted_questions})

            # Parse the response
            if hasattr(response, "content"):
                # For ChatOpenAI which returns a message with content
                content = str(response.content).strip()
            else:
                # For other LLMs that return a string directly
                content = str(response).strip()

            # Extract the JSON array
            import re

            json_match = re.search(r"\[.*\]", content)
            if json_match:
                json_str = json_match.group(0)
                try:
                    results = json.loads(json_str)
                    # Ensure we have the right number of results
                    if len(results) != len(questions):
                        logger.warning(
                            f"Expected {len(questions)} results, got {len(results)}"
                        )
                        # Pad or truncate as needed
                        if len(results) < len(questions):
                            results.extend([False] * (len(questions) - len(results)))
                        else:
                            results = results[: len(questions)]
                    return results
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON from response: {json_str}")

            # Fallback: assume all questions are for clients
            logger.warning("Could not extract valid classification from LLM response")
            return [False] * len(questions)

        except Exception as e:
            logger.warning(f"Error classifying questions: {e}")
            return [False] * len(questions)

    @abstractmethod
    def cluster_questions(self, qa_pairs: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Cluster questions based on semantic similarity.

        Args:
            qa_pairs: List of (question, answer) tuples

        Returns:
            Dict containing clustering results in the requested format
        """
        pass

    @abstractmethod
    def cluster_method(self) -> str:
        """Return the name of the clustering method."""
        pass

    def process_dataset(self, csv_path: str) -> Dict[str, Any]:
        """Process a CSV file containing QA pairs.

        Args:
            csv_path: Path to the CSV file containing question-answer pairs

        Returns:
            Dict containing clustering results and paths to output files
        """
        logger.info(f"Loading QA pairs from {csv_path}")
        qa_pairs = self.load_qa_pairs(csv_path)
        logger.info(f"Loaded {len(qa_pairs)} QA pairs")

        logger.info("Deduplicating questions")
        deduplicated_pairs = self.deduplicate_questions(qa_pairs)
        logger.info(f"Deduplicated to {len(deduplicated_pairs)} QA pairs")

        filtered_pairs = deduplicated_pairs

        # Filter out engineering questions if enabled
        if self.filter_enabled:
            logger.info("Filtering out engineering-focused questions")
            cache_file = os.path.join(self.output_dir, "filter_cache.json")
            filtered_pairs = self.filter_questions(
                deduplicated_pairs,
                batch_size=20,
                use_llm=self.llm is not None,
                cache_file=cache_file,
            )
            logger.info(f"Retained {len(filtered_pairs)} client-focused QA pairs")
        else:
            logger.info("Filtering is disabled, skipping")

        logger.info(f"Clustering questions using {self.cluster_method()}")
        clustering_results = self.cluster_questions(filtered_pairs)

        json_output_path = os.path.join(self.output_dir, "qa_clusters.json")
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(clustering_results, f, indent=2)
        logger.info(f"Saved clustering results to {json_output_path}")

        csv_output_path = os.path.join(self.output_dir, "qa_cleaned.csv")
        with open(csv_output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["question", "answer"])
            for question, answer in filtered_pairs:
                writer.writerow([question, answer])
        logger.info(f"Saved cleaned QA pairs to {csv_output_path}")

        result = {
            "clustering_results": clustering_results,
            "json_output_path": json_output_path,
            "csv_output_path": csv_output_path,
            "deduplicated_count": len(deduplicated_pairs),
            "original_count": len(qa_pairs),
        }

        if self.filter_enabled:
            result["filtered_count"] = len(filtered_pairs)

        return result
