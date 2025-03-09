"""Filtering functionality for QA datasets.

This module provides filters to identify and separate different types of questions
in QA datasets, such as distinguishing between end-user questions and internal
product development team questions.
"""

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from langchain.prompts import PromptTemplate

logger = logging.getLogger(__name__)


class BaseFilter(ABC):
    """Base class for all question filters.

    Filters are used to identify and separate different types of questions
    in QA datasets. Each filter implementation should define its own
    classification logic.
    """

    def __init__(self):
        """Initialize the filter."""
        self.filter_cache = {}

    def load_cache(self, cache_file: str) -> bool:
        """Load filter cache from file if it exists.

        Args:
            cache_file: Path to the cache file

        Returns:
            True if cache was loaded successfully, False otherwise
        """
        if cache_file and os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    self.filter_cache = json.load(f)
                logger.info(f"Loaded {len(self.filter_cache)} cached filter results")
                return True
            except Exception as e:
                logger.warning(f"Error loading filter cache: {e}")
        return False

    def save_cache(self, cache_file: str) -> bool:
        """Save filter cache to file.

        Args:
            cache_file: Path to the cache file

        Returns:
            True if cache was saved successfully, False otherwise
        """
        if cache_file:
            try:
                with open(cache_file, "w") as f:
                    json.dump(self.filter_cache, f)
                logger.info(f"Saved {len(self.filter_cache)} filter results to cache")
                return True
            except Exception as e:
                logger.warning(f"Error saving filter cache: {e}")
        return False

    @abstractmethod
    def classify_questions(self, questions: List[str]) -> List[bool]:
        """Classify a batch of questions.

        Args:
            questions: List of questions to classify

        Returns:
            List of booleans where True means the question should be filtered out
        """
        pass

    def process_questions(
        self,
        qa_pairs: List[Tuple[str, str]],
        batch_size: int = 20,
        cache_file: Optional[str] = None,
    ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """Process questions using the filter.

        Args:
            qa_pairs: List of (question, answer) tuples
            batch_size: Number of questions to process in each batch
            cache_file: Optional path to a cache file to persist filter results

        Returns:
            Tuple of (kept_pairs, filtered_pairs)
        """
        # Load cache if provided
        if cache_file:
            self.load_cache(cache_file)

        # Process questions using cache
        kept_pairs = []
        filtered_pairs = []
        questions_to_process = []

        # First pass: check cache
        for q, a in qa_pairs:
            if q in self.filter_cache:
                if not self.filter_cache[q]:  # False means keep the question
                    kept_pairs.append((q, a))
                else:
                    filtered_pairs.append((q, a))
            else:
                questions_to_process.append((q, a))

        if not questions_to_process:
            logger.info("All questions found in cache, no classification needed")
        else:
            # Process uncached questions in batches
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
                batch_results = self.classify_questions([q for q, _ in batch])

                for (q, a), should_filter in zip(batch, batch_results):
                    self.filter_cache[q] = should_filter

                    if not should_filter:
                        kept_pairs.append((q, a))
                    else:
                        filtered_pairs.append((q, a))

                total_processed += len(batch)
                elapsed = time.time() - start_time
                rate = total_processed / elapsed if elapsed > 0 else 0
                logger.info(
                    f"Processed {total_processed}/{len(questions_to_process)} "
                    f"questions ({rate:.2f} q/s), found {len(filtered_pairs)} "
                    f"filtered questions"
                )

                # Small delay to avoid rate limiting
                time.sleep(0.1)

            # Save cache if provided
            if cache_file:
                self.save_cache(cache_file)

        logger.info(
            f"Filtered out {len(filtered_pairs)} questions "
            f"({len(filtered_pairs)/len(qa_pairs)*100:.1f}%)"
        )

        return kept_pairs, filtered_pairs


class ProductDevelopmentFilter(BaseFilter):
    """Filter for identifying questions related to product development.

    This filter distinguishes between:
    1. Client engineering questions (which should be kept)
    2. Product development team questions (which should be filtered out)

    Client engineering questions are technical questions that clients might ask
    about using the product, while product development questions are internal
    questions about building, maintaining, or deploying the product.
    """

    def __init__(self, llm=None):
        """Initialize the filter.

        Args:
            llm: Language model to use for classification
        """
        super().__init__()
        self.llm = llm

    def classify_questions(self, questions: List[str]) -> List[bool]:
        """Classify questions as product development or client questions.

        Args:
            questions: List of questions to classify

        Returns:
            List of booleans where True means the question is related to product
              development
        """
        if not self.llm:
            return [False] * len(questions)

        formatted_questions = "\n".join(
            [f"{i+1}. {q}" for i, q in enumerate(questions)]
        )

        # Define the prompt template with proper line breaks
        template = (
            "You are an expert at classifying content for the right audience.\n\n"
            "Below is a list of questions about a software product:\n\n"
            "{questions}\n\n"
            "For each question, determine if it is intended for PRODUCT DEVELOPMENT "
            "TEAMS (developers, testers, DevOps) or for CLIENTS (users of the product, "
            "including technical users and engineers who use the product).\n\n"
            "Product development questions typically involve:\n"
            "- Internal development processes\n"
            "- Code implementation details\n"
            "- Deployment infrastructure\n"
            "- Internal testing methodologies\n"
            "- CI/CD pipelines\n"
            "- Source code management\n"
            "- Internal architecture decisions\n\n"
            "Client questions typically involve:\n"
            "- How to use the product (even technical aspects)\n"
            "- Product features and capabilities\n"
            "- API usage and integration\n"
            "- Configuration options\n"
            "- Troubleshooting product issues\n"
            "- Performance characteristics\n"
            "- Technical specifications\n\n"
            "IMPORTANT: Technical or engineering questions from clients should NOT be "
            "filtered out. Only filter questions that would exclusively be asked by "
            "the internal product development team.\n\n"
            "Respond with ONLY a JSON array of boolean values (true/false), where:\n"
            "- true = question is for PRODUCT DEVELOPMENT TEAMS (filter out)\n"
            "- false = question is for CLIENTS (keep)\n\n"
            "Example response format:\n"
            "[false, true, false, false, true]"
        )

        prompt_template = PromptTemplate(
            input_variables=["questions"],
            template=template,
        )

        chain = prompt_template | self.llm

        try:
            response = chain.invoke({"questions": formatted_questions})

            content = ""
            if hasattr(response, "content"):
                content = str(response.content).strip()
            else:
                content = str(response).strip()

            import re

            json_match = re.search(r"\[.*\]", content)
            if json_match:
                json_str = json_match.group(0)
                try:
                    results = json.loads(json_str)
                    if len(results) != len(questions):
                        logger.warning(
                            f"Expected {len(questions)} results, got {len(results)}"
                        )
                        if len(results) < len(questions):
                            results.extend([False] * (len(questions) - len(results)))
                        else:
                            results = results[: len(questions)]
                    return results
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON from response: {json_str}")

            logger.warning("Could not extract valid classification from LLM response")
            return [False] * len(questions)

        except Exception as e:
            logger.warning(f"Error classifying questions: {e}")
            return [False] * len(questions)
