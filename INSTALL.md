# Installation Guide for QA Dataset Clustering Toolkit

This document provides detailed instructions for installing and setting up the QA Dataset Clustering Toolkit (QADST).

## Requirements

- Python 3.12.x
- Poetry (dependency management)

## Basic Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/qa-dataset-clustering.git
cd qa-dataset-clustering

# Install dependencies using Poetry
poetry install

# Activate the virtual environment
poetry shell
```

## Environment Setup

Create a `.env` file in the project root with your API keys:

```
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
```

## Verifying Installation

After installation, you can verify that everything is working correctly by running the tests:

```bash
python -m pytest
```

## Using Different Embedding Models

The toolkit supports any embedding model available through the OpenAI API. You can specify a different model using the `--embedding-model` option when running commands.

## Embedding Caching

The toolkit automatically caches embeddings to avoid recomputing them across runs, which improves performance for repeated operations on the same dataset:

- Embeddings are cached in the `output_dir/embedding_cache` directory
- Each cache file is named based on the embedding model and content hash
- Both deduplication and clustering operations benefit from the cache
- The cache is automatically invalidated when the dataset changes

This means that the first run might take longer as embeddings are computed and cached, but subsequent runs will be much faster as they reuse the cached embeddings.
