# Installation Guide for qadst

This document provides detailed instructions for installing and setting up the `qadst` package.

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

For detailed usage instructions, examples, and advanced configuration options, please see [USAGE.md](USAGE.md).
