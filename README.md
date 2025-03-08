# QA Dataset Clustering Toolkit (QADST)

A toolkit for clustering, analyzing, and benchmarking question-answer datasets using state-of-the-art embedding models and clustering algorithms.

## Overview

QADST is designed to help data engineers, LLM specialists, and researchers organize large question-answer datasets into semantically meaningful clusters. The toolkit provides a comprehensive pipeline for processing QA pairs, including deduplication, filtering, clustering, and quality assessment.

## Key Features

- **Semantic Clustering**: Group semantically similar questions using density-based clustering (HDBSCAN)
- **Intelligent Filtering**: Separate engineering-focused questions from end-user questions using LLM classification
- **Deduplication**: Remove semantically duplicate questions based on embedding similarity
- **Cluster Quality Assessment**: Evaluate clustering results using standard metrics and semantic coherence
- **Topic Labeling**: Generate descriptive topic labels for clusters using LLMs or TF-IDF/NMF
- **Comprehensive Reporting**: Generate detailed reports on cluster quality and composition

## Technical Details

### Algorithms

#### HDBSCAN Clustering

The toolkit implements the Hierarchical Density-Based Spatial Clustering of Applications with Noise (HDBSCAN) algorithm, which offers several advantages for QA dataset clustering:

- Automatically determines the optimal number of clusters
- Identifies outliers as noise points
- Handles clusters of varying densities and shapes
- Adapts to dataset size with configurable minimum cluster size

References:
- Campello, R.J.G.B., Moulavi, D., Sander, J. (2013) Density-Based Clustering Based on Hierarchical Density Estimates
- McInnes, L., Healy, J., Astels, S. (2017) HDBSCAN: Hierarchical density based clustering

#### Post-Processing Techniques

- **Noise Point Recovery**: K-means clustering is applied to noise points to recover potentially useful groups
- **Large Cluster Splitting**: K-means is used to split overly large clusters into more coherent subclusters
- **LLM-based Filtering**: Uses language models to classify questions as engineering-focused or client-focused

#### Evaluation Metrics

The toolkit evaluates clustering quality using:

- **Davies-Bouldin Index**: Measures cluster separation (lower is better)
- **Calinski-Harabasz Index**: Measures cluster definition (higher is better)
- **Silhouette Score**: Measures cluster coherence (-1 to 1, higher is better)
- **Semantic Coherence**: Average pairwise cosine similarity between question embeddings

## Installation

### Requirements

- Python 3.12.x
- Poetry (dependency management)

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

## Usage

### Command Line Interface

The toolkit provides a command-line interface for easy use:

#### Clustering QA Pairs

```bash
qadst cluster --input data/qa_pairs.csv --output-dir ./output --filter
```

#### Benchmarking Clustering Results

```bash
qadst benchmark --clusters output/qa_clusters.json --qa-pairs data/qa_pairs.csv --use-llm
```

### Options

- `--embedding-model`: Embedding model to use (default: text-embedding-3-large)
- `--llm-model`: LLM model to use for filtering and topic labeling (default: gpt-4o)
- `--filter/--no-filter`: Enable/disable filtering of engineering questions
- `--output-dir`: Directory to save output files (default: ./output)
- `--use-llm/--no-llm`: Enable/disable LLM for topic labeling

## Input Format

The toolkit expects a CSV file with at least two columns:
- First column: questions
- Second column: answers

Example:
```csv
question,answer
How do I reset my password?,You can reset your password by clicking on the "Forgot Password" link.
What payment methods do you accept?,We accept credit cards, PayPal, and bank transfers.
```

## Output

### Clustering Output

The clustering process produces:
- A JSON file containing the clustering results
- A cleaned CSV file with deduplicated and filtered QA pairs
- A CSV file with engineering-focused questions (if filtering is enabled)

### Benchmarking Output

The benchmarking process produces:
- A CSV report with cluster quality metrics
- Topic labels for each cluster
- Coherence scores for each cluster

## Example Workflow

1. **Prepare your QA dataset** in CSV format
2. **Run clustering** to group similar questions
   ```bash
   qadst cluster --input data/qa_pairs.csv
   ```
3. **Run benchmarking** to evaluate cluster quality
   ```bash
   qadst benchmark --clusters output/qa_hdbscan_clusters.json --qa-pairs data/qa_pairs.csv
   ```
4. **Analyze the results** in the output directory

## Advanced Usage

### Customizing Clustering Parameters

The HDBSCAN algorithm automatically adapts to your dataset size, but you can customize the behavior by modifying the code:

- Minimum cluster size is calculated based on dataset size
- Cluster selection method uses Excess of Mass (EOM)
- Small epsilon (0.1) is used to merge very similar clusters

### Using Different Embedding Models

The toolkit supports any embedding model available through the OpenAI API. You can specify a different model using the `--embedding-model` option.

## For Researchers

The toolkit is designed to facilitate research on QA datasets:

- **Reproducible Results**: Fixed random seeds ensure reproducible clustering
- **Comprehensive Metrics**: Standard clustering metrics for quantitative evaluation
- **Semantic Analysis**: Tools for analyzing semantic relationships between questions
- **Topic Extraction**: Methods for extracting and analyzing cluster topics

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
