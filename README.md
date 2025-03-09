# QA Dataset Clustering Toolkit (QADST)

A toolkit for clustering, analyzing, and benchmarking question-answer datasets using state-of-the-art embedding models and clustering algorithms.

## Key Features

- **Semantic Clustering**: Group semantically similar questions using density-based clustering (HDBSCAN)
- **Intelligent Filtering**: Separate engineering-focused questions from end-user questions using LLM classification
- **Deduplication**: Remove semantically duplicate questions based on embedding similarity
- **Cluster Quality Assessment**: Evaluate clustering results using standard metrics and semantic coherence
- **Topic Labeling**: Generate descriptive topic labels for clusters using LLMs or TF-IDF/NMF
- **Comprehensive Reporting**: Generate detailed reports on cluster quality and composition

## Introduction

QADST was developed to solve a specific challenge in Retrieval-Augmented Generation (RAG) systems: creating high-quality, diverse, and representative question-answer datasets for evaluation. When building RAG systems, practitioners often struggle with generating reliable benchmark datasets that adequately cover the knowledge domain and provide meaningful evaluation metrics.

This toolkit addresses the following key challenges:

- **Dataset Quality Assessment**: Determining whether a generated QA dataset is "good enough" as a benchmark
- **Redundancy Elimination**: Identifying and removing semantically similar questions that don't add evaluation value
- **Domain Coverage Analysis**: Ensuring the dataset represents all important aspects of source documents
- **Engineering vs. End-User Focus**: Separating technical questions from those that actual users would ask

### Example Use Case

#### Input

Let's assume we've collected company documentation from various sources into a really big RAG system. We then generated an initial dataset based on these documents to use as a benchmark for future evaluations, all compiled into a single CSV file as follows:

```csv
question,answer
How do I configure the API endpoint?,The API endpoint can be configured in the settings.json file under the "endpoints" section.
What's the process for setting up API endpoints?,To set up an API endpoint, navigate to settings.json and modify the "endpoints" section.
How do users reset their passwords?,Users can reset passwords by clicking "Forgot Password" on the login screen.
What's the expected latency for our API when deployed in the EU region with the new load balancer configuration?,The expected latency should be under 100ms for 99% of requests when using the recommended instance types and proper connection pooling.
```

#### Processing Steps

1. **Deduplication**: The first two questions are semantically similar (93% similarity) - both are kept in the same cluster with the first as the representative.
2. **Filtering**: The fourth question is identified as engineering-focused (discussing infrastructure details and performance metrics) and filtered out from the end-user dataset.
3. **Clustering**: The remaining questions are grouped by semantic similarity.

#### Output

Organized clusters in JSON format:

```json
{
  "clusters": [
    {
      "id": 1,
      "representative": [
        {
          "question": "How do I configure the API endpoint?",
          "answer": "The API endpoint can be configured in the settings.json file under the \"endpoints\" section."
        }
      ],
      "source": [
        {
          "question": "How do I configure the API endpoint?",
          "answer": "The API endpoint can be configured in the settings.json file under the \"endpoints\" section."
        },
        {
          "question": "What's the process for setting up API endpoints?",
          "answer": "To set up an API endpoint, navigate to settings.json and modify the \"endpoints\" section."
        }
      ]
    },
    {
      "id": 2,
      "representative": [
        {
          "question": "How do users reset their passwords?",
          "answer": "Users can reset passwords by clicking \"Forgot Password\" on the login screen."
        }
      ],
      "source": [
        {
          "question": "How do users reset their passwords?",
          "answer": "Users can reset passwords by clicking \"Forgot Password\" on the login screen."
        }
      ]
    }
  ]
}
```

#### Additional Outputs

- `qa_cleaned.csv`: CSV file containing deduplicated and filtered QA pairs
- `engineering_questions.csv`: CSV file containing questions identified as engineering-focused (including the latency question)
- `cluster_quality_report.csv`: Detailed metrics on cluster quality and coherence (when running `qadst benchmark`)

This transformation enables RAG system developers to:
1. Evaluate against a diverse, non-redundant set of questions (deduplication)
2. Focus on questions that end-users would actually ask (filtering)
3. Organize questions into meaningful semantic groups (clustering)
4. Quantitatively measure dataset quality before using it for evaluation

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
- Statistics including the number of clusters, original count, and deduplicated count

### Benchmarking Output

The benchmarking process produces:
- A CSV report with cluster quality metrics
- Topic labels for each cluster
- Coherence scores for each cluster
- Enhanced existing clusters JSON file with:
  - Per-cluster metrics (source_count, avg_similarity, coherence_score, topic_label)
  - Global metrics (noise_ratio, davies_bouldin_score, calinski_harabasz_score, silhouette_score)

Example of enhanced clusters JSON (original file is preserved and enhanced with metrics):
```json
{
  "clusters": [
    {
      "id": 1,
      "representative": [...],
      "source": [...],
      "source_count": 15,
      "avg_similarity": 0.82,
      "coherence_score": 0.82,
      "topic_label": "Password Reset Process"
    },
    ...
  ],
  "metrics": {
    "noise_ratio": 0.05,
    "davies_bouldin_score": 0.76,
    "calinski_harabasz_score": 42.3,
    "silhouette_score": 0.68
  }
}
```

## Example Workflow

1. **Prepare your QA dataset** in CSV format
2. **Run clustering** to group similar questions
   ```bash
   qadst cluster --input data/qa_pairs.csv
   ```
3. **Run benchmarking** to evaluate cluster quality
   ```bash
   qadst benchmark --clusters output/qa_hdbscan_clusters.json --qa-pairs data/qa_pairs.csv --use-llm
   ```
4. **Analyze the results** in the output directory

## Advanced Usage

### Customizing Clustering Parameters

The HDBSCAN algorithm parameters are carefully tuned based on academic research:

- **Logarithmic Scaling**: `min_cluster_size` scales logarithmically with dataset size, following research showing that semantic clusters should grow sublinearly with dataset size
- **Optimal Parameters**: Uses `min_samples=5` and `cluster_selection_epsilon=0.3` based on benchmarks from clustering literature
- **Excess of Mass**: Uses EOM cluster selection method for better handling of varying density clusters

For example, with a dataset of 3000 questions, the toolkit automatically sets `min_cluster_size=64`, which represents the smallest meaningful semantic group in the data.

### Embedding Caching

The toolkit automatically caches embeddings to avoid recomputing them across runs, which improves performance for repeated operations on the same dataset:

- Embeddings are cached in the `output_dir/embedding_cache` directory
- Each cache file is named based on the embedding model and content hash
- Both deduplication and clustering operations benefit from the cache
- The cache is automatically invalidated when the dataset changes

This means that the first run might take longer as embeddings are computed and cached, but subsequent runs will be much faster as they reuse the cached embeddings.

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
