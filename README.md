# QA Dataset Clustering Toolkit (QADST)

[![CI](https://github.com/sergeyklay/qa-dataset-clustering/actions/workflows/ci.yml/badge.svg)](https://github.com/sergeyklay/qa-dataset-clustering/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/sergeyklay/qa-dataset-clustering/graph/badge.svg?token=T5d9KTXtqP)](https://codecov.io/gh/sergeyklay/qa-dataset-clustering)

A toolkit for clustering, analyzing, and benchmarking question-answer datasets using state-of-the-art embedding models and clustering algorithms.

## Key Features

- **Semantic Clustering**: Group semantically similar questions using density-based clustering (HDBSCAN)
- **Recursive Cluster Refinement**: Maintain density-based properties when splitting large clusters using recursive HDBSCAN
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

#### Examples of Dataset Cluesting

Bellow are some examples of dataset clustering and benchmarking results interpretations.

**Example 1:** Pour Quality Clustering

| Metric                    | Threshold   | Current Value                 | Interpretation                                           |
| ------------------------- | ----------- | ----------------------------- | -------------------------------------------------------- |
| Davies-Bouldin Index      | <1.0 ideal  | 4.38                          | Poor cluster separation (clusters overlap significantly) |
| Calinski-Harabasz Score   | >100 good   | 54.66                         | Weak cluster density (clusters are not compact)          |
| Low Coherence Clusters    | >0.4 target | 13,18,20,25,26,29 (0.14-0.34) | Mixed/irrelevant QA pairs in same cluster                |

Possible steps to improve: Adjust HDBSCAN Parameters

## Technical Details

### Algorithms

#### HDBSCAN Clustering

The toolkit implements the Hierarchical Density-Based Spatial Clustering of Applications with Noise (HDBSCAN) algorithm, which offers several advantages for QA dataset clustering:

- Automatically determines the optimal number of clusters
- Identifies outliers as noise points
- Handles clusters of varying densities and shapes
- Adapts to dataset size with configurable minimum cluster size

#### Post-Processing Techniques

- **Noise Point Recovery**: K-means clustering is applied to noise points to recover potentially useful groups
- **Large Cluster Handling**: Recursive HDBSCAN with stricter parameters is applied to large clusters to maintain density-based clustering properties, with K-means as a fallback only when necessary
- **LLM-based Filtering**: Uses language models to classify questions as engineering-focused or client-focused

#### Evaluation Metrics

The toolkit evaluates clustering quality using:

- **Davies-Bouldin Index**: Measures cluster separation (lower is better)
- **Calinski-Harabasz Index**: Measures cluster definition (higher is better)
- **Silhouette Score**: Measures cluster coherence (-1 to 1, higher is better)
- **Semantic Coherence**: Average pairwise cosine similarity between question embeddings

## Installation

For detailed installation instructions, please see [INSTALL.md](INSTALL.md).

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/qa-dataset-clustering.git
cd qa-dataset-clustering

# Install dependencies using Poetry
poetry install

# Set up environment variables
cp .env.example .env  # Then edit .env with your API keys
```

## Usage

For detailed usage instructions, examples, and advanced configuration options, please see [USAGE.md](USAGE.md).

## References:

- Moulavi D, Jaskowiak PA, Campello RJGB, Zimek A, Sander J. 2014. Density-based clustering validation. In: Proceedings of the 2014 SIAM International Conference on Data Mining. Philadelphia (PA): Society for Industrial and Applied Mathematics. p. 839–847. [doi:10.1137/1.9781611973440.96](https://doi.org/10.1137/1.9781611973440.96).
- McInnes L, Healy J. 2017. Accelerated hierarchical density clustering. arXiv:1705.07321 [stat.ML]. [doi:10.48550/arXiv.1705.07321](https://doi.org/10.48550/arXiv.1705.07321).
- Schubert E, Sander J, Ester M, Kriegel HP, Xu X. 2017. CM Transactions on Database Systems. Volume 42, Issue 3. New York (NY): Association for Computing Machinery. [doi:10.1145/3129336](https://doi.org/10.1145/3129336).
- Davies DL, Bouldin DW. 1979. A cluster separation measure. IEEE Transactions on Pattern Analysis and Machine Intelligence. 1(2):224–227. [doi:10.1109/TPAMI.1979.4766909](https://doi.org/10.1109/TPAMI.1979.4766909).
- Caliński T, Harabasz J. 1974. A dendrite method for cluster analysis. Communications in Statistics. 3(1):1–27. [doi:10.1080/03610927408827101](https://doi.org/10.1080/03610927408827101).
- Rousseeuw PJ. 1987. Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. Journal of Computational and Applied Mathematics. 20:53–65. [doi:10.1016/0377-0427(87)90125-7](https://doi.org/10.1016/0377-0427(87)90125-7).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
