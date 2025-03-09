# Usage Guide

This document provides detailed usage instructions for the `qadst` package.

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

- `--embedding-model`: Embedding model to use (default: text-embedding-3-large). See [Using Different Embedding Models](#using-different-embedding-models) for details.
- `--llm-model`: LLM model to use for filtering and topic labeling (default: gpt-4o)
- `--filter/--no-filter`: Enable/disable filtering of engineering questions
- `--output-dir`: Directory to save output files (default: ./output)
- `--use-llm/--no-llm`: Enable/disable LLM for topic labeling
- `--min-cluster-size`: Minimum size of clusters (if not provided, calculated automatically)
- `--min-samples`: HDBSCAN min_samples parameter (default: 5)
- `--cluster-selection-epsilon`: HDBSCAN cluster_selection_epsilon parameter (default: 0.3)
- `--cluster-selection-method`: HDBSCAN cluster selection method (default: eom, alternative: leaf)
- `--keep-noise/--cluster-noise`: Keep noise points unclustered or force them into clusters (default: --cluster-noise)

Some options can/should also be configured via environment variables in a `.env` file:

```
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
```

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

## Example Use Case

### Input

Let's assume we've collected company documentation from various sources into a really big RAG system. We then generated an initial dataset based on these documents to use as a benchmark for future evaluations, all compiled into a single CSV file as follows:

```csv
question,answer
How do I configure the API endpoint?,The API endpoint can be configured in the settings.json file under the "endpoints" section.
What's the process for setting up API endpoints?,To set up an API endpoint, navigate to settings.json and modify the "endpoints" section.
How do users reset their passwords?,Users can reset passwords by clicking "Forgot Password" on the login screen.
What's the expected latency for our API when deployed in the EU region with the new load balancer configuration?,The expected latency should be under 100ms for 99% of requests when using the recommended instance types and proper connection pooling.
```

### Processing Steps

1. **Deduplication**: The first two questions are semantically similar (93% similarity) - both are kept in the same cluster with the first as the representative.
2. **Filtering**: The fourth question is identified as engineering-focused (discussing infrastructure details and performance metrics) and filtered out from the end-user dataset.
3. **Clustering**: The remaining questions are grouped by semantic similarity.

### Output

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

### Additional Outputs

- `qa_cleaned.csv`: CSV file containing deduplicated and filtered QA pairs
- `engineering_questions.csv`: CSV file containing questions identified as engineering-focused (including the latency question)
- `cluster_quality_report.csv`: Detailed metrics on cluster quality and coherence (when running `qadst benchmark`)

This transformation enables RAG system developers to:
1. Evaluate against a diverse, non-redundant set of questions (deduplication)
2. Focus on questions that end-users would actually ask (filtering)
3. Organize questions into meaningful semantic groups (clustering)
4. Quantitatively measure dataset quality before using it for evaluation

### Examples of Dataset Clustering Results

Below are some examples of dataset clustering and benchmarking results interpretations.

**Example 1:** Poor Quality Clustering

| Metric                    | Threshold   | Current Value                 | Interpretation                                           |
| ------------------------- | ----------- | ----------------------------- | -------------------------------------------------------- |
| Davies-Bouldin Index      | <1.0 ideal  | 4.38                          | Poor cluster separation (clusters overlap significantly) |
| Calinski-Harabasz Score   | >100 good   | 54.66                         | Weak cluster density (clusters are not compact)          |
| Low Coherence Clusters    | >0.4 target | 13,18,20,25,26,29 (0.14-0.34) | Mixed/irrelevant QA pairs in same cluster                |

Possible steps to improve: Adjust HDBSCAN Parameters

## Example Workflow

1. **Prepare your QA dataset** in CSV format
2. **Run clustering** to group similar questions
   ```bash
   # Basic clustering with default parameters
   qadst cluster --input data/qa_pairs.csv

   # Or with custom HDBSCAN parameters
   qadst cluster --input data/qa_pairs.csv --min-cluster-size 50 --min-samples 3 --cluster-selection-epsilon 0.2

   # Or preserve noise points for better cluster quality
   qadst cluster --input data/qa_pairs.csv --keep-noise

   # Or use leaf cluster selection method for more fine-grained clusters
   qadst cluster --input data/qa_pairs.csv --cluster-selection-method leaf
   ```
   - The clustering process automatically handles large clusters using recursive HDBSCAN to maintain density-based properties
   - Embeddings are automatically cached for faster processing in subsequent runs (see [Embedding Caching](#embedding-caching))
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
- **Cluster Selection Method**: Controls how HDBSCAN selects clusters from the hierarchy:
  - `eom` (default): Excess of Mass - selects clusters based on the stability of clusters, often resulting in varying cluster sizes
  - `leaf`: Selects leaf nodes from the cluster hierarchy, which can result in more fine-grained clusters

For example, with a dataset of 3000 questions, the toolkit automatically sets `min_cluster_size=64`, which represents the smallest meaningful semantic group in the data.

You can override these default parameters using command-line options:

```bash
# Example: Set custom HDBSCAN parameters
qadst cluster --input data/qa_pairs.csv --min-cluster-size 50 --min-samples 3 --cluster-selection-epsilon 0.2

# Example: Use leaf cluster selection method for more fine-grained clusters
qadst cluster --input data/qa_pairs.csv --cluster-selection-method leaf
```

This allows you to fine-tune the clustering process for your specific dataset characteristics.

### Noise Point Handling

HDBSCAN naturally identifies outliers as "noise points" - data points that don't fit well into any cluster based on density criteria. By default, the toolkit attempts to recover potentially useful information from these noise points by applying K-means clustering to them.

You can control this behavior with the `--keep-noise/--cluster-noise` option:

- **--cluster-noise** (default): Force noise points into clusters using K-means
  - Ensures all questions are assigned to a cluster
  - May reduce overall cluster quality by including outliers
  - Results in noise_ratio=0.00 in metrics

- **--keep-noise**: Keep noise points unclustered
  - Preserves HDBSCAN's density-based clustering decisions
  - Improves cluster coherence and separation metrics
  - Creates a special "noise" cluster (id: 0) in the output
  - Shows true noise_ratio in metrics

Example:
```bash
# Keep noise points unclustered for better cluster quality
qadst cluster --input data/qa_pairs.csv --keep-noise
```

This option is particularly useful when:
- You want to identify truly outlier questions
- You're prioritizing cluster quality over complete coverage
- Your dataset contains questions that genuinely don't fit into semantic groups

### Large Cluster Handling

The toolkit employs a sophisticated approach to handle large clusters:

- **Recursive HDBSCAN**: Large clusters (exceeding 20% of total questions or 50 questions, whichever is larger) are processed using a recursive application of HDBSCAN with stricter parameters:
  - More aggressive scaling for `min_cluster_size` using `int(np.log(cluster_size) ** 1.5)`
  - Tighter `cluster_selection_epsilon` (0.2 instead of the default 0.3)
  - Slightly lower `min_samples` (3) to allow for smaller but still meaningful subclusters
  - Maintains the density-based nature of the original clustering

- **Adaptive Fallback**: If recursive HDBSCAN cannot effectively split a large cluster (produces 1 or fewer subclusters), the system falls back to K-means:
  - Number of subclusters is determined adaptively based on cluster size
  - Aims for approximately 30 questions per subcluster
  - Caps at 10 subclusters to avoid excessive fragmentation

This approach ensures that the natural density structure of the data is preserved whenever possible, while still providing effective splitting of large, unwieldy clusters.

### Using Different Embedding Models

The toolkit supports various embedding models through the `--embedding-model` parameter. The choice of embedding model can significantly impact clustering quality and performance.

#### Available Models

By default, the toolkit uses OpenAI's `text-embedding-3-large` model, which provides high-quality embeddings for semantic clustering. The toolkit supports any embedding model available through the OpenAI API, including:

```bash
# Use OpenAI's text-embedding-3-small model (faster, smaller)
qadst cluster --input data/qa_pairs.csv --embedding-model text-embedding-3-small

# Use OpenAI's text-embedding-ada-002 model (legacy)
qadst cluster --input data/qa_pairs.csv --embedding-model text-embedding-ada-002
```

You can also configure the default embedding model in your `.env` file:

```
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
```

#### Model Selection Considerations

- **Quality vs. Speed**: Larger models generally provide better semantic understanding but may be slower and more expensive.
- **Dimensionality**: Different models produce embeddings with different dimensions, which can affect clustering behavior.
- **Domain Specificity**: Some models may perform better for specific domains or languages.

For optimal results, consider benchmarking different embedding models on a subset of your data to determine which provides the best clustering for your specific use case.

### Embedding Caching

The toolkit implements an efficient caching system for embeddings to improve performance and reduce API costs when working with large datasets or running multiple experiments.

#### How Caching Works

1. **Memory Cache**: Embeddings are stored in memory during a session to avoid redundant API calls.
2. **Disk Cache**: Embeddings are also saved to disk in the output directory, allowing reuse across different runs.
3. **Deterministic Hashing**: A hash of the input questions is used as a cache key, ensuring that the same questions always use cached embeddings.
4. **Automatic Invalidation**: The cache is automatically invalidated when the dataset changes, ensuring you always get correct results.

#### Cache Files

Cache files are stored in the output directory with names following this pattern:
```
embeddings_{model_name}_{hash}.npy
```

For example:
```
embeddings_text-embedding-3-large_a1b2c3d4.npy
```

#### Benefits of Caching

- **Reduced API Costs**: Minimizes the number of API calls to embedding services.
- **Faster Execution**: Significantly speeds up repeated runs with the same or overlapping datasets.
- **Consistent Results**: Ensures the same embeddings are used across different runs for reproducibility.
- **Performance Optimization**: The first run might take longer as embeddings are computed and cached, but subsequent runs will be much faster as they reuse the cached embeddings.

Both deduplication and clustering operations benefit from the cache, making iterative experimentation with different parameters much more efficient.

#### Clearing the Cache

If you need to regenerate embeddings (e.g., after updating to a new version of an embedding model), simply delete the cache files from your output directory:

```bash
# Remove all embedding cache files
rm output/embeddings_*.npy
```

### Customizing Report Output

The toolkit provides a flexible reporting system that allows you to control how clustering results are presented. By default, two reporters are enabled:

1. **CSV Reporter**: Generates a CSV file with detailed metrics for each cluster
2. **Console Reporter**: Displays summary information and top clusters in the console

You can customize which reporters are enabled using the `--reporters` option:

```bash
# Use only the CSV reporter
qadst benchmark --clusters output/qa_clusters.json --qa-pairs data/qa_pairs.csv --reporters csv

# Use only the console reporter
qadst benchmark --clusters output/qa_clusters.json --qa-pairs data/qa_pairs.csv --reporters console

# Use both reporters (default)
qadst benchmark --clusters output/qa_clusters.json --qa-pairs data/qa_pairs.csv --reporters csv,console
```

#### CSV Reporter Output

The CSV reporter generates a file named `cluster_quality_report.csv` in the output directory with the following columns:

- **Cluster_ID**: Unique identifier for each cluster
- **Num_QA_Pairs**: Number of question-answer pairs in the cluster
- **Avg_Similarity**: Average pairwise similarity between questions in the cluster
- **Coherence_Score**: Semantic coherence score for the cluster (higher is better)
- **Topic_Label**: Descriptive label for the cluster content

The file also includes a summary row with global metrics.

#### Console Reporter Output

The console reporter displays a formatted table view with:

1. **Summary Section**:
   - Total number of QA pairs in the dataset
   - Path to the clusters JSON file
   - Global metrics (Noise Ratio, Davies-Bouldin Index, Calinski-Harabasz Index, Silhouette Score)

2. **Top Clusters Table**:
   - Displays the top 5 clusters by size in a formatted table
   - Columns include Cluster ID, Size, Coherence, and Topic
   - Topics are truncated if they're too long to fit in the display

Example output:
```
--------------------------------------------------------------------------------
|                           CLUSTER ANALYSIS SUMMARY                           |
--------------------------------------------------------------------------------
Total QA pairs: 5175
Clusters JSON: output/qa_clusters.json

Global Metrics:
  Noise Ratio: 0.00
  Davies-Bouldin Index: 3.95
  Calinski-Harabasz Index: 35.91
  Silhouette Score: 0.12

--------------------------------------------------------------------------------
|                            TOP 5 CLUSTERS BY SIZE                            |
--------------------------------------------------------------------------------
| Cluster ID | Size         | Coherence    | Topic                             |
--------------------------------------------------------------------------------
| 1          | 582          | 0.37         | API Endpoint Behavior             |
| 2          | 537          | 0.49         | Digital Document Workflows        |
| 17         | 206          | 0.14         | Customer Trust Insights           |
| 6          | 168          | 0.43         | API Integration Guidelines        |
| 41         | 158          | 0.52         | Feature-Specific Integrations     |
--------------------------------------------------------------------------------
```

#### Extending the Reporting System

The reporting system is designed to be extensible. Developers can create custom reporters by:

1. Implementing a class that inherits from `BaseReporter`
2. Implementing the `generate_report` method
3. Registering the reporter with the `ReporterRegistry`

This allows for additional output formats such as HTML, JSON, or integration with other systems.
