# QA Dataset Clustering Toolkit (QADST) - Usage Guide

This document provides detailed usage instructions for the QA Dataset Clustering Toolkit (QADST).

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
- `--min-cluster-size`: Minimum size of clusters (if not provided, calculated automatically)
- `--min-samples`: HDBSCAN min_samples parameter (default: 5)
- `--cluster-selection-epsilon`: HDBSCAN cluster_selection_epsilon parameter (default: 0.3)

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
   # Basic clustering with default parameters
   qadst cluster --input data/qa_pairs.csv

   # Or with custom HDBSCAN parameters
   qadst cluster --input data/qa_pairs.csv --min-cluster-size 50 --min-samples 3 --cluster-selection-epsilon 0.2
   ```
   - The clustering process automatically handles large clusters using recursive HDBSCAN to maintain density-based properties
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

You can override these default parameters using command-line options:

```bash
# Example: Set custom HDBSCAN parameters
qadst cluster --input data/qa_pairs.csv --min-cluster-size 50 --min-samples 3 --cluster-selection-epsilon 0.2
```

This allows you to fine-tune the clustering process for your specific dataset characteristics.

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
