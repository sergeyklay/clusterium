# Usage Guide

This document provides detailed instructions for using the QA Dataset Clustering Tool (qadst).

## Command Line Interface

The `qadst` command-line tool provides a simple interface for clustering question-answer datasets.

### Basic Usage

```bash
qadst cluster --input your_data.csv --output clusters.csv
```

### Command Structure

The `qadst` tool uses a command-based structure:

```bash
qadst [global-options] COMMAND [command-options]
```

Available commands:
- `cluster`: Cluster text data using Dirichlet Process and Pitman-Yor Process
- `evaluate`: Evaluate clustering results using established metrics

### Command Line Options for `cluster`

| Option | Description | Default |
|--------|-------------|---------|
| `--input` | Path to input CSV file (required) | - |
| `--column` | Column name to use for clustering | "question" |
| `--output` | Output CSV file path | "clusters_output.csv" |
| `--output-dir` | Directory to save output files | "output" |
| `--alpha` | Concentration parameter for clustering | 1.0 |
| `--sigma` | Discount parameter for Pitman-Yor Process | 0.5 |
| `--cache-dir` | Directory to cache embeddings | ".cache" |

### Command Line Options for `evaluate`

| Option | Description | Default |
|--------|-------------|---------|
| `--input` | Path to input CSV file (required) | - |
| `--column` | Column name to use for clustering | "question" |
| `--dp-clusters` | Path to Dirichlet Process clustering results CSV (required) | - |
| `--pyp-clusters` | Path to Pitman-Yor Process clustering results CSV (required) | - |
| `--plot` | Generate evaluation plots | True |
| `--output-dir` | Directory to save output files | "output" |
| `--cache-dir` | Directory to cache embeddings | ".cache" |

### Examples

#### Basic Clustering

```bash
qadst cluster --input your_data.csv
```

#### Specifying Column Names

If your CSV file has a different column name for questions:

```bash
qadst cluster --input your_data.csv --column question_text --output clusters.csv
```

#### Adjusting Clustering Parameters

Fine-tune the clustering by adjusting the alpha and sigma parameters:

```bash
qadst cluster --input your_data.csv --alpha 0.5 --sigma 0.3
```

#### Generating Visualizations

Generate plots showing the distribution of cluster sizes:

```bash
# Generate linear scale plots
qadst cluster --input your_data.csv --plot linear

# Generate log-log scale plots
qadst cluster --input your_data.csv --plot log-log
```

These commands will save the plots as `cluster_distribution.png` and `cluster_distribution_log.png` in the output directory.

#### Specifying Output Directory

Save all output files to a specific directory:

```bash
qadst cluster --input your_data.csv --output-dir results
```

#### Evaluating Clustering Results

After running clustering, you can evaluate the quality of the clusters:

```bash
# Basic evaluation with interactive visualization dashboard
qadst evaluate --input your_data.csv --dp-clusters output/clusters_output_dp.csv --pyp-clusters output/clusters_output_pyp.csv

# Evaluation without visualizations
qadst evaluate --input your_data.csv --dp-clusters output/clusters_output_dp.csv --pyp-clusters output/clusters_output_pyp.csv --no-plot
```

This will generate evaluation metrics and visualizations comparing the quality of the Dirichlet Process and Pitman-Yor Process clustering results. When `--plot` is enabled (the default), the visualization will be displayed interactively in a Matplotlib window. The visualization dashboard includes:

1. Cluster size distribution (log-log scale)
2. Silhouette score comparison
3. Similarity metrics comparison (intra vs. inter-cluster)
4. Power-law fit visualization with Clauset's method

For cluster distribution visualizations, use the `cluster` command with the `--plot` option.

## Python API

You can also use the clustering functionality directly in your Python code.

### Basic Usage

```python
from qadst.clustering import DirichletProcess, PitmanYorProcess, EmbeddingCache
from qadst.clustering.utils import load_data_from_csv, save_clusters_to_json

# Load data
texts, data = load_data_from_csv("your_data.csv", column="question")

# Create cache provider
cache = EmbeddingCache(cache_dir=".cache")

# Perform Dirichlet Process clustering
dp = DirichletProcess(alpha=1.0, cache=cache)
clusters, params = dp.fit(texts)

# Save results
save_clusters_to_json("clusters.json", texts, clusters, "DP", data)
```

### Using Pitman-Yor Process

The Pitman-Yor Process often produces better clustering results for text data:

```python
# Perform Pitman-Yor Process clustering
pyp = PitmanYorProcess(alpha=1.0, sigma=0.5, cache=cache)
clusters_pyp, params_pyp = pyp.fit(texts)

# Save results
save_clusters_to_json("pyp_clusters.json", texts, clusters_pyp, "PYP", data)
```

### Evaluating Clusters

You can evaluate the quality of your clusters using the evaluation module:

```python
from qadst.evaluation import ClusterEvaluator, save_evaluation_report
from qadst.visualization import visualize_silhouette_score, visualize_evaluation_dashboard
import numpy as np

# Get embeddings for evaluation
embeddings = np.array([dp.get_embedding(text).cpu().numpy() for text in texts])

# Evaluate DP clusters
dp_evaluator = ClusterEvaluator(texts, embeddings, clusters, "DirichletProcess")
dp_report = dp_evaluator.generate_report()

# Check if clusters follow power-law distribution
powerlaw_params = dp_report["powerlaw_params"]
if powerlaw_params["is_powerlaw"]:
    print(f"DP clusters follow power-law with alpha={powerlaw_params['alpha']:.2f}")
else:
    print("DP clusters do not follow power-law distribution")

# Evaluate PYP clusters
pyp_evaluator = ClusterEvaluator(texts, embeddings, clusters_pyp, "PitmanYorProcess")
pyp_report = pyp_evaluator.generate_report()

# Compare results
reports = {
    "DirichletProcess": dp_report,
    "PitmanYorProcess": pyp_report,
}
save_evaluation_report(reports, "output")

# Generate visualizations
visualize_silhouette_score(reports, "output")
visualize_evaluation_dashboard(reports, "output")
```

### Customizing the Clustering Process

You can customize various aspects of the clustering process:

```python
# Custom alpha and sigma values
dp = DirichletProcess(alpha=0.5, cache=cache)
pyp = PitmanYorProcess(alpha=0.5, sigma=0.3, cache=cache)

# Custom embedding model (advanced)
from sentence_transformers import SentenceTransformer
custom_model = SentenceTransformer("all-mpnet-base-v2")  # Different model

# Custom similarity function (advanced)
def custom_similarity(text, cluster_param):
    # Your custom similarity logic here
    pass
```

## Output Files

The tool generates several output files:

- `*_dp.csv`: CSV file with Dirichlet Process clustering results
- `*_pyp.csv`: CSV file with Pitman-Yor Process clustering results
- `*_dp.json`: JSON file with Dirichlet Process clustering results
- `*_pyp.json`: JSON file with Pitman-Yor Process clustering results
- `qa_clusters.json`: Combined JSON file with clustering results
- `*_clusters.png`: Visualization of cluster size distributions (if `--plot` is specified)

### Evaluation Output Files

When using the `evaluate` command, the following files are generated:

- `evaluation_report.json`: JSON file containing evaluation metrics for both clustering methods
- `silhouette_comparison.png`: Visualization comparing silhouette scores of both methods (with `--plot silhouette`)
- `evaluation_dashboard.png`: Comprehensive dashboard visualization with multiple metrics (with `--plot dashboard`)

### JSON Output Format

The JSON output follows this structure:

```json
{
  "clusters": [
    {
      "id": 1,
      "representative": [
        {
          "question": "What is the capital of France?",
          "answer": "Paris is the capital of France."
        }
      ],
      "source": [
        {
          "question": "What is the capital of France?",
          "answer": "Paris is the capital of France."
        },
        {
          "question": "What city is the capital of France?",
          "answer": "Paris is the capital city of France."
        }
      ]
    }
  ]
}
```

Each cluster has:
- A unique ID
- A representative question-answer pair (typically the first item in the cluster)
- A list of source question-answer pairs that belong to the cluster

## Performance Considerations

- **Caching**: Embeddings are cached to speed up repeated runs. Use the `--cache-dir` option to specify a cache directory.
- **Memory Usage**: Large datasets may require significant memory, especially for the embedding model.
- **Processing Time**: The clustering process can be time-consuming for large datasets. The Pitman-Yor Process is typically faster than the Dirichlet Process.

## Troubleshooting

If you encounter issues:

1. Check your input CSV file format
2. Ensure you have sufficient memory for large datasets
3. Try adjusting the alpha and sigma parameters for better clustering results
4. Remember to use the correct command structure: `qadst cluster [options]` instead of just `qadst [options]`

For more help, please open an issue on the [GitHub repository](https://github.com/sergeyklay/qa-dataset-clustering/issues).
