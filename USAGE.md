# Usage Guide

This document provides detailed instructions for using the `clusx` package.

## Command Line Interface

The `clusx` command-line tool provides a simple interface for clustering text data.

### Basic Usage

```bash
clusx cluster --input your_data.csv --column your_column --output clusters.csv
```

### Command Structure

The `clusx` tool uses a command-based structure:

```bash
clusx [global-options] COMMAND [command-options]
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
clusx cluster --input your_data.csv --column your_column --output clusters.csv
```

#### Specifying Column Names

If your CSV file has a different column name for questions:

```bash
clusx cluster --input your_data.csv --column question_text --output clusters.csv
```

#### Adjusting Clustering Parameters

Fine-tune the clustering by adjusting the alpha and sigma parameters:

```bash
clusx cluster --input your_data.csv --alpha 0.5 --sigma 0.3
```

#### Generating Visualizations

Visualizations are generated using the `evaluate` command, not the `cluster` command:

```bash
# Generate evaluation dashboard with visualizations (default)
clusx evaluate --input your_data.csv --dp-clusters output/clusters_output_dp.csv --pyp-clusters output/clusters_output_pyp.csv

# Skip visualization generation
clusx evaluate --input your_data.csv --dp-clusters output/clusters_output_dp.csv --pyp-clusters output/clusters_output_pyp.csv --no-plot
```

The evaluation dashboard includes visualizations of cluster size distributions, silhouette scores, similarity metrics, and power-law fits. These visualizations are saved as `evaluation_dashboard.png` in the output directory.

#### Specifying Output Directory

Save all output files to a specific directory:

```bash
clusx cluster --input your_data.csv --output-dir results
```

#### Evaluating Clustering Results

After running clustering, you can evaluate the quality of the clusters:

```bash
# Basic evaluation with interactive visualization dashboard
clusx evaluate --input your_data.csv --dp-clusters output/clusters_output_dp.csv --pyp-clusters output/clusters_output_pyp.csv

# Evaluation without visualizations
clusx evaluate --input your_data.csv --dp-clusters output/clusters_output_dp.csv --pyp-clusters output/clusters_output_pyp.csv --no-plot
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
from clusx.clustering import DirichletProcess, PitmanYorProcess, EmbeddingCache
from clusx.clustering.utils import load_data_from_csv, save_clusters_to_json

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
from clusx.evaluation import ClusterEvaluator, save_evaluation_report
from clusx.visualization import visualize_evaluation_dashboard
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

# Generate visualization dashboard
visualize_evaluation_dashboard(reports, "output", show_plot=True)
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

### Evaluation Output Files

When using the `evaluate` command, the following files are generated:

- `evaluation_report.json`: JSON file containing evaluation metrics for both clustering methods
- `evaluation_dashboard.png`: Comprehensive dashboard visualization with multiple metrics (when using `--plot`)

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

## Understanding Parameters and Output Files

### Key Parameters

The tool uses several important parameters that appear in different contexts:

1. **Clustering Parameters** (inputs to the clustering algorithms):
   - **alpha**: Concentration parameter that controls how likely the algorithm is to create new clusters. Higher values lead to more clusters.
     - Range: Typically 0.1 to 10.0
     - Default: 1.0
   - **sigma**: Discount parameter used only in the Pitman-Yor Process. Controls the power-law behavior of the cluster sizes.
     - Range: 0.0 to 0.9 (must be less than 1)
     - Default: 0.5
     - Note: When sigma=0, Pitman-Yor reduces to Dirichlet Process

2. **Power Law Parameters** (detected in the evaluation results):
   - **alpha**: Power law exponent that describes how quickly the probability of finding larger clusters decreases.
     - Typical values in natural phenomena: 2.0 to 3.0
     - Note: This is different from the clustering alpha parameter
   - **sigma_error**: Standard error of the power law alpha estimate, representing the uncertainty in the estimate.

### Example Output Files

#### Example CSV Output (`clusters_output_dp.csv`):

```csv
Text,Cluster_DP,Alpha,Sigma
"What is the capital of France?",0,1.0,0.0
"What city is the capital of France?",0,1.0,0.0
"How tall is the Eiffel Tower?",1,1.0,0.0
"What is the height of the Eiffel Tower?",1,1.0,0.0
"Who was the first president of the United States?",2,1.0,0.0
```

#### Example JSON Output (`clusters_output_dp.json`):

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
    },
    {
      "id": 2,
      "representative": [
        {
          "question": "How tall is the Eiffel Tower?",
          "answer": "The Eiffel Tower is 330 meters tall."
        }
      ],
      "source": [
        {
          "question": "How tall is the Eiffel Tower?",
          "answer": "The Eiffel Tower is 330 meters tall."
        },
        {
          "question": "What is the height of the Eiffel Tower?",
          "answer": "The Eiffel Tower stands at a height of 330 meters."
        }
      ]
    }
  ],
  "metadata": {
    "model_name": "DP",
    "alpha": 1.0,
    "sigma": 0.0
  }
}
```

#### Example Evaluation Report (excerpt from `evaluation_report.json`):

```json
{
  "Dirichlet": {
    "basic_metrics": {
      "model_name": "Dirichlet",
      "num_texts": 500,
      "num_clusters": 42,
      "alpha": 1.0,
      "sigma": 0.0
    },
    "silhouette_score": 0.32,
    "powerlaw_params": {
      "alpha": 2.45,
      "sigma_error": 0.18,
      "xmin": 1.0,
      "is_powerlaw": true
    }
  },
  "Pitman-Yor": {
    "basic_metrics": {
      "model_name": "Pitman-Yor",
      "num_texts": 500,
      "num_clusters": 38,
      "alpha": 1.0,
      "sigma": 0.5
    },
    "silhouette_score": 0.38,
    "powerlaw_params": {
      "alpha": 2.21,
      "sigma_error": 0.15,
      "xmin": 1.0,
      "is_powerlaw": true
    }
  }
}
```

### Interpreting the Parameters

- **Clustering alpha**: Higher values (e.g., 5.0) create more clusters, while lower values (e.g., 0.1) create fewer, larger clusters.
- **Sigma**: When sigma=0, the Pitman-Yor Process behaves like the Dirichlet Process. As sigma increases toward 1, the cluster size distribution becomes more power-law-like.
- **Power law alpha**: Values around 2.0 indicate a strong power-law behavior in the cluster sizes. The higher this value, the more rapidly the frequency of large clusters decreases.
- **sigma_error**: Smaller values indicate more confidence in the power law alpha estimate.

### Choosing Optimal Parameters

The best parameters depend on your specific dataset and clustering goals:

1. Start with the defaults (alpha=1.0, sigma=0.5)
2. If you want more clusters, increase alpha
3. If you want fewer clusters, decrease alpha
4. To get a more power-law-like distribution, increase sigma (for PYP only)
5. Evaluate the results using the evaluation metrics, especially silhouette score

The evaluation dashboard will help you compare different parameter settings and choose the optimal configuration for your dataset.

## Performance Considerations

- **Caching**: Embeddings are cached to speed up repeated runs. Use the `--cache-dir` option to specify a cache directory.
- **Memory Usage**: Large datasets may require significant memory, especially for the embedding model.
- **Processing Time**: The clustering process can be time-consuming for large datasets. The Pitman-Yor Process is typically faster than the Dirichlet Process.

## Troubleshooting

If you encounter issues:

1. Check your input CSV file format
2. Ensure you have sufficient memory for large datasets
3. Try adjusting the alpha and sigma parameters for better clustering results
4. Remember to use the correct command structure: `clusx cluster [options]` instead of just `clusx [options]`

For more help, please open an issue on the [GitHub repository](https://github.com/sergeyklay/clusterium/issues).
