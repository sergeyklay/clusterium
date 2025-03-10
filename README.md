# QA Dataset Clustering

A tool for clustering question-answer datasets using Dirichlet Process and Pitman-Yor Process.

## Features

- **Dirichlet Process Clustering**: Implements the Dirichlet Process for text clustering
- **Pitman-Yor Process Clustering**: Implements the Pitman-Yor Process for text clustering with improved performance
- **Embedding Caching**: Efficiently caches text embeddings to speed up repeated runs
- **CSV and JSON Output**: Saves clustering results in both CSV and JSON formats
- **Visualization**: Generates plots of cluster size distributions
- **Real Answers**: Includes real answers from the source data in the output

## Installation

```bash
# Clone the repository
git clone https://github.com/sergeyklay/qa-dataset-clustering.git
cd qa-dataset-clustering

# Install with Poetry
poetry install
```

## Usage

### Command Line Interface

```bash
# Basic usage
qa-cluster --input your_data.csv --output clusters.csv

# Specify column names
qa-cluster --input your_data.csv --column question --output clusters.csv

# Adjust clustering parameters
qa-cluster --input your_data.csv --alpha 0.5 --sigma 0.3

# Generate visualization
qa-cluster --input your_data.csv --plot

# Specify output directory
qa-cluster --input your_data.csv --output-dir results
```

### Python API

```python
from qa_dataset_clustering.clustering import DirichletProcess, PitmanYorProcess, EmbeddingCache
from qa_dataset_clustering.clustering.utils import load_data_from_csv, save_clusters_to_json

# Load data
texts, data = load_data_from_csv("your_data.csv", column="question")

# Create cache provider
cache = EmbeddingCache(cache_dir=".cache")

# Perform Dirichlet Process clustering
dp = DirichletProcess(alpha=1.0, cache=cache)
clusters, params = dp.fit(texts)

# Perform Pitman-Yor Process clustering
pyp = PitmanYorProcess(alpha=1.0, sigma=0.5, cache=cache)
clusters_pyp, params_pyp = pyp.fit(texts)

# Save results
save_clusters_to_json("clusters.json", texts, clusters, "DP", data)
```

## Output Format

The tool generates several output files:

- `*_dp.csv`: CSV file with Dirichlet Process clustering results
- `*_pyp.csv`: CSV file with Pitman-Yor Process clustering results
- `*_dp.json`: JSON file with Dirichlet Process clustering results
- `*_pyp.json`: JSON file with Pitman-Yor Process clustering results
- `qa_clusters.json`: Combined JSON file with clustering results
- `*_clusters.png`: Visualization of cluster size distributions (if `--plot` is specified)

The JSON format follows this structure:

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

## License

This project is licensed under the MIT License - see the LICENSE file for details.
