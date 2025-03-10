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

For detailed installation instructions, please see [INSTALL.md](INSTALL.md).

Quick start:
```bash
git clone https://github.com/sergeyklay/qa-dataset-clustering.git
cd qa-dataset-clustering
poetry install
```

## Usage

For detailed usage instructions, please see [USAGE.md](USAGE.md).

### Quick Start

```bash
# Basic usage
qadst --input your_data.csv --output clusters.csv

# Generate visualization
qadst --input your_data.csv --plot
```

### Python API Example

```python
from qadst.clustering import DirichletProcess
from qadst.clustering.utils import load_data_from_csv, save_clusters_to_json

# Load data
texts, data = load_data_from_csv("your_data.csv")

# Perform clustering
dp = DirichletProcess(alpha=1.0)
clusters, params = dp.fit(texts)

# Save results
save_clusters_to_json("clusters.json", texts, clusters, "DP", data)
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
