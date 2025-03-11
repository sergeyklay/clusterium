# Clusterium

[![CI](https://github.com/sergeyklay/clusterium/actions/workflows/ci.yml/badge.svg)](https://github.com/sergeyklay/clusterium/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/sergeyklay/clusterium/graph/badge.svg?token=T5d9KTXtqP)](https://codecov.io/gh/sergeyklay/clusterium)

A toolkit for clustering, analyzing, and benchmarking text data using state-of-the-art embedding models and clustering algorithms.

## Features

- **Dirichlet Process Clustering**: Implements the Dirichlet Process for text clustering
- **Pitman-Yor Process Clustering**: Implements the Pitman-Yor Process for text clustering with improved performance
- **Evaluation**: Evaluates clustering results using a variety of metrics, including Silhouette Score, Davies-Bouldin Index, and Power-law Analysis
- **Visualization**: Generates plots of cluster size distributions

## Installation

For detailed installation instructions, please see [INSTALL.md](INSTALL.md).

### Quick Start

```bash
git clone https://github.com/sergeyklay/clusterium.git
cd clusterium
poetry install
```

## Usage

For detailed usage instructions, use cases, examples, and advanced configuration options, please see [USAGE.md](USAGE.md).

### Quick Start

```bash
# Run clustering
clusx --input your_data.csv --column your_column --output clusters.csv

# Evaluate clustering results and generate visualizations
clusx evaluate \
  --input input.csv \
  --column your_column \
  --dp-clusters output_dp.csv \
  --pyp-clusters output_pyp.csv \
  --plot
```

### Python API Example

```python
from clusx.clustering import DirichletProcess
from clusx.clustering.utils import load_data_from_csv, save_clusters_to_json

# Load data
texts, data = load_data_from_csv("your_data.csv", column="your_column")

# Perform clustering
dp = DirichletProcess(alpha=1.0)
clusters, params = dp.fit(texts)

# Save results
save_clusters_to_json("clusters.json", texts, clusters, "DP", data)
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
