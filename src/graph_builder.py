"""
Builds a PyTorch Geometric graph from the disease co-occurrence matrix.
Each node = disease, each edge = co-occurrence above threshold.
"""

import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data


def build_graph(cooccurrence_csv: str, threshold: int = 10):
    """
    Converts co-occurrence matrix into a PyTorch Geometric Data object.

    Args:
        cooccurrence_csv: Path to the co-occurrence matrix CSV
        threshold: Minimum co-occurrence count to create an edge

    Returns:
        data: PyTorch Geometric Data object
        disease_names: List of disease names (node labels)
    """
    df = pd.read_csv(cooccurrence_csv, index_col=0)
    disease_names = list(df.index)
    n = len(disease_names)

    # Node features: degree (how many comorbidities each disease has)
    matrix = df.values
    degree = (matrix > threshold).sum(axis=1).astype(np.float32)
    x = torch.tensor(degree, dtype=torch.float).unsqueeze(1)  # shape [n, 1]

    # Build edge list from co-occurrence pairs above threshold
    edge_index = []
    edge_weight = []
    for i in range(n):
        for j in range(i + 1, n):
            if matrix[i][j] > threshold:
                edge_index.append([i, j])
                edge_index.append([j, i])  # undirected
                edge_weight.append(matrix[i][j])
                edge_weight.append(matrix[i][j])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    # Positive link prediction labels: edges that exist
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)
    data.num_nodes = n

    print(f"Graph: {n} nodes, {len(edge_weight)//2} unique edges")
    return data, disease_names
