"""
Graph Convolutional Network (GCN) for node embedding.
Paper: Kipf & Welling, 2017 — https://arxiv.org/abs/1609.02907

How it works:
  Each layer aggregates features from a node's neighbors and updates the node's representation.
  GCN treats all neighbors equally (no attention weights).
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        """Produce node embeddings."""
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_index):
        """Predict link probability via dot product."""
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        return z
