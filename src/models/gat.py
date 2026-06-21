"""
Graph Attention Network (GAT) for node embedding.
Paper: Velickovic et al., 2018 — https://arxiv.org/abs/1710.10903

How it works:
  Unlike GCN, GAT learns ATTENTION WEIGHTS for each neighbor.
  More important neighbors get higher weights during aggregation.
  Uses multi-head attention for stability.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.3)
        # After multi-head attention, feature size = hidden_channels * heads
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.3)

    def encode(self, x, edge_index):
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_index):
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

    def forward(self, x, edge_index):
        return self.encode(x, edge_index)
