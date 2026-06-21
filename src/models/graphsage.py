"""
GraphSAGE (Sample and Aggregate) for node embedding.
Paper: Hamilton et al., 2017 — https://arxiv.org/abs/1706.02216

How it works:
  Instead of using ALL neighbors (like GCN/GAT), GraphSAGE SAMPLES a fixed number.
  This makes it scalable to large graphs — it doesn't need to know the full graph structure.
  Great for inductive learning (can generalize to unseen nodes).
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_index):
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

    def forward(self, x, edge_index):
        return self.encode(x, edge_index)
