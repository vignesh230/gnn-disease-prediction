"""
Graph Transformer for node embedding.
Paper: Shi et al., 2020 — https://arxiv.org/abs/2009.03509

How it works:
  Applies the Transformer self-attention mechanism to graphs.
  Each node attends to ALL other nodes (not just neighbors), then combines with local structure.
  Most expressive but also most computationally expensive.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv


class GraphTransformer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super().__init__()
        self.conv1 = TransformerConv(in_channels, hidden_channels, heads=heads, dropout=0.3)
        self.conv2 = TransformerConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.3)

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
