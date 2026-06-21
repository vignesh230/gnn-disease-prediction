"""
Training loop for link prediction on the disease graph.
Uses negative sampling: for every real edge, create a fake edge as a negative example.
"""

import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling


def train_epoch(model, data, optimizer):
    """One training step."""
    model.train()
    optimizer.zero_grad()

    # Get node embeddings from the model
    z = model(data.x, data.edge_index)

    # Positive edges (real comorbidities)
    pos_edge_index = data.edge_index

    # Negative edges (random pairs that are NOT comorbid)
    neg_edge_index = negative_sampling(
        edge_index=pos_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=pos_edge_index.size(1)
    )

    # Predict link probability for positive and negative edges
    pos_scores = model.decode(z, pos_edge_index)
    neg_scores = model.decode(z, neg_edge_index)

    # Binary cross-entropy loss
    pos_labels = torch.ones(pos_scores.size(0))
    neg_labels = torch.zeros(neg_scores.size(0))
    scores = torch.cat([pos_scores, neg_scores])
    labels = torch.cat([pos_labels, neg_labels])

    loss = F.binary_cross_entropy_with_logits(scores, labels)
    loss.backward()
    optimizer.step()

    return loss.item()


def train(model, data, epochs=200, lr=0.01):
    """Full training loop."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    print(f"Training {model.__class__.__name__} for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, data, optimizer)
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d} | Loss: {loss:.4f}")

    return model
