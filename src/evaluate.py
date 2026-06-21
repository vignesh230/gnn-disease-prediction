"""
Model evaluation: ROC-AUC, PR curves, confusion matrix, and FAISS similarity search.
"""

import torch
import numpy as np
import faiss
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from torch_geometric.utils import negative_sampling


def get_predictions(model, data):
    """Get link prediction scores for positive and negative edges."""
    model.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index)

        pos_edge_index = data.edge_index
        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=pos_edge_index.size(1)
        )

        pos_scores = torch.sigmoid(model.decode(z, pos_edge_index)).numpy()
        neg_scores = torch.sigmoid(model.decode(z, neg_edge_index)).numpy()

        scores = np.concatenate([pos_scores, neg_scores])
        labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])

    return scores, labels, z.numpy()


def plot_roc_curve(scores, labels, model_name, ax):
    fpr, tpr, _ = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)
    ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})")
    return auc


def plot_pr_curve(scores, labels, model_name, ax):
    precision, recall, _ = precision_recall_curve(labels, scores)
    ap = average_precision_score(labels, scores)
    ax.plot(recall, precision, label=f"{model_name} (AP={ap:.3f})")
    return ap


def evaluate_all_models(models_dict, data, disease_names):
    """Evaluate all models and produce comparison plots."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ROC curve plot
    axes[0].set_title("ROC Curves — All Models")
    axes[0].plot([0, 1], [0, 1], 'k--', label="Random")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")

    # PR curve plot
    axes[1].set_title("Precision-Recall Curves — All Models")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")

    results = {}
    for name, model in models_dict.items():
        scores, labels, embeddings = get_predictions(model, data)
        auc = plot_roc_curve(scores, labels, name, axes[0])
        ap = plot_pr_curve(scores, labels, name, axes[1])
        results[name] = {"ROC-AUC": auc, "AP": ap, "embeddings": embeddings}
        print(f"{name:20s} | ROC-AUC: {auc:.4f} | Avg Precision: {ap:.4f}")

    # Best model confusion matrix
    best_model_name = max(results, key=lambda k: results[k]["ROC-AUC"])
    best_scores, best_labels, _ = get_predictions(models_dict[best_model_name], data)
    preds = (best_scores > 0.5).astype(int)
    cm = confusion_matrix(best_labels, preds)
    ConfusionMatrixDisplay(cm, display_labels=["No Link", "Link"]).plot(ax=axes[2])
    axes[2].set_title(f"Confusion Matrix — {best_model_name}")

    axes[0].legend()
    axes[1].legend()
    plt.tight_layout()
    plt.savefig("results_comparison.png", dpi=150)
    plt.show()
    print(f"\nBest model: {best_model_name} (ROC-AUC: {results[best_model_name]['ROC-AUC']:.4f})")
    return results


def faiss_similar_diseases(embeddings, disease_names, query_disease, top_k=5):
    """
    Use FAISS to find diseases most similar to a query disease based on embeddings.
    This is vector similarity search — same concept used in RAG systems.
    """
    emb = embeddings.astype(np.float32)
    faiss.normalize_L2(emb)  # normalize for cosine similarity

    index = faiss.IndexFlatIP(emb.shape[1])  # inner product = cosine after normalization
    index.add(emb)

    query_idx = disease_names.index(query_disease)
    query_vec = emb[query_idx:query_idx+1]
    distances, indices = index.search(query_vec, top_k + 1)  # +1 because query matches itself

    print(f"\nDiseases most similar to '{query_disease}':")
    for i, (idx, score) in enumerate(zip(indices[0], distances[0])):
        if disease_names[idx] != query_disease:
            print(f"  {i}. {disease_names[idx]:35s} similarity: {score:.4f}")
