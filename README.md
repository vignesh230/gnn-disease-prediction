# Transformer-Based GNN Pipeline for Disease Comorbidity Prediction

End-to-end graph neural network pipeline that predicts disease co-occurrence (comorbidity) from patient EHR records. Implements and benchmarks four GNN architectures — GCN, GAT, GraphSAGE, and Graph Transformer — using PyTorch Geometric.

## What This Project Does

1. Generates synthetic patient EHR records (2,000+ patients, 20 diseases)
2. Constructs a disease co-occurrence graph with community detection
3. Trains four GNN architectures for link prediction
4. Evaluates models with ROC-AUC, PR curves, and confusion matrices
5. Uses FAISS vector search to find similar diseases from learned embeddings

## Results

| Model | ROC-AUC | Avg Precision |
|---|---|---|
| GCN | ~0.88 | ~0.85 |
| GAT | ~0.89 | ~0.86 |
| GraphSAGE | ~0.87 | ~0.84 |
| Graph Transformer | ~0.90 | ~0.88 |

## Project Structure

```
gnn-disease-prediction/
├── data/
│   └── generate_data.py       # Synthetic EHR data generation
├── src/
│   ├── graph_builder.py       # Build PyG graph from co-occurrence matrix
│   ├── train.py               # Training loop with negative sampling
│   ├── evaluate.py            # ROC-AUC, PR curves, FAISS search
│   └── models/
│       ├── gcn.py             # Graph Convolutional Network
│       ├── gat.py             # Graph Attention Network
│       ├── graphsage.py       # GraphSAGE
│       └── graph_transformer.py  # Graph Transformer
├── main.py                    # Run full pipeline
└── requirements.txt
```

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python main.py
```

## Key Concepts You'll Learn

- **Graph Neural Networks**: How nodes aggregate information from neighbors
- **Link Prediction**: Predicting edges using node embeddings + dot product
- **Negative Sampling**: Creating fake edges as training negatives
- **Attention Mechanisms**: How GAT learns to weight different neighbors
- **FAISS**: Vector similarity search on learned embeddings
- **Model Evaluation**: ROC-AUC, Precision-Recall, Confusion Matrix

## Dataset

Uses synthetic EHR data generated from clinical comorbidity literature. To use real data, replace `data/generate_data.py` with MIMIC-IV loader: https://physionet.org/content/mimiciv/

## Tech Stack

Python · PyTorch · PyTorch Geometric · FAISS · Scikit-learn · NetworkX · Matplotlib
