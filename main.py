"""
Main entry point — runs the full GNN pipeline:
  1. Generate synthetic EHR data
  2. Build disease co-occurrence graph
  3. Train all 4 GNN architectures
  4. Evaluate and compare with ROC-AUC, PR curves, confusion matrix
  5. FAISS similarity search on node embeddings
"""

import torch
from data.generate_data import generate_patient_records, build_cooccurrence_matrix, DISEASES
from src.graph_builder import build_graph
from src.train import train
from src.evaluate import evaluate_all_models, faiss_similar_diseases
from src.models.gcn import GCN
from src.models.gat import GAT
from src.models.graphsage import GraphSAGE
from src.models.graph_transformer import GraphTransformer
import pandas as pd

# --- Step 1: Generate data ---
print("=" * 50)
print("Step 1: Generating synthetic EHR data...")
records = generate_patient_records(n_patients=2000)
matrix, _ = build_cooccurrence_matrix(records)
df = pd.DataFrame(matrix, index=DISEASES, columns=DISEASES)
df.to_csv("data/cooccurrence_matrix.csv")
print(f"  {len(records)} patient records generated.")

# --- Step 2: Build graph ---
print("\nStep 2: Building disease co-occurrence graph...")
data, disease_names = build_graph("data/cooccurrence_matrix.csv", threshold=10)

# --- Step 3: Train all 4 models ---
IN = data.x.size(1)   # input feature size (1 — degree)
HIDDEN = 64
OUT = 32

models = {
    "GCN":               GCN(IN, HIDDEN, OUT),
    "GAT":               GAT(IN, HIDDEN, OUT),
    "GraphSAGE":         GraphSAGE(IN, HIDDEN, OUT),
    "Graph Transformer": GraphTransformer(IN, HIDDEN, OUT),
}

print("\nStep 3: Training models...")
for name, model in models.items():
    print(f"\n--- {name} ---")
    train(model, data, epochs=200, lr=0.01)

# --- Step 4: Evaluate ---
print("\nStep 4: Evaluating all models...")
results = evaluate_all_models(models, data, disease_names)

# --- Step 5: FAISS similarity search ---
print("\nStep 5: FAISS similarity search on embeddings...")
best_model = max(results, key=lambda k: results[k]["ROC-AUC"])
embeddings = results[best_model]["embeddings"]
faiss_similar_diseases(embeddings, disease_names, "Type 2 Diabetes")
faiss_similar_diseases(embeddings, disease_names, "Heart Failure")
