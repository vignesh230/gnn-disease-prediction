"""
Generates synthetic EHR-like data for the disease comorbidity pipeline.
In a real project, replace this with MIMIC-IV data (https://physionet.org/content/mimiciv/).
"""

import numpy as np
import pandas as pd

# ICD-10 disease codes we'll simulate
DISEASES = [
    "Type 2 Diabetes", "Hypertension", "Heart Failure", "COPD",
    "Chronic Kidney Disease", "Depression", "Obesity", "Asthma",
    "Atrial Fibrillation", "Osteoarthritis", "Hypothyroidism",
    "Coronary Artery Disease", "Anxiety Disorder", "Sleep Apnea",
    "Peripheral Artery Disease", "Stroke", "Cancer (unspecified)",
    "Anemia", "Hyperlipidemia", "GERD"
]

# Real-world co-occurrence probabilities (based on clinical literature)
COMORBIDITY_PAIRS = {
    ("Type 2 Diabetes", "Hypertension"): 0.75,
    ("Type 2 Diabetes", "Obesity"): 0.70,
    ("Type 2 Diabetes", "Hyperlipidemia"): 0.65,
    ("Type 2 Diabetes", "Chronic Kidney Disease"): 0.40,
    ("Heart Failure", "Atrial Fibrillation"): 0.60,
    ("Heart Failure", "Hypertension"): 0.70,
    ("COPD", "Asthma"): 0.30,
    ("COPD", "Heart Failure"): 0.25,
    ("Depression", "Anxiety Disorder"): 0.65,
    ("Obesity", "Sleep Apnea"): 0.55,
    ("Hypertension", "Coronary Artery Disease"): 0.50,
    ("Stroke", "Atrial Fibrillation"): 0.45,
}


def generate_patient_records(n_patients=2000, seed=42):
    """Generate synthetic patient EHR records."""
    np.random.seed(seed)
    records = []

    for patient_id in range(n_patients):
        age = np.random.randint(30, 85)
        n_diseases = np.random.choice([1, 2, 3, 4], p=[0.3, 0.35, 0.25, 0.1])
        base_diseases = list(np.random.choice(DISEASES, size=min(n_diseases, len(DISEASES)), replace=False))

        # Add comorbid diseases based on known clinical relationships
        extra = []
        for d in base_diseases:
            for (d1, d2), prob in COMORBIDITY_PAIRS.items():
                if d == d1 and d2 not in base_diseases and np.random.random() < prob * 0.5:
                    extra.append(d2)
                elif d == d2 and d1 not in base_diseases and np.random.random() < prob * 0.5:
                    extra.append(d1)

        all_diseases = list(set(base_diseases + extra))
        records.append({
            "patient_id": f"P{patient_id:04d}",
            "age": age,
            "diseases": all_diseases
        })

    return records


def build_cooccurrence_matrix(records):
    """Build disease co-occurrence matrix from patient records."""
    disease_to_idx = {d: i for i, d in enumerate(DISEASES)}
    n = len(DISEASES)
    matrix = np.zeros((n, n), dtype=int)

    for record in records:
        diseases = [d for d in record["diseases"] if d in disease_to_idx]
        for i, d1 in enumerate(diseases):
            for d2 in diseases[i+1:]:
                idx1, idx2 = disease_to_idx[d1], disease_to_idx[d2]
                matrix[idx1][idx2] += 1
                matrix[idx2][idx1] += 1

    return matrix, disease_to_idx


if __name__ == "__main__":
    records = generate_patient_records(2000)
    matrix, disease_to_idx = build_cooccurrence_matrix(records)

    df = pd.DataFrame(matrix, index=DISEASES, columns=DISEASES)
    df.to_csv("data/cooccurrence_matrix.csv")

    records_df = pd.DataFrame([
        {"patient_id": r["patient_id"], "age": r["age"], "diseases": "|".join(r["diseases"])}
        for r in records
    ])
    records_df.to_csv("data/patient_records.csv", index=False)
    print(f"Generated {len(records)} patient records, {len(DISEASES)} diseases")
    print(f"Saved to data/cooccurrence_matrix.csv and data/patient_records.csv")
