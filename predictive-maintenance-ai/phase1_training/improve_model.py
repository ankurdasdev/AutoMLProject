"""
M1 — Model Improvement Script
==============================
1. Data-driven contamination tuning (elbow detection)
2. Increase n_estimators: 100 → 200
3. LOF competitor comparison
4. Retrain and save improved model
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline

DATA_DIR  = r"D:\IOT Project\predictive-maintenance-ai\data"
MODEL_DIR = r"D:\IOT Project\predictive-maintenance-ai\models"

# ──────────────────────────────────────────────
# STEP 1 — Load Splits
# ──────────────────────────────────────────────
train_df = pd.read_csv(os.path.join(DATA_DIR, "train_80_of_half_a.csv"))
bench_df = pd.read_csv(os.path.join(DATA_DIR, "benchmark_20_of_half_a.csv"))
sensor_cols = [c for c in train_df.columns if c not in ["timestamp", "original_index"]]
X_train = train_df[sensor_cols].values
X_bench = bench_df[sensor_cols].values

print(f"Training size: {len(X_train)} | Benchmark size: {len(X_bench)}\n")

# ──────────────────────────────────────────────
# STEP 2 — Data-Driven Contamination Tuning
# ──────────────────────────────────────────────
print("STEP 2: Finding optimal contamination via score histogram elbow...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Fit a probe model with generous contamination to see the full score distribution
probe = IsolationForest(n_estimators=200, contamination=0.1, random_state=42)
probe.fit(X_train_scaled)
probe_scores = probe.decision_function(X_train_scaled)

# Histogram-based elbow: find the natural "valley" between normal and anomaly scores
hist, bin_edges = np.histogram(probe_scores, bins=50)
# Find the bin with the lowest count in the negative score zone (the "trough")
negative_mask = bin_edges[:-1] < 0
if negative_mask.any():
    trough_idx = np.argmin(hist[negative_mask])
    trough_score = bin_edges[:-1][negative_mask][trough_idx]
    # Convert score to contamination: what fraction of training data is below trough?
    optimal_contamination = float(np.mean(probe_scores < trough_score))
    optimal_contamination = max(0.01, min(optimal_contamination, 0.15))  # clamp 1–15%
else:
    optimal_contamination = 0.05

print(f"  Score mean:    {probe_scores.mean():.4f}")
print(f"  Score std:     {probe_scores.std():.4f}")
print(f"  Score min:     {probe_scores.min():.4f}")
print(f"  Trough score:  {trough_score:.4f}")
print(f"  -> Optimal contamination: {optimal_contamination:.4f} ({optimal_contamination*100:.1f}%)\n")

# ──────────────────────────────────────────────
# STEP 3 — Train Improved Isolation Forest
# ──────────────────────────────────────────────
print("STEP 3: Training Improved Isolation Forest (n_estimators=200)...")
improved_iforest = Pipeline([
    ('scaler', StandardScaler()),
    ('model', IsolationForest(
        n_estimators=200,
        contamination=optimal_contamination,
        max_samples='auto',
        random_state=42
    ))
])
improved_iforest.fit(X_train, y=None)

bench_preds_if  = improved_iforest.predict(X_bench)
bench_scores_if = improved_iforest.decision_function(X_bench)
anomaly_count_if = np.sum(bench_preds_if == -1)
separation_if = bench_scores_if[bench_preds_if == 1].mean() - bench_scores_if[bench_preds_if == -1].mean()

print(f"  Isolation Forest -> Anomalies: {anomaly_count_if}/{len(X_bench)} | Score separation: {separation_if:.4f}\n")

# ──────────────────────────────────────────────
# STEP 4 — LOF Competitor
# ──────────────────────────────────────────────
print("STEP 4: Running LOF competitor model...")
scaler_lof = StandardScaler()
X_bench_scaled = scaler_lof.fit_transform(X_bench)  # LOF is transductive — fits on test data

lof = LocalOutlierFactor(n_neighbors=20, contamination=optimal_contamination, novelty=False)
bench_preds_lof = lof.fit_predict(X_bench_scaled)
anomaly_count_lof = np.sum(bench_preds_lof == -1)
lof_scores = lof.negative_outlier_factor_
separation_lof = lof_scores[bench_preds_lof == 1].mean() - lof_scores[bench_preds_lof == -1].mean()

print(f"  LOF             -> Anomalies: {anomaly_count_lof}/{len(X_bench)} | Score separation: {separation_lof:.4f}\n")

# ──────────────────────────────────────────────
# STEP 5 — Winner Selection
# ──────────────────────────────────────────────
print("STEP 5: Selecting winner...")
# Primary criterion: score separation (higher = better discrimination)
# Secondary: production-readiness (Isolation Forest supports novelty detection on new data)
print(f"  Isolation Forest score separation: {separation_if:.4f}")
print(f"  LOF score separation:              {separation_lof:.4f}")

winner = "IsolationForest"
reasoning = ""
if separation_if >= separation_lof * 0.9:  # IF within 10% of LOF → pick IF (production-ready)
    winner = "IsolationForest"
    reasoning = "Isolation Forest is within 10% of LOF performance AND supports novelty=True for live inference on new unseen data. LOF does not support this production use-case."
else:
    winner = "LOF"
    reasoning = "LOF significantly outperforms Isolation Forest on score separation."

print(f"\n  WINNER: {winner}")
print(f"  REASON: {reasoning}\n")

# ──────────────────────────────────────────────
# STEP 6 — Save Improved Model
# ──────────────────────────────────────────────
model_path = os.path.join(MODEL_DIR, "fsl_heavy_motor_model.joblib")
joblib.dump(improved_iforest, model_path)

# Also save model metadata for the UI
import json
metadata = {
    "algorithm": "IsolationForest",
    "n_estimators": 200,
    "contamination": round(optimal_contamination, 4),
    "features": list(sensor_cols),
    "training_samples": len(X_train),
    "benchmark_anomalies": int(anomaly_count_if),
    "benchmark_total": len(X_bench),
    "score_separation": round(float(separation_if), 4),
    "winner_reasoning": reasoning,
    "competitor": "LocalOutlierFactor",
    "competitor_separation": round(float(separation_lof), 4),
}
with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Improved model saved -> {model_path}")
print(f"Model metadata saved -> {os.path.join(MODEL_DIR, 'model_metadata.json')}")
print("\n--- FINAL SUMMARY ---")
print(f"  Algorithm : IsolationForest (improved)")
print(f"  Contamination: {optimal_contamination*100:.1f}% (was 5.0%)")
print(f"  n_estimators : 200 (was 100)")
print(f"  Anomalies on Benchmark: {anomaly_count_if}/{len(X_bench)}")
