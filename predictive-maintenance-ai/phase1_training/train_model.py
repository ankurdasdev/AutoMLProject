"""
IoT Predictive Maintenance — Phase 1: Model Training
====================================================
Dataset : train_80_of_half_a.csv
Internal Test : benchmark_20_of_half_a.csv
Model   : Scikit-Learn Isolation Forest (iforest)
Output  : models/fsl_heavy_motor_model.joblib
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

# Paths
DATA_DIR   = r"D:\IOT Project\predictive-maintenance-ai\data"
MODEL_DIR  = r"D:\IOT Project\predictive-maintenance-ai\models"
MODEL_NAME = "fsl_heavy_motor_model.joblib"

os.makedirs(MODEL_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# STEP 1 — Load the Pre-Processed Splits
# ──────────────────────────────────────────────
print("Loading randomized training and benchmark sets...")
train_df = pd.read_csv(os.path.join(DATA_DIR, "train_80_of_half_a.csv"))
bench_df = pd.read_csv(os.path.join(DATA_DIR, "benchmark_20_of_half_a.csv"))

# Define sensor columns (exclude metadata)
sensor_cols = [c for c in train_df.columns if c not in ["timestamp", "original_index"]]

X_train = train_df[sensor_cols]
X_bench = bench_df[sensor_cols]

print(f"Training set size  : {len(X_train)}")
print(f"Benchmark set size : {len(X_bench)}")


# ──────────────────────────────────────────────
# STEP 2 — Build and Train Pipeline
# ──────────────────────────────────────────────
print("\nBuilding Scikit-Learn Pipeline (StandardScaler + IsolationForest)...")

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', IsolationForest(contamination=0.05, random_state=42))
])

print("Training Isolation Forest on randomized 80% split...")
pipeline.fit(X_train)


# ──────────────────────────────────────────────
# STEP 3 — Save the Model
# ──────────────────────────────────────────────
output_path = os.path.join(MODEL_DIR, MODEL_NAME)
joblib.dump(pipeline, output_path)
print(f"Model saved -> {output_path}")


# ──────────────────────────────────────────────
# STEP 4 — Benchmark Evaluation (Internal Test)
# ──────────────────────────────────────────────
print("\n--- BENCHMARK EVALUATION (on 20% Unseen Data from Half A) ---")
# predict(): 1 = normal, -1 = anomaly
preds = pipeline.predict(X_bench)
anomaly_count = np.sum(preds == -1)
total = len(preds)

print(f"Total samples tested : {total}")
print(f"Anomalies detected   : {anomaly_count} ({anomaly_count/total*100:.1f}%)")
print(f"Normal samples       : {total - anomaly_count}")

# Heuristic Check: Are anomalies physically different?
bench_df['is_anomaly'] = (preds == -1)
if 'fwd_rms_vel' in bench_df.columns:
    avg_rms_normal = bench_df[bench_df['is_anomaly'] == False]['fwd_rms_vel'].mean()
    avg_rms_anomaly = bench_df[bench_df['is_anomaly'] == True]['fwd_rms_vel'].mean()
    print(f"\nPhysical Heuristic Check:")
    print(f"  Avg RMS Velocity (Normal)  : {avg_rms_normal:.4f}")
    print(f"  Avg RMS Velocity (Anomaly) : {avg_rms_anomaly:.4f}")
    print(f"  Multiplier                 : {avg_rms_anomaly/avg_rms_normal:.1f}x")