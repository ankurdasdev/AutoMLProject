"""
IoT Predictive Maintenance API — v3.0
======================================
Multi-Model Registry Architecture:
  - Each unique device type (identified by its feature column signature) gets
    its OWN model file. Models are NEVER overwritten by a different device type.
  - If a model already exists for the incoming data's column signature, it is
    reused directly — no retraining.
  - If the column signature is new, AutoML trains a fresh model and saves it
    to a new file: models/model_<device_key>.joblib

Endpoints:
  POST /api/upload        — Accept CSV/NPZ, run inference or train new model
  GET  /api/model-info    — Return registry of all known device models
  GET  /api/health        — Heartbeat
  GET  /api/models        — List all registered device models
  WebSocket /ws/stream    — Real-time streaming
"""

import os, io, json, time, uuid, shutil, hashlib, joblib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as scipy_stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ─────────────────────────────────────────────────────────────────────────────
# PATHS & CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR             = Path(__file__).parent.parent
MODEL_DIR            = BASE_DIR / "models"
MODEL_REGISTRY_PATH  = MODEL_DIR / "model_registry.json"
SAMPLING_RATE        = 32_000

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Fan Coil I NPZ extraction prefix names
FANCOIL_SENSOR_COLS = [
    "fwd_rms_vel", "fwd_mean", "fwd_std", "fwd_min", "fwd_max", "fwd_skew", "fwd_kurtosis",
    "rr_rms_vel",  "rr_mean",  "rr_std",  "rr_min",  "rr_max",  "rr_skew",  "rr_kurtosis",
]

# Columns that are metadata, not sensor features
NON_FEATURE_COLS = {
    "timestamp", "original_index", "label", "fault_code",
    "health_index", "is_anomaly", "raw_score", "index"
}

app = FastAPI(title="IoT Predictive Maintenance API", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ─────────────────────────────────────────────────────────────────────────────
# IN-MEMORY MODEL CACHE  {device_key -> (pipeline, feature_cols)}
# ─────────────────────────────────────────────────────────────────────────────
_model_cache: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# REGISTRY HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def get_device_key(feature_cols: list) -> str:
    """Stable 12-char hash of the sorted column list — unique per device type."""
    sig = ",".join(sorted(feature_cols))
    return "dev_" + hashlib.md5(sig.encode()).hexdigest()[:10]


def load_registry() -> dict:
    if MODEL_REGISTRY_PATH.exists():
        with open(MODEL_REGISTRY_PATH) as f:
            return json.load(f)
    return {}


def save_registry(registry: dict):
    with open(MODEL_REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


def _migrate_legacy_model():
    """
    One-time migration: if the old single-model file exists and is not yet
    in the registry, register it under its feature column signature.
    """
    legacy_path = MODEL_DIR / "fsl_heavy_motor_model.joblib"
    legacy_meta = MODEL_DIR / "model_metadata.json"
    if not legacy_path.exists():
        return

    registry = load_registry()
    # Check if already migrated
    for entry in registry.values():
        if entry.get("model_file") == "fsl_heavy_motor_model.joblib":
            return  # already registered

    # Read feature cols from old metadata
    feature_cols = FANCOIL_SENSOR_COLS  # default
    meta = {}
    if legacy_meta.exists():
        with open(legacy_meta) as f:
            meta = json.load(f)
        if meta.get("trained_feature_cols"):
            feature_cols = meta["trained_feature_cols"]

    device_key = get_device_key(feature_cols)
    registry[device_key] = {
        "device_key"    : device_key,
        "device_label"  : "Fan Coil I (Legacy)",
        "model_file"    : "fsl_heavy_motor_model.joblib",
        "feature_cols"  : feature_cols,
        "algorithm"     : meta.get("algorithm", "Unknown"),
        "contamination" : meta.get("contamination", 0.014),
        "score_separation": meta.get("score_separation", 0),
        "training_samples": meta.get("training_samples", 0),
        "trained_at"    : meta.get("trained_at", "unknown"),
        "upload_count"  : 0,
    }
    save_registry(registry)


# Run migration at import time
_migrate_legacy_model()


def get_model_for_device(device_key: str):
    """
    Return (pipeline, feature_cols) for the given device_key.
    Loads from disk on first access, then caches in memory.
    Returns (None, None) if not registered.
    """
    if device_key in _model_cache:
        return _model_cache[device_key]

    registry = load_registry()
    if device_key not in registry:
        return None, None

    entry      = registry[device_key]
    model_path = MODEL_DIR / entry["model_file"]
    if not model_path.exists():
        return None, None

    pipeline     = joblib.load(str(model_path))
    feature_cols = entry["feature_cols"]
    _model_cache[device_key] = (pipeline, feature_cols)
    return pipeline, feature_cols


def save_model_for_device(device_key: str, pipeline, feature_cols: list,
                           automl_meta: dict, device_label: str = None):
    """
    Save a newly trained model to its own file and register it.
    Never overwrites a model file that belongs to a DIFFERENT device_key.
    """
    model_filename = f"model_{device_key}.joblib"
    model_path     = MODEL_DIR / model_filename

    joblib.dump(pipeline, str(model_path))

    registry = load_registry()
    registry[device_key] = {
        "device_key"      : device_key,
        "device_label"    : device_label or f"Device {device_key}",
        "model_file"      : model_filename,
        "feature_cols"    : feature_cols,
        "algorithm"       : automl_meta.get("algorithm"),
        "contamination"   : automl_meta.get("contamination"),
        "score_separation": automl_meta.get("score_separation"),
        "training_samples": automl_meta.get("training_samples"),
        "trained_at"      : automl_meta.get("trained_at"),
        "upload_count"    : 0,
    }
    save_registry(registry)
    _model_cache[device_key] = (pipeline, feature_cols)


def increment_upload_count(device_key: str):
    registry = load_registry()
    if device_key in registry:
        registry[device_key]["upload_count"] = registry[device_key].get("upload_count", 0) + 1
        registry[device_key]["last_used"]    = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        save_registry(registry)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION (for NPZ / raw signal uploads)
# ─────────────────────────────────────────────────────────────────────────────
def compute_rms_velocity(accel_row: np.ndarray) -> float:
    vel_mm_s = np.cumsum(accel_row) / SAMPLING_RATE * 1000
    return float(np.sqrt(np.mean(vel_mm_s ** 2)))


def extract_features_from_signal(accel_row: np.ndarray, prefix: str) -> dict:
    return {
        f"{prefix}_rms_vel"  : compute_rms_velocity(accel_row),
        f"{prefix}_mean"     : float(np.mean(accel_row)),
        f"{prefix}_std"      : float(np.std(accel_row)),
        f"{prefix}_min"      : float(np.min(accel_row)),
        f"{prefix}_max"      : float(np.max(accel_row)),
        f"{prefix}_skew"     : float(scipy_stats.skew(accel_row)),
        f"{prefix}_kurtosis" : float(scipy_stats.kurtosis(accel_row)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# AUTOML TRAINING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def automl_train_and_select(X: np.ndarray, feature_cols: list, steps_log: list) -> tuple:
    """
    Train IsolationForest + LOF, select winner by separation score.
    Contamination floor: 1.4% (preserves industrial sensor sensitivity).
    Returns (pipeline, metadata_dict).
    """
    steps_log.append({"step": "Scaling features with StandardScaler", "status": "done"})
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Contamination Optimizer ───────────────────────────────────────────────
    steps_log.append({"step": "Running contamination optimizer (histogram elbow)", "status": "done"})
    probe       = IsolationForest(n_estimators=200, contamination=0.1, random_state=42)
    probe.fit(X_scaled)
    probe_scores = probe.decision_function(X_scaled)
    hist, bin_edges = np.histogram(probe_scores, bins=50)
    negative_mask   = bin_edges[:-1] < 0

    if negative_mask.any():
        trough_idx   = np.argmin(hist[negative_mask])
        trough_score = bin_edges[:-1][negative_mask][trough_idx]
        optimal_cont = float(np.mean(probe_scores < trough_score))
        # Floor 1.4% preserves original Fan Coil sensitivity (~36 anomalies / 2619 samples)
        optimal_cont = max(0.014, min(optimal_cont, 0.15))
    else:
        optimal_cont = 0.05
    steps_log.append({"step": f"Optimal contamination: {optimal_cont * 100:.2f}%", "status": "done"})

    # ── Train Isolation Forest ────────────────────────────────────────────────
    steps_log.append({"step": "Training Isolation Forest (n_estimators=200)", "status": "done"})
    iforest  = IsolationForest(n_estimators=200, contamination=optimal_cont, random_state=42)
    iforest.fit(X_scaled)
    if_scores = iforest.decision_function(X_scaled)
    if_preds  = iforest.predict(X_scaled)
    sep_if = (float(if_scores[if_preds == 1].mean() - if_scores[if_preds == -1].mean())
              if np.any(if_preds == 1) and np.any(if_preds == -1) else 0.0)

    # ── Train LOF ─────────────────────────────────────────────────────────────
    steps_log.append({"step": "Training LOF competitor (n_neighbors=20)", "status": "done"})
    lof = LocalOutlierFactor(n_neighbors=20, contamination=optimal_cont, novelty=True)
    lof.fit(X_scaled)
    lof_scores = lof.decision_function(X_scaled)
    lof_preds  = lof.predict(X_scaled)
    sep_lof = (float(lof_scores[lof_preds == 1].mean() - lof_scores[lof_preds == -1].mean())
               if np.any(lof_preds == 1) and np.any(lof_preds == -1) else 0.0)

    steps_log.append({
        "step"  : f"Model comparison: IForest sep={sep_if:.3f} vs LOF sep={sep_lof:.3f}",
        "status": "done"
    })

    # ── Winner Selection ──────────────────────────────────────────────────────
    if sep_if >= sep_lof * 0.9:
        winner_name, winner_model, winner_sep = "IsolationForest", iforest, sep_if
        reasoning = (f"Isolation Forest ({sep_if:.3f}) within 10% of LOF ({sep_lof:.3f}). "
                     f"Selected — supports live novelty detection.")
    else:
        winner_name, winner_model, winner_sep = "LocalOutlierFactor", lof, sep_lof
        reasoning = (f"LOF ({sep_lof:.3f}) significantly exceeds IForest ({sep_if:.3f}). "
                     f"Selected for superior anomaly discrimination.")

    steps_log.append({"step": f"Winner: {winner_name}. {reasoning}", "status": "done"})

    final_pipeline = Pipeline([("scaler", scaler), ("model", winner_model)])
    metadata = {
        "algorithm"           : winner_name,
        "contamination"       : round(optimal_cont, 4),
        "score_separation"    : round(winner_sep, 4),
        "competitor_separation": round(sep_lof if winner_name == "IsolationForest" else sep_if, 4),
        "winner_reasoning"    : reasoning,
        "training_samples"    : len(X),
        "trained_feature_cols": feature_cols,
        "trained_at"          : time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return final_pipeline, metadata


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
def run_inference(pipeline, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    cols = [c for c in feature_cols if c in df.columns]
    if not cols:
        raise ValueError(f"No matching columns. Model expects: {feature_cols}. Got: {list(df.columns)}")

    X            = df[cols].values
    preds        = pipeline.predict(X)
    raw_scores   = pipeline.decision_function(X)
    health_index = np.clip((raw_scores + 0.5) * 100, 0, 100)

    df = df.copy()
    df["is_anomaly"]   = preds == -1
    df["raw_score"]    = raw_scores
    df["health_index"] = health_index
    df["fault_code"]   = df["health_index"].apply(lambda h:
        "HEALTHY"          if h >= 75 else
        "DEGRADED_WATCH"   if h >= 50 else
        "FAULT_IMMINENT"   if h >= 25 else
        "CRITICAL_FAILURE"
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    registry = load_registry()
    return {
        "status"         : "ok",
        "api_version"    : "3.0.0",
        "registered_models": len(registry),
        "device_keys"    : list(registry.keys()),
    }


@app.get("/api/models")
async def list_models():
    """Return all registered device models and their metadata."""
    return load_registry()


@app.get("/api/model-info")
async def model_info():
    """Legacy endpoint — returns full registry."""
    registry = load_registry()
    if not registry:
        return JSONResponse(status_code=404, content={"error": "No models registered yet."})
    return registry


@app.post("/api/upload")
async def upload_and_infer(file: UploadFile = File(...)):
    job_id    = str(uuid.uuid4())[:8]
    steps_log = []
    is_new_device = False

    try:
        # ── 1. Read file ───────────────────────────────────────────────────────
        content = await file.read()
        steps_log.append({
            "step"  : f"File received: {file.filename} ({len(content)//1024} KB)",
            "status": "done"
        })

        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
            steps_log.append({
                "step"  : f"CSV parsed: {len(df)} rows, {len(df.columns)} columns",
                "status": "done"
            })
            numeric_cols = [
                c for c in df.select_dtypes(include=[np.number]).columns
                if c.lower() not in NON_FEATURE_COLS
            ]

        elif file.filename.endswith(".npz"):
            import zipfile, tempfile
            steps_log.append({"step": "NPZ detected — extracting vibration features...", "status": "done"})
            with tempfile.TemporaryDirectory() as tmpdir:
                with zipfile.ZipFile(io.BytesIO(content)) as zf:
                    zf.extractall(tmpdir)
                keys    = [f.replace(".npy", "") for f in os.listdir(tmpdir) if f.endswith(".npy")]
                fwd_key = next((k for k in keys if any(x in k for x in ["fwd","foward","forward"])), None)
                rr_key  = next((k for k in keys if any(x in k for x in ["rear","rr"])), None)
                fwd_data = np.load(os.path.join(tmpdir, f"{fwd_key}.npy"), mmap_mode="r") if fwd_key else None
                rr_data  = np.load(os.path.join(tmpdir, f"{rr_key}.npy"),  mmap_mode="r") if rr_key  else None
                n = min(len(fwd_data) if fwd_data is not None else 500,
                        len(rr_data)  if rr_data  is not None else 500, 500)
                rows = []
                for i in range(n):
                    row = {}
                    if fwd_data is not None: row.update(extract_features_from_signal(fwd_data[i], "fwd"))
                    if rr_data  is not None: row.update(extract_features_from_signal(rr_data[i],  "rr"))
                    rows.append(row)
                df = pd.DataFrame(rows).dropna()
            numeric_cols = list(df.columns)
            steps_log.append({"step": f"Features extracted: {len(df)} valid samples", "status": "done"})
        else:
            return JSONResponse(status_code=400, content={"error": "Only .csv and .npz files are supported."})

        if not numeric_cols:
            return JSONResponse(status_code=422, content={"error": "No numeric feature columns found in upload."})

        # ── 2. Identify device type ────────────────────────────────────────────
        device_key   = get_device_key(numeric_cols)
        registry     = load_registry()
        device_label = registry.get(device_key, {}).get("device_label", f"Device {device_key}")

        steps_log.append({
            "step"  : f"Device signature: {device_key} | Columns: {numeric_cols}",
            "status": "done"
        })

        # ── 3. Load existing model OR train new one ────────────────────────────
        pipeline, feature_cols = get_model_for_device(device_key)
        automl_metadata        = None

        if pipeline is not None:
            # ── REUSE: model already trained for this exact device type ──────
            steps_log.append({
                "step"  : f"Existing model found for {device_label}. Reusing — no retraining needed.",
                "status": "done"
            })
            automl_metadata = registry.get(device_key, {})
            increment_upload_count(device_key)
        else:
            # ── NEW DEVICE: train fresh model, save to its own file ──────────
            is_new_device = True
            steps_log.append({
                "step"  : f"New device type detected ({device_key}). Starting AutoML training...",
                "status": "training"
            })

            df_clean   = df.dropna(subset=numeric_cols)
            nan_count  = len(df) - len(df_clean)
            if nan_count > 0:
                steps_log.append({"step": f"Dropped {nan_count} NaN rows", "status": "warning"})

            if len(df_clean) < 10:
                return JSONResponse(status_code=422, content={
                    "error": f"Too few clean rows ({len(df_clean)}). Need at least 10."
                })

            X_train         = df_clean[numeric_cols].values
            pipeline, automl_metadata = automl_train_and_select(X_train, numeric_cols, steps_log)

            # Give the device a meaningful label based on its columns
            if set(numeric_cols) == set(FANCOIL_SENSOR_COLS):
                device_label = "Fan Coil I"
            elif any("accel" in c for c in numeric_cols):
                device_label = "CPU Cooling Fan (MPU-6050)"
            else:
                device_label = f"Device {device_key}"

            save_model_for_device(device_key, pipeline, numeric_cols,
                                  automl_metadata, device_label)
            feature_cols = numeric_cols
            steps_log.append({
                "step"  : f"Model saved: models/model_{device_key}.joblib",
                "status": "done"
            })

        # ── 4. Clean data and run inference ────────────────────────────────────
        df_clean  = df.dropna(subset=[c for c in feature_cols if c in df.columns])
        result_df = run_inference(pipeline, df_clean, feature_cols)
        steps_log.append({"step": "Inference complete. Generating report...", "status": "done"})

        # ── 5. Build response ──────────────────────────────────────────────────
        n_total    = len(result_df)
        n_anomaly  = int(result_df["is_anomaly"].sum())
        avg_health = float(result_df["health_index"].mean())

        feature_stats = {}
        for col in feature_cols:
            if col not in result_df.columns:
                continue
            normal  = result_df[result_df["is_anomaly"] == False][col]
            anomaly = result_df[result_df["is_anomaly"] == True][col]
            feature_stats[col] = {
                "normal_mean" : float(normal.mean())  if len(normal)  > 0 else None,
                "anomaly_mean": float(anomaly.mean()) if len(anomaly) > 0 else None,
                "all_values"  : [round(v, 4) for v in result_df[col].tolist()],
            }

        health_timeline = [
            {"index": int(i), "health_index": round(float(h), 2),
             "is_anomaly": bool(a), "fault_code": fc}
            for i, h, a, fc in zip(result_df.index, result_df["health_index"],
                                   result_df["is_anomaly"], result_df["fault_code"])
        ]

        top_anomalies = result_df[result_df["is_anomaly"]].sort_values("health_index").head(10)
        top_anomaly_rows = []
        for _, row in top_anomalies.iterrows():
            entry = {"index": int(row.name),
                     "health_index": round(float(row["health_index"]), 2),
                     "fault_code": row["fault_code"]}
            for col in feature_cols[:4]:
                if col in row:
                    entry[col] = round(float(row[col]), 4)
            top_anomaly_rows.append(entry)

        return {
            "job_id": job_id,
            "summary": {
                "total_samples"      : n_total,
                "anomalies_detected" : n_anomaly,
                "anomaly_rate_pct"   : round(n_anomaly / n_total * 100, 2),
                "avg_health_index"   : round(avg_health, 2),
                "overall_status"     : (
                    "HEALTHY"          if avg_health >= 75 else
                    "DEGRADED_WATCH"   if avg_health >= 50 else
                    "FAULT_IMMINENT"   if avg_health >= 25 else
                    "CRITICAL_FAILURE"
                ),
                "is_new_device"      : is_new_device,
                "device_key"         : device_key,
                "device_label"       : device_label,
                "feature_cols_used"  : feature_cols,
            },
            "automl_steps"   : steps_log,
            "automl_metadata": automl_metadata,
            "health_timeline": health_timeline,
            "feature_stats"  : feature_stats,
            "top_anomalies"  : top_anomaly_rows,
        }

    except Exception as e:
        import traceback
        return JSONResponse(status_code=500, content={"error": str(e), "trace": traceback.format_exc()})


# ─────────────────────────────────────────────────────────────────────────────
# WEBSOCKET — Live Streaming
# ─────────────────────────────────────────────────────────────────────────────
@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    registry = load_registry()
    if not registry:
        await websocket.send_json({"error": "No models registered. Upload a dataset first."})
        await websocket.close()
        return

    try:
        while True:
            data     = await websocket.receive_json()
            features = data.get("features", {})
            # Identify which registered model matches the incoming feature keys
            incoming_cols = list(features.keys())
            device_key    = get_device_key(incoming_cols)
            pipeline, feature_cols = get_model_for_device(device_key)

            if pipeline is None:
                # Try to find a model with the best column overlap
                best_key, best_overlap = None, 0
                for dk, entry in registry.items():
                    overlap = len(set(entry["feature_cols"]) & set(incoming_cols))
                    if overlap > best_overlap:
                        best_overlap, best_key = overlap, dk
                if best_key:
                    pipeline, feature_cols = get_model_for_device(best_key)
                    device_key = best_key

            if pipeline is None:
                await websocket.send_json({"error": "No matching model found for incoming features."})
                continue

            df_row = pd.DataFrame([features])
            scored = run_inference(pipeline, df_row, feature_cols)
            row    = scored.iloc[0]
            await websocket.send_json({
                "device_id"   : data.get("device_id", "UNKNOWN"),
                "device_key"  : device_key,
                "health_index": round(float(row["health_index"]), 2),
                "fault_code"  : row["fault_code"],
                "is_anomaly"  : bool(row["is_anomaly"]),
                "timestamp"   : time.time(),
            })
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
