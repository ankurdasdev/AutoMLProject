"""
IoT Predictive Maintenance — Phase 2: Unified Python Orchestrator (Scikit-Learn)
==============================================================================
This single script acts as:
  1. Inference engine  — loads the Scikit-Learn pipeline and scores live payloads
  2. Integration layer — publishes Salesforce Platform Events via REST API

NOTE: Updated to use Scikit-Learn (joblib) for compatibility with Python 3.12.
"""

import os
import sys
import json
import time
import logging
import datetime
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# LOAD ENVIRONMENT VARIABLES
# ─────────────────────────────────────────────────────────────
ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(ENV_PATH)

SF_CLIENT_ID      = os.getenv("SF_CLIENT_ID", "")
SF_CLIENT_SECRET  = os.getenv("SF_CLIENT_SECRET", "")
SF_USERNAME       = os.getenv("SF_USERNAME", "")
SF_PASSWORD       = os.getenv("SF_PASSWORD", "")
SF_SECURITY_TOKEN = os.getenv("SF_SECURITY_TOKEN", "")
SF_LOGIN_URL      = os.getenv("SF_LOGIN_URL", "https://login.salesforce.com")
MODEL_PATH        = os.getenv("MODEL_PATH", "../models/fsl_heavy_motor_model.joblib")
ALERT_THRESHOLD   = float(os.getenv("ALERT_THRESHOLD", "50"))

# Resolve relative path against this file's directory
MODEL_PATH = str((Path(__file__).parent / MODEL_PATH).resolve())

# ─────────────────────────────────────────────────────────────
# FEATURE COLUMNS (must match Phase 1 training)
# ─────────────────────────────────────────────────────────────
SENSOR_COLS = [
    "fwd_rms_vel", "fwd_mean", "fwd_std", "fwd_min", "fwd_max",
    "fwd_skew", "fwd_kurtosis",
    "rr_rms_vel",  "rr_mean",  "rr_std",  "rr_min",  "rr_max",
    "rr_skew",  "rr_kurtosis",
]

# ─────────────────────────────────────────────────────────────
# FAULT CODE TABLE
# ─────────────────────────────────────────────────────────────
FAULT_CODE_TABLE = [
    (75, 100, "HEALTHY"),
    (50,  74, "DEGRADED_WATCH"),
    (25,  49, "FAULT_IMMINENT"),
    ( 0,  24, "CRITICAL_FAILURE"),
]

def health_to_fault_code(health_index: float) -> str:
    for lo, hi, code in FAULT_CODE_TABLE:
        if lo <= health_index <= hi:
            return code
    return "UNKNOWN"


# ─────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────
_pipeline = None

def load_model():
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    log.info(f"Loading model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        log.error("Model file not found! Run phase1_training/train_model.py first.")
        sys.exit(1)

    _pipeline = joblib.load(MODEL_PATH)
    log.info("Model loaded successfully.")
    return _pipeline


# ─────────────────────────────────────────────────────────────
# TELEMETRY SIMULATION
# ─────────────────────────────────────────────────────────────
SAMPLING_RATE = 32_000

def _compute_rms_velocity(accel_array: np.ndarray) -> float:
    vel_mm_s = np.cumsum(accel_array) / SAMPLING_RATE * 1000
    return float(np.sqrt(np.mean(vel_mm_s ** 2)))

def simulate_payload(device_id: str, degraded: bool = False) -> dict:
    rng   = np.random.default_rng(seed=int(time.time()))
    scale = 0.45 if not degraded else 0.75

    fwd = rng.normal(0, scale, 320_000).astype(np.float32)
    rr  = rng.normal(0, scale, 320_000).astype(np.float32)

    from scipy import stats as scipy_stats
    payload = {
        "device_id"    : device_id,
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "fwd_rms_vel"  : _compute_rms_velocity(fwd),
        "fwd_mean"     : float(np.mean(fwd)),
        "fwd_std"      : float(np.std(fwd)),
        "fwd_min"      : float(np.min(fwd)),
        "fwd_max"      : float(np.max(fwd)),
        "fwd_skew"     : float(scipy_stats.skew(fwd)),
        "fwd_kurtosis" : float(scipy_stats.kurtosis(fwd)),
        "rr_rms_vel"   : _compute_rms_velocity(rr),
        "rr_mean"      : float(np.mean(rr)),
        "rr_std"       : float(np.std(rr)),
        "rr_min"       : float(np.min(rr)),
        "rr_max"       : float(np.max(rr)),
        "rr_skew"      : float(scipy_stats.skew(rr)),
        "rr_kurtosis"  : float(scipy_stats.kurtosis(rr)),
    }
    return payload


# ─────────────────────────────────────────────────────────────
# INFERENCE ENGINE
# ─────────────────────────────────────────────────────────────
def score_payload(payload: dict) -> dict:
    pipeline = load_model()

    # Build input DataFrame
    df_input = pd.DataFrame([{col: payload[col] for col in SENSOR_COLS}])

    # Predict: 1 = normal, -1 = anomaly
    pred = pipeline.predict(df_input)[0]
    anomaly = 1 if pred == -1 else 0

    # Decision function score (lower = more anomalous)
    # Range is usually [-0.5, 0.5]
    raw_score = pipeline.decision_function(df_input)[0]

    # Map to 0-100 health index
    health_index = float(np.clip((raw_score + 0.5) * 100, 0, 100))
    fault_code   = health_to_fault_code(health_index)

    return {
        "device_id"    : payload["device_id"],
        "anomaly"      : anomaly,
        "raw_score"    : raw_score,
        "health_index" : round(health_index, 2),
        "fault_code"   : fault_code,
        "timestamp_utc": payload.get("timestamp_utc", ""),
    }


# ─────────────────────────────────────────────────────────────
# SALESFORCE INTEGRATION
# ─────────────────────────────────────────────────────────────
_sf_session = None

def get_salesforce_session():
    global _sf_session
    if _sf_session is not None:
        return _sf_session

    from simple_salesforce import Salesforce
    log.info(f"Authenticating with Salesforce at {SF_LOGIN_URL} …")
    _sf_session = Salesforce(
        username       = SF_USERNAME,
        password       = SF_PASSWORD,
        security_token = SF_SECURITY_TOKEN,
        consumer_key   = SF_CLIENT_ID,
        consumer_secret= SF_CLIENT_SECRET,
        domain         = "test" if "test.salesforce.com" in SF_LOGIN_URL else "login",
    )
    log.info("Salesforce authentication successful.")
    return _sf_session

def publish_platform_event(scored: dict) -> bool:
    try:
        sf = get_salesforce_session()
        event_payload = {
            "Device_ID__c"    : scored["device_id"],
            "Health_Index__c" : scored["health_index"],
            "Fault_Code__c"   : scored["fault_code"],
            "Raw_Score__c"    : scored["raw_score"],
            "Timestamp_UTC__c": scored["timestamp_utc"],
        }
        log.info(f"Publishing Platform Event: {json.dumps(event_payload)}")
        response = sf.IoT_Telemetry__e.create(event_payload)
        log.info(f"Platform Event published. Response: {response}")
        return True
    except Exception as exc:
        log.error(f"Failed to publish Platform Event: {exc}")
        return False

# ─────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────
def run_once(device_id: str, degraded: bool = False):
    payload = simulate_payload(device_id, degraded=degraded)
    scored  = score_payload(payload)
    
    # Create a simple visual health bar for the console
    bar_len = 20
    filled = int(scored['health_index'] / 100 * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)
    
    status_msg = f"[{device_id}] Health: {scored['health_index']:5.1f}% |{bar}| Code: {scored['fault_code']}"
    
    if scored["health_index"] < ALERT_THRESHOLD:
        log.warning(f"{status_msg} -> !!! ALERT TRIGGERED !!!")
        # Attempt to publish, but don't crash if Salesforce isn't configured
        publish_platform_event(scored)
    else:
        log.info(status_msg)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", default="FAN_COIL_001")
    parser.add_argument("--degraded", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()

    if args.loop:
        while True:
            try: run_once(args.device_id, degraded=args.degraded)
            except KeyboardInterrupt: break
            except Exception as e: log.error(e)
            time.sleep(args.interval)
    else:
        run_once(args.device_id, degraded=args.degraded)
