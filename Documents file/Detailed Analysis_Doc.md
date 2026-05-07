# IoT Predictive Maintenance — Full Implementation Plan

## Background

The project trains an unsupervised anomaly detection model on the industrial **Fan Coil I** dataset (large NPZ files with `foward.npy` / `rear.npy` vibration channels at 32kHz). The next phase involves bringing in a **CPU cooling fan** as the target device, collecting new sensor data, feeding it into the same AutoML pipeline to create **Model M2**, and ensuring the dashboard does not break in the process.

---

## Item 1 — Fix the Breaking Site

### Root Cause (Diagnosed)

The current `api.py` has **two hardcoded assumptions** that cause it to crash when data for a new device (e.g. a CPU fan) is uploaded:

**Bug 1 — Hardcoded feature schema for inference**
```python
# Line 39-42 in api.py — these 14 columns are ALWAYS expected
SENSOR_COLS = [
    "fwd_rms_vel", "fwd_mean", "fwd_std", "fwd_min", "fwd_max", "fwd_skew", "fwd_kurtosis",
    "rr_rms_vel",  "rr_mean",  "rr_std",  "rr_min",  "rr_max",  "rr_skew",  "rr_kurtosis",
]
```
When a CPU fan CSV with columns `accel_x`, `accel_y`, `accel_z` is uploaded, the pipeline detects "missing" columns and triggers AutoML training — **but then `run_inference()` still tries to slice using the old `SENSOR_COLS`**, causing a `KeyError` crash.

**Bug 2 — Training metadata not persisted**
After AutoML trains a new model for an unknown device, the feature columns it used are never saved. On the next upload (or after server restart), the system can't know which columns the loaded model was trained on.

**Bug 3 — WebSocket inference crashes if no model is loaded**
`websocket_stream` calls `run_inference(pipeline, ...)` without checking if `pipeline is None`, causing an `AttributeError` if no model file exists yet.

### Fix Plan

#### [MODIFY] [api.py](file:///d:/IOT%20Project/predictive-maintenance-ai/backend/api.py)

| Change | What it does |
| :--- | :--- |
| **Load `feature_cols` from `model_metadata.json`** | At startup, read `trained_feature_cols` from metadata so the correct columns are always used for inference |
| **Save `feature_cols` in metadata after every AutoML run** | Persist column names alongside algorithm/contamination so the model is self-describing |
| **Pass `feature_cols` into `run_inference()`** | Make inference use the model's actual schema, not the hardcoded Fan Coil list |
| **Add null-check in WebSocket endpoint** | Guard against crash when no model is loaded |
| **Add `fastapi` + `uvicorn` to requirements.txt** | They are missing from `requirements.txt`, which causes `pip install -r requirements.txt` to fail |

> [!IMPORTANT]
> After this fix, the pipeline becomes truly "device-agnostic" — any CSV with any numeric feature columns will work without code changes.

---

## Item 2 — Hardware Shopping List & Data Collection Protocol (CPU Fan)

### Why we need NEW data

The existing model (`fsl_heavy_motor_model.joblib`) was trained on vibration signatures of a **large industrial fan coil** running at high RPM with heavy mechanical loads. A **CPU cooling fan** has:
- Much lower mass (~25g vs. ~5kg)
- Different RPM range (1200–2500 RPM vs. 400–1200 RPM)
- Different vibration amplitude (milliG vs. full G)
- Only **one** vibration axis is dominant (radial, vs. forward + rear for the industrial unit)

**The M1 model will be completely wrong for the CPU fan.** We must train a new Model M2.

### 2.1 — Exact Shopping List

| # | Component | Exact Model / Part # | Price (INR) | Where to Buy | Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **CPU Cooling Fan** | Any standard **12V, 4-pin PWM** fan (e.g., **Noctua NF-F12 PWM** or a cheap generic 120mm fan) | ₹300–₹2,500 | Amazon India / local PC shop | The device under test |
| 2 | **Microcontroller** | **ESP32 DevKit V1** (WROOM-32, 30-pin) | ₹380–₹450 | RoboElements / Robu.in | Reads the sensor and sends data over Wi-Fi |
| 3 | **Vibration Sensor** | **MPU-6050 (GY-521)** — 3-axis MEMS accelerometer + gyroscope | ₹120–₹180 | Robu.in / Amazon | Measures vibration in X, Y, Z |
| 4 | **Fan Power Supply** | **12V 2A DC Adapter** (barrel jack) | ₹200–₹300 | Amazon / local electronics | Powers the 12V fan |
| 5 | **Voltage Regulator** | **LM2596 DC-DC Step-Down Module** | ₹50–₹100 | Robu.in | Converts 12V → 5V for the ESP32 |
| 6 | **Breadboard** | **MB-102 830-point Breadboard** | ₹80–₹120 | Any electronics shop | Prototyping connections |
| 7 | **Jumper Wires** | **Dupont 20cm M-M and M-F set** | ₹80–₹100 | Any electronics shop | Wiring |
| 8 | **Mounting** | **3M Mounting Tape (double-sided foam)** or **hot glue** | ₹50 | Any stationery / hardware store | Rigidly mounting MPU6050 to fan frame |

> [!IMPORTANT]
> **Do NOT use the MPU-6050 at 3.3V with 5V logic directly.** The ESP32 runs at 3.3V which is correct for the GY-521 module (it has an onboard regulator). Connect VCC to ESP32 **3V3** pin, **not** the 5V pin.

### 2.2 — Wiring Diagram

```
MPU-6050 (GY-521)          ESP32 DevKit V1
─────────────────          ───────────────
VCC          ──────────►  3V3
GND          ──────────►  GND
SCL          ──────────►  GPIO 22
SDA          ──────────►  GPIO 21
INT          (not needed)
```

### 2.3 — Step-by-Step Data Collection Protocol

**Step 1 — Physical Mounting (Critical)**
Mount the MPU-6050 flush against the **plastic fan housing frame** using double-sided foam tape or a tiny dab of hot glue. The sensor must not wobble. A loose sensor adds "rattle" noise that corrupts the dataset.

**Step 2 — Power On & Baseline (Normal State)**
- Run the fan at 100% speed (12V, no PWM throttling) for **3 minutes** before recording.
- This allows the bearing grease to warm up and the RPM to stabilize.
- Record 5 minutes of data at steady-state. This is your **HEALTHY** baseline.

**Step 3 — Early Anomaly Simulation**
- Take a small strip of **electrical tape (~1cm)** and attach it to the **outer edge of one blade only**.
- This creates a slight rotational imbalance — mimicking early-stage blade erosion.
- Record 3 minutes of data. Label this: **DEGRADED_WATCH**

**Step 4 — Fault State Simulation**
- Add a second strip of tape to the **same blade** (double the mass), or use a small piece of Blu-Tack.
- This represents significant imbalance or a cracked blade.
- Record 3 minutes of data. Label: **FAULT_IMMINENT**

**Step 5 — Critical Failure Simulation**
- Block ~30% of the fan air intake with a piece of cardboard.
- This creates **vortex shedding** and turbulent load spikes — mimicking a seized bearing or obstruction.
- Record 2 minutes of data. Label: **CRITICAL_FAILURE**

**Step 6 — Export CSV**
Save all data as a CSV with columns:
```
timestamp, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, temp
```

---

## Item 3 — How the AutoML M2 Pipeline Works (End-to-End)

### Overview Flow

```
CSV Upload (CPU fan data)
        │
        ▼
Feature Validation — Are the column names in SENSOR_COLS?
        │ NO — New Device Detected
        ▼
AutoML Training Pipeline (automl_train_and_select)
        │
        ├─ Step 1: StandardScaler normalization
        │          (so accel_x doesn't overpower gyro_z)
        │
        ├─ Step 2: Contamination Optimizer
        │          Probe IForest at 10% contamination
        │          Plot histogram of decision scores
        │          Find the "elbow" in the negative score region
        │          → Sets optimal contamination (typically 3–12%)
        │
        ├─ Step 3: Train IsolationForest (n_estimators=200)
        │          Calculate "Separation Score" = 
        │          mean(normal_scores) - mean(anomaly_scores)
        │
        ├─ Step 4: Train LocalOutlierFactor (n_neighbors=20, novelty=True)
        │          Calculate Separation Score for LOF
        │
        ├─ Step 5: Compare
        │          IF IForest_sep >= LOF_sep * 0.9 → IForest WINS
        │          (prefer IForest because it supports live streaming)
        │          ELSE → LOF WINS
        │
        └─ Step 6: Save Pipeline + Metadata
                   ├─ fsl_heavy_motor_model.joblib (scaler + model)
                   └─ model_metadata.json (algorithm, contamination, 
                      score_separation, trained_feature_cols ← FIX)
        │
        ▼
Inference on the uploaded data using the NEW M2 model
        │
        ▼
Health Index = clip((raw_score + 0.5) * 100, 0, 100)
Fault Code assignment → Response JSON
```

### Why This Works "At All Costs"

| Concern | How the Pipeline Handles It |
| :--- | :--- |
| **Small dataset (< 500 rows)** | Both IForest and LOF work on as few as 50 samples |
| **Unknown feature names** | Dynamic column detection — no hardcoded schema |
| **All data looks normal** | Contamination optimizer prevents over-flagging |
| **All data looks anomalous** | Min contamination floor of 1% prevents the model from labeling everything as bad |
| **Model file missing** | AutoML trains fresh from the uploaded file |
| **Server restart** | Model is persisted to disk, reloaded on startup |

---

## Item 4 — Feature Analysis: Anomaly vs. Failure Chart

### 4.1 — Feature Dictionary (What Each Feature Measures Physically)

| Feature | Mathematical Definition | Physical Meaning | What "High" Means | What "Low" Means |
| :--- | :--- | :--- | :--- | :--- |
| **`rms_vel`** | √(mean(velocity²)) — "Energy" | Total vibrational energy of the fan | **Imbalance or Misalignment** — the fan is shaking violently. Often the first sign of a bent blade or loose mount | Fan is at rest or perfectly balanced — good sign |
| **`kurtosis`** | 4th statistical moment — "Spikiness" | Presence of sharp, sudden impact events | **Bearing damage** — a cracked ball in the bearing creates sharp impacts on every revolution. Kurtosis > 4 is a strong early warning | Smooth sinusoidal vibration — healthy operation |
| **`skewness`** | 3rd statistical moment — "Lopsidedness" | Asymmetry of the vibration waveform | **Structural looseness** — mounting screws are loose, causing one-sided wobble on each spin cycle | Perfectly symmetric vibration — healthy |
| **`std`** | Standard deviation of acceleration | Spread / variability of vibration | **Erratic load** — airflow is turbulent or blocked, causing the motor to fight against variable resistance | Consistent, predictable vibration — healthy |
| **`min` / `max`** | Raw peak values | Extreme displacement events | **Blade strike** — a blade is hitting the housing. This creates periodic sharp amplitude spikes | Bounded, consistent range — healthy |
| **`mean`** | DC offset of acceleration | Gravitational or positional bias | Fan is mounted at an angle, or sensor is drifting (temperature effect) | Near zero (sensor properly zeroed) |

### 4.2 — Health Classification Chart

| Health Index | Fault Code | Feature Signature | Physical Diagnosis | Action |
| :--- | :--- | :--- | :--- | :--- |
| **75–100%** | `HEALTHY` | All features within 1 standard deviation of baseline. Low RMS, Kurtosis < 3, Skew ≈ 0 | Fan is operating normally. Bearings are lubricated. Blades are balanced. | ✅ No action. Continue monitoring. |
| **50–74%** | `DEGRADED_WATCH` | Kurtosis begins rising (3–6). RMS slightly elevated. Skew starts drifting from 0. | **Early bearing wear**. Micro-pitting on bearing races. Surface finish deteriorating. Imbalance developing. | ⚠️ Schedule inspection within **30 days**. Log in Salesforce as a Preventive Maintenance case. |
| **25–49%** | `FAULT_IMMINENT` | RMS elevated 2–3× baseline. Kurtosis > 8. Std rising significantly. Periodic spikes in min/max. | **Active bearing failure** or **moderate imbalance**. Rotor is shaking. Metal-on-metal contact occurring. | 🟠 Dispatch technician within **24–48 hours**. Prepare replacement bearing kit. Create FSL Work Order. |
| **0–24%** | `CRITICAL_FAILURE` | RMS elevated 5–10× baseline. Kurtosis > 15. All features far from normal cluster. Massive energy in std and peak-to-peak. | **Catastrophic failure**. Bearing has seized, blade has fractured, or motor winding is burning. Risk of fire or cascade damage. | 🚨 **Immediate shutdown**. Emergency dispatch. Create high-priority Case + Work Order in Salesforce. |

### 4.3 — The "Anomaly" vs "Failure" Distinction

> [!NOTE]
> These two terms are often confused. Here is the precise distinction for this system:

| Term | Definition | Feature Values | Example |
| :--- | :--- | :--- | :--- |
| **Anomaly** | A data point the model has not seen before — outside the normal cluster. It COULD be a fault, or just an unusual operating condition. | Isolation score is negative (model flags it), but Health Index > 25% | Fan ran at unusual speed briefly; Kurtosis spiked but returned to normal |
| **Failure** | A confirmed degraded state that persists over multiple consecutive samples. Health Index consistently below 25%. | All features persistently outside normal bounds across many samples | Bearing cracked; every revolution generates a shock spike |

The pipeline currently reports **Anomaly Rate %** (% of samples flagged). A single spike is an anomaly. A sustained pattern of anomalies = a failure.

---

## Verification Plan

### Step 1 — Fix the Bug
- Apply the `api.py` changes.
- Start the backend: `uvicorn api:app --reload`
- Upload a CSV with columns `accel_x, accel_y, accel_z` — it should train M2 and return results without crashing.

### Step 2 — Verify Metadata
- After upload, check `models/model_metadata.json` contains `trained_feature_cols`.

### Step 3 — Re-upload Same File
- Upload again — the system should **re-use M2** without re-training.

### Step 4 — Open Frontend
- Open `frontend/index.html` and confirm dashboard renders correctly.

---

## Open Questions

> [!IMPORTANT]
> **Q1**: Is the "site breaking" happening in the browser (e.g. a white screen, JavaScript error, "Failed to fetch") or in the backend terminal (a Python traceback)? This will confirm which bug is the primary cause.

> [!IMPORTANT]
> **Q2**: For the CPU fan data collection — do you have an ESP32 + MPU-6050 already, or do you need to purchase them first? This determines the timeline.

> [!IMPORTANT]
> **Q3**: Should the AutoML M2 model **replace** M1 on disk (overwrite `fsl_heavy_motor_model.joblib`), or should we maintain **separate model files** per device type (e.g. `model_fancoil.joblib`, `model_cpufan.joblib`)?
