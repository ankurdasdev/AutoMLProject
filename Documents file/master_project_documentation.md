# IoT Predictive Maintenance AI: Master Project Documentation

This document serves as the **Single Source of Truth** for the entire project. It integrates technical architecture, machine learning logic, hardware implementation, and cloud orchestration into a comprehensive deep-dive.

---

## 1. Project Vision
The goal of this system is to provide **Zero-Touch Predictive Maintenance**. By simply uploading sensor data (CSV/NPZ), the system "self-discovers" the hardware signature, trains a customized AI model, and begins monitoring for mechanical failures before they happen.

---

## 2. Technical Architecture & Data Flow

### A. The Signature Engine
Every machine has a unique "Data Fingerprint." The system hashes the incoming column names (e.g., `accel_x`, `temp`) into a **Device Key**. 
*   **Purpose**: This prevents the "Fan Coil" model from being overwritten by a "CPU Fan" model. Each hardware type lives in its own isolated logic container.

### B. The Multi-Model Registry
*   **Persistence**: Models are saved as `.joblib` files.
*   **Registry**: A `model_registry.json` file acts as the router, mapping Device Keys to their respective model files and training metadata.

---

## 3. The ML Pipeline: "The Winner Duel"
The heart of the project is the **AutoML v3.0** engine.

1.  **Preprocessing**: Features are scaled using `StandardScaler`.
2.  **Contamination Optimization**: Uses a **Histogram Elbow** method. It automatically detects the natural "noise floor" of the sensor and sets the anomaly threshold (default floor: 1.4%).
3.  **The Competitors**:
    *   **Isolation Forest**: Best at finding "Global Outliers" (sudden spikes).
    *   **Local Outlier Factor (LOF)**: Best at finding "Local Density Shifts" (subtle mechanical wear).
4.  **Winner Selection**: The model with the highest **Shadow Separation** (distance between normal and anomaly clusters) is selected for production.

---

## 4. Hardware Implementation (CPU Cooling Fan)
For smaller-scale testing and deployment:

*   **Controller**: ESP32 DevKit V1.
*   **Sensor**: MPU-6050 (Accelerometer + Gyro).
*   **Target**: 12V 4-Pin CPU Cooling Fan.
*   **Data Collection**:
    *   **Healthy**: Clean spinning at 2000 RPM.
    *   **Degraded**: Blade imbalance (added weight) or bearing friction (dust).
    *   **Critical**: Structural looseness or stalled rotor.

---

## 5. UI/UX & State Management
The dashboard is a **State Machine** that adapts dynamically:
*   **State 1: Idle**: Waiting for upload.
*   **State 2: Training**: Shows real-time logs from the AutoML pipeline.
*   **State 3: Insights**: Renders the Health Gauge and the "Anomalous Feature Distribution" chart.
*   **Dynamic UI**: If a sensor name changes, the UI automatically updates table headers and chart legends without a page reload.

---

## 6. Cloud Integration (Salesforce Orchestration)
The system bridges the **Edge (Local AI)** with the **Cloud (Salesforce)**:
1.  **Local Inference**: Backend calculates a Health Score of 15%.
2.  **Orchestrator**: `orchestrator.py` detects the critical score.
3.  **Platform Event**: Sends a secure signal to Salesforce.
4.  **FSL Automation**: Salesforce Field Service Lightning automatically generates a **Work Order** and assigns a technician.

---

## 7. Comprehensive File Inventory

### Backend Core (`/backend`)
| File | Role |
| :--- | :--- |
| `api.py` | The main engine. Handles API, WebSockets, and ML Logic. |
| `requirements.txt` | Dependency list (FastAPI, Scikit-Learn, Joblib). |

### Models & Memory (`/models`)
| File | Role |
| :--- | :--- |
| `model_registry.json` | The "Phonebook" mapping hardware to models. |
| `*.joblib` | Binary weights for trained AI models. |

### Frontend (`/frontend`)
| File | Role |
| :--- | :--- |
| `index.html` | The Single-Page Dashboard (Vanilla JS + Chart.js). |

### Integration (`/phase2_orchestrator`)
| File | Role |
| :--- | :--- |
| `orchestrator.py` | The Salesforce bridge script. |

---

## 8. Operational Boundaries (Risk Matrix)

### 🟢 Works Perfectly When:
*   Data is numeric and features are consistent.
*   Dataset size is between 100 and 10,000 rows.
*   Machine failures are reflected in vibration or temperature shifts.

### 🔴 Breaks When:
*   **Missing Numeric Data**: Uploading non-numeric CSVs will crash the feature extractor.
*   **Zero Variance**: If a sensor is "stuck" on one value (e.g., 0.0), the Scaler may fail to normalize.
*   **RAM Limits**: Files larger than 500MB may cause memory exhaustion on local hardware.

---

> [!TIP]
> **Pro-Tip**: To restore the original Fan Coil sensitivity, always ensure the `contamination` floor in `api.py` is set to **0.014 (1.4%)**. This yields the industry-standard ~37 anomalies for the Fan Coil I dataset.
