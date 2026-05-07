# File Structure Deep-Dive: IoT Predictive Maintenance AI

This document provides a detailed inventory of the project's file structure, explaining the purpose, contents, and functional role of each component.

---

## 📂 Project Root
The root directory contains project-wide configuration, environment setup, and automation scripts.

*   **`requirements.txt`**: 
    *   *Functionality*: Lists all Python dependencies (FastAPI, Scikit-Learn, Joblib, etc.).
    *   *Usage*: Used to build the `ai_env` virtual environment.
*   **`logs.log`**: 
    *   *Functionality*: Centralized application log file.
    *   *Usage*: Tracks runtime errors, model training steps, and API request status.
*   **`push_code.ps1`**: 
    *   *Functionality*: PowerShell deployment script.
    *   *Usage*: Automates git staging and pushing for version control.
*   **`model_training.pynb`**: 
    *   *Functionality*: Jupyter Notebook for initial data exploration and manual model prototyping.
    *   *Usage*: Used by data scientists to validate the "Winner Duel" logic before it was moved to the production API.

---

## 📂 `backend/`
The "Brain" of the application, responsible for API handling and AutoML logic.

*   **`api.py`**: 
    *   *Functionality*: The main FastAPI application.
    *   *Contents*:
        *   **Endpoints**: `/api/upload`, `/ws/stream`, `/api/models`.
        *   **Core Logic**: Feature extraction for raw signals (.npz), AutoML model selection (LOF vs. IForest), and the health scoring algorithm.
    *   *Usage*: Must be running (`uvicorn api:app`) for the dashboard to function.

---

## 📂 `frontend/`
The visual interface for monitoring hardware health.

*   **`index.html`**: 
    *   *Functionality*: A "Single-Page Application" (SPA) built with Vanilla JS and CSS.
    *   *Contents*:
        *   **UI Components**: Drag-and-drop upload zone, AutoML log console, Health Gauge (Chart.js), and dynamic Anomaly tables.
        *   **Logic**: Communicates with the backend via Fetch API and WebSockets.
    *   *Usage*: Opened in any modern browser to view real-time predictive insights.

---

## 📂 `models/`
The persistence layer where AI models and their "Device Signatures" are stored.

*   **`model_registry.json`**: 
    *   *Functionality*: The "Phonebook" for the AI.
    *   *Contents*: Maps unique device column signatures (MD5 hashes) to specific `.joblib` files.
    *   *Usage*: Ensures that if you upload Fan Coil data, the system knows exactly which model to load.
*   **`fsl_heavy_motor_model.joblib`**: 
    *   *Functionality*: Binary weights for the **Fan Coil I** industrial model.
*   **`model_dev_7cfe5eb7a2.joblib`**: 
    *   *Functionality*: Binary weights for the **CPU Cooling Fan** model.
*   **`model_metadata.json`**: 
    *   *Functionality*: Stores global training parameters (Contamination %, Accuracy scores).

---

## 📂 `phase2_orchestrator/`
Bridges the gap between local AI and the Cloud (Salesforce).

*   **`orchestrator.py`**: 
    *   *Functionality*: A middleware script that monitors local health scores.
    *   *Usage*: Triggers Salesforce Platform Events when health falls below 25%, automatically creating Work Orders in FSL (Field Service Lightning).
*   **`.env.example`**: 
    *   *Functionality*: Template for Salesforce credentials (Username, Password, Security Token).

---

## 📂 `data/` & `eda_images/`
*   **`data/`**: Stores raw `.csv` and `.npz` samples used for initial training.
*   **`eda_images/`**: Contains generated plots (Correlation Heatmaps, Feature Importance) used for documentation and reports.

---

## 📂 `ai_env/`
*   **Functionality**: Python Virtual Environment (VENV).
*   **Usage**: Isolates the project's dependencies from the system-wide Python installation to prevent version conflicts.
