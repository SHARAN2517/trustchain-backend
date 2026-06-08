# Trustchain-Med AI: Proof-of-Intelligence for Federated Clinical AI

[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.0%2B-green.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Trustchain-Med AI is a federated clinical intelligence backend that combines multimodal diagnosis, explainability, privacy accounting, accelerated audit logging, and blockchain-style proof generation for distributed healthcare deployments.

---

## 🔧 What’s New in This Backend

*   **API Authentication & Rate Limiting**: `X-API-Key` protected endpoints with per-key request throttling.
*   **Adaptive Rényi Differential Privacy**: `privacy.py` now includes an RDP accountant and privacy summaries for inference flows.
*   **Byzantine-Robust Federated Aggregation**: `federated.py` now uses Multi-Krum to resist malicious client updates.
*   **Asynchronous Audit & Anchoring**: Audit proof submission is queued asynchronously and progress can be streamed via WebSocket.
*   **Patient Anonymization**: Patient IDs are salted and hashed before storage for improved privacy.
*   **Model Versioning & Governance**: Version metadata, rollback capability, and governance vote tracking are now available.
*   **ZK-Proof Metadata Simulation**: Generates lightweight zero-knowledge proof commitments for audit transparency.
*   **DICOM Support**: Robust `.dcm` input parsing with window-level normalization.

---

## 📁 Project Structure

```text
backend/
├── app.py               # Local FastAPI server with auth, audit, DP, explainability
├── main.py              # Alternate FastAPI entrypoint with specialty routing and governance
├── model.py             # Multimodal PyTorch model definitions
├── privacy.py           # Differential privacy accountant and EWC helper
├── federated.py         # Federation simulation with Multi-Krum aggregation
├── explainability.py    # Grad-CAM and token-level explainability helpers
├── requirements.txt     # Python dependency manifest
├── README.md            # Project overview and usage guide
└── trustchain.db        # Local SQLite audit and prediction ledger (generated at runtime)
```

---

## 🚀 Quick Start

1.  Clone the repository:

    ```bash
    git clone https://github.com/SHARAN2517/trustchain-backend.git
    cd trustchain-backend
    ```

2.  Create and activate a Python virtual environment:

    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4.  Run the local server:

    ```bash
    python app.py
    ```

    or with Uvicorn:

    ```bash
    uvicorn app:app --host 127.0.0.1 --port 8000 --reload
    ```

---

## 📡 API Overview

### Authentication

All protected endpoints require the `X-API-Key` header.

Example keys:

*   `demo-key`
*   `trusted-partner`

### Inference Endpoint

*   `POST /predict`
*   `Headers`: `X-API-Key`
*   `multipart/form-data`
*   Required fields:
    *   `image` - upload image file or DICOM file
    *   `clinical_text` - patient note
    *   `hospital_id` - hospital node identifier
*   Optional fields:
    *   `department` - `radiology`, `cardiology`, `pediatrics`, `oncology`
    *   `target_disease`
    *   `epsilon_budget`
    *   `patient_id`

### Grad-CAM Heatmap

*   `POST /gradcam`
*   `Headers`: `X-API-Key`
*   `multipart/form-data`
*   Required fields:
    *   `file`
    *   `department`

### Text Explainability

*   `POST /explain`
*   `Headers`: `X-API-Key`
*   `application/x-www-form-urlencoded`
*   Required fields:
    *   `note`

### Audit Trail

*   `GET /blockchain/proofs`
*   `Headers`: `X-API-Key`
*   Optional query:
    *   `hospital_id`
    *   `limit`

### Reputation

*   `GET /reputation/{hospital_id}`
*   `Headers`: `X-API-Key`

### Model Versioning & Governance

*   `GET /model/version`
*   `POST /model/rollback`
*   `POST /governance/vote`
*   `GET /governance/status`

### WebSocket Streaming

*   `GET /ws/progress`
*   Streams progress and heartbeat events for async audit and fine-tune workflows.

---

## 🧠 Backend Capabilities

*   **Multimodal inference** combining image and clinical text.
*   **Explainability** via Grad-CAM and token importance.
*   **Adaptive privacy accounting** with Rényi DP summaries.
*   **Byzantine-robust federation** using Multi-Krum.
*   **Audit logging** with local SQLite and proof metadata.
*   **Patient anonymization** before storage.
*   **Governance workflows** for voting and policy updates.

---

## ⚠️ Notes

*   `app.py` is the primary local API server entrypoint.
*   `main.py` is an alternate FastAPI deployment script and includes additional specialty routing features.
*   If `trustchain_med_model.pth` is missing, the backend still runs with randomized weights for development.
