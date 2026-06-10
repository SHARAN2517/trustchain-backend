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

    `app.py` requires trained weights at `models/trustchain_med_model.pth`.
    For local API wiring only, set `TRUSTCHAIN_ALLOW_RANDOM_WEIGHTS=1`; do not
    use that mode for demos, validation, or clinical inference.

---

## Training

`train.py` trains only from a real CSV manifest by default. The manifest must
contain `image_path` and `note` columns plus either:

*   one JSON column named `labels`, for example `[0, 1, 0, 0, 0, 0, 0, 0]`
*   or one binary column per class name supplied through `--labels`

Optional metadata columns `age`, `sex`, and `study_description` are consumed by
the metadata fusion gate during training.

Example:

```bash
python train.py ^
  --manifest data/train_manifest.csv ^
  --image-root data/images ^
  --epochs 20 ^
  --min-samples 1000 ^
  --save-model
```

The trained state dict is saved to `models/trustchain_med_model.pth`.

Differential privacy is opt-in and guarded against tiny datasets:

```bash
python train.py --manifest data/train_manifest.csv --image-root data/images --dp --min-dp-samples 5000 --save-model
```

For a non-deployable plumbing check:

```bash
python train.py --smoke-test
```

Smoke-test mode uses random tensors and refuses to save model weights.

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
    *   `patient_age` - optional metadata, for example `045Y`
    *   `patient_sex` - optional metadata, for example `M` or `F`
    *   `study_description` - optional study metadata
    *   `mc_samples` - Monte Carlo dropout samples for uncertainty, default `20`
    *   `uncertainty_threshold` - manual-review trigger threshold, default `0.12`

Inference responses include `uncertainty.manual_review_required` and
`proof.proof_id`. The proof links the model version, model weights, input hash,
metadata hash, and output hash for audit verification.

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

### Inference Proof Lookup

*   `GET /audit/inference/{proof_id}`
*   `Headers`: `X-API-Key`
*   Returns the stored proof of inference and verifies its commitment.

### HITL Clinician Review

*   `POST /hitl/review`
*   `Headers`: `X-API-Key`
*   `application/x-www-form-urlencoded`
*   Required fields:
    *   `proof_id`
    *   `action` - `verify` or `overrule`
    *   `clinician_id`
*   Optional fields:
    *   `original_diagnosis`
    *   `corrected_diagnosis` - required for `overrule`
    *   `corrected_department`
    *   `notes`

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
*   Benchmark endpoints report `not_evaluated` until real held-out predictions are passed through `evaluator.compute_multilabel_metrics()`.
