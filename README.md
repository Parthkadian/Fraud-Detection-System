<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Inter&weight=700&size=32&pause=1200&color=C49A2E&center=true&vCenter=true&width=800&lines=Enterprise+Fraud+Detection+Platform;Production+ML+%7C+Real-Time+Risk+Scoring;SHAP+%E2%80%A2+Drift+Monitor+%E2%80%A2+Audit+Trail+%E2%80%A2+Async+Pipeline" alt="Fraud Detection System" />

<br/>

# Fraud Detection System

**A production-grade, end-to-end machine learning platform for real-time credit card fraud detection.**  
From raw transactions to ranked fraud insights — with full explainability, drift monitoring, audit trails, async processing, and a human review queue.

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1.1-FF6600?style=flat-square)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-7C3AED?style=flat-square)](https://shap.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?style=flat-square&logo=mlflow&logoColor=white)](https://mlflow.org)
[![CI](https://img.shields.io/github/actions/workflow/status/Parthkadian/Fraud-Detection-System/ci.yml?branch=main&style=flat-square&label=CI&logo=github)](https://github.com/Parthkadian/Fraud-Detection-System/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-C49A2E?style=flat-square)](LICENSE)

<br/>

> *"Not just a model — a complete fraud intelligence platform built to production engineering standards."*

</div>

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Key Capabilities](#key-capabilities)
- [ML Pipeline](#ml-pipeline)
- [API Reference](#api-reference)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Docker Deployment](#docker-deployment)
- [Configuration](#configuration)
- [Model Performance](#model-performance)
- [Explainability & Fairness](#explainability--fairness)
- [Drift Monitoring](#drift-monitoring)
- [Testing](#testing)
- [CI/CD Pipeline](#cicd-pipeline)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This platform implements a **full-stack fraud detection system** built around the [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (284,807 transactions — 0.173% fraud rate). It demonstrates a production ML engineering workflow covering the complete lifecycle: raw data ingestion → feature engineering → model training → serving → monitoring.

### What Differentiates This Platform

Most fraud detection repositories are Jupyter notebooks. This is a **complete, deployable platform**:

| Dimension | Implementation |
|-----------|----------------|
| **Production ML** | Config-driven pipeline: ingestion → cleaning → features → training → serving |
| **Explainability** | SHAP `TreeExplainer` attribution on every prediction — fully auditable |
| **Business Rules** | YAML-defined rule engine layered on top of ML score with severity tiers |
| **Audit Trail** | Every prediction logged with full provenance, request ID, and risk classification |
| **Drift Detection** | Evidently AI monitors feature distribution shift against training reference |
| **Async Processing** | Celery + Redis for non-blocking high-volume batch scoring |
| **Human Review** | Case management queue for analyst escalation workflows |
| **Observability** | Prometheus metrics, structured JSON logging, distributed request tracing |
| **Infrastructure** | Docker Compose multi-service stack, GitHub Actions CI/CD, Railway deployment |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FRAUD DETECTION PLATFORM v2.0                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   TRAINING PIPELINE                                                 │
│   ┌────────────┐  ┌────────────┐  ┌─────────────┐  ┌───────────┐  │
│   │  Data       │→ │  Data      │→ │  Feature    │→ │  XGBoost  │  │
│   │  Ingestion  │  │  Cleaning  │  │  Engineering│  │  Training │  │
│   │            │  │            │  │             │  │           │  │
│   │ Schema     │  │ Outlier    │  │ log_amount  │  │ 5-Fold    │  │
│   │ Validation │  │ Detection  │  │ time feats  │  │ Strat. CV │  │
│   │ NLP Memos  │  │ Imputation │  │ TF-IDF+SVD  │  │ Threshold │  │
│   └────────────┘  └────────────┘  └─────────────┘  │ Optimise  │  │
│                                                     │ MLflow    │  │
│                                                     └───────────┘  │
│                                                           │        │
│   SERVING LAYER                              ┌────────────┘        │
│   ┌───────────────────┬──────────────────────┼──────────────────┐  │
│   │  FastAPI REST API │  Celery + Redis       │  Streamlit UI    │  │
│   │  v2.0             │  Async Workers        │  Dashboard       │  │
│   │                   │                       │                  │  │
│   │  POST /predict    │  Non-blocking          │  Single scoring  │  │
│   │  POST /explain    │  batch scoring         │  Batch upload    │  │
│   │  POST /batch      │                       │  Drift monitor   │  │
│   │  GET  /audit      │                       │  Analytics       │  │
│   │  GET  /health     │                       │  Audit trail     │  │
│   │  GET  /metrics    │                       │  Business rules  │  │
│   └───────────────────┴───────────────────────┴──────────────────┘  │
│                                                                     │
│   MONITORING & DATA LAYER                                           │
│   ┌──────────────┬──────────────────┬──────────────────────────┐   │
│   │  Evidently   │  Prometheus      │  SQLite / PostgreSQL      │   │
│   │  Drift AI    │  /metrics        │  Audit DB  │  Platform DB │   │
│   └──────────────┴──────────────────┴──────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

**Data Flow Summary:**
```
creditcard.csv → Ingestion → Cleaning → Feature Engineering → XGBoost CV Training
     → Threshold Optimisation → MLflow Tracking → models/artifacts/
     → FastAPI /predict → Business Rules → SHAP → Audit Log → Response
```

---

## Key Capabilities

### Real-Time Single Transaction Scoring
Submit any transaction with 30 features (`Time`, `Amount`, `V1–V28`) and receive:
- **Fraud probability** (calibrated, 0.0 → 1.0)
- **Risk classification** (`LOW` / `MEDIUM` / `HIGH`)
- **Business rule flags** with triggered rule names and severity
- **SHAP waterfall explanation** (top feature contributions, per-prediction)
- **Latency measurement** (millisecond precision, returned in response)

### Batch CSV Scoring
Upload a CSV of N transactions and receive:
- Full batch scoring with risk distribution
- ROC curve, Precision-Recall curve, Confusion Matrix
- KDE fraud probability distribution, fraud time trends
- Feature importance proxy chart
- Downloadable scored results CSV

### Data Drift Monitoring
- Evidently AI HTML report comparing production data to training reference
- 3-tier reference loading: raw CSV → committed snapshot → synthetic fallback
- Per-feature drift scores (Wasserstein distance, chi-square)
- Dataset-level drift summary and alert thresholds

### Asynchronous Batch Processing
- Celery + Redis backed `/predict_async` endpoint
- Task ID polling via `/task_status/{id}` for non-blocking workflows
- Designed for high-volume production throughput

### Human Review Workflow
- Fraud cases created automatically for HIGH-risk predictions
- Analyst queue at `/review/queue` with full case context
- `PATCH /review/{case_id}/decision` to submit analyst verdicts
- Complete decision audit trail persisted to database

### Audit Trail & Observability
- Every prediction stored in SQLite/PostgreSQL with request ID, timestamp, risk level, rule triggered
- Prometheus `/metrics` endpoint for Grafana integration
- Structured JSON logging with log rotation
- `X-Request-ID` and `X-Process-Time` headers on every response

---

## ML Pipeline

### Feature Engineering

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `log_amount` | `log1p(Amount)` | Reduces right-skew in transaction amounts |
| `hour` | Hour of day from elapsed seconds | Captures time-of-day fraud patterns |
| `is_night_transaction` | Binary flag for 00:00–06:00 | Night transactions correlate with higher fraud risk |
| `amount_zscore` | Z-score of Amount | Detects anomalous transaction amounts relative to batch |
| `v_features_magnitude` | L2 norm of V1–V28 | Single scalar capturing PCA component strength |
| `amount_time_ratio` | `Amount / (Time + 1)` | Interaction feature: unusual amount velocity |
| `nlp_svd_0..4` | TF-IDF + TruncatedSVD on transaction memos | NLP features from synthetic transaction descriptions |

### Training Configuration

```yaml
model: XGBoostClassifier
  n_estimators: 150
  max_depth: 4
  learning_rate: 0.03
  subsample: 0.8
  colsample_bytree: 0.8
  reg_alpha: 1.0           # L1 regularisation
  reg_lambda: 5.0          # L2 regularisation
  min_child_weight: 5
  gamma: 0.3
  scale_pos_weight: ~577   # Dynamic — computed from training class ratio

cross_validation: StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
threshold: Grid search over [0.01..0.99] maximising F1 on validation set
split: 70% train / 15% validation / 15% test (stratified)
```

### Business Rules Engine

Rules are defined in `configs/business_rules.yaml` and evaluated **after** the ML score. Rules can:
- `flag` — keep ML prediction, mark as rule-triggered
- `override_high` — force `prediction=1, risk_level=HIGH` regardless of model output

Rules are evaluated by severity: `CRITICAL > HIGH > MEDIUM > LOW`. Example ruleset:

```yaml
rules:
  - name: "Large Amount Threshold"
    field: Amount
    operator: ">"
    value: 5000
    severity: CRITICAL
    action: override_high

  - name: "High Amount Night Transaction"
    field: Amount
    operator: ">"
    value: 2000
    secondary_field: hour
    secondary_operator: "<"
    secondary_value: 5
    severity: HIGH
    action: flag

  - name: "PCA Anomaly Burst"
    field: v_features_magnitude
    operator: ">"
    value: 8.0
    severity: MEDIUM
    action: flag
```

---

## API Reference

FastAPI v2.0 · Auto-generated interactive docs at **`/docs`** · ReDoc at **`/redoc`**

### Authentication

Set `FRAUD_API_KEY` environment variable to enable API key authentication. Protected endpoints require an `X-API-Key` header. Unset in development for open access.

### Endpoints

#### System

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/` | — | API root, version info |
| `GET` | `/health` | — | Model, DB, and Redis status |
| `GET` | `/metrics` | — | Prometheus metrics scrape |
| `GET` | `/model_card` | — | Google Model Card (JSON) |

#### Prediction

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/predict` | Optional | Score a single transaction (synchronous) |
| `POST` | `/predict_batch` | Optional | Score multiple transactions (synchronous) |
| `POST` | `/predict_compare` | Optional | Champion-challenger model comparison |
| `POST` | `/predict_async` | Optional | Queue async prediction via Celery |
| `GET` | `/task_status/{id}` | Optional | Poll async task result |

#### Explainability

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/explain` | Optional | SHAP explanation for one transaction |
| `GET` | `/explain/global` | Optional | Global feature importance |
| `POST` | `/drift_report` | Optional | Generate Evidently HTML drift report |

#### Audit & Monitoring

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/audit/history` | 🔒 | Paginated prediction audit log |
| `GET` | `/rules` | — | Active business rules listing |
| `GET` | `/alerts/recent` | 🔒 | Recent fraud alert records |

#### Case Management

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/cases` | 🔒 | Create a fraud investigation case |
| `GET` | `/cases` | 🔒 | List cases with filters |
| `PATCH` | `/cases/{id}` | 🔒 | Update case status |
| `GET` | `/review/queue` | 🔒 | Human analyst review queue |
| `PATCH` | `/review/{case_id}/decision` | 🔒 | Submit analyst decision |

#### Risk Intelligence

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/customers/{id}/risk-profile` | 🔒 | Customer risk profile |
| `GET` | `/merchants/{id}/risk-profile` | 🔒 | Merchant risk profile |
| `GET` | `/velocity/{entity_type}/{value}` | 🔒 | Transaction velocity signals |
| `POST` | `/blacklist` | 🔒 | Add entity to blacklist |
| `POST` | `/whitelist` | 🔒 | Add entity to whitelist |

### Example: Score a Transaction

```bash
curl -X POST https://your-api-host/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 10000, "Amount": 150.50,
    "V1": -1.2, "V2": 0.3, "V3": 1.1, "V4": 0.5,
    "V5": -0.2, "V6": 0.1, "V7": 0.2, "V8": -0.1,
    "V9": 0.4, "V10": -0.3, "V11": 0.2, "V12": -0.5,
    "V13": 0.1, "V14": -0.2, "V15": 0.3, "V16": -0.1,
    "V17": 0.2, "V18": 0.1, "V19": -0.3, "V20": 0.05,
    "V21": -0.02, "V22": 0.1, "V23": -0.03, "V24": 0.2,
    "V25": -0.1, "V26": 0.05, "V27": 0.02, "V28": -0.01
  }'
```

**Response:**

```json
{
  "transaction_id": "txn_a3f7c2e1",
  "fraud_probability": 0.0023,
  "prediction": 0,
  "risk_level": "LOW",
  "rule_triggered": false,
  "rule_name": null,
  "model_version": "2.0.0",
  "latency_ms": 4.2
}
```

### Example: SHAP Explanation

```bash
curl -X POST https://your-api-host/explain \
  -H "Content-Type: application/json" \
  -d '{ ... same payload as /predict ... }'
```

**Response:**

```json
{
  "transaction_id": "txn_b9k1p0d4",
  "fraud_probability": 0.847,
  "risk_level": "HIGH",
  "explanation": [
    { "feature": "V14",    "shap_value": -0.412, "description": "High-value transaction signal" },
    { "feature": "V17",    "shap_value": -0.318, "description": "Session behaviour anomaly" },
    { "feature": "Amount", "shap_value":  0.229, "description": "Transaction amount £4,980" },
    { "feature": "V12",    "shap_value": -0.198, "description": "Cross-border indicator" },
    { "feature": "V4",     "shap_value":  0.142, "description": "Merchant category risk score" }
  ]
}
```

---

## Streamlit Dashboard

The Streamlit dashboard (`dashboard/app.py`) is a premium dark-mode fraud intelligence workspace.

### Navigation Tabs

| Tab | Description |
|-----|-------------|
| 🎯 **Single Scoring** | 30-field transaction form with PCA tooltips, outlier warnings, preset examples. Returns fraud gauge, risk badge, SHAP waterfall |
| 📁 **Batch CSV** | Drag-drop CSV upload → full batch scoring → 8+ analytics charts + CSV download |
| 🌊 **Drift Monitor** | Evidently HTML report with per-feature drift scores and dataset-level summary |
| 🔁 **Async Stream** | Celery task submission + polling interface for non-blocking high-volume scoring |
| 💬 **NLP Insights** | Word cloud analysis of fraud vs legitimate transaction memo language patterns |
| 📊 **Model Performance** | Live ROC-AUC, F1, MCC, PR-AUC with all evaluation charts rendered on-demand |
| 🗂️ **Audit Trail** | Searchable, filterable prediction history — exportable to CSV |
| 📋 **Business Rules** | Active rules table with trigger counts and last-fired timestamps |
| 🪪 **Model Card** | Full Google Model Card with intended use, limitations, and ethical considerations |
| ⚡ **System Health** | Real-time API health, model status, Redis and DB connectivity |

### Running the Dashboard

```bash
# Ensure API is running first (port 8000)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Launch dashboard in a new terminal
streamlit run dashboard/app.py
```

Dashboard available at **http://localhost:8501**

---

## Technology Stack

| Layer | Technology | Version | Purpose |
|-------|------------|---------|---------|
| **ML Core** | XGBoost | 2.1.1 | Gradient boosted decision tree classifier |
| **ML Framework** | scikit-learn | 1.5.1 | Preprocessing, CV, metrics |
| **Explainability** | SHAP | 0.46.0 | Per-prediction feature attribution |
| **NLP** | TF-IDF + TruncatedSVD | — | Transaction memo feature extraction |
| **API** | FastAPI | 0.115.0 | REST microservice with OpenAPI docs |
| **Server** | Uvicorn | 0.30.6 | ASGI server |
| **Validation** | Pydantic | 2.9.2 | Request/response schema validation |
| **Dashboard** | Streamlit | 1.38.0 | Interactive fraud analytics UI |
| **Async** | Celery + Redis | 5.4.0 + 5.0.8 | Non-blocking batch prediction workers |
| **Database** | SQLite / PostgreSQL | — | Audit trail and platform data |
| **Monitoring** | Evidently AI | 0.4.39 | Feature drift detection and reporting |
| **Experiment Tracking** | MLflow | 2.16.2 | Parameters, metrics, model registry |
| **Metrics** | Prometheus Client | 0.21.0 | Instrumentation and scraping |
| **Infrastructure** | Docker + Compose | — | Containerised multi-service deployment |
| **CI/CD** | GitHub Actions | — | Lint, test, Docker build pipeline |
| **Linting** | Ruff | 0.6.9 | Fast Python linter and formatter |
| **Deployment** | Railway | — | Cloud API hosting |

---

## Project Structure

```
fraud-detection-system/
│
├── api/                              # FastAPI microservice (v2.0)
│   ├── main.py                       # App factory, lifespan hooks, middleware
│   ├── schemas.py                    # 40+ Pydantic request/response models
│   ├── middleware.py                 # Request ID, timing, rate limiting, API key auth
│   ├── database.py                   # SQLAlchemy ORM — audit/case/alert/profile tables
│   ├── celery_worker.py              # Async Celery task definitions
│   ├── router.py                     # Route aggregation
│   ├── requirements.txt              # API-specific dependencies
│   ├── dockerfile                    # API service Dockerfile
│   └── routes/                       # 10 modular route files
│       ├── prediction.py             # /predict, /predict_batch, /predict_async
│       ├── explainability.py         # /explain, /explain/global
│       ├── monitoring.py             # /health, /metrics, /audit, /rules, /drift_report
│       ├── cases.py                  # /cases/* CRUD
│       ├── review.py                 # /review/* human analyst queue
│       ├── lists.py                  # /blacklist, /whitelist
│       ├── profiles.py               # /customers/*, /merchants/*, /velocity/*
│       ├── alerts.py                 # /alerts/*
│       ├── business.py               # /dashboard/summary, /business/*
│       └── model_ops.py              # /model/version, /model/performance
│
├── dashboard/
│   ├── app.py                        # Streamlit dashboard (3,200+ lines)
│   ├── dockerfile                    # Dashboard service Dockerfile
│   └── requirements.txt              # Dashboard-specific dependencies
│
├── src/                              # Core ML and business logic
│   ├── ingestion/
│   │   └── data_ingestion.py         # Dataset loading + NLP memo synthesis
│   ├── preprocessing/
│   │   ├── data_validator.py         # Schema, type, and range validation
│   │   ├── data_cleaning.py          # Cleaning + IQR-based outlier detection
│   │   └── splitter.py               # Stratified 70/15/15 train/val/test split
│   ├── features/
│   │   └── feature_engineering.py   # Tabular + NLP feature pipeline (stateful transformer)
│   ├── training/
│   │   ├── train.py                  # CV training loop + early stopping + MLflow
│   │   ├── model_factory.py          # Config-driven model instantiation
│   │   └── thresholding.py           # F1-optimised threshold grid search
│   ├── evaluation/
│   │   └── evaluate.py               # F1, MCC, Cohen's Kappa, ROC-AUC, PR-AUC
│   ├── inference/
│   │   ├── predict.py                # Single-transaction predictor with rule engine
│   │   └── batch_predict.py          # DataFrame batch predictor
│   ├── explainability/
│   │   └── shap_explainer.py         # SHAP TreeExplainer integration
│   ├── monitoring/
│   │   ├── drift_detection.py        # Evidently AI drift reports (3-tier reference loading)
│   │   ├── alerting.py               # Threshold-based alert engine
│   │   └── logger.py                 # Structured logging with rotation
│   ├── rules/
│   │   └── rule_engine.py            # YAML-driven business rule engine
│   └── utils/
│       ├── common.py                 # I/O helpers (joblib, JSON)
│       ├── config_loader.py          # YAML config loader
│       └── exceptions.py             # Custom exception hierarchy
│
├── configs/
│   ├── config.yaml                   # Project-wide paths, training, monitoring config
│   ├── model_params.yaml             # XGBoost hyperparameters
│   └── business_rules.yaml           # YAML-defined business rules
│
├── tests/                            # Comprehensive test suite (pytest)
│   ├── conftest.py                   # Shared fixtures
│   ├── test_api.py                   # API endpoint tests
│   ├── test_data_ingestion.py        # Data loading and NLP synthesis
│   ├── test_preprocessing.py         # Cleaning, validation, outlier detection
│   ├── test_training.py              # CV training, early stopping, model save/load
│   ├── test_evaluation.py            # All metrics, serialisation, edge cases
│   ├── test_inference.py             # Risk logic, SHAP output, batch predictor
│   └── test_transformer.py           # Feature engineering pipeline tests
│
├── docs/
│   └── model_card.json               # Google Model Card (machine-readable JSON)
│
├── configs/
│   ├── config.yaml
│   ├── model_params.yaml
│   └── business_rules.yaml
│
├── data/
│   ├── raw/                          # creditcard.csv (not tracked in git — download separately)
│   ├── fraud_platform.db             # SQLite platform database
│   ├── audit_log.db                  # Prediction audit log
│   └── schemas/
│       └── data_schema.yaml          # Formal column schema definition
│
├── models/
│   ├── artifacts/                    # fraud_model.pkl, scaler.pkl, metrics.json, threshold.json
│   └── trained/                      # Versioned model checkpoints
│
├── reports/figures/                  # EDA visualisation outputs
├── logs/                             # Structured application logs
├── notebooks/
│   └── generate_eda.py               # EDA visualisation generator
│
├── .github/workflows/
│   └── ci.yml                        # CI: lint → test → Docker build
│
├── docker-compose.yml                # Full stack: API + PostgreSQL + Redis + Celery
├── Dockerfile                        # Production API container
├── railway.toml                      # Railway.app cloud deployment config
├── pyproject.toml                    # Ruff linter + pytest configuration
├── requirements.txt                  # Python dependencies (pinned versions)
├── run_pipeline.py                   # One-command training pipeline CLI
├── CONTRIBUTING.md                   # Developer contribution guide
└── README.md                         # This document
```

---

## Quick Start

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.10+ |
| pip | Latest |
| Docker + Docker Compose | Optional (for containerised run) |
| Redis | Optional (for async Celery features) |

### 1. Clone and Install

```bash
git clone https://github.com/Parthkadian/Fraud-Detection-System.git
cd Fraud-Detection-System

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

# Install all dependencies
pip install -r requirements.txt
```

### 2. Obtain the Dataset

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it at:

```
data/raw/creditcard.csv
```

> **Note:** This file is not included in the repository. It contains 284,807 real credit card transactions from September 2013.

### 3. Train the Model

```bash
# Full pipeline — XGBoost (default)
python run_pipeline.py

# Specify an alternative model
python run_pipeline.py --model logistic_regression
python run_pipeline.py --model random_forest

# Skip cross-validation for faster iteration during development
python run_pipeline.py --skip-cv
```

The pipeline executes:
1. Schema validation and data cleaning
2. Feature engineering (tabular + NLP)
3. Stratified 5-fold cross-validation
4. Final model training with early stopping
5. Threshold optimisation (F1-grid search on validation set)
6. MLflow experiment logging
7. Artefact persistence to `models/artifacts/`

### 4. Start the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

- API: **http://localhost:8000**
- Swagger UI: **http://localhost:8000/docs**
- ReDoc: **http://localhost:8000/redoc**

### 5. Launch the Dashboard

```bash
streamlit run dashboard/app.py
```

- Dashboard: **http://localhost:8501**

---

## Docker Deployment

### Full Stack

```bash
docker-compose up --build
```

| Service | Port | Description |
|---------|------|-------------|
| `api` | `8000` | FastAPI REST microservice |
| `postgres` | `5432` | PostgreSQL production database |
| `redis` | `6379` | Celery message broker |
| `celery_worker` | — | Async prediction worker |

### Environment Variables

Create a `.env` file in the project root (see `.env.example` for template):

```bash
# API Authentication
FRAUD_API_KEY=your-secret-api-key-here

# API Settings
API_BASE_URL=http://localhost:8000
RATE_LIMIT_PER_MINUTE=120

# Database (omit for SQLite in development)
DATABASE_URL=postgresql://fraud:your_password@postgres:5432/fraud_db

# Celery / Redis
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/0

# Dashboard
NEXT_PUBLIC_API_URL=http://localhost:8000
```

> **Security Notice:** Never commit credentials to version control. Use environment variables or a secrets manager in production.

---

## Configuration

All system behaviour is driven by YAML configuration files in `configs/`:

### `configs/config.yaml` — Main Configuration

```yaml
project:
  name: "fraud_detection_system"
  version: "2.0.0"
  random_state: 42

training:
  model_name: "xgboost"       # xgboost | logistic_regression | random_forest
  threshold_metric: "f1"      # Metric to optimise during threshold search

monitoring:
  alert_fraud_rate_threshold: 0.05   # Alert when fraud rate exceeds 5%
  alert_high_risk_threshold: 0.10    # Alert when high-risk rate exceeds 10%
```

### `configs/business_rules.yaml` — Rule Engine

Rules are hot-reloadable — modify and restart the API to apply changes without retraining.

```yaml
rules:
  - name: "Large Amount Threshold"
    field: Amount
    operator: ">"
    value: 5000
    severity: CRITICAL
    action: override_high
    active: true
```

---

## Model Performance

> **Dataset note:** The [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) uses pre-anonymised PCA features (V1–V28). This is a well-documented characteristic of this dataset — tree-based models achieve near-perfect separation due to the PCA preprocessing applied by the card network. This is not data leakage. See the [Kaggle discussion](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/discussion) for community acknowledgement.
>
> **The engineering value of this project lies in the platform capabilities**, not the metric numbers: threshold optimisation, SHAP explainability, drift monitoring, business rules, audit trails, and human review workflows.

### Dataset Statistics

| Property | Value |
|----------|-------|
| Total transactions | 284,807 |
| Fraudulent | 492 (0.173%) |
| Class imbalance ratio | 1 : 578 |
| Feature set | 28 PCA components (V1–V28) + Time + Amount |
| Time window | 48-hour capture (September 2013) |

### Benchmark Results

| Metric | Validation | Test (Held-Out) | Notes |
|--------|------------|-----------------|-------|
| **F1-Score** | ~0.99 | ~0.99 | Near-perfect on this PCA dataset |
| **ROC-AUC** | ~0.99 | ~0.99 | Features are highly separable |
| **PR-AUC** | ~0.85 | ~0.85 | Most meaningful metric for imbalanced data |
| **MCC** | ~0.95 | ~0.95 | Best single metric for fraud detection |
| **Decision Threshold** | 0.10 | — | Optimised via F1 grid search (not default 0.5) |
| **CV F1 (5-Fold)** | ~0.99 ± 0.00 | — | Stratified to preserve class balance |

### What Matters in Production

| Capability | Implementation |
|-----------|----------------|
| **Threshold Optimisation** | Grid search over F1 on validation set — not the naive 0.5 default |
| **Class Imbalance** | `scale_pos_weight` computed dynamically per training fold |
| **SHAP Explainability** | Per-transaction feature attribution for auditability and analyst review |
| **Drift Monitoring** | Evidently AI detects when production data diverges from training distribution |
| **Human Review Queue** | Escalation workflow with analyst decision capture and audit trail |
| **Business Rules** | YAML-defined hard guardrails that override model scores |
| **Async Throughput** | Celery + Redis for non-blocking bulk scoring pipelines |

---

## Explainability & Fairness

Every prediction is accompanied by a SHAP explanation, making every model decision auditable.

### Per-Prediction SHAP

The `/explain` endpoint uses `shap.TreeExplainer` on the trained XGBoost model and returns the top feature contributions ranked by absolute SHAP value. The Streamlit dashboard renders these as a horizontal waterfall bar chart.

### Model Card

The full [Google Model Card](docs/model_card.json) documents:

- **Intended use:** Real-time decision support for financial fraud analysts — not sole basis for adverse action
- **Out of scope:** Identity verification, creditworthiness assessment, non-credit-card transactions
- **Bias and fairness:** PCA anonymisation prevents direct demographic analysis; human review is required before adverse action
- **Privacy:** Audit logs store SHA-256 hashes of inputs, not raw data; data minimisation principles applied
- **Limitations:** Trained on 2013 European data; recommend retraining when drift share exceeds 20%

Served via API at `GET /model_card`.

---

## Drift Monitoring

The `DriftMonitor` class uses **Evidently AI** to detect when production transaction distributions diverge from training data.

### Reference Data Loading (3-Tier Fallback)

1. **Raw CSV** (`data/raw/creditcard.csv`) — available after training; sampled to 1,000 rows
2. **Committed snapshot** (`data/drift_reference.csv`) — lightweight reference committed to the repo
3. **Synthetic fallback** — generated from the published UCI dataset feature statistics (always available, even on fresh clone)

### Generating a Drift Report

```bash
# Via API
curl -X POST http://localhost:8000/drift_report \
  -H "Content-Type: application/json" \
  -d '{"current_data": [ ...transaction dicts... ]}'
```

Or upload a CSV in the **🌊 Drift Monitor** tab of the Streamlit dashboard. The HTML report includes:
- Dataset-level drift score and share of drifted features
- Per-feature drift metrics (Wasserstein distance, chi-square tests)
- Distribution comparison plots for every feature

**Recommended retraining trigger:** when `drift_share > 0.20` (20% of features show statistically significant drift).

---

## Testing

```bash
# Run full test suite
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=src --cov=api --cov-report=term-missing

# Run a specific module
pytest tests/test_api.py -v
pytest tests/test_inference.py -v
pytest tests/test_training.py -v
```

### Coverage Areas

| Test Module | What It Covers |
|-------------|----------------|
| `test_api.py` | API endpoints, middleware headers, validation errors |
| `test_data_ingestion.py` | Dataset loading, NLP memo synthesis, schema edge cases |
| `test_preprocessing.py` | Data cleaning, outlier detection, schema validation |
| `test_training.py` | CV training loop, early stopping, model save and load |
| `test_evaluation.py` | All metrics, serialisation, edge cases (zero positives) |
| `test_inference.py` | Risk level logic, SHAP output format, batch predictor |
| `test_transformer.py` | Feature engineering pipeline, NLP transform, shape consistency |

---

## CI/CD Pipeline

GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push and pull request to `main`:

```
Push / PR to main
       │
       ├─► Lint (ruff check)
       │         │
       │         └─► Tests (pytest + coverage → Codecov)
       │
       └─► Docker Build Check (no push — validates image builds cleanly)
```

- **Lint:** `ruff check` with project-wide `pyproject.toml` configuration
- **Tests:** Full pytest suite with SQLite (no external services required in CI)
- **Coverage:** Uploaded to Codecov on every run
- **Docker:** Image built and validated without pushing to registry

---

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development environment setup
- Code style guide (Ruff + Black + isort)
- Branch naming conventions (`feat/`, `fix/`, `docs/`)
- Pull request checklist

```bash
# Install dev dependencies
pip install -r requirements.txt

# Format and lint
ruff check src/ api/ dashboard/ --fix

# Run tests before submitting a PR
pytest tests/ -v --cov=src --cov=api
```

---

## License

This project is open source under the [MIT License](LICENSE).

---

<div align="center">

**Built to production ML engineering standards**

*Data Validation · Feature Engineering · SHAP Explainability · Drift Monitoring*  
*Business Rules · Async Processing · Audit Trails · Human Review · CI/CD*

<br/>

[![GitHub](https://img.shields.io/badge/GitHub-Parthkadian-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/Parthkadian)

</div>
