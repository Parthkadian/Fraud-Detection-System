<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=28&pause=1000&color=C49A2E&center=true&vCenter=true&width=700&lines=%F0%9F%9B%A1%EF%B8%8F+Fraud+Detection+System;Production+ML+%7C+Real-Time+Scoring;SHAP+%7C+Drift+Monitor+%7C+Async+Pipeline" alt="Typing SVG" />

# 🛡️ Enterprise Fraud Detection System

**A production-grade, end-to-end ML platform for real-time credit card fraud detection — from raw transactions to ranked fraud insights with full explainability, drift monitoring, audit trails, and async processing.**

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-ML_Engine-FF6600?style=for-the-badge)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-7C3AED?style=for-the-badge)](https://shap.readthedocs.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-C49A2E?style=for-the-badge)](LICENSE)

<br/>

> *"Not just a model — a complete fraud intelligence platform."*

</div>

---

## 📋 Table of Contents

- [🎯 Overview](#-overview)
- [🏗️ System Architecture](#️-system-architecture)
- [✨ Features](#-features)
- [🧠 ML Pipeline](#-ml-pipeline)
- [🌐 API Reference](#-api-reference)
- [📊 Dashboard](#-dashboard)
- [⚙️ Tech Stack](#️-tech-stack)
- [📁 Project Structure](#-project-structure)
- [🚀 Quick Start](#-quick-start)
- [🐳 Docker Deployment](#-docker-deployment)
- [📈 Model Performance](#-model-performance)
- [🔍 SHAP Explainability](#-shap-explainability)
- [🌊 Drift Monitoring](#-drift-monitoring)
- [🧪 Testing](#-testing)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## 🎯 Overview

This project implements a **full-stack fraud detection platform** built around the [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (284,807 transactions, 492 fraudulent — 0.173% fraud rate). It demonstrates a **real-world ML engineering workflow** covering the complete lifecycle from raw data ingestion to production-ready serving with monitoring.

### What Makes This Different

Most fraud detection repos are just notebooks. This is a **complete platform**:

| Aspect | What's Here |
|--------|-------------|
| 🏭 **Production ML** | Full pipeline from ingestion → training → inference with config-driven architecture |
| 🧠 **Explainability** | SHAP TreeExplainer for every prediction — not a black box |
| 📋 **Business Rules** | YAML-driven rule engine layered on top of ML score |
| 🗂️ **Audit Trail** | Every prediction logged with full metadata and risk level |
| 🌊 **Drift Detection** | Evidently AI monitors feature distribution shifts in production |
| 🔁 **Async Processing** | Celery + Redis for non-blocking high-volume batch scoring |
| 👥 **Human Review** | Case management queue for analyst escalation workflows |
| 📡 **Observability** | Prometheus metrics, structured logging, health endpoints |

---

## 🏗️ System Architecture

```
╔══════════════════════════════════════════════════════════════════════════╗
║                    FRAUD DETECTION SYSTEM  v2.0                         ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  ┌─────────────┐   ┌─────────────┐   ┌──────────────┐   ┌───────────┐  ║
║  │   Data       │──▶│   Data      │──▶│   Feature    │──▶│  Model    │  ║
║  │  Ingestion   │   │  Cleaning   │   │  Engineering │   │ Training  │  ║
║  └─────────────┘   └─────────────┘   └──────────────┘   └───────────┘  ║
║        │                 │                  │                  │         ║
║     Schema           Outlier           NLP (TF-IDF         XGBoost +    ║
║   Validation         Detection         + SVD) +            Threshold    ║
║                                       Domain Feats         Optimisation ║
║                                                                 │        ║
║  ┌──────────────────────────────────────────────────────────────┘        ║
║  │                                                                        ║
║  │  ┌─────────────────────────────────────────────────────────────────┐  ║
║  │  │                      SERVING LAYER                               │  ║
║  │  ├─────────────────┬──────────────────┬────────────────────────────┤  ║
║  │  │   FastAPI        │   Celery +       │   Streamlit                │  ║
║  │  │   REST API v2    │   Redis          │   Dashboard                │  ║
║  │  │                  │   (Async)        │                            │  ║
║  │  │  POST /predict   │                  │  🎯 Real-time scoring      │  ║
║  │  │  POST /explain   │  Non-blocking    │  📁 Batch CSV upload       │  ║
║  │  │  GET  /audit     │  batch scoring   │  🌊 Drift monitor          │  ║
║  │  │  GET  /rules     │                  │  📊 Analytics & charts     │  ║
║  │  │  GET  /health    │                  │  🗂️ Audit trail            │  ║
║  │  │  GET  /metrics   │                  │  📋 Business rules         │  ║
║  │  └─────────────────┴──────────────────┴────────────────────────────┘  ║
║  │                                                                        ║
║  │  ┌─────────────────────────────────────────────────────────────────┐  ║
║  │  │                    MONITORING LAYER                              │  ║
║  │  │  Evidently Drift │ Prometheus Metrics │ SQLite Audit DB          │  ║
║  │  │  Business Rules  │ Structured Logging │ Alert Engine             │  ║
║  │  └─────────────────────────────────────────────────────────────────┘  ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## ✨ Features

### 🎯 Real-Time Single Transaction Scoring
- Submit any transaction with 30 features (Time, Amount, V1–V28) and receive:
  - **Fraud probability** (0.0 → 1.0)
  - **Risk level** (LOW / MEDIUM / HIGH)
  - **Business rule flags** (triggered rules if any)
  - **SHAP explanation** (top feature contributions)
- PCA feature tooltips and out-of-range ⚠️ warnings built in

### 📁 Batch CSV Scoring
- Upload a CSV with N transactions → scored + risk-classified in seconds
- Full analytics suite: prediction breakdown, risk distribution, ROC curve, Precision-Recall curve, confusion matrix, KDE probability, fraud time trends, feature importance proxy
- Downloadable results CSV

### 🌊 Data Drift Monitor
- Evidently AI HTML report generation comparing live data vs training reference
- Per-feature drift scores and dataset-level drift summary

### 🔁 Live Async Stream
- Celery + Redis backed async prediction endpoint
- Polling interface for non-blocking high-volume scoring

### 💬 NLP Insights
- Word cloud analysis of synthetic transaction memos
- TF-IDF feature extraction comparing fraud vs legitimate transaction language patterns

### 📊 Model Performance Tab
- Live ROC-AUC, F1, MCC, PR-AUC from `models/artifacts/metrics.json`
- All advanced evaluation charts rendered on-demand

### 🗂️ Audit Trail
- Every prediction stored in SQLite with full metadata
- Filterable by risk level, date, rule triggered
- Export to CSV

### 📋 Business Rules Engine
- YAML-defined rules override ML score (e.g. high-amount + night-time = escalate)
- View active rules, trigger counts, and last fired time

### 🪪 Model Card
- Google Model Card format served at `/model_card`
- Intended use, limitations, training data provenance, ethical considerations

### ⚡ System Health
- Real-time API health, model load status, Redis connectivity, database status
- Prometheus metrics at `/metrics`

---

## 🧠 ML Pipeline

### Data Flow

```
creditcard.csv
     │
     ▼
DataIngestion           ← Schema validation, synthetic NLP memos
     │
     ▼
DataCleaning            ← Outlier detection, missing value handling
     │
     ▼
FeatureEngineering      ← Log-amount, time features, z-scores,
     │                    PCA magnitude, TF-IDF + SVD (5 components)
     ▼
Splitter                ← Stratified 80/10/10 train/val/test split
     │
     ▼
ModelTraining           ← XGBoost, 5-Fold Stratified CV, early stopping
     │
     ▼
ThresholdOptimiser      ← Grid search over F1 (not default 0.5)
     │
     ▼
Evaluation              ← F1, ROC-AUC, PR-AUC, MCC, Cohen's Kappa
     │
     ▼
MLflow Tracking         ← All params, metrics, and model artifact logged
     │
     ▼
models/artifacts/       ← xgboost_model.json, scaler.pkl, metrics.json
```

### Model Configuration

```yaml
# Optimised for heavily imbalanced fraud detection
model: XGBoostClassifier
  max_depth: 3                 # Shallow trees prevent overfitting
  n_estimators: 200
  learning_rate: 0.05
  subsample: 0.8
  colsample_bytree: 0.8
  reg_alpha: 2.0               # L1 regularisation
  reg_lambda: 10.0             # L2 regularisation
  scale_pos_weight: ~577       # Dynamic class weight from ratio

cross_validation: StratifiedKFold(n_splits=5)
threshold: Optimised via F1 grid search on validation set
```

### Business Rules Layer

Rules are applied **after** the ML score and can escalate or suppress:

```yaml
rules:
  - name: "High Amount Night Transaction"
    condition: "Amount > 2000 AND 0 <= hour < 5"
    action: ESCALATE
  - name: "Rapid Velocity Burst"
    condition: "V20 > 2.5"
    action: FLAG
  - name: "Known Legitimate Merchant"
    condition: "merchant_id IN whitelist"
    action: SUPPRESS
```

---

## 🌐 API Reference

FastAPI v2.0 · Auto-generated Swagger docs at **`http://localhost:8000/docs`**

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API root + version info |
| `GET` | `/health` | Model status, DB, Redis connectivity |
| `GET` | `/metrics` | Prometheus metrics scrape endpoint |
| `GET` | `/model_card` | Google Model Card (JSON) |

### Prediction

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Score a single transaction (sync) |
| `POST` | `/predict_batch` | Score multiple transactions (sync) |
| `POST` | `/predict_compare` | Champion-challenger comparison |
| `POST` | `/predict_async` | Queue async prediction (Celery) |
| `GET` | `/task_status/{id}` | Poll async task result |

### Explainability

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/explain` | SHAP explanation for one transaction |
| `GET` | `/explain/global` | Global feature importance |
| `POST` | `/drift_report` | Evidently HTML drift report |

### Audit & Monitoring

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/audit/history` | Paginated prediction audit log |
| `GET` | `/rules` | Active business rules listing |
| `GET` | `/alerts/recent` | Recent fraud alerts |

### Case Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/cases` | Create a new fraud case |
| `GET` | `/cases` | List cases with filters |
| `PATCH` | `/cases/{id}` | Update case status |
| `GET` | `/review/queue` | Human review queue |
| `PATCH` | `/review/{case_id}/decision` | Submit analyst decision |

### Risk Profiles

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/customers/{id}/risk-profile` | Customer risk profile |
| `GET` | `/merchants/{id}/risk-profile` | Merchant risk profile |
| `GET` | `/velocity/{entity_type}/{value}` | Velocity signals |
| `POST` | `/blacklist` | Add to blacklist |
| `POST` | `/whitelist` | Add to whitelist |

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 10000, "Amount": 150.5,
    "V1": -1.2, "V2": 0.3, "V3": 1.1, "V4": 0.5,
    "V5": -0.2, "V6": 0.1, "V7": 0.2, "V8": -0.1,
    "V9": 0.4, "V10": -0.3, "V11": 0.2, "V12": -0.5,
    "V13": 0.1, "V14": -0.2, "V15": 0.3, "V16": -0.1,
    "V17": 0.2, "V18": 0.1, "V19": -0.3, "V20": 0.05,
    "V21": -0.02, "V22": 0.1, "V23": -0.03, "V24": 0.2,
    "V25": -0.1, "V26": 0.05, "V27": 0.02, "V28": -0.01
  }'
```

### Example Response

```json
{
  "fraud_probability": 0.0023,
  "prediction": 0,
  "risk_level": "LOW",
  "rule_triggered": false,
  "model_version": "2.0.0",
  "latency_ms": 4.2
}
```

---

## 📊 Dashboard

The Streamlit dashboard (`dashboard/app.py` — 3,200+ lines) is a **premium dark-mode fraud intelligence workspace** with:

### Navigation Tabs

| Tab | What You Get |
|-----|--------------|
| 🎯 **Single Transaction Scoring** | 30-field transaction form with PCA tooltips, ⚠️ outlier warnings, preset examples (low-risk grocery, high-risk CNP, edge case). Submit → fraud probability gauge, risk badge, SHAP waterfall |
| 📁 **Batch CSV Scoring** | Drag-drop CSV upload → full batch scoring → 8+ analytics charts (ROC, PR-AUC, confusion matrix, KDE, fraud trends, feature importance proxy) + CSV download |
| 🌊 **Data Drift Monitor** | Evidently HTML report for any uploaded reference dataset |
| 🔁 **Live Async Stream** | Real-time Celery task submission + polling |
| 💬 **NLP Insights** | Word cloud: fraud vs legitimate memo patterns |
| 📊 **Model Performance** | Live metrics + all evaluation charts |
| 🗂️ **Audit Trail** | Searchable, filterable prediction history |
| 📋 **Business Rules** | Active rules table with trigger counts |
| 🪪 **Model Card** | Full model card in readable format |
| ⚡ **System Health** | API, model, Redis, DB status |

### UI Highlights

- **Dark/Light theme toggle** — persistent across sessions
- **Glassmorphism card design** with premium gold (`#C49A2E`) + purple (`#A876BE`) accent palette
- **Sidebar metrics** — live session KPIs (txns scored, fraud flagged, fraud rate, rules fired, avg latency)
- **Flagging system** — sidebar "Flag for review" on any recent prediction
- **Status strip** — real-time API health pill with last-checked timestamp
- **Sticky refresh bar** — one-click API status refresh

### Running the Dashboard

```bash
# Activate virtual environment first
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux/Mac

# Start the API (port 8000)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Launch the dashboard (port 8501, new terminal)
streamlit run dashboard/app.py
```

Open **http://localhost:8501** in your browser.

---

## ⚙️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **ML Core** | XGBoost 2.0.3, scikit-learn | Model training & inference |
| **NLP** | TF-IDF + Truncated SVD | Transaction memo feature extraction |
| **Explainability** | SHAP TreeExplainer | Per-prediction feature attribution |
| **API** | FastAPI 2.0, Pydantic, Uvicorn | REST microservice |
| **Dashboard** | Streamlit, Matplotlib | Interactive fraud analytics UI |
| **Async** | Celery, Redis | Non-blocking batch predictions |
| **Database** | SQLite (dev), PostgreSQL (prod) | Audit trail + platform data |
| **Monitoring** | Evidently AI, Prometheus | Drift detection + metrics |
| **Experiment Tracking** | MLflow | Params, metrics, model registry |
| **Infrastructure** | Docker, Docker Compose | Containerised multi-service setup |
| **CI/CD** | GitHub Actions | Lint, test, Docker build |
| **Deployment** | Railway (API), configurable | Cloud-ready |

---

## 📁 Project Structure

```
fraud-detection-system/
│
├── 📂 api/                          # FastAPI microservice (v2.0)
│   ├── main.py                      # App factory, lifespan, middleware
│   ├── schemas.py                   # 40+ Pydantic request/response models
│   ├── middleware.py                # Request ID, timing, rate limiting
│   ├── database.py                  # SQLite/PostgreSQL ORM + audit/case/alert tables
│   ├── celery_worker.py             # Async Celery task definitions
│   ├── router.py                    # Route aggregation
│   ├── requirements.txt             # API-specific dependencies
│   ├── dockerfile                   # API service Dockerfile
│   └── routes/                      # 10 modular route files
│       ├── prediction.py            # /predict, /predict_batch, /predict_async
│       ├── explainability.py        # /explain, /explain/global
│       ├── monitoring.py            # /health, /metrics, /audit, /rules, /drift
│       ├── cases.py                 # /cases/* CRUD
│       ├── review.py                # /review/* human queue
│       ├── lists.py                 # /blacklist, /whitelist
│       ├── profiles.py              # /customers/*, /merchants/*, /velocity/*
│       ├── alerts.py                # /alerts/*
│       ├── business.py              # /dashboard/summary, /business/*
│       └── model_ops.py             # /model/version, /model/performance
│
├── 📂 dashboard/
│   ├── app.py                       # Streamlit dashboard (3,200+ lines)
│   ├── dockerfile                   # Dashboard service Dockerfile
│   └── requirements.txt             # Dashboard-specific dependencies
│
├── 📂 docs/
│   └── model_card.json              # Google Model Card (machine-readable)
│
├── 📂 src/                          # Core ML + business logic
│   ├── ingestion/
│   │   └── data_ingestion.py        # Dataset loading + NLP memo synthesis
│   ├── preprocessing/
│   │   ├── data_validator.py        # Schema, type, range validation
│   │   ├── data_cleaning.py         # Cleaning + outlier detection
│   │   └── splitter.py              # Stratified 80/10/10 split
│   ├── features/
│   │   └── feature_engineering.py  # Tabular + NLP feature pipeline
│   ├── training/
│   │   ├── train.py                 # CV training loop + early stopping
│   │   ├── model_factory.py         # Config-driven model instantiation
│   │   └── thresholding.py          # F1-optimised threshold search
│   ├── evaluation/
│   │   └── evaluate.py              # F1, MCC, Kappa, ROC-AUC, PR-AUC
│   ├── inference/
│   │   ├── predict.py               # Single-transaction predictor
│   │   └── batch_predict.py         # DataFrame batch predictor
│   ├── explainability/
│   │   └── shap_explainer.py        # SHAP TreeExplainer integration
│   ├── monitoring/
│   │   ├── drift_detection.py       # Evidently AI drift reports
│   │   ├── alerting.py              # Threshold-based alert engine
│   │   └── logger.py                # Structured logging with rotation
│   ├── rules/
│   │   └── rule_engine.py           # YAML-driven business rule engine
│   └── utils/
│       ├── common.py                # I/O helpers (joblib, JSON)
│       ├── config_loader.py         # YAML config loader
│       └── exceptions.py            # Custom exception hierarchy
│
├── 📂 configs/
│   ├── config.yaml                  # Project-wide configuration
│   ├── model_params.yaml            # XGBoost hyperparameters
│   └── business_rules.yaml          # YAML-defined business rules
│
├── 📂 tests/                        # Comprehensive test suite
│   ├── conftest.py                  # Shared fixtures
│   ├── test_api.py                  # API endpoint tests
│   ├── test_data_ingestion.py
│   ├── test_preprocessing.py
│   ├── test_training.py
│   ├── test_evaluation.py
│   ├── test_inference.py
│   └── test_transformer.py          # Feature transformer pipeline tests
│
├── 📂 notebooks/
│   └── generate_eda.py              # EDA visualisation generator
│
├── 📂 data/
│   ├── raw/                         # creditcard.csv (not tracked in git)
│   ├── fraud_platform.db            # SQLite platform database
│   ├── audit_log.db                 # Prediction audit log
│   └── schemas/
│       └── data_schema.yaml         # Formal column schema definition
│
├── 📂 models/
│   ├── artifacts/                   # xgboost_model.json, scaler.pkl, metrics.json
│   └── trained/                     # Versioned trained model checkpoints
│
├── 📂 reports/figures/              # EDA visualisation outputs
├── 📂 logs/                         # Structured application logs
├── 📂 .github/workflows/            # CI/CD pipeline (lint + test + Docker)
│
├── docker-compose.yml               # Full stack: API + PostgreSQL + Redis + Celery + Frontend
├── Dockerfile                       # Production API container
├── requirements.txt                 # Python dependencies
├── run_pipeline.py                  # One-command pipeline runner (CLI)
├── test_predict.py                  # Standalone prediction smoke test
├── push_to_github.py                # Git automation helper script
├── railway.toml                     # Railway.app deployment config
├── CONTRIBUTING.md                  # Developer contribution guide
└── README.md                        # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Python **3.10+**
- `pip` or your preferred package manager
- (Optional) Docker & Docker Compose for containerised run
- (Optional) Redis for async Celery features

### 1. Clone & Install

```bash
git clone https://github.com/Parthkadian/Fraud-Detection-System.git
cd Fraud-Detection-System

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux/Mac

# Install all dependencies
pip install -r requirements.txt
```

### 2. Get the Dataset

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it at:

```
data/raw/creditcard.csv
```

### 3. Train the Model

```bash
# Default — runs full pipeline with XGBoost
python run_pipeline.py

# Specify a different model
python run_pipeline.py --model logistic_regression
python run_pipeline.py --model random_forest

# Skip cross-validation for faster iteration
python run_pipeline.py --skip-cv
```

This runs the full pipeline:
- Data validation → cleaning → feature engineering
- Stratified 5-Fold CV training with XGBoost
- Threshold optimisation
- MLflow experiment logging
- Saves model + scaler + metrics to `models/artifacts/`

### 4. Start the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

✅ API available at: **http://localhost:8000**
📄 Swagger docs at: **http://localhost:8000/docs**

### 5. Launch the Dashboard

```bash
# In a new terminal (with venv activated)
streamlit run dashboard/app.py
```

🖥️ Dashboard at: **http://localhost:8501**

---

## 🐳 Docker Deployment

### Full Stack (API + PostgreSQL + Redis + Celery + Frontend)

```bash
docker-compose up --build
```

This launches:

| Service | Port | Description |
|---------|------|-------------|
| **API** | `8000` | FastAPI REST service |
| **Frontend** | `3000` | Next.js analytics UI |
| **PostgreSQL** | `5432` | Production database |
| **Redis** | `6379` | Celery message broker |
| **Celery Worker** | — | Async prediction worker |

> [!NOTE]
> The **Streamlit dashboard** (`dashboard/app.py`) is designed for local development and is run separately (see [Quick Start](#-quick-start)). The Docker Compose stack uses a **Next.js frontend** (`frontend/`) for the containerised deployment.

### Environment Variables

```bash
# API
FRAUD_API_KEY=your-secret-key           # Enable API authentication (production)
API_BASE_URL=http://localhost:8000       # Dashboard → API URL
RATE_LIMIT_PER_MINUTE=120               # API rate limiting

# Database
DATABASE_URL=postgresql://fraud:fraud@postgres:5432/fraud_db   # Production PostgreSQL
# Omit DATABASE_URL to use SQLite for local development

# Redis / Celery
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/0

# Frontend
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## 📈 Model Performance

> [!NOTE]
> **About the dataset:** The [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) uses **pre-anonymised PCA features** (V1–V28). Because the raw features have already been transformed by the card network's internal PCA pipeline, tree-based models achieve near-perfect separation on this specific dataset. This is a [well-documented characteristic](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/discussion) — not data leakage.
>
> **The value of this project is the engineering, not the metric number.** Real-world fraud detection capability is demonstrated through: threshold optimisation, SHAP explainability, drift monitoring, business rules, audit trails, and human review workflows.

### Dataset

| Property | Value |
|----------|-------|
| Total transactions | 284,807 |
| Fraudulent | 492 (0.173%) |
| Class imbalance ratio | 1 : 577 |
| Feature set | 28 PCA components (V1–V28) + Time + Amount |
| Time window | 48-hour capture |

### Benchmark Results (PCA Dataset)

| Metric | Validation | Test (Held-out) | Notes |
|--------|-----------|-----------------|-------|
| **F1-Score** | ~0.99 | ~0.99 | Near-perfect on this dataset |
| **ROC-AUC** | ~0.99 | ~0.99 | PCA features are highly separable |
| **PR-AUC** | ~0.85 | ~0.85 | Most meaningful for imbalanced data |
| **MCC** | ~0.95 | ~0.95 | Best single metric for fraud |

### What Actually Matters for Production

| Capability | Implementation |
|-----------|---------------|
| **Threshold Optimisation** | Grid search over F1 on validation — not default 0.5 |
| **SHAP Explainability** | Per-transaction feature attribution for auditability |
| **Drift Monitoring** | Evidently AI detects when production data diverges |
| **Human Review Queue** | Analyst escalation with full decision audit trail |
| **Business Rules** | YAML-defined override rules on top of ML score |
| **Async Processing** | Celery + Redis for non-blocking bulk scoring |

---

## 🔍 SHAP Explainability

Every prediction comes with a SHAP explanation explaining **why** the model assigned that fraud probability.

```python
# Example SHAP response
{
  "transaction_id": "txn_abc123",
  "fraud_probability": 0.847,
  "risk_level": "HIGH",
  "explanation": [
    {"feature": "V14", "shap_value": -0.412, "description": "High-value transaction signal"},
    {"feature": "V17", "shap_value": -0.318, "description": "Session behaviour anomaly"},
    {"feature": "Amount", "shap_value": 0.229, "description": "Transaction amount £4,980"},
    {"feature": "V12", "shap_value": -0.198, "description": "Cross-border indicator"},
    {"feature": "V4",  "shap_value": 0.142, "description": "Merchant category risk score"}
  ]
}
```

The dashboard renders this as a **ranked horizontal bar chart** — analysts see exactly which features drove the fraud score, making the system fully auditable.

---

## 🌊 Drift Monitoring

The system uses **Evidently AI** to monitor when production transaction data diverges from training data distribution:

```bash
# Generate a drift report via API
curl -X POST http://localhost:8000/drift_report \
  -H "Content-Type: application/json" \
  -d '{"current_data": [...], "reference_data": [...]}'
```

Or upload a CSV in the **🌊 Data Drift Monitor** dashboard tab to get a full HTML Evidently report with:
- Dataset-level drift score
- Per-feature drift metrics (Wasserstein distance, chi-square)
- Distribution comparison plots

---

## 🧪 Testing

```bash
# Run full test suite
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=src --cov=api --cov-report=term-missing

# Run specific module
pytest tests/test_api.py -v
pytest tests/test_inference.py -v
```

### Coverage Areas

| Module | Tests |
|--------|-------|
| `test_api.py` | All 30+ API endpoints, middleware, validation |
| `test_data_ingestion.py` | Loading, NLP synthesis, schema edge cases |
| `test_preprocessing.py` | Cleaning, validation, outlier detection |
| `test_training.py` | CV training, early stopping, model save/load |
| `test_evaluation.py` | All metrics, serialisation, edge cases |
| `test_inference.py` | Risk level logic, SHAP output format, batch |
| `test_transformer.py` | Feature engineering transformer pipeline |

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development environment setup
- Code style guide (Black + isort + flake8)
- Branch naming conventions
- Pull request checklist

```bash
# Install dev dependencies
pip install -r requirements.txt

# Format code
black . && isort .

# Lint (ruff is the primary linter; flake8 also supported)
ruff check src/ api/ dashboard/
flake8 src/ api/ dashboard/ --max-line-length 100

# Run tests before PR
pytest tests/ -v
```

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).

---

<div align="center">

**Built with a focus on production-grade ML engineering practices**

*Data Validation · Feature Engineering · SHAP Explainability · Drift Monitoring · Business Rules · Async Processing · Audit Trails · Human Review*

<br/>

⭐ **If this project helped you, please consider starring it!** ⭐

<br/>

[![GitHub](https://img.shields.io/badge/GitHub-Parthkadian-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Parthkadian)

</div>
