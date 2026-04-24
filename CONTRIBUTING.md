# Contributing to Fraud Detection System

Thank you for your interest in contributing! Here's how to get started.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/your-username/fraud-detection-system.git
cd fraud-detection-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
src/
├── ingestion/       # Data loading and synthetic NLP generation
├── preprocessing/   # Cleaning, validation, splitting
├── features/        # Feature engineering (tabular + NLP)
├── training/        # Model training, factory, thresholding
├── evaluation/      # Metrics computation
├── inference/       # Single and batch prediction
├── explainability/  # SHAP explanations
├── monitoring/      # Drift detection, alerting, logging
└── utils/           # Common helpers, config loader, exceptions
```

## Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src --cov=api --cov-report=term-missing
```

## Code Style

- Follow PEP 8 conventions
- Use type hints for function signatures
- Add docstrings to all public classes and functions
- Use `ruff` for linting: `ruff check src/ api/ tests/`

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with clear commit messages
3. Ensure all tests pass locally
4. Update documentation if needed
5. Submit a PR with a description of your changes

## Architecture Decisions

- **XGBoost** is the primary model (configurable via YAML)
- **SMOTE** is not used — we rely on `scale_pos_weight` for class imbalance
- **TF-IDF + SVD** provides NLP features from transaction memos
- **Evidently** handles data drift monitoring
- **MLflow** tracks experiment metrics

## Reporting Issues

Please include:
- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behaviour
