from fastapi import FastAPI
from api.schemas import TransactionInput
from src.inference.predict import FraudPredictor
from src.inference.batch_predict import BatchPredictor
from src.explainability.shap_explainer import ShapExplainer
import pandas as pd
import logging

# =========================
# LOGGING (for Render logs)
# =========================
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud risk scoring API",
    version="1.0.0"
)

# =========================
# SAFE MODEL LOADING
# =========================
try:
    predictor = FraudPredictor()
    batch_predictor = BatchPredictor()
    shap_explainer = ShapExplainer()

    MODEL_LOADED = True
    MODEL_ERROR = None

    logging.info("✅ Models loaded successfully")

except Exception as e:
    predictor = None
    batch_predictor = None
    shap_explainer = None

    MODEL_LOADED = False
    MODEL_ERROR = str(e)

    logging.error(f"❌ Model loading failed: {e}")


# =========================
# STARTUP EVENT
# =========================
@app.on_event("startup")
def startup_event():
    logging.info("🚀 Fraud Detection API Started")


# =========================
# ROUTES
# =========================
@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}


@app.get("/health")
def health():
    return {
        "status": "healthy" if MODEL_LOADED else "unhealthy",
        "model_loaded": MODEL_LOADED,
        "error": MODEL_ERROR
    }


@app.post("/predict")
def predict(transaction: TransactionInput):
    if not MODEL_LOADED:
        return {
            "error": "Model not loaded",
            "details": MODEL_ERROR
        }

    try:
        result = predictor.predict(transaction.model_dump())
        return result
    except Exception as e:
        return {"error": str(e)}


@app.post("/predict_batch")
def predict_batch(transactions: list[dict]):
    if not MODEL_LOADED:
        return {
            "error": "Model not loaded",
            "details": MODEL_ERROR
        }

    try:
        df = pd.DataFrame(transactions)
        result_df = batch_predictor.predict_dataframe(df)
        return result_df.to_dict(orient="records")
    except Exception as e:
        return {"error": str(e)}


@app.post("/explain")
def explain(transaction: TransactionInput):
    if not MODEL_LOADED:
        return {
            "error": "Model not loaded",
            "details": MODEL_ERROR
        }

    try:
        explanation = shap_explainer.explain_single(
            transaction.model_dump(),
            top_n=10
        )
        return explanation
    except Exception as e:
        return {"error": str(e)}