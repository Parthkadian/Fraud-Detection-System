import os
from celery import Celery
from src.inference.predict import FraudPredictor

REDIS_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "fraud_worker",
    broker=REDIS_URL,
    backend=REDIS_URL
)

# Lazy load model in worker
predictor = None

@celery_app.task(name="api.celery_worker.predict_transaction")
def predict_transaction(transaction_dict: dict):
    global predictor
    if predictor is None:
        try:
            predictor = FraudPredictor()
        except Exception as e:
            return {"error": f"Failed to load model in worker: {str(e)}"}
            
    try:
        result = predictor.predict(transaction_dict)
        return result
    except Exception as e:
        return {"error": str(e)}
