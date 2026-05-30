from src.inference.predict import FraudPredictor

predictor = FraudPredictor()

sample = {
    "Time": 10000,
    "Amount": 150.5,
    "V1": -1.2,
    "V2": 0.3,
    "V3": 1.1,
    "V4": 0.5,
    "V5": -0.2,
    "V6": 0.1,
    "V7": 0.2,
    "V8": -0.1,
    "V9": 0.4,
    "V10": -0.3,
    "V11": 0.2,
    "V12": -0.5,
    "V13": 0.1,
    "V14": -0.2,
    "V15": 0.3,
    "V16": -0.1,
    "V17": 0.2,
    "V18": 0.1,
    "V19": -0.3,
    "V20": 0.05,
    "V21": -0.02,
    "V22": 0.1,
    "V23": -0.03,
    "V24": 0.2,
    "V25": -0.1,
    "V26": 0.05,
    "V27": 0.02,
    "V28": -0.01
}

result = predictor.predict(sample)
print(result)