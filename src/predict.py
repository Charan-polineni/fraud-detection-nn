"""Score a single example transaction with the saved model."""
import joblib
import pandas as pd

model = joblib.load("models/fraud_model.joblib")

transaction = pd.DataFrame([{
    "amount": 950.0,
    "hour": 3,
    "distance_from_home_km": 400.0,
    "transactions_last_24h": 9,
    "is_online": 1,
}])

score = model.predict_proba(transaction)[0, 1]
print(f"Fraud probability: {score:.1%}")
print("FLAG FOR REVIEW" if score > 0.5 else "Looks normal")
