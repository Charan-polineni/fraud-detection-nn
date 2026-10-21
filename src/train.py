"""Train a neural network to detect fraudulent transactions."""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

df = pd.read_csv("data/transactions.csv")
X = df.drop(columns="is_fraud")
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Fraud is rare, so oversample fraud cases in the TRAINING set only.
train = pd.concat([X_train, y_train], axis=1)
fraud = train[train.is_fraud == 1]
legit = train[train.is_fraud == 0]
fraud_up = resample(fraud, replace=True, n_samples=len(legit) // 4, random_state=42)
train_bal = pd.concat([legit, fraud_up]).sample(frac=1, random_state=42)

model = make_pipeline(
    StandardScaler(),
    MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200,
                  early_stopping=True, random_state=42),
)
model.fit(train_bal.drop(columns="is_fraud"), train_bal["is_fraud"])

preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, preds, digits=3))
print(f"ROC-AUC: {roc_auc_score(y_test, probs):.3f}")

ConfusionMatrixDisplay.from_predictions(y_test, preds)
plt.title("Fraud detection - confusion matrix")
plt.savefig("results/confusion_matrix.png", dpi=150, bbox_inches="tight")

joblib.dump(model, "models/fraud_model.joblib")
print("Model saved to models/fraud_model.joblib")
