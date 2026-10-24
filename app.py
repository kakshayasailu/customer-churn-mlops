
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request


app = Flask(__name__)

MODEL_PATH = Path("customer_churn_model.pkl")


def load_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file not found. Train the model first."
        )
    return joblib.load(MODEL_PATH)


@app.get("/")
def home():
    return jsonify({
        "status": "ok",
        "service": "customer-churn-prediction"
    })


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict) or not payload:
        return jsonify({
            "error": "A JSON customer record is required"
        }), 400

    bundle = load_bundle()
    features = bundle["feature_columns"]
    numeric_features = bundle["numeric_features"]

    missing = [
        column for column in features
        if column not in payload
    ]

    if missing:
        return jsonify({
            "error": "Missing customer fields",
            "missing_fields": missing
        }), 400

    customer = pd.DataFrame(
        [{column: payload[column] for column in features}]
    )

    for column in numeric_features:
        customer[column] = pd.to_numeric(
            customer[column], errors="coerce"
        )

    model = bundle["model"]
    prediction = int(model.predict(customer)[0])

    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(
            model.predict_proba(customer)[0][1]
        )

    return jsonify({
        "prediction": "Yes" if prediction == 1 else "No",
        "churn_probability": probability
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
