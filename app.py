
from pathlib import Path
import logging
import os
import time

import joblib
import pandas as pd
from flask import Flask, jsonify, request, g


# --------------------------------------------------
# 1. Flask Application Configuration
# --------------------------------------------------

app = Flask(__name__)

# Configure logging for Render and Docker
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True
)

logger = logging.getLogger("customer_churn_api")

MODEL_PATH = Path("customer_churn_model.pkl")


# --------------------------------------------------
# 2. Request Logging and Response-Time Tracking
# --------------------------------------------------

@app.before_request
def log_request_start():
    g.request_start_time = time.time()

    logger.info(
        "Request started | method=%s | path=%s",
        request.method,
        request.path
    )


@app.after_request
def log_request_end(response):
    start_time = getattr(g, "request_start_time", time.time())
    duration = time.time() - start_time

    logger.info(
        "Request completed | method=%s | path=%s | status=%s | duration=%.3fs",
        request.method,
        request.path,
        response.status_code,
        duration
    )

    return response


# --------------------------------------------------
# 3. Load Trained Customer Churn Model
# --------------------------------------------------

def load_bundle():
    if not MODEL_PATH.exists():
        logger.error("Model file not found: %s", MODEL_PATH)
        raise FileNotFoundError(
            "Model file not found. Train the model first."
        )

    logger.info("Loading customer churn model")
    return joblib.load(MODEL_PATH)


# --------------------------------------------------
# 4. Home Endpoint
# --------------------------------------------------

@app.get("/")
def home():
    logger.info("Home endpoint accessed")

    return jsonify({
        "status": "ok",
        "service": "customer-churn-prediction"
    })


# --------------------------------------------------
# 5. Health Check Endpoint
# --------------------------------------------------

@app.get("/health")
def health():
    logger.info("Health check requested")

    return jsonify({
        "status": "ok"
    })


# --------------------------------------------------
# 6. Customer Churn Prediction Endpoint
# --------------------------------------------------

@app.post("/predict")
def predict():
    logger.info("Customer churn prediction requested")

    payload = request.get_json(silent=True)

    # Validate request payload
    if not isinstance(payload, dict) or not payload:
        logger.warning(
            "Invalid prediction request: missing or invalid JSON"
        )

        return jsonify({
            "error": "A JSON customer record is required"
        }), 400

    try:
        # Load model and feature information
        bundle = load_bundle()

        features = bundle["feature_columns"]
        numeric_features = bundle["numeric_features"]

        # Validate required customer fields
        missing = [
            column for column in features
            if column not in payload
        ]

        if missing:
            logger.warning(
                "Prediction request missing %d required fields",
                len(missing)
            )

            return jsonify({
                "error": "Missing customer fields",
                "missing_fields": missing
            }), 400

        # Prepare customer data
        customer = pd.DataFrame(
            [{column: payload[column] for column in features}]
        )

        for column in numeric_features:
            customer[column] = pd.to_numeric(
                customer[column],
                errors="coerce"
            )

        # Generate prediction
        model = bundle["model"]
        prediction = int(model.predict(customer)[0])

        probability = None

        if hasattr(model, "predict_proba"):
            probability = float(
                model.predict_proba(customer)[0][1]
            )

        result = {
            "prediction": "Yes" if prediction == 1 else "No",
            "churn_probability": probability
        }

        logger.info("Customer churn prediction completed successfully")

        return jsonify(result)

    except (FileNotFoundError, KeyError, ValueError) as error:
        logger.exception("Prediction processing failed")

        return jsonify({
            "error": "Unable to process prediction request"
        }), 500

    except Exception:
        logger.exception("Unexpected error during prediction")

        return jsonify({
            "error": "Internal server error"
        }), 500


# --------------------------------------------------
# 7. Global Error Handling
# --------------------------------------------------

@app.errorhandler(404)
def not_found(error):
    logger.warning(
        "Endpoint not found | method=%s | path=%s",
        request.method,
        request.path
    )

    return jsonify({
        "error": "Endpoint not found"
    }), 404


@app.errorhandler(500)
def internal_server_error(error):
    logger.error("Internal server error occurred")

    return jsonify({
        "error": "Internal server error"
    }), 500


# --------------------------------------------------
# 8. Start Flask Application
# --------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))

    logger.info("Starting Customer Churn Prediction API")

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
