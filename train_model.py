
import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("customer_churn.csv")
TARGET = "Churn"


def train_model():
    print("Loading customer churn dataset...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "customer_churn.csv is missing from the repository root."
        )

    data = pd.read_csv(DATA_PATH)
    data.columns = data.columns.str.strip()

    if TARGET not in data.columns:
        raise ValueError(
            "Target column 'Churn' was not found. "
            f"Available columns: {list(data.columns)}"
        )

    # Remove rows without a target value.
    data = data.dropna(subset=[TARGET]).copy()

    # Standardize Yes/No labels.
    data[TARGET] = (
        data[TARGET].astype(str).str.strip().str.lower()
    )

    data[TARGET] = data[TARGET].map({
        "yes": 1,
        "no": 0
    })

    data = data.dropna(subset=[TARGET])
    data[TARGET] = data[TARGET].astype(int)

    # Customer ID identifies a person; it is not a useful
    # customer behavior feature for this baseline model.
    X = data.drop(
        columns=[TARGET, "customerID"],
        errors="ignore"
    ).copy()

    y = data[TARGET]

    if y.nunique() != 2:
        raise ValueError(
            "Churn must contain two classes: Yes and No."
        )

    # In the common Telco dataset, TotalCharges may contain
    # blank strings. Convert it to numeric when present.
    if "TotalCharges" in X.columns:
        X["TotalCharges"] = pd.to_numeric(
            X["TotalCharges"], errors="coerce"
        )

    # Identify numeric and categorical columns.
    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            strategy="most_frequent"
        )),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore"
        )),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline,
         categorical_features),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        )),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("Total records:", len(data))
    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))
    print("Training churn model...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "precision": float(
            precision_score(
                y_test, predictions, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_test, predictions, zero_division=0
            )
        ),
        "f1_score": float(
            f1_score(
                y_test, predictions, zero_division=0
            )
        ),
        "training_records": int(len(X_train)),
        "testing_records": int(len(X_test)),
    }

    # Save model and preprocessing information together.
    joblib.dump({
        "model": model,
        "feature_columns": X.columns.tolist(),
        "numeric_features": numeric_features,
    }, "customer_churn_model.pkl")

    with open("metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    print("Evaluation results:")
    print(json.dumps(metrics, indent=4))
    print("Model saved: customer_churn_model.pkl")
    print("Metrics saved: metrics.json")


if __name__ == "__main__":
    train_model()
