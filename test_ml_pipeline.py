
import json
import unittest
from pathlib import Path

import joblib
import pandas as pd


class TestChurnPipeline(unittest.TestCase):

    def test_dataset_exists(self):
        self.assertTrue(
            Path("customer_churn.csv").exists()
        )

    def test_model_exists(self):
        self.assertTrue(
            Path("customer_churn_model.pkl").exists()
        )

    def test_metrics_exist(self):
        self.assertTrue(Path("metrics.json").exists())

    def test_metrics_are_valid(self):
        with open("metrics.json") as file:
            metrics = json.load(file)

        for key in [
            "accuracy", "precision", "recall", "f1_score"
        ]:
            self.assertIn(key, metrics)
            self.assertGreaterEqual(metrics[key], 0.0)
            self.assertLessEqual(metrics[key], 1.0)

    def test_model_predicts_churn(self):
        bundle = joblib.load("customer_churn_model.pkl")
        model = bundle["model"]
        features = bundle["feature_columns"]

        data = pd.read_csv("customer_churn.csv")
        data.columns = data.columns.str.strip()

        if "TotalCharges" in data.columns:
            data["TotalCharges"] = pd.to_numeric(
                data["TotalCharges"], errors="coerce"
            )

        sample = data[features].head(5).copy()
        predictions = model.predict(sample)

        self.assertEqual(len(predictions), len(sample))
        self.assertTrue(
            set(predictions).issubset({0, 1})
        )


if __name__ == "__main__":
    unittest.main()
