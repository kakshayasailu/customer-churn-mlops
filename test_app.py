
import unittest

import pandas as pd
from app import app


class TestChurnAPI(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json()["status"], "ok"
        )

    def test_missing_fields_rejected(self):
        response = self.client.post(
            "/predict",
            json={"tenure": 12}
        )
        self.assertEqual(response.status_code, 400)

    def test_valid_customer_prediction(self):
        bundle = __import__("joblib").load(
            "customer_churn_model.pkl"
        )
        features = bundle["feature_columns"]

        data = pd.read_csv("customer_churn.csv")
        data.columns = data.columns.str.strip()

        customer = data[features].iloc[0].to_dict()

        # Convert missing values to JSON null.
        customer = {
            key: (
                None if pd.isna(value)
                else value.item()
                if hasattr(value, "item")
                else value
            )
            for key, value in customer.items()
        }

        response = self.client.post(
            "/predict", json=customer
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(
            response.get_json()["prediction"], ["Yes", "No"]
        )


if __name__ == "__main__":
    unittest.main()
