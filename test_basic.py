
import unittest


class TestBasicCI(unittest.TestCase):

    def test_python_is_working(self):
        self.assertEqual(1 + 1, 2)

    def test_customer_churn_project(self):
        self.assertEqual("Churn".lower(), "churn")


if __name__ == "__main__":
    unittest.main()
