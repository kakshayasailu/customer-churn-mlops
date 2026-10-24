
import json
import sys

MINIMUM_F1 = 0.45

with open("metrics.json") as file:
    metrics = json.load(file)

f1 = metrics["f1_score"]

print("Customer Churn F1 Score:", round(f1, 4))
print("Required minimum F1:", MINIMUM_F1)

if f1 < MINIMUM_F1:
    print("QUALITY GATE FAILED")
    print("The model does not meet the required F1 score.")
    sys.exit(1)

print("QUALITY GATE PASSED")
sys.exit(0)
