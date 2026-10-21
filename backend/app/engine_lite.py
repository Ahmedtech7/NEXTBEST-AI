
from pathlib import Path
from functools import lru_cache
import json

FEATURES = [
    "progress", "inactive", "sessions", "errors",
    "stage_code", "product_code", "previous_contact"
]
ACTIONS = ["none", "reminder", "assistance", "incentive"]
COSTS = [0, .15, 2.50, 8.00]
STAGES = [
    "Personal details", "Document upload",
    "Income verification", "Employment information",
    "Final review"
]
PRODUCTS = ["Personal loan", "Credit card", "Auto finance"]

@lru_cache(maxsize=1)
def load():
    path = Path(__file__).resolve().parents[1] / "artifacts/model-lite.json"
    return json.loads(path.read_text())

def customer_rows(artifact, count=120):
    return artifact["customers"][:count]

def predict(artifact, features):
    if isinstance(features[0], (int, float)):
        features = [features]

    results = []
    for x in features:
        scores = []

        for model in artifact["models"]:
            value = model["baseline"]

            for tree in model["trees"]:
                pos = 0

                while True:
                    feature, threshold, left, right, leaf, missing_left = tree[pos]

                    if left == 0 and right == 0:
                        value += leaf
                        break

                    v = x[feature]
                    pos = left if (
                        (v is None and missing_left)
                        or (v is not None and v <= threshold)
                    ) else right

            scores.append(max(.01, min(.99, value)))

        results.append(scores)

    return results
