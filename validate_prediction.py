"""Quick local validation for the persisted housing prediction bundle."""

from pathlib import Path

import joblib
import pandas as pd

from train_model import FEATURE_COLUMNS, MODEL_PATH

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
assert bundle["feature_columns"] == FEATURE_COLUMNS

base = {
    "longitude": -122.23,
    "latitude": 37.88,
    "housing_median_age": 41,
    "total_rooms": 880,
    "total_bedrooms": 129,
    "population": 322,
    "households": 126,
    "median_income": 8.3252,
    "ocean_proximity": "NEAR BAY",
}
wealthier = {**base, "median_income": 12.0}
inputs = pd.DataFrame([base, wealthier], columns=FEATURE_COLUMNS)
predictions = model.predict(inputs)
assert len(predictions) == 2
assert predictions[0] != predictions[1]
assert all(float(value) > 0 for value in predictions)

print(f"Artifact: {Path(MODEL_PATH).relative_to(Path.cwd())}")
print(f"Model: {bundle['model_name']}")
print(f"Predictions: {[round(float(value), 2) for value in predictions]}")
print("Validation passed: predictions come from the fitted pipeline and respond to changed inputs.")
