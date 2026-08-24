"""Train and persist the housing-price prediction pipeline.

This module is used both by the command line training workflow and by the
Streamlit app when a model artifact is not already present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - requirements.txt includes xgboost
    XGBRegressor = None


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "housing.csv"
MODEL_PATH = BASE_DIR / "model" / "house_price_model.pkl"
TARGET_COLUMN = "median_house_value"
NUMERIC_FEATURES = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]
CATEGORICAL_FEATURES = ["ocean_proximity"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_pipeline(regressor: Any) -> Pipeline:
    """Build the exact preprocessing-plus-regressor pipeline used for inference."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )


def make_models() -> dict[str, Any]:
    """Return the candidate regressors used by the original notebook."""
    models: dict[str, Any] = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),
    }
    if XGBRegressor is not None:
        models["XGBoost"] = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            objective="reg:squarederror",
        )
    return models


def train_and_save_model(
    data_path: Path = DATA_PATH,
    model_path: Path = MODEL_PATH,
) -> dict[str, Any]:
    """Train, evaluate, and persist the best complete preprocessing pipeline."""
    frame = pd.read_csv(data_path)
    required_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing_columns = sorted(required_columns.difference(frame.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing_columns)}")

    features = frame[FEATURE_COLUMNS]
    target = frame[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
    )

    results: dict[str, dict[str, float]] = {}
    fitted_models: dict[str, Pipeline] = {}
    for name, regressor in make_models().items():
        pipeline = build_pipeline(regressor)
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        results[name] = {
            "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
            "r2": float(r2_score(y_test, predictions)),
        }
        fitted_models[name] = pipeline

    best_model_name = min(results, key=lambda name: results[name]["rmse"])
    bundle = {
        "model": fitted_models[best_model_name],
        "model_name": best_model_name,
        "metrics": results,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    return bundle


if __name__ == "__main__":
    trained_bundle = train_and_save_model()
    print(f"Saved {trained_bundle['model_name']} to {MODEL_PATH}")
    for name, metrics in trained_bundle["metrics"].items():
        print(f"{name}: RMSE={metrics['rmse']:.2f}, R²={metrics['r2']:.3f}")
