"""Streamlit interface for the trained housing-price prediction pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st

from train_model import DATA_PATH, FEATURE_COLUMNS, MODEL_PATH, train_and_save_model


st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏡",
    layout="wide",
)

OCEAN_PROXIMITIES = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
PRESETS: dict[str, dict[str, Any]] = {
    "Custom": {
        "longitude": -122.23,
        "latitude": 37.88,
        "housing_median_age": 41,
        "total_rooms": 880,
        "total_bedrooms": 129,
        "population": 322,
        "households": 126,
        "median_income": 8.3252,
        "ocean_proximity": "NEAR BAY",
    },
    "Affordable Home": {
        "longitude": -122.23,
        "latitude": 37.88,
        "housing_median_age": 30,
        "total_rooms": 1200,
        "total_bedrooms": 220,
        "population": 500,
        "households": 220,
        "median_income": 4.2,
        "ocean_proximity": "INLAND",
    },
    "Large Family Home": {
        "longitude": -121.95,
        "latitude": 37.75,
        "housing_median_age": 45,
        "total_rooms": 2600,
        "total_bedrooms": 500,
        "population": 900,
        "households": 400,
        "median_income": 6.8,
        "ocean_proximity": "<1H OCEAN",
    },
    "Coastal Property": {
        "longitude": -122.40,
        "latitude": 37.95,
        "housing_median_age": 35,
        "total_rooms": 1800,
        "total_bedrooms": 300,
        "population": 650,
        "households": 280,
        "median_income": 7.5,
        "ocean_proximity": "NEAR BAY",
    },
}


@st.cache_resource(show_spinner="Loading the trained model…")
def load_model_bundle() -> dict[str, Any]:
    """Load the persisted bundle, training it once when no artifact is present."""
    if MODEL_PATH.exists():
        artifact = joblib.load(MODEL_PATH)
        if isinstance(artifact, dict) and "model" in artifact:
            return artifact
        # Keep compatibility with a legacy artifact containing only a pipeline.
        return {
            "model": artifact,
            "model_name": "Saved model",
            "metrics": {},
            "feature_columns": FEATURE_COLUMNS,
        }

    # Streamlit Cloud receives the source repository but may not receive a local
    # ignored model artifact. Train from the checked-in real dataset in that case.
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training dataset not found at {DATA_PATH}")
    return train_and_save_model(DATA_PATH, MODEL_PATH)


try:
    bundle = load_model_bundle()
    model = bundle["model"]
except Exception as exc:
    st.error(f"The prediction model could not be loaded or trained: {exc}")
    st.stop()

st.title("🏡 California House Price Predictor")
st.write(
    "Enter district characteristics to get an estimate from the trained "
    "machine-learning pipeline. The preprocessing steps are stored with the model."
)

with st.sidebar:
    st.header("About this model")
    st.success(f"Model in use: {bundle.get('model_name', 'Saved model')}")
    st.caption(
        "The model is trained from the repository's California housing dataset "
        "and selected by lowest holdout RMSE."
    )
    metrics = bundle.get("metrics", {})
    if metrics:
        st.subheader("Validation results")
        metrics_frame = (
            pd.DataFrame(metrics)
            .T.rename(columns={"rmse": "RMSE", "r2": "R²"})
            .sort_values("RMSE")
        )
        st.dataframe(metrics_frame.style.format({"RMSE": "${:,.0f}", "R²": "{:.3f}"}))

preset_name = st.selectbox("Choose a sample scenario", list(PRESETS))
defaults = PRESETS[preset_name]

with st.form("prediction_form"):
    st.subheader("District characteristics")
    col1, col2 = st.columns(2)

    with col1:
        longitude = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=float(defaults["longitude"]),
            step=0.01,
            format="%.4f",
        )
        latitude = st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=float(defaults["latitude"]),
            step=0.01,
            format="%.4f",
        )
        housing_median_age = st.number_input(
            "Housing median age",
            min_value=0,
            max_value=200,
            value=int(defaults["housing_median_age"]),
            step=1,
        )
        total_rooms = st.number_input(
            "Total rooms",
            min_value=1,
            value=int(defaults["total_rooms"]),
            step=1,
        )
        total_bedrooms = st.number_input(
            "Total bedrooms",
            min_value=1,
            value=int(defaults["total_bedrooms"]),
            step=1,
        )

    with col2:
        population = st.number_input(
            "Population",
            min_value=1,
            value=int(defaults["population"]),
            step=1,
        )
        households = st.number_input(
            "Households",
            min_value=1,
            value=int(defaults["households"]),
            step=1,
        )
        median_income = st.number_input(
            "Median income (in tens of thousands USD)",
            min_value=0.0,
            value=float(defaults["median_income"]),
            step=0.1,
            format="%.4f",
        )
        ocean_proximity = st.selectbox(
            "Ocean proximity",
            OCEAN_PROXIMITIES,
            index=OCEAN_PROXIMITIES.index(defaults["ocean_proximity"]),
        )

    submitted = st.form_submit_button("Predict house price", use_container_width=True)

if submitted:
    if total_bedrooms > total_rooms:
        st.error("Total bedrooms cannot be greater than total rooms.")
        st.stop()
    if households > population:
        st.error("Households cannot be greater than population.")
        st.stop()

    input_frame = pd.DataFrame(
        [
            {
                "longitude": longitude,
                "latitude": latitude,
                "housing_median_age": housing_median_age,
                "total_rooms": total_rooms,
                "total_bedrooms": total_bedrooms,
                "population": population,
                "households": households,
                "median_income": median_income,
                "ocean_proximity": ocean_proximity,
            }
        ],
        columns=FEATURE_COLUMNS,
    )
    prediction = float(model.predict(input_frame)[0])

    st.success("Prediction complete")
    st.metric("Estimated house value", f"${prediction:,.0f}")
    st.caption(
        "This value is the direct output of the persisted/trained regression "
        "pipeline for the inputs above; it is not a hard-coded estimate."
    )
