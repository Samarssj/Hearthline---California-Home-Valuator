"""Streamlit interface for the trained housing-price prediction pipeline."""

from __future__ import annotations

from typing import Any

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from train_model import DATA_PATH, FEATURE_COLUMNS, MODEL_PATH, train_and_save_model


st.set_page_config(
    page_title="Hearthline | California Home Valuator",
    page_icon="⌂",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- DESIGN SYSTEM --------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

        :root {
            --ink: #172433;
            --muted: #66717e;
            --navy: #13283d;
            --navy-2: #1b3954;
            --cream: #faf8f3;
            --sand: #eee8dc;
            --gold: #c9974e;
            --gold-light: #f4e6cc;
            --teal: #2b746c;
            --line: #e7e1d6;
        }

        [data-testid="stAppViewContainer"] {
            background: var(--cream);
        }

        [data-testid="stHeader"] {
            background: rgba(250, 248, 243, 0.92);
        }

        [data-testid="stSidebar"] {
            background: var(--navy);
            border-right: 0;
        }

        [data-testid="stSidebar"] * {
            color: #f6f2e9;
        }

        [data-testid="stSidebar"] .stCaption,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: #c4d0d9;
        }

        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] [data-testid="stNumberInputContainer"] {
            background: #213e57;
            border-color: #426078;
        }

        h1, h2, h3 {
            color: var(--ink);
            letter-spacing: -0.025em;
        }

        h1, h2 {
            font-family: 'Playfair Display', Georgia, serif;
        }

        .hero {
            background: linear-gradient(135deg, #13283d 0%, #234863 100%);
            border-radius: 24px;
            padding: 2.2rem 2.4rem 2rem;
            margin: 0.5rem 0 1.5rem;
            color: white;
            position: relative;
            overflow: hidden;
            box-shadow: 0 18px 45px rgba(19, 40, 61, 0.16);
        }

        .hero::after {
            content: '⌂';
            position: absolute;
            right: 3rem;
            top: -1.4rem;
            color: rgba(255,255,255,0.08);
            font: 18rem/1 Georgia, serif;
        }

        .hero-kicker {
            color: #e6c58f;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            margin-bottom: 0.7rem;
        }

        .hero-title {
            color: white;
            font: 700 clamp(2rem, 4vw, 3.55rem)/1.05 'Playfair Display', Georgia, serif;
            max-width: 680px;
            margin: 0;
        }

        .hero-copy {
            color: #d8e1e7;
            font-size: 1.03rem;
            line-height: 1.65;
            max-width: 620px;
            margin: 1rem 0 0;
        }

        .eyebrow {
            color: var(--gold);
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            margin: 0 0 0.45rem;
        }

        .section-title {
            color: var(--ink);
            font: 700 1.8rem/1.15 'Playfair Display', Georgia, serif;
            margin: 0;
        }

        .section-copy {
            color: var(--muted);
            margin: 0.45rem 0 1.15rem;
        }

        .model-pill {
            display: inline-block;
            background: var(--gold-light);
            color: #80591e;
            border-radius: 999px;
            padding: 0.38rem 0.78rem;
            font-size: 0.78rem;
            font-weight: 700;
            margin: 0.4rem 0 0.8rem;
        }

        .trust-note {
            border-left: 3px solid var(--gold);
            background: rgba(244, 230, 204, 0.38);
            color: var(--muted);
            padding: 0.75rem 0.9rem;
            border-radius: 0 10px 10px 0;
            font-size: 0.88rem;
            line-height: 1.5;
        }

        div[data-testid="stForm"] {
            background: white;
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 1.2rem 1.35rem 0.75rem;
            box-shadow: 0 10px 30px rgba(23, 36, 51, 0.05);
        }

        div[data-testid="stMetric"] {
            background: white;
            border: 1px solid var(--line);
            border-radius: 15px;
            padding: 0.9rem 1rem;
            box-shadow: 0 8px 22px rgba(23, 36, 51, 0.04);
        }

        div[data-testid="stMetricLabel"] p {
            color: var(--muted);
        }

        div[data-testid="stMetricValue"] {
            color: var(--navy);
        }

        .prediction-card {
            background: linear-gradient(135deg, #f4e6cc, #fbf4e7);
            border: 1px solid #e7d0a7;
            border-radius: 18px;
            padding: 1.25rem 1.4rem;
            margin-top: 1rem;
        }

        .prediction-label {
            color: #80591e;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.14em;
            text-transform: uppercase;
        }

        .prediction-value {
            color: var(--navy);
            font: 700 clamp(2rem, 5vw, 3.2rem)/1.1 'Playfair Display', Georgia, serif;
            margin-top: 0.35rem;
        }

        .stButton > button,
        [data-testid="stFormSubmitButton"] button {
            background: var(--gold);
            border: 0;
            border-radius: 10px;
            color: #182735;
            font-weight: 700;
            min-height: 2.8rem;
            transition: all 0.2s ease;
        }

        .stButton > button:hover,
        [data-testid="stFormSubmitButton"] button:hover {
            background: #ddb36f;
            border: 0;
            color: #182735;
            transform: translateY(-1px);
        }

        .chart-card {
            background: white;
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 1.2rem 1.25rem 0.55rem;
            box-shadow: 0 10px 30px rgba(23, 36, 51, 0.05);
        }

        .footer-note {
            color: #87919a;
            font-size: 0.8rem;
            text-align: center;
            padding: 1.6rem 0 0.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
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


@st.cache_resource(show_spinner="Preparing the valuation engine…")
def load_model_bundle() -> dict[str, Any]:
    """Load the persisted bundle, training it once when no artifact is present."""
    if MODEL_PATH.exists():
        artifact = joblib.load(MODEL_PATH)
        if isinstance(artifact, dict) and "model" in artifact:
            return artifact
        return {
            "model": artifact,
            "model_name": "Saved model",
            "metrics": {},
            "feature_columns": FEATURE_COLUMNS,
        }

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training dataset not found at {DATA_PATH}")
    return train_and_save_model(DATA_PATH, MODEL_PATH)


try:
    bundle = load_model_bundle()
    model = bundle["model"]
except Exception as exc:
    st.error(f"The prediction model could not be loaded or trained: {exc}")
    st.stop()

metrics = bundle.get("metrics", {})
metrics_frame = (
    pd.DataFrame(metrics)
    .T.rename_axis("Model")
    .reset_index()
    if metrics
    else pd.DataFrame()
)
if not metrics_frame.empty:
    metrics_frame["Production"] = metrics_frame["Model"].eq(bundle.get("model_name"))
    best_rmse = float(metrics_frame["rmse"].min())
    best_r2 = float(metrics_frame.loc[metrics_frame["rmse"].idxmin(), "r2"])
else:
    best_rmse = None
    best_r2 = None

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.markdown("## ⌂ Hearthline")
    st.caption("California residential valuation studio")
    st.divider()
    st.markdown("### Your scenario")
    preset_name = st.selectbox(
        "Start from a property profile",
        list(PRESETS),
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("### Model credentials")
    st.markdown(f'<span class="model-pill">{bundle.get("model_name", "Saved model")} · production</span>', unsafe_allow_html=True)
    st.caption("Selected by the lowest holdout RMSE across the evaluated regressors.")
    if best_r2 is not None:
        st.metric("Holdout R²", f"{best_r2:.3f}")
    st.markdown(
        '<div class="trust-note">The estimate is generated from the trained pipeline and the property details you provide.</div>',
        unsafe_allow_html=True,
    )

# -------------------- HERO --------------------
st.markdown(
    """
    <section class="hero">
        <div class="hero-kicker">Data-led home valuation</div>
        <h1 class="hero-title">Know the value of where you live.</h1>
        <p class="hero-copy">A considered estimate for California homes, powered by location, income, space, and a production-ready machine-learning model.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

if best_rmse is not None:
    stat1, stat2, stat3 = st.columns(3)
    with stat1:
        st.metric("Production model", bundle.get("model_name", "Saved model"))
    with stat2:
        st.metric("Validation RMSE", f"${best_rmse:,.0f}")
    with stat3:
        st.metric("Models evaluated", f"{len(metrics_frame)}")

# -------------------- MODEL PERFORMANCE --------------------
st.markdown('<p class="eyebrow">Why this model</p>', unsafe_allow_html=True)
st.markdown('<p class="section-title">Performance, side by side</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-copy">Lower RMSE means a smaller average prediction error on the holdout set. The highlighted bar is the model powering this valuation tool.</p>',
    unsafe_allow_html=True,
)

if not metrics_frame.empty:
    chart_data = metrics_frame[["Model", "rmse", "r2", "Production"]].rename(
        columns={"rmse": "RMSE", "r2": "R²"}
    )
    chart = (
        alt.Chart(chart_data)
        .mark_bar(size=34, cornerRadiusEnd=6)
        .encode(
            x=alt.X(
                "RMSE:Q",
                title="Holdout RMSE (USD)",
                axis=alt.Axis(format="$,.0f", titleColor="#66717e", labelColor="#66717e"),
                scale=alt.Scale(domain=[0, None]),
            ),
            y=alt.Y("Model:N", title=None, sort="-x", axis=alt.Axis(labelColor="#172433")),
            color=alt.condition(
                alt.datum.Production,
                alt.value("#c9974e"),
                alt.value("#b8c5ce"),
            ),
            tooltip=[
                alt.Tooltip("Model:N", title="Model"),
                alt.Tooltip("RMSE:Q", title="RMSE", format="$,.0f"),
                alt.Tooltip("R²:Q", title="R²", format=".3f"),
                alt.Tooltip("Production:N", title="Used in app"),
            ],
        )
        .properties(height=225)
        .configure_view(strokeOpacity=0)
        .configure_axis(gridColor="#eee9df", domain=False, tickColor="#eee9df")
    )
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.altair_chart(chart, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("Validation metrics are not available for this legacy model artifact.")

st.markdown("<br>", unsafe_allow_html=True)

# -------------------- INPUT FORM --------------------
defaults = PRESETS[preset_name]
st.markdown('<p class="eyebrow">Property profile</p>', unsafe_allow_html=True)
st.markdown('<p class="section-title">Tell us about the home</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-copy">Use a sample profile or fine-tune the district details for a more tailored estimate.</p>',
    unsafe_allow_html=True,
)

with st.form("prediction_form"):
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("**Location & age**")
        longitude = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=float(defaults["longitude"]),
            step=0.01,
            format="%.4f",
            help="The district's east-west coordinate.",
        )
        latitude = st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=float(defaults["latitude"]),
            step=0.01,
            format="%.4f",
            help="The district's north-south coordinate.",
        )
        housing_median_age = st.number_input(
            "Housing median age",
            min_value=0,
            max_value=200,
            value=int(defaults["housing_median_age"]),
            step=1,
        )
        ocean_proximity = st.selectbox(
            "Ocean proximity",
            OCEAN_PROXIMITIES,
            index=OCEAN_PROXIMITIES.index(defaults["ocean_proximity"]),
        )

    with col2:
        st.markdown("**Space & household profile**")
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
            "Median income (× $10,000)",
            min_value=0.0,
            value=float(defaults["median_income"]),
            step=0.1,
            format="%.4f",
            help="The dataset's median-income measure, expressed in tens of thousands of US dollars.",
        )

    submitted = st.form_submit_button("Estimate home value  →", use_container_width=True)

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
    st.markdown(
        f'<div class="prediction-card"><div class="prediction-label">Estimated market value</div><div class="prediction-value">${prediction:,.0f}</div></div>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"Direct output from the {bundle.get('model_name', 'trained')} regression pipeline for the property profile above."
    )

st.markdown(
    '<div class="footer-note">Hearthline · A machine-learning valuation experience · Estimates are for exploration, not a formal appraisal.</div>',
    unsafe_allow_html=True,
)
