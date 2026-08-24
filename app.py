"""Streamlit interface for the trained housing-price prediction pipeline."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from train_model import DATA_PATH, FEATURE_COLUMNS, MODEL_PATH, train_and_save_model


BASE_DIR = Path(__file__).resolve().parent
BACKGROUND_PATH = BASE_DIR / "assets" / "real_estate_hero.jpg"

st.set_page_config(
    page_title="Hearthline | California Home Valuator",
    page_icon="⌂",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def background_data_uri(path: Path) -> str:
    """Return a local image as a CSS-safe data URI for deployment portability."""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


background_image = background_data_uri(BACKGROUND_PATH) if BACKGROUND_PATH.exists() else ""
background_rule = (
    f"background-image: linear-gradient(90deg, rgba(11, 29, 45, .92) 0%, rgba(18, 48, 69, .73) 48%, rgba(18, 48, 69, .34) 100%), url('{background_image}');"
    if background_image
    else "background-image: linear-gradient(135deg, #13283d, #2d5a68);"
)

# -------------------- THEME CONTROL --------------------
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Light"
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "prediction_model_name" not in st.session_state:
    st.session_state.prediction_model_name = None
if "prediction_comparison" not in st.session_state:
    st.session_state.prediction_comparison = None

st.markdown('<div class="theme-kicker">Appearance</div>', unsafe_allow_html=True)
theme_left, theme_right = st.columns(2, gap="small")
with theme_left:
    if st.button("☀  Light mode", key="light_mode_button", use_container_width=True):
        st.session_state.theme_mode = "Light"
with theme_right:
    if st.button("◐  Dark mode", key="dark_mode_button", use_container_width=True):
        st.session_state.theme_mode = "Dark"

theme_mode = st.session_state.theme_mode
dark_mode = theme_mode == "Dark"
st.markdown(
    '<span class="dark-mode-marker"></span>' if dark_mode else '<span class="light-mode-marker"></span>',
    unsafe_allow_html=True,
)

# -------------------- DESIGN SYSTEM --------------------
st.markdown(
    f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

        :root {{
            --ink: #172433;
            --muted: #66717e;
            --navy: #13283d;
            --navy-2: #1b3954;
            --cream: #f7f5ef;
            --sand: #eee8dc;
            --gold: #c9974e;
            --gold-light: #f4e6cc;
            --teal: #2b746c;
            --line: #e5dfd4;
        }}

        [data-testid="stAppViewContainer"] {{
            background: var(--cream);
        }}

        [data-testid="stHeader"] {{
            background: transparent;
        }}

        main .block-container {{
            max-width: 1240px;
            padding: 1.5rem 2.4rem 2rem;
        }}

        h1, h2, h3 {{
            color: var(--ink);
            letter-spacing: -0.025em;
        }}

        h1, h2 {{
            font-family: 'Playfair Display', Georgia, serif;
        }}

        .topbar {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.25rem 0 1.25rem;
        }}

        .brand {{
            color: var(--navy);
            font: 700 1.25rem/1 'Playfair Display', Georgia, serif;
            letter-spacing: -0.02em;
        }}

        .brand-mark {{
            color: var(--gold);
            font-family: Georgia, serif;
            font-size: 1.45rem;
            margin-right: 0.35rem;
        }}

        .topbar-meta {{
            color: var(--muted);
            font-size: 0.78rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }}

        .theme-kicker {{
            color: var(--muted);
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.14em;
            margin: 0 0 0.34rem;
            text-transform: uppercase;
        }}

        [data-testid="stRadio"] {{
            margin: 0 0 0.65rem;
        }}

        [data-testid="stRadio"] [role="radiogroup"] {{
            align-items: center;
            display: flex;
            gap: 0.45rem;
        }}

        [data-testid="stRadio"] label {{
            align-items: center;
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 999px;
            color: var(--ink) !important;
            cursor: pointer;
            display: inline-flex;
            font-size: 0.82rem;
            font-weight: 700;
            min-height: 2.25rem;
            padding: 0.32rem 0.82rem;
            transition: all 0.2s ease;
        }}

        [data-testid="stRadio"] label:hover {{
            border-color: var(--gold);
            transform: translateY(-1px);
        }}

        [data-testid="stRadio"] label p,
        [data-testid="stRadio"] label span {{
            color: var(--ink) !important;
            font-size: 0.82rem !important;
            font-weight: 700 !important;
        }}

        [data-testid="stRadio"] label:has(input:checked) {{
            background: var(--navy);
            border-color: var(--navy);
            color: #ffffff !important;
        }}

        [data-testid="stRadio"] label:has(input:checked) p,
        [data-testid="stRadio"] label:has(input:checked) span {{
            color: #ffffff !important;
        }}

        [data-testid="stRadio"] label > div:first-child {{
            display: none;
        }}

        .hero {{
            {background_rule}
            background-size: cover;
            background-position: center;
            border-radius: 24px;
            min-height: 330px;
            padding: 3.4rem 3.2rem 2.9rem;
            margin: 0 0 1.35rem;
            color: white;
            position: relative;
            overflow: hidden;
            box-shadow: 0 22px 50px rgba(19, 40, 61, 0.18);
        }}

        .hero::after {{
            content: '⌂';
            position: absolute;
            right: 3rem;
            bottom: -4rem;
            color: rgba(255,255,255,0.11);
            font: 19rem/1 Georgia, serif;
        }}

        .hero-content {{
            position: relative;
            z-index: 1;
            max-width: 720px;
        }}

        .hero-kicker, .eyebrow {{
            color: #e6c58f;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.17em;
            text-transform: uppercase;
        }}

        .hero-kicker {{ margin-bottom: 0.85rem; }}

        .hero-title {{
            color: white;
            font: 700 clamp(2.25rem, 5vw, 4.3rem)/1.02 'Playfair Display', Georgia, serif;
            max-width: 700px;
            margin: 0;
        }}

        .hero-copy {{
            color: #e0e8eb;
            font-size: 1.05rem;
            line-height: 1.65;
            max-width: 650px;
            margin: 1rem 0 1.35rem;
        }}

        .hero-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            background: rgba(255, 255, 255, 0.16);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 999px;
            color: #fff8ec;
            padding: 0.48rem 0.8rem;
            font-size: 0.78rem;
            font-weight: 600;
            backdrop-filter: blur(8px);
        }}

        .section-head {{
            margin: 2rem 0 0.9rem;
        }}

        .eyebrow {{
            color: var(--gold);
            margin: 0 0 0.35rem;
        }}

        .section-title {{
            color: var(--ink);
            font: 700 1.85rem/1.15 'Playfair Display', Georgia, serif;
            margin: 0;
        }}

        .section-copy {{
            color: var(--muted);
            line-height: 1.55;
            margin: 0.45rem 0 1rem;
        }}

        .model-intro {{
            color: var(--muted);
            font-size: 0.91rem;
            line-height: 1.55;
            margin: 0.2rem 0 1rem;
        }}

        div[data-testid="stMetric"] {{
            background: white;
            border: 1px solid var(--line);
            border-radius: 15px;
            min-height: 98px;
            padding: 0.9rem 1rem;
            box-shadow: 0 9px 25px rgba(23, 36, 51, 0.045);
        }}

        div[data-testid="stMetricLabel"] p {{ color: var(--muted); }}
        div[data-testid="stMetricValue"] {{ color: var(--navy); }}

        .chart-card, .glossary-card {{
            background: white;
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 1.15rem 1.25rem 0.6rem;
            box-shadow: 0 10px 30px rgba(23, 36, 51, 0.045);
        }}

        .chart-note {{
            color: var(--muted);
            font-size: 0.78rem;
            margin: 0 0 0.2rem;
        }}

        div[data-testid="stForm"] {{
            background: white;
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 1.25rem 1.5rem 0.9rem;
            box-shadow: 0 12px 34px rgba(23, 36, 51, 0.055);
        }}

        .form-column-label {{
            color: var(--navy);
            font: 700 1.05rem/1.2 'Playfair Display', Georgia, serif;
            margin: 0.15rem 0 0.9rem;
        }}

        .input-note {{
            color: var(--muted);
            font-size: 0.77rem;
            line-height: 1.45;
            margin: -0.28rem 0 0.7rem;
        }}

        .glossary-item {{
            border-bottom: 1px solid #eee9df;
            padding: 0.62rem 0;
        }}

        .glossary-item:last-child {{ border-bottom: 0; }}

        .glossary-term {{
            color: var(--navy);
            font-weight: 700;
            font-size: 0.87rem;
        }}

        .glossary-definition {{
            color: var(--muted);
            font-size: 0.78rem;
            line-height: 1.45;
            margin-top: 0.16rem;
        }}

        .prediction-card {{
            background: linear-gradient(135deg, #f4e6cc, #fbf4e7);
            border: 1px solid #e7d0a7;
            border-radius: 19px;
            padding: 1.35rem 1.5rem;
            margin-top: 1.1rem;
            box-shadow: 0 10px 25px rgba(168, 121, 52, 0.08);
        }}

        .prediction-label {{
            color: #80591e;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.14em;
            text-transform: uppercase;
        }}

        .prediction-value {{
            color: var(--navy);
            font: 700 clamp(2.3rem, 6vw, 3.8rem)/1.05 'Playfair Display', Georgia, serif;
            margin-top: 0.32rem;
        }}

        .stButton > button, [data-testid="stFormSubmitButton"] button {{
            background: var(--gold);
            border: 0;
            border-radius: 10px;
            color: #182735;
            font-weight: 700;
            min-height: 2.85rem;
            transition: all 0.2s ease;
        }}

        .stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover {{
            background: #ddb36f;
            border: 0;
            color: #182735;
            transform: translateY(-1px);
        }}

        .disclaimer {{
            background: #edf4f2;
            border-left: 3px solid var(--teal);
            border-radius: 0 10px 10px 0;
            color: #4c6865;
            font-size: 0.82rem;
            line-height: 1.5;
            padding: 0.75rem 0.9rem;
            margin-top: 1rem;
        }}

        .footer-note {{
            color: #87919a;
            font-size: 0.78rem;
            text-align: center;
            padding: 1.8rem 0 0.5rem;
        }}

        /* Dark mode: all surfaces, labels, inputs, cards, and chart-adjacent text
           use explicit high-contrast colors so content remains visible. */
        body:has(.dark-mode-marker) {{
            --ink: #edf4f8;
            --muted: #b7c5cf;
            --navy: #edf4f8;
            --navy-2: #d4e0e7;
            --cream: #0d151e;
            --sand: #172532;
            --gold: #e4b86d;
            --gold-light: #4d3a1f;
            --teal: #78c8bc;
            --line: #314554;
        }}

        body:has(.dark-mode-marker) [data-testid="stAppViewContainer"],
        body:has(.dark-mode-marker) [data-testid="stMain"] {{
            background: var(--cream);
            color: var(--ink);
        }}

        body:has(.dark-mode-marker) .topbar-meta,
        body:has(.dark-mode-marker) .section-copy,
        body:has(.dark-mode-marker) .model-intro,
        body:has(.dark-mode-marker) .chart-note,
        body:has(.dark-mode-marker) .input-note,
        body:has(.dark-mode-marker) .glossary-definition,
        body:has(.dark-mode-marker) .footer-note {{
            color: var(--muted) !important;
        }}

        body:has(.dark-mode-marker) div[data-testid="stMetric"],
        body:has(.dark-mode-marker) .chart-card,
        body:has(.dark-mode-marker) .glossary-card,
        body:has(.dark-mode-marker) div[data-testid="stForm"] {{
            background: #16232f !important;
            border-color: var(--line) !important;
            color: var(--ink) !important;
        }}

        body:has(.dark-mode-marker) .form-column-label,
        body:has(.dark-mode-marker) .glossary-term,
        body:has(.dark-mode-marker) div[data-testid="stMetricValue"] {{
            color: var(--ink) !important;
        }}

        body:has(.dark-mode-marker) div[data-testid="stMetricLabel"] p,
        body:has(.dark-mode-marker) label,
        body:has(.dark-mode-marker) [data-testid="stWidgetLabel"] p,
        body:has(.dark-mode-marker) [data-testid="stMarkdownContainer"] p,
        body:has(.dark-mode-marker) [data-testid="stMarkdownContainer"] strong {{
            color: var(--muted) !important;
        }}

        body:has(.dark-mode-marker) [data-baseweb="select"] > div,
        body:has(.dark-mode-marker) [data-testid="stNumberInputContainer"],
        body:has(.dark-mode-marker) input,
        body:has(.dark-mode-marker) textarea {{
            background: #20313f !important;
            color: #f4f8fa !important;
            border-color: #4a6070 !important;
        }}

        body:has(.dark-mode-marker) input::placeholder,
        body:has(.dark-mode-marker) textarea::placeholder {{
            color: #aebdc7 !important;
        }}

        body:has(.dark-mode-marker) [data-baseweb="select"] svg,
        body:has(.dark-mode-marker) [data-testid="stNumberInputContainer"] svg {{
            fill: #e3edf2 !important;
        }}

        body:has(.dark-mode-marker) [data-testid="stRadio"] label {{
            background: #20313f !important;
            border-color: #4a6070 !important;
            color: #f4f8fa !important;
        }}

        body:has(.dark-mode-marker) [data-testid="stRadio"] label p,
        body:has(.dark-mode-marker) [data-testid="stRadio"] label span {{
            color: #f4f8fa !important;
        }}

        body:has(.dark-mode-marker) [data-testid="stRadio"] label:has(input:checked) {{
            background: #e4b86d !important;
            border-color: #e4b86d !important;
            color: #13283d !important;
        }}

        body:has(.dark-mode-marker) [data-testid="stRadio"] label:has(input:checked) p,
        body:has(.dark-mode-marker) [data-testid="stRadio"] label:has(input:checked) span {{
            color: #13283d !important;
        }}

        body:has(.dark-mode-marker) [data-testid="stFormSubmitButton"] button,
        body:has(.dark-mode-marker) .stButton > button {{
            background: #e4b86d !important;
            color: #13283d !important;
            text-shadow: none !important;
        }}

        body:has(.dark-mode-marker) [data-testid="stFormSubmitButton"] button p,
        body:has(.dark-mode-marker) [data-testid="stFormSubmitButton"] button span,
        body:has(.dark-mode-marker) .stButton > button p,
        body:has(.dark-mode-marker) .stButton > button span {{
            color: #13283d !important;
        }}

        body:has(.dark-mode-marker) .prediction-card {{
            background: linear-gradient(135deg, #4d3a1f, #342b20) !important;
            border-color: #a87836 !important;
        }}

        body:has(.dark-mode-marker) .prediction-label {{
            color: #f3ce8b !important;
        }}

        body:has(.dark-mode-marker) .prediction-value {{
            color: #fff4dc !important;
        }}

        body:has(.dark-mode-marker) .glossary-item {{
            border-bottom-color: #2d4050 !important;
        }}

        body:has(.dark-mode-marker) .disclaimer {{
            background: #173431 !important;
            color: #b9ddd7 !important;
            border-left-color: var(--teal) !important;
        }}

        body:has(.dark-mode-marker) [data-testid="stAlert"] {{
            color: #edf4f8 !important;
        }}

        @media (max-width: 720px) {{
            main .block-container {{ padding: 1rem 1rem 1.5rem; }}
            .hero {{ min-height: 360px; padding: 2.3rem 1.45rem 2rem; }}
            .topbar-meta {{ display: none; }}
        }}
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
    best_row = metrics_frame.loc[metrics_frame["rmse"].idxmin()]
    best_rmse = float(best_row["rmse"])
    best_r2 = float(best_row["r2"])
else:
    best_rmse = None
    best_r2 = None

chart_axis_color = "#b7c5cf" if dark_mode else "#66717e"
chart_other_bar = "#587080" if dark_mode else "#b8c5ce"
chart_grid = "#304554" if dark_mode else "#eee9df"
chart_background = "#16232f" if dark_mode else "#ffffff"

# -------------------- TOP NAVIGATION --------------------
st.markdown(
    """
    <div class="topbar">
        <div class="brand"><span class="brand-mark">⌂</span>Hearthline</div>
        <div class="topbar-meta">California residential valuation studio&nbsp;&nbsp; / &nbsp;&nbsp;Model-led insights</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------- HERO --------------------
st.markdown(
    f"""
    <section class="hero">
        <div class="hero-content">
            <div class="hero-kicker">Data-led home valuation</div>
            <h1 class="hero-title">Know the value of where you live.</h1>
            <p class="hero-copy">A considered estimate for California homes, powered by location, income, space, and a production-ready machine-learning model.</p>
            <span class="hero-badge">●&nbsp; Live valuation engine &nbsp;·&nbsp; {bundle.get('model_name', 'Saved model')} in production</span>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

# -------------------- MODEL OVERVIEW --------------------
st.markdown('<div class="section-head"><p class="eyebrow">The valuation engine</p><p class="section-title">A model you can see into</p></div>', unsafe_allow_html=True)
st.markdown(
    '<p class="model-intro">We compare several regressors on a held-out portion of the California housing data. The model with the lowest RMSE becomes the production model used for every estimate below.</p>',
    unsafe_allow_html=True,
)

if best_rmse is not None:
    stat1, stat2, stat3, stat4 = st.columns(4)
    with stat1:
        st.metric("Production model", bundle.get("model_name", "Saved model"))
    with stat2:
        st.metric("Validation RMSE", f"${best_rmse:,.0f}")
    with stat3:
        st.metric("Holdout R²", f"{best_r2:.3f}")
    with stat4:
        st.metric("Models evaluated", f"{len(metrics_frame)}")

# -------------------- MODEL PERFORMANCE --------------------
st.markdown('<div class="section-head"><p class="eyebrow">Model selection</p><p class="section-title">Performance, side by side</p></div>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-copy">Lower RMSE means a smaller average prediction error on the holdout set. The gold bar marks the model powering this valuation tool.</p>',
    unsafe_allow_html=True,
)

if not metrics_frame.empty:
    chart_data = metrics_frame[["Model", "rmse", "r2", "Production"]].rename(
        columns={"rmse": "RMSE", "r2": "R²"}
    )
    chart_data["Series"] = chart_data["Production"].map(
        {True: "Production model", False: "Comparison model"}
    )
    chart_max = float(chart_data["RMSE"].max()) * 1.18
    chart_base = alt.Chart(chart_data)
    chart_bars = chart_base.mark_bar(size=34, cornerRadiusEnd=6).encode(
        x=alt.X(
            "RMSE:Q",
            title="Holdout RMSE (USD)",
            axis=alt.Axis(format="$,.0f", titleColor=chart_axis_color, labelColor=chart_axis_color),
            scale=alt.Scale(domain=[0, chart_max]),
        ),
        y=alt.Y("Model:N", title=None, sort="-x", axis=alt.Axis(labelColor=chart_axis_color)),
        color=alt.Color(
            "Series:N",
            scale=alt.Scale(
                domain=["Production model", "Comparison model"],
                range=["#c9974e", chart_other_bar],
            ),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("Model:N", title="Model"),
            alt.Tooltip("RMSE:Q", title="RMSE", format="$,.0f"),
            alt.Tooltip("R²:Q", title="R²", format=".3f"),
            alt.Tooltip("Production:N", title="Used in app"),
        ],
    )
    chart_labels = chart_base.mark_text(
        align="left", baseline="middle", dx=7, color=chart_axis_color, fontSize=12
    ).encode(
        x=alt.X("RMSE:Q", scale=alt.Scale(domain=[0, chart_max])),
        y=alt.Y("Model:N", sort="-x"),
        text=alt.Text("RMSE:Q", format="$,.0f"),
    )
    chart = (
        (chart_bars + chart_labels)
        .properties(height=230, background=chart_background)
        .configure_view(strokeOpacity=0)
        .configure_axis(gridColor=chart_grid, domain=False, tickColor=chart_grid)
    )
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<p class="chart-note">Gold = production model · Gray = comparison models</p>', unsafe_allow_html=True)
    st.altair_chart(chart, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("Validation metrics are not available for this legacy model artifact.")

# -------------------- PROPERTY FORM --------------------
st.markdown('<div class="section-head"><p class="eyebrow">Your property profile</p><p class="section-title">Tell us about the home</p></div>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-copy">Start with a curated profile, then adjust the district details. Every field below is used by the trained model to calculate the estimate.</p>',
    unsafe_allow_html=True,
)

preset_name = st.selectbox(
    "Start from a sample profile",
    list(PRESETS),
    label_visibility="visible",
    key="preset_name",
    help="Choose a profile to populate the form, or select Custom to enter your own values.",
)
defaults = PRESETS[preset_name]

# A theme button causes a Streamlit rerun. Store every input under a stable key
# so both the form and the last prediction survive that rerun.
if st.session_state.get("active_preset") != preset_name:
    for field_name, field_value in defaults.items():
        st.session_state[f"input_{field_name}"] = field_value
    st.session_state.active_preset = preset_name

with st.form("prediction_form"):
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<p class="form-column-label">Location & character</p>', unsafe_allow_html=True)
        longitude = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            key="input_longitude",
            step=0.01,
            format="%.4f",
            help="East-west location of the district. California values are generally negative.",
        )
        st.markdown('<p class="input-note">Where is the district on the map?</p>', unsafe_allow_html=True)
        latitude = st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            key="input_latitude",
            step=0.01,
            format="%.4f",
            help="North-south location of the district. Higher values are farther north.",
        )
        st.markdown('<p class="input-note">Latitude helps the model capture regional price patterns.</p>', unsafe_allow_html=True)
        housing_median_age = st.number_input(
            "Housing median age",
            min_value=0,
            max_value=200,
            key="input_housing_median_age",
            step=1,
            help="The median age of homes in the district, measured in years.",
        )
        st.markdown('<p class="input-note">Typical construction age across the district.</p>', unsafe_allow_html=True)
        ocean_proximity = st.selectbox(
            "Ocean proximity",
            OCEAN_PROXIMITIES,
            key="input_ocean_proximity",
            help="The housing dataset’s geographic proximity category.",
        )
        st.markdown('<p class="input-note">A location signal that captures access to the coast.</p>', unsafe_allow_html=True)

    with col2:
        st.markdown('<p class="form-column-label">Space & household profile</p>', unsafe_allow_html=True)
        total_rooms = st.number_input(
            "Total rooms",
            min_value=1,
            key="input_total_rooms",
            step=1,
            help="Total number of rooms reported across homes in the district.",
        )
        st.markdown('<p class="input-note">The overall amount of interior space in the district.</p>', unsafe_allow_html=True)
        total_bedrooms = st.number_input(
            "Total bedrooms",
            min_value=1,
            key="input_total_bedrooms",
            step=1,
            help="Total bedrooms reported across homes in the district.",
        )
        st.markdown('<p class="input-note">A rough measure of sleeping capacity and home size.</p>', unsafe_allow_html=True)
        population = st.number_input(
            "Population",
            min_value=1,
            key="input_population",
            step=1,
            help="Total number of people living in the district.",
        )
        st.markdown('<p class="input-note">The number of residents represented in the district.</p>', unsafe_allow_html=True)
        households = st.number_input(
            "Households",
            min_value=1,
            key="input_households",
            step=1,
            help="Total number of households in the district.",
        )
        st.markdown('<p class="input-note">Household density helps describe local housing demand.</p>', unsafe_allow_html=True)
        median_income = st.number_input(
            "Median income (× $10,000)",
            min_value=0.0,
            key="input_median_income",
            step=0.1,
            format="%.4f",
            help="The dataset’s median household income measure, expressed in tens of thousands of US dollars.",
        )
        st.markdown('<p class="input-note">The district’s median income, scaled in tens of thousands of dollars.</p>', unsafe_allow_html=True)

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
    st.session_state.prediction_result = prediction
    st.session_state.prediction_model_name = bundle.get("model_name", "trained")

    comparison_profiles = {
        "Your estimate": input_frame.iloc[0].to_dict(),
        "Affordable Home": PRESETS["Affordable Home"],
        "Large Family Home": PRESETS["Large Family Home"],
        "Coastal Property": PRESETS["Coastal Property"],
    }
    comparison_frame = pd.DataFrame.from_dict(comparison_profiles, orient="index")
    comparison_frame = comparison_frame[FEATURE_COLUMNS]
    comparison_values = model.predict(comparison_frame)
    st.session_state.prediction_comparison = [
        {
            "Profile": profile_name,
            "Estimated value": float(profile_prediction),
            "Current": profile_name == "Your estimate",
        }
        for profile_name, profile_prediction in zip(comparison_profiles, comparison_values)
    ]

if st.session_state.prediction_result is not None:
    saved_prediction = float(st.session_state.prediction_result)
    saved_model_name = st.session_state.prediction_model_name or bundle.get("model_name", "trained")
    st.markdown(
        f'<div class="prediction-card"><div class="prediction-label">Estimated market value</div><div class="prediction-value">${saved_prediction:,.0f}</div></div>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"Direct output from the {saved_model_name} regression pipeline. This result is preserved when you switch Light/Dark mode."
    )
    if best_rmse is not None:
        st.info(
            f"How to read the model chart above: it compares model accuracy, not your property. "
            f"The gold {saved_model_name} bar is the production model; its typical validation error is about ${best_rmse:,.0f}. "
            "Changing your inputs changes the estimated market value, while the RMSE comparison stays fixed because it describes model performance."
        )

    comparison_data = pd.DataFrame(st.session_state.prediction_comparison or [])
    if not comparison_data.empty:
        st.markdown('<div class="section-head comparison-head"><p class="eyebrow">Your estimate in context</p><p class="section-title">See the possibilities side by side</p></div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-copy">These are four fresh predictions from the same production pipeline. Your submitted property is highlighted in gold; the other bars show how the model responds to different district profiles.</p>',
            unsafe_allow_html=True,
        )
        comparison_max = float(comparison_data["Estimated value"].max()) * 1.2
        comparison_base = alt.Chart(comparison_data)
        comparison_bars = comparison_base.mark_bar(size=34, cornerRadiusEnd=6).encode(
            x=alt.X(
                "Estimated value:Q",
                title="Model-predicted value (USD)",
                axis=alt.Axis(format="$,.0f", titleColor=chart_axis_color, labelColor=chart_axis_color),
                scale=alt.Scale(domain=[0, comparison_max]),
            ),
            y=alt.Y("Profile:N", title=None, sort="-x", axis=alt.Axis(labelColor=chart_axis_color)),
            color=alt.Color(
                "Current:N",
                scale=alt.Scale(domain=[True, False], range=["#c9974e", chart_other_bar]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Profile:N", title="Profile"),
                alt.Tooltip("Estimated value:Q", title="Predicted value", format="$,.0f"),
                alt.Tooltip("Current:N", title="Your submitted property"),
            ],
        )
        comparison_labels = comparison_base.mark_text(
            align="left", baseline="middle", dx=7, color=chart_axis_color, fontSize=12
        ).encode(
            x=alt.X("Estimated value:Q", scale=alt.Scale(domain=[0, comparison_max])),
            y=alt.Y("Profile:N", sort="-x"),
            text=alt.Text("Estimated value:Q", format="$,.0f"),
        )
        comparison_chart = (
            (comparison_bars + comparison_labels)
            .properties(height=245, background=chart_background)
            .configure_view(strokeOpacity=0)
            .configure_axis(gridColor=chart_grid, domain=False, tickColor=chart_grid)
        )
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<p class="chart-note">Gold = your submitted property · Gray = alternative profiles</p>', unsafe_allow_html=True)
        st.altair_chart(comparison_chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# -------------------- INPUT GLOSSARY --------------------
st.markdown('<div class="section-head"><p class="eyebrow">A closer look</p><p class="section-title">What each input means</p></div>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-copy">These are district-level signals from the California housing dataset—not a room-by-room appraisal. Together, they give the model context about location, space, residents, and purchasing power.</p>',
    unsafe_allow_html=True,
)

glossary_columns = st.columns(3, gap="medium")
glossary = [
    (
        "Location",
        [
            ("Longitude", "East-west position of the district."),
            ("Latitude", "North-south position of the district."),
            ("Ocean proximity", "Categorical distance-to-coast signal."),
        ],
    ),
    (
        "Home profile",
        [
            ("Housing median age", "Typical age of homes in years."),
            ("Total rooms", "Total rooms reported across the district."),
            ("Total bedrooms", "Total bedrooms reported across the district."),
        ],
    ),
    (
        "Community profile",
        [
            ("Population", "Residents represented in the district."),
            ("Households", "Households represented in the district."),
            ("Median income", "Median income, scaled by $10,000."),
        ],
    ),
]
for column, (heading, items) in zip(glossary_columns, glossary):
    with column:
        content = f'<div class="glossary-card"><p class="form-column-label">{heading}</p>'
        for term, definition in items:
            content += f'<div class="glossary-item"><div class="glossary-term">{term}</div><div class="glossary-definition">{definition}</div></div>'
        content += "</div>"
        st.markdown(content, unsafe_allow_html=True)

st.markdown(
    '<div class="disclaimer"><strong>Good to know:</strong> This is a machine-learning estimate for exploration, not a formal appraisal, inspection, offer, or financial advice. The estimate is the direct output of the trained production pipeline.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="footer-note">Hearthline · California home valuation intelligence · Built with a reproducible training pipeline</div>',
    unsafe_allow_html=True,
)
