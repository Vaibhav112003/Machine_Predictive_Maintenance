"""
dashboard/app.py

Streamlit dashboard for Industrial Predictive Maintenance.

Run from project root:

    streamlit run dashboard/app.py
"""

from pathlib import Path
import sys
import warnings

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# PROJECT PATH
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(
    str(PROJECT_ROOT)
)

from src.predict import (
    PredictiveMaintenanceModel
)

# ---------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Industrial Predictive Maintenance",
    page_icon="🏭",
    layout="wide"
)

# ---------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 700;
    }

    .subtitle {
        font-size: 18px;
        color: #666;
    }

    .metric-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #ddd;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------

st.markdown(
    '<div class="main-title">'
    '🏭 Industrial Predictive Maintenance'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Deep Learning Early-Warning System | '
    '2–6 Hour Failure Warning'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

# ---------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------

FEATURE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features.csv"
)

if not FEATURE_PATH.exists():

    st.error(
        "features.csv not found. "
        "Run the feature-engineering script first."
    )

    st.stop()

df = pd.read_csv(
    FEATURE_PATH
)

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

# ---------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------

try:

    model_service = (
        PredictiveMaintenanceModel()
    )

except Exception as error:

    st.error(
        f"Model could not be loaded: {error}"
    )

    st.stop()

# ---------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------

st.sidebar.header(
    "Machine Selection"
)

machine_ids = sorted(
    df["machine_id"]
    .dropna()
    .unique()
)

selected_machine = st.sidebar.selectbox(
    "Select Machine",
    machine_ids
)

machine_df = df[
    df["machine_id"] == selected_machine
].copy()

machine_df = machine_df.sort_values(
    "timestamp"
)

# ---------------------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------------------

try:

    result = model_service.predict(
        machine_df
    )

except Exception as error:

    st.error(
        f"Prediction failed: {error}"
    )

    st.stop()

probability = (
    result["failure_probability"]
)

threshold = (
    result["threshold"]
)

severity = (
    result["severity"]
)

alert = (
    result["alert"]
)

# ---------------------------------------------------------------------
# KPI ROW
# ---------------------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Machine",
        selected_machine
    )

with col2:

    st.metric(
        "Failure Probability",
        f"{probability * 100:.2f}%"
    )

with col3:

    st.metric(
        "Alert Threshold",
        f"{threshold * 100:.2f}%"
    )

with col4:

    st.metric(
        "Warning Status",
        "🚨 ALERT"
        if alert
        else "✅ NORMAL"
    )

# ---------------------------------------------------------------------
# STATUS
# ---------------------------------------------------------------------

if severity == "CRITICAL":

    st.error(
        "🚨 CRITICAL: High probability of "
        "an upcoming maintenance/failure event."
    )

elif severity == "HIGH":

    st.warning(
        "⚠️ HIGH RISK: Investigate machine condition."
    )

elif severity == "WARNING":

    st.warning(
        "⚠️ WARNING: Model probability exceeds "
        "the operational threshold."
    )

else:

    st.success(
        "✅ NORMAL: No early-warning alert currently."
    )

# ---------------------------------------------------------------------
# PROBABILITY HISTORY
# ---------------------------------------------------------------------

st.subheader(
    "Failure Probability Trend"
)

# For dashboard visualization, generate predictions
# over recent windows.

history_size = min(
    len(machine_df),
    720
)

recent = machine_df.tail(
    history_size
).copy()

probabilities = []

timestamps = []

if len(recent) >= 120:

    for i in range(
        120,
        len(recent) + 1,
        5
    ):

        window = recent.iloc[
            i - 120:i
        ]

        try:

            prediction = (
                model_service.predict(
                    window
                )
            )

            probabilities.append(
                prediction[
                    "failure_probability"
                ]
            )

            timestamps.append(
                window["timestamp"].iloc[-1]
            )

        except Exception:
            pass

if probabilities:

    probability_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "probability": probabilities
        }
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=probability_df["timestamp"],
            y=probability_df["probability"],
            mode="lines",
            name="Failure Probability"
        )
    )

    fig.add_hline(
        y=threshold,
        line_dash="dash",
        annotation_text="Alert Threshold"
    )

    fig.update_layout(
        yaxis_title="Failure Probability",
        xaxis_title="Time",
        yaxis_range=[0, 1],
        height=450
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

else:

    st.info(
        "Not enough historical data to display "
        "the probability trend."
    )

# ---------------------------------------------------------------------
# SENSOR MONITORING
# ---------------------------------------------------------------------

st.subheader(
    "Machine Sensor Monitoring"
)

sensor_columns = [
    "vibration_x",
    "vibration_y",
    "temperature_bearing",
    "ambient_temp",
    "rotational_speed_rpm",
    "pressure_psi"
]

available_sensors = [
    c
    for c in sensor_columns
    if c in recent.columns
]

for sensor in available_sensors:

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=recent["timestamp"],
            y=recent[sensor],
            mode="lines",
            name=sensor
        )
    )

    fig.update_layout(
        title=sensor,
        height=300,
        xaxis_title="Time",
        yaxis_title=sensor
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ---------------------------------------------------------------------
# CURRENT TELEMETRY
# ---------------------------------------------------------------------

st.subheader(
    "Latest Machine Telemetry"
)

latest_columns = [
    "timestamp"
] + available_sensors

st.dataframe(
    machine_df[
        latest_columns
    ].tail(10),
    use_container_width=True,
    hide_index=True
)

# ---------------------------------------------------------------------
# MODEL INFORMATION
# ---------------------------------------------------------------------

st.sidebar.divider()

st.sidebar.subheader(
    "Model Information"
)

st.sidebar.write(
    "Model: LSTM"
)

st.sidebar.write(
    "History Window: 120 minutes"
)

st.sidebar.write(
    "Prediction Horizon: 2–6 hours"
)

st.sidebar.write(
    f"Threshold: {threshold:.4f}"
)

st.sidebar.info(
    "The event label is based on the "
    "maintenance-event field supplied in "
    "the case-study dataset."
)