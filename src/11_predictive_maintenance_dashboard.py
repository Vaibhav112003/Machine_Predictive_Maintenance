"""
11_predictive_maintenance_dashboard.py

Simple Streamlit dashboard for the predictive-maintenance pipeline.

Reads:
    reports/alert_engine/machine_alerts.csv
    reports/alert_engine/machine_alert_machine_summary.csv
    reports/alert_engine/confirmed_alerts.csv
    reports/production_inference_test.csv

Run:
    streamlit run src/11_predictive_maintenance_dashboard.py
"""

from pathlib import Path
import pandas as pd
import streamlit as st


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
ALERT_DIR = REPORTS_DIR / "alert_engine"

MACHINE_ALERT_FILE = ALERT_DIR / "machine_alerts.csv"
MACHINE_SUMMARY_FILE = ALERT_DIR / "machine_alert_machine_summary.csv"
CONFIRMED_ALERT_FILE = ALERT_DIR / "confirmed_alerts.csv"
INFERENCE_FILE = REPORTS_DIR / "production_inference_test.csv"

st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="🏭",
    layout="wide",
)


# ============================================================
# HELPERS
# ============================================================

@st.cache_data
def load_csv(path):
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def metric_delta(value, suffix=""):
    return f"{value}{suffix}"


# ============================================================
# LOAD DATA
# ============================================================

machine_alerts = load_csv(MACHINE_ALERT_FILE)
machine_summary = load_csv(MACHINE_SUMMARY_FILE)
confirmed_alerts = load_csv(CONFIRMED_ALERT_FILE)
inference = load_csv(INFERENCE_FILE)

for df in [machine_alerts, confirmed_alerts, inference]:
    if not df.empty and "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

if not machine_alerts.empty:
    machine_alerts["episode_start"] = pd.to_datetime(
        machine_alerts["episode_start"], errors="coerce"
    )
    machine_alerts["episode_end"] = pd.to_datetime(
        machine_alerts["episode_end"], errors="coerce"
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🏭 Predictive Maintenance")

st.sidebar.caption("Early-warning anomaly detection")

if not machine_summary.empty:
    machines = ["All Machines"] + sorted(
        machine_summary["machine_id"].astype(str).unique().tolist()
    )
else:
    machines = ["All Machines"]

selected_machine = st.sidebar.selectbox(
    "Machine",
    machines
)

if st.sidebar.button("🔄 Refresh data"):
    st.cache_data.clear()
    st.rerun()


# ============================================================
# TITLE
# ============================================================

st.title("🏭 Predictive Maintenance Dashboard")
st.caption(
    "Machine-level early-warning monitoring using frozen LSTM, GRU and XGBoost models"
)

st.divider()


# ============================================================
# FILTER
# ============================================================

if selected_machine != "All Machines":
    filtered_alerts = machine_alerts[
        machine_alerts["machine_id"].astype(str) == selected_machine
    ].copy()

    filtered_confirmed = confirmed_alerts[
        confirmed_alerts["machine_id"].astype(str) == selected_machine
    ].copy()
else:
    filtered_alerts = machine_alerts.copy()
    filtered_confirmed = confirmed_alerts.copy()


# ============================================================
# KPI CARDS
# ============================================================

total_machines = (
    machine_summary["machine_id"].nunique()
    if not machine_summary.empty
    else 0
)

total_machine_alerts = len(filtered_alerts)

multi_model_alerts = (
    int((filtered_alerts["model_consensus_count"] >= 2).sum())
    if not filtered_alerts.empty
    else 0
)

highest_risk = (
    float(filtered_alerts["max_risk_score"].max())
    if not filtered_alerts.empty
    else 0.0
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Machines Monitored",
    total_machines
)

c2.metric(
    "Machine Alert Episodes",
    total_machine_alerts
)

c3.metric(
    "Multi-Model Alerts",
    multi_model_alerts
)

c4.metric(
    "Highest Risk Score",
    f"{highest_risk:.3f}"
)


# ============================================================
# ACTIVE / RECENT MACHINE ALERTS
# ============================================================

st.subheader("🚨 Machine Risk Alerts")

if filtered_alerts.empty:
    st.success("No machine-level alerts found.")
else:
    display = filtered_alerts[
        [
            "machine_alert_id",
            "machine_id",
            "episode_start",
            "episode_end",
            "contributing_models",
            "model_consensus_count",
            "total_confirmed_model_alerts",
            "max_risk_score",
            "max_risk_model",
        ]
    ].copy()

    display["episode_start"] = display["episode_start"].dt.strftime(
        "%Y-%m-%d %H:%M"
    )
    display["episode_end"] = display["episode_end"].dt.strftime(
        "%Y-%m-%d %H:%M"
    )

    display = display.rename(
        columns={
            "machine_alert_id": "Alert ID",
            "machine_id": "Machine",
            "episode_start": "Start",
            "episode_end": "End",
            "contributing_models": "Models",
            "model_consensus_count": "Consensus",
            "total_confirmed_model_alerts": "Model Alerts",
            "max_risk_score": "Max Risk",
            "max_risk_model": "Peak Model",
        }
    )

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# MACHINE RISK SUMMARY
# ============================================================

st.subheader("📊 Machine Summary")

if not machine_summary.empty:
    summary_display = machine_summary.copy()

    if selected_machine != "All Machines":
        summary_display = summary_display[
            summary_display["machine_id"].astype(str)
            == selected_machine
        ]

    summary_display = summary_display.rename(
        columns={
            "machine_id": "Machine",
            "machine_alert_episodes": "Alert Episodes",
            "total_confirmed_model_alerts": "Model Alerts",
            "max_risk_score": "Max Risk",
            "models_contributed": "Models",
            "multi_model_episodes": "Multi-Model Episodes",
        }
    )

    st.dataframe(
        summary_display,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# RISK BY MACHINE
# ============================================================

if not machine_alerts.empty:
    st.subheader("📈 Maximum Risk by Machine")

    risk_chart = (
        machine_alerts.groupby("machine_id", as_index=True)["max_risk_score"]
        .max()
        .sort_values(ascending=False)
    )

    st.bar_chart(risk_chart)


# ============================================================
# MODEL CONTRIBUTION
# ============================================================

if not filtered_confirmed.empty:
    st.subheader("🤖 Model Contribution")

    model_counts = (
        filtered_confirmed["model"]
        .value_counts()
        .rename_axis("model")
        .rename("confirmed_alerts")
    )

    st.bar_chart(model_counts)


# ============================================================
# ALERT TIMELINE
# ============================================================

if not filtered_alerts.empty:
    st.subheader("⏱️ Alert Timeline")

    timeline = filtered_alerts[
        ["episode_start", "max_risk_score"]
    ].copy()

    timeline = timeline.set_index("episode_start")

    st.line_chart(timeline)


# ============================================================
# MODEL RISK SCORES
# ============================================================

if not inference.empty:
    st.subheader("🧠 Model Risk Scores")

    available_models = [
        col for col in
        ["lstm_risk", "gru_risk", "xgboost_risk"]
        if col in inference.columns
    ]

    if available_models:
        risk_data = inference.copy()

        if selected_machine != "All Machines" and "machine_id" in risk_data:
            risk_data = risk_data[
                risk_data["machine_id"].astype(str)
                == selected_machine
            ]

        if "timestamp" in risk_data.columns:
            risk_data = risk_data.set_index("timestamp")

        st.line_chart(
            risk_data[available_models]
        )


# ============================================================
# ALERT DETAILS
# ============================================================

with st.expander("🔍 Alert Details / Audit Information"):
    if not filtered_alerts.empty:
        st.dataframe(
            filtered_alerts,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No alert details available.")


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Predictive Maintenance | Frozen model thresholds | "
    "Operational confirmation + machine-level aggregation"
)
