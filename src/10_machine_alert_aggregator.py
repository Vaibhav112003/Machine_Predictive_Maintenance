"""
10_machine_alert_aggregator.py

Machine-level alert aggregation for predictive maintenance.

Purpose
-------
Convert model-level confirmed alerts from 09_alert_engine.py into
machine-level actionable alert episodes.

Why this layer exists
---------------------
LSTM, GRU and XGBoost can independently raise alerts for the same
underlying machine-risk episode. Treating those as three separate
maintenance incidents can overstate the number of operational events.

This script:
    1. Loads confirmed model alerts from 09_alert_engine.py.
    2. Groups alerts by machine_id.
    3. Merges alerts occurring within AGGREGATION_WINDOW_MINUTES.
    4. Preserves all contributing models and their risk scores.
    5. Produces one machine-level alert episode per grouped period.
    6. Saves both detailed episode output and a machine summary.

IMPORTANT
---------
- No model training.
- No threshold tuning.
- Uses the already-confirmed alerts from 09_alert_engine.py.
- The aggregation window is an operational policy, not an ML parameter.
- Model-level alerts remain available in contributing_alert_ids/models.
"""

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORTS_DIR = PROJECT_ROOT / "reports"
INPUT_FILE = REPORTS_DIR / "alert_engine" / "confirmed_alerts.csv"

OUTPUT_DIR = REPORTS_DIR / "alert_engine"

MACHINE_ALERT_OUTPUT = OUTPUT_DIR / "machine_alerts.csv"
MACHINE_SUMMARY_OUTPUT = OUTPUT_DIR / "machine_alert_machine_summary.csv"

# Operational grouping policy.
# Confirmed alerts for the same machine that are close in time
# are treated as one machine-risk episode.
AGGREGATION_WINDOW_MINUTES = 15


# ============================================================
# VALIDATION / INPUT
# ============================================================

def check_input_file():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file was not found:\n{INPUT_FILE}\n\n"
            "Run first:\n"
            "python src/09_alert_engine.py"
        )


def load_confirmed_alerts():
    check_input_file()

    df = pd.read_csv(INPUT_FILE)

    required = [
        "alert_id",
        "machine_id",
        "model",
        "alert_timestamp",
        "risk_score",
        "threshold",
        "confirmation_count",
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            "confirmed_alerts.csv is missing required columns:\n"
            + "\n".join(f"  - {col}" for col in missing)
        )

    df["alert_timestamp"] = pd.to_datetime(
        df["alert_timestamp"],
        errors="coerce"
    )

    if df["alert_timestamp"].isna().any():
        bad = int(df["alert_timestamp"].isna().sum())
        raise ValueError(
            f"{bad} rows contain invalid alert timestamps."
        )

    if df["machine_id"].isna().any():
        raise ValueError("machine_id contains missing values.")

    if df["model"].isna().any():
        raise ValueError("model contains missing values.")

    if df["risk_score"].isna().any():
        raise ValueError("risk_score contains missing values.")

    if df.empty:
        print("No confirmed alerts found.")
        return df

    df = df.sort_values(
        ["machine_id", "alert_timestamp", "model"]
    ).reset_index(drop=True)

    return df


# ============================================================
# MACHINE-LEVEL EPISODE AGGREGATION
# ============================================================

def aggregate_machine_alerts(df):
    """
    Group confirmed model alerts into machine-level risk episodes.

    Rule:
        Same machine + alert gap <= AGGREGATION_WINDOW_MINUTES
        => same machine-risk episode.

    The comparison is made with the previous confirmed alert in the
    same machine. This creates chronological, non-overlapping episodes.
    """

    if df.empty:
        return pd.DataFrame()

    episode_rows = []

    for machine_id, group in df.groupby("machine_id", sort=True):
        group = group.sort_values(
            "alert_timestamp"
        ).reset_index(drop=True)

        current_alerts = []
        previous_timestamp = None
        episode_number = 0

        for _, alert in group.iterrows():
            timestamp = alert["alert_timestamp"]

            if (
                previous_timestamp is None
                or timestamp - previous_timestamp
                > pd.Timedelta(minutes=AGGREGATION_WINDOW_MINUTES)
            ):
                # Close previous episode.
                if current_alerts:
                    episode_rows.append(
                        build_episode_record(
                            machine_id,
                            episode_number,
                            current_alerts
                        )
                    )

                episode_number += 1
                current_alerts = []

            current_alerts.append(alert)
            previous_timestamp = timestamp

        # Close final episode.
        if current_alerts:
            episode_rows.append(
                build_episode_record(
                    machine_id,
                    episode_number,
                    current_alerts
                )
            )

    result = pd.DataFrame(episode_rows)

    if not result.empty:
        result = result.sort_values(
            ["machine_id", "episode_start"]
        ).reset_index(drop=True)

        result.insert(
            0,
            "machine_alert_id",
            [
                f"MA-{i:05d}"
                for i in range(1, len(result) + 1)
            ]
        )

    return result


def build_episode_record(machine_id, episode_number, alerts):
    """
    Build one machine-level alert episode from model-level alerts.
    """

    alert_df = pd.DataFrame(alerts)

    start_time = alert_df["alert_timestamp"].min()
    end_time = alert_df["alert_timestamp"].max()

    models = sorted(
        alert_df["model"].astype(str).unique().tolist()
    )

    model_risk = {}

    for _, row in alert_df.iterrows():
        model = str(row["model"])
        risk = float(row["risk_score"])

        # Keep the maximum risk observed for each model
        # during this machine-risk episode.
        model_risk[model] = max(
            model_risk.get(model, float("-inf")),
            risk
        )

    contributing_models = ",".join(models)

    model_risk_text = ";".join(
        f"{model}:{model_risk[model]:.6f}"
        for model in models
    )

    max_risk = float(
        alert_df["risk_score"].max()
    )

    max_risk_model = str(
        alert_df.loc[
            alert_df["risk_score"].idxmax(),
            "model"
        ]
    )

    return {
        "machine_id": machine_id,
        "episode_number": episode_number,
        "episode_start": start_time,
        "episode_end": end_time,
        "episode_duration_minutes": (
            (end_time - start_time).total_seconds() / 60.0
        ),
        "contributing_models": contributing_models,
        "model_consensus_count": len(models),
        "total_confirmed_model_alerts": len(alert_df),
        "max_risk_score": max_risk,
        "max_risk_model": max_risk_model,
        "model_max_risk_scores": model_risk_text,
        "contributing_alert_ids": ",".join(
            alert_df["alert_id"].astype(str).tolist()
        ),
        "aggregation_window_minutes": (
            AGGREGATION_WINDOW_MINUTES
        ),
    }


# ============================================================
# MACHINE SUMMARY
# ============================================================

def build_machine_summary(machine_alerts):
    if machine_alerts.empty:
        return pd.DataFrame(
            columns=[
                "machine_id",
                "machine_alert_episodes",
                "total_confirmed_model_alerts",
                "max_risk_score",
                "models_contributed",
                "multi_model_episodes",
            ]
        )

    rows = []

    for machine_id, group in machine_alerts.groupby(
        "machine_id",
        sort=True
    ):
        all_models = set()

        for value in group["contributing_models"]:
            all_models.update(
                str(value).split(",")
            )

        rows.append({
            "machine_id": machine_id,
            "machine_alert_episodes": len(group),
            "total_confirmed_model_alerts": int(
                group["total_confirmed_model_alerts"].sum()
            ),
            "max_risk_score": float(
                group["max_risk_score"].max()
            ),
            "models_contributed": ",".join(
                sorted(all_models)
            ),
            "multi_model_episodes": int(
                (group["model_consensus_count"] >= 2).sum()
            ),
        })

    return pd.DataFrame(rows)


# ============================================================
# PRINT RESULTS
# ============================================================

def print_summary(
    confirmed_df,
    machine_alerts,
    machine_summary
):
    print("\n" + "=" * 78)
    print("10 - MACHINE-LEVEL ALERT AGGREGATOR")
    print("=" * 78)

    print(f"\nInput confirmed model alerts : {len(confirmed_df)}")
    if not confirmed_df.empty:
        print(
            f"Machines                    : "
            f"{confirmed_df['machine_id'].nunique()}"
        )
    else:
        print("Machines                    : 0")

    print(
        f"Aggregation window          : "
        f"{AGGREGATION_WINDOW_MINUTES} minutes"
    )

    print(
        "\nMachine-level alert episodes : "
        f"{len(machine_alerts)}"
    )

    if machine_alerts.empty:
        print("\nNo machine-level alerts to aggregate.")
        print("=" * 78)
        return

    print("\nMachine alert episodes:")

    display_cols = [
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

    print(
        machine_alerts[display_cols].to_string(
            index=False
        )
    )

    print("\nMachine summary:")

    print(
        machine_summary.to_string(
            index=False
        )
    )

    print("\nInterpretation:")
    print(
        "  - One machine-level episode may contain alerts from "
        "multiple models."
    )
    print(
        "  - model_consensus_count shows how many distinct models "
        "contributed."
    )
    print(
        "  - contributing_alert_ids preserves the original "
        "model-level alerts for auditability."
    )
    print(
        "  - No model predictions or thresholds were changed."
    )

    print("=" * 78)


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_outputs(machine_alerts, machine_summary):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    machine_alerts.to_csv(
        MACHINE_ALERT_OUTPUT,
        index=False
    )

    machine_summary.to_csv(
        MACHINE_SUMMARY_OUTPUT,
        index=False
    )

    print("\nSaved:")
    print(f"  {MACHINE_ALERT_OUTPUT}")
    print(f"  {MACHINE_SUMMARY_OUTPUT}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 78)
    print("10 - MACHINE-LEVEL ALERT AGGREGATOR")
    print("=" * 78)

    print(f"\nProject root : {PROJECT_ROOT}")
    print(f"Input        : {INPUT_FILE}")

    # 1. Load confirmed model-level alerts.
    confirmed_df = load_confirmed_alerts()

    if confirmed_df.empty:
        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        empty_alerts = pd.DataFrame()
        empty_summary = build_machine_summary(empty_alerts)

        save_outputs(
            empty_alerts,
            empty_summary
        )

        return

    # 2. Aggregate model alerts into machine-risk episodes.
    machine_alerts = aggregate_machine_alerts(
        confirmed_df
    )

    # 3. Build per-machine summary.
    machine_summary = build_machine_summary(
        machine_alerts
    )

    # 4. Save results.
    save_outputs(
        machine_alerts,
        machine_summary
    )

    # 5. Print operational summary.
    print_summary(
        confirmed_df,
        machine_alerts,
        machine_summary
    )

    print("\nSUCCESS")
    print(
        "Machine-level alerts are ready for the next "
        "production/API layer."
    )


if __name__ == "__main__":
    main()
