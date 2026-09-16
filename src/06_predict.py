"""
06_predict.py

Inference utility for the trained predictive-maintenance model.

This file can be imported by the dashboard or another
production inference service.

Run:

    python src/06_predict.py
"""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "best_lstm.keras"
)

SCALER_PATH = (
    PROJECT_ROOT
    / "models"
    / "scaler.pkl"
)

FEATURE_PATH = (
    PROJECT_ROOT
    / "models"
    / "feature_columns.json"
)

THRESHOLD_PATH = (
    PROJECT_ROOT
    / "models"
    / "threshold.json"
)

FEATURE_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features.csv"
)

SEQUENCE_LENGTH = 120


# ---------------------------------------------------------------------
# MODEL SERVICE
# ---------------------------------------------------------------------

class PredictiveMaintenanceModel:

    def __init__(self):

        self.model = load_model(
            MODEL_PATH
        )

        self.scaler = joblib.load(
            SCALER_PATH
        )

        with open(
            FEATURE_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            self.feature_columns = json.load(
                f
            )

        with open(
            THRESHOLD_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            self.threshold = float(
                json.load(f)["threshold"]
            )

    # -----------------------------------------------------------------
    # VALIDATE INPUT
    # -----------------------------------------------------------------

    def validate_input(
        self,
        data: pd.DataFrame
    ):

        missing = [
            column
            for column
            in self.feature_columns
            if column not in data.columns
        ]

        if missing:

            raise ValueError(
                "Missing model features:\n"
                + "\n".join(missing)
            )

        if len(data) < SEQUENCE_LENGTH:

            raise ValueError(
                f"At least "
                f"{SEQUENCE_LENGTH} "
                "rows are required."
            )

    # -----------------------------------------------------------------
    # PREDICT
    # -----------------------------------------------------------------

    def predict(
        self,
        data: pd.DataFrame
    ):

        self.validate_input(
            data
        )

        latest = (
            data
            .sort_values("timestamp")
            .tail(SEQUENCE_LENGTH)
            .copy()
        )

        X = latest[
            self.feature_columns
        ].astype(float)

        X_scaled = self.scaler.transform(
            X
        )

        X_sequence = np.expand_dims(
            X_scaled,
            axis=0
        )

        probability = float(
            self.model.predict(
                X_sequence,
                verbose=0
            )[0][0]
        )

        alert = (
            probability
            >= self.threshold
        )

        if probability >= 0.80:

            severity = "CRITICAL"

        elif probability >= 0.60:

            severity = "HIGH"

        elif probability >= self.threshold:

            severity = "WARNING"

        else:

            severity = "NORMAL"

        return {

            "failure_probability": probability,

            "threshold": self.threshold,

            "alert": bool(alert),

            "severity": severity,

            "history_minutes": SEQUENCE_LENGTH,

            "timestamp": str(
                latest["timestamp"].iloc[-1]
            )
        }


# ---------------------------------------------------------------------
# COMMAND-LINE TEST
# ---------------------------------------------------------------------

def main():

    print("=" * 80)
    print("PREDICTIVE MAINTENANCE INFERENCE")
    print("=" * 80)

    if not FEATURE_DATA_PATH.exists():

        raise FileNotFoundError(
            "features.csv not found.\n"
            "Run feature engineering first."
        )

    data = pd.read_csv(
        FEATURE_DATA_PATH
    )

    data["timestamp"] = pd.to_datetime(
        data["timestamp"]
    )

    machine_ids = (
        data["machine_id"]
        .unique()
    )

    if len(machine_ids) == 0:

        raise ValueError(
            "No machine IDs found."
        )

    machine_id = machine_ids[0]

    machine_data = data[
        data["machine_id"] == machine_id
    ].copy()

    service = PredictiveMaintenanceModel()

    result = service.predict(
        machine_data
    )

    print(
        f"\nMachine: {machine_id}"
    )

    print(
        f"Failure probability: "
        f"{result['failure_probability']:.4f}"
    )

    print(
        f"Threshold: "
        f"{result['threshold']:.4f}"
    )

    print(
        f"Alert: "
        f"{result['alert']}"
    )

    print(
        f"Severity: "
        f"{result['severity']}"
    )


if __name__ == "__main__":
    main()