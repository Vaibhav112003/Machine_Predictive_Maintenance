"""
07_event_level_evaluation.py

Rigorous frozen-model evaluation for the Industrial Predictive Maintenance case study.

IMPORTANT
---------
- NO model training is performed.
- LSTM, GRU and XGBoost are loaded from disk.
- Validation-selected thresholds remain frozen.
- TEST data is used only for final evaluation.
- The saved sequence convention is:
      X[i] = rows[i-LOOKBACK:i]
      y[i] = target at row[i]
  therefore metadata starts at group.iloc[LOOKBACK:].
- Event-level evaluation is built from the FULL feature dataset, not only
  the sequence metadata. This prevents the previous "1 test event" artifact
  caused by detecting events only inside the truncated metadata range.
- An event is eligible only when its complete 2-6 hour warning window is
  observable inside the available TEST prediction range.
- is_maintenance_event=1 is treated as a maintenance/failure-proxy event,
  because the supplied case study defines it as active maintenance/repair.
- No test-set threshold tuning is performed.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb

from tensorflow.keras.models import load_model

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "features.csv"
SEQUENCE_PATH = PROJECT_ROOT / "data" / "processed" / "sequences.npz"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

LOOKBACK = 120

WARNING_MINUTES = 120
WARNING_MAX_MINUTES = 360

TEST_START = pd.Timestamp("2024-01-26 12:45:00")
TEST_END = pd.Timestamp("2024-01-30 17:59:00")

# Frozen validation-selected thresholds.
THRESHOLDS = {
    "LSTM": 0.050773,
    "GRU": 0.060276,
    "XGBoost": 0.033000,
}

# Operational alert simulation.
# These settings are NOT used to change model thresholds.
# They are reported separately from raw model metrics.
CONFIRMATION_COUNT = 3
ALERT_COOLDOWN_MINUTES = 60


# ============================================================
# UTILITY
# ============================================================

def print_section(title, width=100):
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def require_file(path, description):
    if not path.exists():
        raise FileNotFoundError(
            f"\n{description} not found:\n{path}"
        )


def wilson_interval(successes, trials, z=1.96):
    """95% Wilson confidence interval for a binomial proportion."""
    if trials == 0:
        return np.nan, np.nan

    p = successes / trials
    denominator = 1 + (z ** 2) / trials
    centre = (
        p + (z ** 2) / (2 * trials)
    ) / denominator

    margin = (
        z
        * np.sqrt(
            (
                p * (1 - p) / trials
                + (z ** 2) / (4 * trials ** 2)
            )
        )
        / denominator
    )

    return max(0.0, centre - margin), min(1.0, centre + margin)


# ============================================================
# LOAD FEATURE DATA
# ============================================================

def load_feature_data():
    require_file(DATA_PATH, "Feature dataset")

    print("\nLoading feature dataset...")

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["timestamp"],
    )

    required = [
        "timestamp",
        "machine_id",
        "is_maintenance_event",
        "failure_2_6h",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing)
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    if df["timestamp"].isna().any():
        raise ValueError(
            "Feature dataset contains invalid timestamps."
        )

    df = (
        df.sort_values(["machine_id", "timestamp"])
        .reset_index(drop=True)
    )

    print("Feature dataset shape:", df.shape)
    print("Machines:", df["machine_id"].unique().tolist())

    return df


# ============================================================
# LOAD SAVED SEQUENCES
# ============================================================

def load_sequences():
    require_file(SEQUENCE_PATH, "Saved sequence file")

    print("\nLoading saved sequences...")

    data = np.load(SEQUENCE_PATH)

    required = [
        "X_train",
        "y_train",
        "X_validation",
        "y_validation",
        "X_test",
        "y_test",
    ]

    missing = [name for name in required if name not in data.files]

    if missing:
        raise ValueError(
            "Missing arrays in sequences.npz:\n"
            + "\n".join(missing)
        )

    sequences = {
        name: data[name]
        for name in required
    }

    X_test = sequences["X_test"]
    y_test = sequences["y_test"]

    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    if X_test.ndim != 3:
        raise ValueError(
            f"X_test must be 3-dimensional. Received {X_test.shape}"
        )

    if X_test.shape[1] != LOOKBACK:
        raise ValueError(
            f"LOOKBACK mismatch. "
            f"Configured={LOOKBACK}, "
            f"X_test timesteps={X_test.shape[1]}"
        )

    if len(X_test) != len(y_test):
        raise ValueError(
            "X_test and y_test have different lengths."
        )

    return sequences


# ============================================================
# RECONSTRUCT EXACT TEST METADATA
# ============================================================

def create_test_metadata(df, expected_count):
    """
    Reconstruct the timestamp/machine corresponding to every X_test sequence.

    Saved sequence convention:
        X[i] = rows[i-LOOKBACK:i]
        y[i] = target at row[i]

    Therefore metadata begins at group.iloc[LOOKBACK:].
    """

    print("\nCreating exact test metadata...")

    test_df = df[
        (df["timestamp"] >= TEST_START)
        & (df["timestamp"] <= TEST_END)
    ].copy()

    if test_df.empty:
        raise ValueError(
            "No rows found in configured test period."
        )

    print(
        "Test rows before sequence alignment:",
        len(test_df),
    )

    metadata_parts = []

    for machine_id, group in test_df.groupby(
        "machine_id",
        sort=True,
    ):
        group = (
            group.sort_values("timestamp")
            .reset_index(drop=True)
        )

        print(
            f"  {machine_id}: {len(group)} test rows"
        )

        if len(group) <= LOOKBACK:
            print(
                f"  WARNING: {machine_id} does not have enough "
                "rows for sequence metadata."
            )
            continue

        machine_metadata = group.iloc[LOOKBACK:].copy()
        metadata_parts.append(machine_metadata)

    if not metadata_parts:
        raise ValueError(
            "No test sequence metadata was created."
        )

    metadata = pd.concat(
        metadata_parts,
        ignore_index=True,
    )

    metadata = (
        metadata.sort_values(["machine_id", "timestamp"])
        .reset_index(drop=True)
    )

    print(
        "\nTest metadata sequences:",
        len(metadata),
    )

    print("\nTest metadata by machine:")
    print(
        metadata["machine_id"]
        .value_counts()
        .sort_index()
    )

    if metadata["failure_2_6h"].isna().any():
        raise ValueError(
            "Metadata contains missing failure_2_6h targets."
        )

    if len(metadata) != expected_count:
        raise ValueError(
            "\nEXACT METADATA RECONSTRUCTION FAILED\n"
            f"Expected X_test sequences : {expected_count}\n"
            f"Metadata rows              : {len(metadata)}"
        )

    print("\n✓ Exact metadata count matches X_test.")

    return metadata


# ============================================================
# ALIGNMENT VALIDATION
# ============================================================

def validate_alignment(metadata, X_test, y_test):
    print_section("SEQUENCE / METADATA ALIGNMENT CHECK")

    print("X_test sequences :", len(X_test))
    print("y_test labels    :", len(y_test))
    print("metadata rows    :", len(metadata))

    if len(X_test) != len(y_test):
        raise ValueError(
            "X_test and y_test lengths do not match."
        )

    if len(metadata) != len(X_test):
        raise ValueError(
            "SEQUENCE / METADATA ALIGNMENT MISMATCH\n"
            f"X_test={len(X_test)}, "
            f"y_test={len(y_test)}, "
            f"metadata={len(metadata)}"
        )

    required = [
        "timestamp",
        "machine_id",
        "is_maintenance_event",
        "failure_2_6h",
    ]

    missing = [c for c in required if c not in metadata.columns]

    if missing:
        raise ValueError(
            "Metadata missing columns:\n"
            + "\n".join(missing)
        )

    if metadata["timestamp"].isna().any():
        raise ValueError("Metadata contains missing timestamps.")

    if metadata["machine_id"].isna().any():
        raise ValueError("Metadata contains missing machine IDs.")

    print("\n✓ Alignment: PASS")
    print("✓ Every test prediction has exactly one metadata row.")


# ============================================================
# MODEL PATH HELPERS
# ============================================================

def find_xgboost_model():
    candidates = [
        MODEL_DIR / "xgboost_model.json",
        MODEL_DIR / "xgboost_model.pkl",
        MODEL_DIR / "best_xgboost.pkl",
        MODEL_DIR / "xgb_model.pkl",
        MODEL_DIR / "xgboost_model.joblib",
        MODEL_DIR / "best_xgboost.joblib",
        MODEL_DIR / "xgb_model.joblib",
    ]

    for path in candidates:
        if path.exists():
            print(f"✓ XGBoost model found: {path}")
            return path

    raise FileNotFoundError(
        "\nXGBoost model could not be found.\n\n"
        "Checked:\n"
        + "\n".join(str(p) for p in candidates)
    )


# ============================================================
# LSTM PREDICTION
# ============================================================

def predict_lstm(X_test):
    model_path = MODEL_DIR / "best_lstm.keras"
    require_file(model_path, "LSTM model")

    print("\nLoading frozen LSTM model...")

    model = load_model(model_path)

    probabilities = (
        model.predict(
            X_test,
            verbose=0,
        )
        .ravel()
    )

    if len(probabilities) != len(X_test):
        raise ValueError(
            "LSTM prediction length mismatch."
        )

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "LSTM produced NaN/Inf probabilities."
        )

    print("LSTM predictions:", len(probabilities))

    return probabilities


# ============================================================
# GRU PREDICTION
# ============================================================

def predict_gru(X_test):
    model_path = MODEL_DIR / "best_gru.keras"
    require_file(model_path, "GRU model")

    print("\nLoading frozen GRU model...")

    model = load_model(model_path)

    probabilities = (
        model.predict(
            X_test,
            verbose=0,
        )
        .ravel()
    )

    if len(probabilities) != len(X_test):
        raise ValueError(
            "GRU prediction length mismatch."
        )

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "GRU produced NaN/Inf probabilities."
        )

    print("GRU predictions:", len(probabilities))

    return probabilities


# ============================================================
# XGBOOST FEATURE TRANSFORMATION
# ============================================================

def create_xgb_features(X):
    """
    Convert:
        (samples, 120, 174)

    into:
        (samples, 1740)

    using the SAME 10 statistics used during XGBoost training:

        mean
        std
        min
        max
        median
        first
        last
        change
        absolute change
        range
    """

    if X.ndim != 3:
        raise ValueError(
            f"Expected 3D X. Received {X.shape}"
        )

    n_samples, timesteps, n_features = X.shape

    print("\nCreating XGBoost temporal features...")
    print("Samples   :", n_samples)
    print("Timesteps :", timesteps)
    print("Features  :", n_features)

    transformed = np.empty(
        (n_samples, n_features * 10),
        dtype=np.float32,
    )

    for i in range(n_samples):
        sequence = X[i]

        mean = np.mean(sequence, axis=0)
        std = np.std(sequence, axis=0)
        minimum = np.min(sequence, axis=0)
        maximum = np.max(sequence, axis=0)
        median = np.median(sequence, axis=0)

        first = sequence[0]
        last = sequence[-1]

        change = last - first
        absolute_change = np.abs(change)
        data_range = maximum - minimum

        transformed[i] = np.concatenate([
            mean,
            std,
            minimum,
            maximum,
            median,
            first,
            last,
            change,
            absolute_change,
            data_range,
        ])

    expected = n_features * 10

    if transformed.shape[1] != expected:
        raise ValueError(
            f"XGBoost feature count mismatch. "
            f"Expected={expected}, "
            f"Actual={transformed.shape[1]}"
        )

    print("XGBoost feature shape:", transformed.shape)

    return transformed


# ============================================================
# XGBOOST PREDICTION
# ============================================================

def predict_xgboost(X_test):
    model_path = find_xgboost_model()

    print("\nLoading frozen XGBoost model...")
    print("Model:", model_path.name)

    X_xgb = create_xgb_features(X_test)

    if model_path.suffix.lower() == ".json":
        model = xgb.XGBClassifier()
        model.load_model(str(model_path))
    else:
        import joblib
        model = joblib.load(model_path)

    if hasattr(model, "n_features_in_"):
        expected = int(model.n_features_in_)
        actual = int(X_xgb.shape[1])

        print("Expected XGBoost features:", expected)
        print("Actual XGBoost features  :", actual)

        if expected != actual:
            raise ValueError(
                f"XGBoost feature mismatch. "
                f"Model expects {expected}; "
                f"generated {actual}."
            )
    else:
        print(
            "Expected XGBoost features:",
            X_xgb.shape[1],
        )
        print(
            "Actual XGBoost features  :",
            X_xgb.shape[1],
        )

    probabilities = model.predict_proba(X_xgb)[:, 1]

    if len(probabilities) != len(X_test):
        raise ValueError(
            "XGBoost prediction length mismatch."
        )

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "XGBoost produced NaN/Inf probabilities."
        )

    print("XGBoost predictions:", len(probabilities))

    return probabilities


# ============================================================
# CREATE MAINTENANCE EVENTS FROM FULL DATA
# ============================================================

def create_events(df):
    """
    Convert contiguous is_maintenance_event=1 rows into maintenance
    episodes using the FULL feature dataset.
    """

    events = []

    for machine_id, group in df.groupby(
        "machine_id",
        sort=True,
    ):
        group = (
            group.sort_values("timestamp")
            .reset_index(drop=True)
        )

        active = (
            group["is_maintenance_event"]
            .astype(int)
            .to_numpy()
        )

        timestamps = group["timestamp"].to_numpy()

        in_event = False
        start_time = None
        end_time = None

        for timestamp, value in zip(
            timestamps,
            active,
        ):
            timestamp = pd.Timestamp(timestamp)

            if value == 1 and not in_event:
                in_event = True
                start_time = timestamp
                end_time = timestamp

            elif value == 1 and in_event:
                end_time = timestamp

            elif value == 0 and in_event:
                events.append({
                    "machine_id": machine_id,
                    "event_start": start_time,
                    "event_end": end_time,
                })

                in_event = False
                start_time = None
                end_time = None

        if in_event:
            events.append({
                "machine_id": machine_id,
                "event_start": start_time,
                "event_end": end_time,
            })

    events_df = pd.DataFrame(events)

    if not events_df.empty:
        events_df = (
            events_df
            .sort_values(
                ["machine_id", "event_start"]
            )
            .reset_index(drop=True)
        )

    return events_df


# ============================================================
# WARNING WINDOW
# ============================================================

def get_warning_window(event_start):
    return (
        event_start - pd.Timedelta(
            minutes=WARNING_MAX_MINUTES
        ),
        event_start - pd.Timedelta(
            minutes=WARNING_MINUTES
        ),
    )


# ============================================================
# ELIGIBLE EVENT SELECTION
# ============================================================

def select_eligible_test_events(
    full_events,
    prediction_results,
):
    """
    Select only events whose COMPLETE 2-6 hour warning window is
    observable in the prediction dataset.

    This avoids counting boundary-truncated events.
    """

    prediction_start = (
        prediction_results["timestamp"].min()
    )

    prediction_end = (
        prediction_results["timestamp"].max()
    )

    eligible = []

    for _, event in full_events.iterrows():
        event_start = pd.Timestamp(
            event["event_start"]
        )

        # Event itself must belong to the configured test period.
        if event_start < TEST_START:
            continue

        if event_start > TEST_END:
            continue

        window_start, window_end = get_warning_window(
            event_start
        )

        # Complete warning window must be observable.
        if window_start < prediction_start:
            continue

        if window_end > prediction_end:
            continue

        record = event.to_dict()
        record["warning_window_start"] = window_start
        record["warning_window_end"] = window_end
        record["eligible"] = 1

        eligible.append(record)

    eligible_df = pd.DataFrame(eligible)

    if eligible_df.empty:
        return eligible_df

    return (
        eligible_df
        .sort_values(
            ["machine_id", "event_start"]
        )
        .reset_index(drop=True)
    )


# ============================================================
# SAMPLE-LEVEL METRICS
# ============================================================

def calculate_sample_metrics(
    y_true,
    probabilities,
    threshold,
    model_name,
):
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities)

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    metrics = {
        "model": model_name,
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_true,
                probabilities,
            )
        ),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "positive_samples": int(y_true.sum()),
        "negative_samples": int((y_true == 0).sum()),
    }

    return metrics


def print_sample_metrics(metrics):
    print_section(
        f"{metrics['model']} SAMPLE-LEVEL TEST METRICS"
    )

    print(
        f"Frozen threshold : "
        f"{metrics['threshold']:.6f}"
    )
    print(
        f"Accuracy         : "
        f"{metrics['accuracy']:.4f}"
    )
    print(
        f"Precision        : "
        f"{metrics['precision']:.4f}"
    )
    print(
        f"Recall           : "
        f"{metrics['recall']:.4f}"
    )
    print(
        f"F1               : "
        f"{metrics['f1']:.4f}"
    )
    print(
        f"ROC-AUC          : "
        f"{metrics['roc_auc']:.4f}"
    )
    print(
        f"PR-AUC           : "
        f"{metrics['pr_auc']:.4f}"
    )

    print("\nConfusion matrix:")
    print(
        f"TN={metrics['true_negative']}  "
        f"FP={metrics['false_positive']}"
    )
    print(
        f"FN={metrics['false_negative']}  "
        f"TP={metrics['true_positive']}"
    )


# ============================================================
# EVENT-LEVEL + ALERT-LEVEL EVALUATION
# ============================================================

def evaluate_events(
    prediction_results,
    eligible_events,
    threshold,
    model_name,
):
    """
    Evaluate frozen predictions against eligible maintenance events.

    Raw model alerts:
        probability >= frozen threshold.

    Valid event alert:
        alert timestamp is inside the event's 2-6 hour warning window.

    Operational alert simulation:
        consecutive confirmation + cooldown.
        Reported separately and does not modify model metrics.
    """

    results = prediction_results.copy()

    results["alert"] = (
        results["probability"] >= threshold
    ).astype(int)

    print_section(
        f"{model_name} EVENT-LEVEL EVALUATION"
    )

    print(
        f"Frozen threshold       : {threshold:.6f}"
    )
    print(
        f"Eligible test events   : "
        f"{len(eligible_events)}"
    )

    # --------------------------------------------------------
    # Event-level evaluation
    # --------------------------------------------------------

    detected_events = 0
    lead_times = []
    event_records = []

    for _, event in eligible_events.iterrows():
        machine = event["machine_id"]

        event_start = pd.Timestamp(
            event["event_start"]
        )

        event_end = pd.Timestamp(
            event["event_end"]
        )

        window_start = pd.Timestamp(
            event["warning_window_start"]
        )

        window_end = pd.Timestamp(
            event["warning_window_end"]
        )

        candidates = results[
            (results["machine_id"] == machine)
            & (results["timestamp"] >= window_start)
            & (results["timestamp"] <= window_end)
            & (results["alert"] == 1)
        ].sort_values("timestamp")

        if len(candidates) > 0:
            detected_events += 1

            first_alert = pd.Timestamp(
                candidates.iloc[0]["timestamp"]
            )

            last_alert = pd.Timestamp(
                candidates.iloc[-1]["timestamp"]
            )

            lead_time = (
                event_start - first_alert
            ).total_seconds() / 60.0

            lead_times.append(lead_time)

            event_records.append({
                "model": model_name,
                "machine_id": machine,
                "event_start": event_start,
                "event_end": event_end,
                "warning_window_start": window_start,
                "warning_window_end": window_end,
                "detected": 1,
                "first_alert": first_alert,
                "last_alert": last_alert,
                "lead_time_minutes": lead_time,
                "number_of_alerts_in_window": len(
                    candidates
                ),
            })

        else:
            event_records.append({
                "model": model_name,
                "machine_id": machine,
                "event_start": event_start,
                "event_end": event_end,
                "warning_window_start": window_start,
                "warning_window_end": window_end,
                "detected": 0,
                "first_alert": pd.NaT,
                "last_alert": pd.NaT,
                "lead_time_minutes": np.nan,
                "number_of_alerts_in_window": 0,
            })

    # --------------------------------------------------------
    # Raw alert-level analysis
    # --------------------------------------------------------

    alert_rows = results[
        results["alert"] == 1
    ].copy()

    total_alerts = len(alert_rows)
    valid_alerts = 0
    false_alerts = 0

    for _, alert in alert_rows.iterrows():
        machine = alert["machine_id"]
        alert_time = pd.Timestamp(
            alert["timestamp"]
        )

        machine_events = eligible_events[
            eligible_events["machine_id"] == machine
        ]

        is_valid = False

        for _, event in machine_events.iterrows():
            window_start = pd.Timestamp(
                event["warning_window_start"]
            )

            window_end = pd.Timestamp(
                event["warning_window_end"]
            )

            if (
                alert_time >= window_start
                and alert_time <= window_end
            ):
                is_valid = True
                break

        if is_valid:
            valid_alerts += 1
        else:
            false_alerts += 1

    if valid_alerts + false_alerts != total_alerts:
        raise RuntimeError(
            "Raw alert accounting error."
        )

    # --------------------------------------------------------
    # Operational alert simulation
    # --------------------------------------------------------

    operational_alerts = generate_operational_alerts(
        results,
        threshold=threshold,
        confirmation_count=CONFIRMATION_COUNT,
        cooldown_minutes=ALERT_COOLDOWN_MINUTES,
    )

    operational_total = len(
        operational_alerts
    )

    operational_valid = 0

    for _, alert in operational_alerts.iterrows():
        machine = alert["machine_id"]
        alert_time = pd.Timestamp(
            alert["timestamp"]
        )

        machine_events = eligible_events[
            eligible_events["machine_id"] == machine
        ]

        for _, event in machine_events.iterrows():
            if (
                alert_time
                >= pd.Timestamp(
                    event["warning_window_start"]
                )
                and alert_time
                <= pd.Timestamp(
                    event["warning_window_end"]
                )
            ):
                operational_valid += 1
                break

    operational_false = (
        operational_total - operational_valid
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    total_events = len(eligible_events)

    missed_events = (
        total_events - detected_events
    )

    event_detection_rate = (
        detected_events / total_events
        if total_events
        else np.nan
    )

    event_ci_low, event_ci_high = wilson_interval(
        detected_events,
        total_events,
    )

    false_alarm_rate = (
        false_alerts / total_alerts
        if total_alerts
        else np.nan
    )

    false_alarm_ci_low, false_alarm_ci_high = (
        wilson_interval(
            false_alerts,
            total_alerts,
        )
    )

    valid_alert_rate = (
        valid_alerts / total_alerts
        if total_alerts
        else np.nan
    )

    alerts_per_event = (
        total_alerts / total_events
        if total_events
        else np.nan
    )

    operational_false_alarm_rate = (
        operational_false / operational_total
        if operational_total
        else np.nan
    )

    operational_valid_rate = (
        operational_valid / operational_total
        if operational_total
        else np.nan
    )

    if lead_times:
        mean_lead_time = float(
            np.mean(lead_times)
        )
        median_lead_time = float(
            np.median(lead_times)
        )
        min_lead_time = float(
            np.min(lead_times)
        )
        max_lead_time = float(
            np.max(lead_times)
        )
    else:
        mean_lead_time = np.nan
        median_lead_time = np.nan
        min_lead_time = np.nan
        max_lead_time = np.nan

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        f"\nTotal eligible events      : "
        f"{total_events}"
    )
    print(
        f"Detected events            : "
        f"{detected_events}"
    )
    print(
        f"Missed events              : "
        f"{missed_events}"
    )
    print(
        f"Event detection rate       : "
        f"{event_detection_rate:.2%}"
        if not np.isnan(event_detection_rate)
        else "Event detection rate       : N/A"
    )

    if not np.isnan(event_ci_low):
        print(
            f"95% Wilson CI              : "
            f"{event_ci_low:.2%} - "
            f"{event_ci_high:.2%}"
        )

    print(
        f"\nRaw threshold alerts       : "
        f"{total_alerts}"
    )
    print(
        f"Valid alerts in 2-6h       : "
        f"{valid_alerts}"
    )
    print(
        f"False alerts               : "
        f"{false_alerts}"
    )

    print(
        f"Valid alert rate           : "
        f"{valid_alert_rate:.2%}"
        if not np.isnan(valid_alert_rate)
        else "Valid alert rate           : N/A"
    )

    print(
        f"False alarm rate           : "
        f"{false_alarm_rate:.2%}"
        if not np.isnan(false_alarm_rate)
        else "False alarm rate           : N/A"
    )

    if not np.isnan(false_alarm_ci_low):
        print(
            f"95% false-alarm CI         : "
            f"{false_alarm_ci_low:.2%} - "
            f"{false_alarm_ci_high:.2%}"
        )

    print(
        f"Raw alerts per event       : "
        f"{alerts_per_event:.2f}"
        if not np.isnan(alerts_per_event)
        else "Raw alerts per event       : N/A"
    )

    print("\nOperational alert simulation:")
    print(
        f"Confirmation count         : "
        f"{CONFIRMATION_COUNT}"
    )
    print(
        f"Cooldown                   : "
        f"{ALERT_COOLDOWN_MINUTES} minutes"
    )
    print(
        f"Operational alerts         : "
        f"{operational_total}"
    )
    print(
        f"Operational valid alerts   : "
        f"{operational_valid}"
    )
    print(
        f"Operational false alerts   : "
        f"{operational_false}"
    )
    print(
        f"Operational valid rate     : "
        f"{operational_valid_rate:.2%}"
        if not np.isnan(operational_valid_rate)
        else "Operational valid rate     : N/A"
    )
    print(
        f"Operational false rate     : "
        f"{operational_false_alarm_rate:.2%}"
        if not np.isnan(
            operational_false_alarm_rate
        )
        else "Operational false rate     : N/A"
    )

    if lead_times:
        print(
            f"\nMean lead time             : "
            f"{mean_lead_time:.1f} min"
        )
        print(
            f"Median lead time           : "
            f"{median_lead_time:.1f} min"
        )
        print(
            f"Minimum lead time          : "
            f"{min_lead_time:.1f} min"
        )
        print(
            f"Maximum lead time          : "
            f"{max_lead_time:.1f} min"
        )
    else:
        print("\nLead time                  : N/A")

    summary = {
        "model": model_name,
        "threshold": float(threshold),
        "eligible_events": int(total_events),
        "detected_events": int(detected_events),
        "missed_events": int(missed_events),
        "event_detection_rate": float(
            event_detection_rate
        )
        if not np.isnan(event_detection_rate)
        else np.nan,
        "event_detection_ci95_low": float(
            event_ci_low
        )
        if not np.isnan(event_ci_low)
        else np.nan,
        "event_detection_ci95_high": float(
            event_ci_high
        )
        if not np.isnan(event_ci_high)
        else np.nan,
        "raw_alerts": int(total_alerts),
        "valid_alerts": int(valid_alerts),
        "false_alerts": int(false_alerts),
        "valid_alert_rate": float(
            valid_alert_rate
        )
        if not np.isnan(valid_alert_rate)
        else np.nan,
        "false_alarm_rate": float(
            false_alarm_rate
        )
        if not np.isnan(false_alarm_rate)
        else np.nan,
        "false_alarm_ci95_low": float(
            false_alarm_ci_low
        )
        if not np.isnan(false_alarm_ci_low)
        else np.nan,
        "false_alarm_ci95_high": float(
            false_alarm_ci_high
        )
        if not np.isnan(false_alarm_ci_high)
        else np.nan,
        "raw_alerts_per_event": float(
            alerts_per_event
        )
        if not np.isnan(alerts_per_event)
        else np.nan,
        "operational_alerts": int(
            operational_total
        ),
        "operational_valid_alerts": int(
            operational_valid
        ),
        "operational_false_alerts": int(
            operational_false
        ),
        "operational_valid_alert_rate": float(
            operational_valid_rate
        )
        if not np.isnan(operational_valid_rate)
        else np.nan,
        "operational_false_alarm_rate": float(
            operational_false_alarm_rate
        )
        if not np.isnan(
            operational_false_alarm_rate
        )
        else np.nan,
        "mean_lead_time_minutes": mean_lead_time,
        "median_lead_time_minutes": median_lead_time,
        "min_lead_time_minutes": min_lead_time,
        "max_lead_time_minutes": max_lead_time,
    }

    event_results = pd.DataFrame(
        event_records
    )

    return (
        summary,
        event_results,
        results,
        operational_alerts,
    )


# ============================================================
# OPERATIONAL ALERT ENGINE
# ============================================================

def generate_operational_alerts(
    results,
    threshold,
    confirmation_count=3,
    cooldown_minutes=60,
):
    """
    Convert raw threshold crossings into operational alert episodes.

    Rule:
      1. Probability must be >= threshold for N consecutive observations.
      2. Emit one alert at the first timestamp satisfying confirmation.
      3. Suppress new alerts for the cooldown period for that machine.

    This is an operational simulation only. It does not retrain or
    retune the underlying model.
    """

    alerts = []

    for machine_id, group in results.groupby(
        "machine_id",
        sort=True,
    ):
        group = (
            group.sort_values("timestamp")
            .reset_index(drop=True)
        )

        consecutive = 0
        cooldown_until = None

        for _, row in group.iterrows():
            timestamp = pd.Timestamp(
                row["timestamp"]
            )

            probability = float(
                row["probability"]
            )

            if probability >= threshold:
                consecutive += 1
            else:
                consecutive = 0
                continue

            if consecutive < confirmation_count:
                continue

            if (
                cooldown_until is not None
                and timestamp < cooldown_until
            ):
                continue

            alert_time = timestamp

            alerts.append({
                "machine_id": machine_id,
                "timestamp": alert_time,
                "probability": probability,
                "threshold": threshold,
                "confirmation_count": confirmation_count,
                "cooldown_minutes": cooldown_minutes,
            })

            cooldown_until = (
                alert_time
                + pd.Timedelta(
                    minutes=cooldown_minutes
                )
            )

            consecutive = 0

    if not alerts:
        return pd.DataFrame(
            columns=[
                "machine_id",
                "timestamp",
                "probability",
                "threshold",
                "confirmation_count",
                "cooldown_minutes",
            ]
        )

    return (
        pd.DataFrame(alerts)
        .sort_values(
            ["machine_id", "timestamp"]
        )
        .reset_index(drop=True)
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_model_results(
    model_name,
    sample_results,
    event_results,
    alert_results,
    operational_alerts,
):
    slug = model_name.lower()

    sample_path = (
        REPORT_DIR
        / f"{slug}_sample_metrics.json"
    )

    event_path = (
        REPORT_DIR
        / f"{slug}_event_results.csv"
    )

    alert_path = (
        REPORT_DIR
        / f"{slug}_alert_results.csv"
    )

    operational_path = (
        REPORT_DIR
        / f"{slug}_operational_alerts.csv"
    )

    with open(
        sample_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            sample_results,
            f,
            indent=2,
            allow_nan=True,
        )

    event_results.to_csv(
        event_path,
        index=False,
    )

    alert_results.to_csv(
        alert_path,
        index=False,
    )

    operational_alerts.to_csv(
        operational_path,
        index=False,
    )

    print(f"\nSaved {model_name} sample metrics:")
    print(sample_path)

    print(f"Saved {model_name} event results:")
    print(event_path)

    print(f"Saved {model_name} raw alert results:")
    print(alert_path)

    print(
        f"Saved {model_name} operational alerts:"
    )
    print(operational_path)


# ============================================================
# MAIN
# ============================================================

def main():
    print_section(
        "INDUSTRIAL PREDICTIVE MAINTENANCE\n"
        "RIGOROUS EVENT-LEVEL EVALUATION"
    )

    print("\nFROZEN MODELS")
    print("LSTM + GRU + XGBoost")
    print("\nNo model training will be performed.")

    # --------------------------------------------------------
    # 1. Load feature dataset
    # --------------------------------------------------------

    df = load_feature_data()

    # --------------------------------------------------------
    # 2. Load saved sequences
    # --------------------------------------------------------

    sequences = load_sequences()

    X_test = sequences["X_test"]
    y_test = sequences["y_test"]

    print("\nTest sequence shape:", X_test.shape)
    print("Test target shape  :", y_test.shape)

    # --------------------------------------------------------
    # 3. Reconstruct exact prediction metadata
    # --------------------------------------------------------

    metadata = create_test_metadata(
        df=df,
        expected_count=len(X_test),
    )

    validate_alignment(
        metadata=metadata,
        X_test=X_test,
        y_test=y_test,
    )

    # --------------------------------------------------------
    # 4. Generate frozen model predictions
    # --------------------------------------------------------

    print_section(
        "GENERATING FROZEN MODEL PREDICTIONS"
    )

    predictions = {}

    predictions["LSTM"] = predict_lstm(
        X_test
    )

    predictions["GRU"] = predict_gru(
        X_test
    )

    predictions["XGBoost"] = predict_xgboost(
        X_test
    )

    for model_name, probabilities in predictions.items():
        if len(probabilities) != len(metadata):
            raise ValueError(
                f"{model_name} prediction length mismatch. "
                f"Predictions={len(probabilities)}, "
                f"Metadata={len(metadata)}"
            )

    print(
        "\n✓ All model predictions aligned successfully."
    )

    # --------------------------------------------------------
    # 5. Create prediction table
    # --------------------------------------------------------

    prediction_base = metadata[
        [
            "timestamp",
            "machine_id",
            "is_maintenance_event",
            "failure_2_6h",
        ]
    ].copy()

    # --------------------------------------------------------
    # 6. Full-dataset event discovery
    # --------------------------------------------------------

    print_section(
        "FULL-DATA MAINTENANCE EVENT DISCOVERY"
    )

    all_events = create_events(df)

    print(
        "Maintenance/failure-proxy episodes in full dataset:",
        len(all_events),
    )

    if not all_events.empty:
        print("\nEvents by machine:")
        print(
            all_events["machine_id"]
            .value_counts()
            .sort_index()
        )

    # --------------------------------------------------------
    # 7. Eligible event selection
    # --------------------------------------------------------

    eligible_events = select_eligible_test_events(
        full_events=all_events,
        prediction_results=prediction_base,
    )

    print_section(
        "ELIGIBLE TEST EVENT SELECTION"
    )

    prediction_start = prediction_base[
        "timestamp"
    ].min()

    prediction_end = prediction_base[
        "timestamp"
    ].max()

    print(
        "Prediction availability:",
        prediction_start,
        "→",
        prediction_end,
    )

    print(
        "Configured test period:",
        TEST_START,
        "→",
        TEST_END,
    )

    print(
        "All maintenance episodes:",
        len(all_events),
    )

    print(
        "Eligible events with complete 2-6h window:",
        len(eligible_events),
    )

    if not eligible_events.empty:
        print("\nEligible events:")
        print(
            eligible_events[
                [
                    "machine_id",
                    "event_start",
                    "event_end",
                    "warning_window_start",
                    "warning_window_end",
                ]
            ].to_string(index=False)
        )
    else:
        print(
            "\nWARNING: No event has a complete "
            "2-6 hour warning window inside the "
            "available prediction range."
        )

    # --------------------------------------------------------
    # 8. Evaluate all models
    # --------------------------------------------------------

    summaries = []
    sample_summaries = []

    for model_name, probabilities in predictions.items():

        prediction_results = prediction_base.copy()
        prediction_results["probability"] = (
            probabilities
        )

        threshold = THRESHOLDS[model_name]

        # ----------------------------------------------------
        # Sample-level metrics
        # ----------------------------------------------------

        sample_metrics = calculate_sample_metrics(
            y_true=y_test,
            probabilities=probabilities,
            threshold=threshold,
            model_name=model_name,
        )

        print_sample_metrics(sample_metrics)

        sample_summaries.append(
            sample_metrics
        )

        # ----------------------------------------------------
        # Event-level + operational metrics
        # ----------------------------------------------------

        (
            summary,
            event_results,
            alert_results,
            operational_alerts,
        ) = evaluate_events(
            prediction_results=prediction_results,
            eligible_events=eligible_events,
            threshold=threshold,
            model_name=model_name,
        )

        summaries.append(summary)

        save_model_results(
            model_name=model_name,
            sample_results=sample_metrics,
            event_results=event_results,
            alert_results=alert_results,
            operational_alerts=operational_alerts,
        )

    # --------------------------------------------------------
    # 9. Save model comparisons
    # --------------------------------------------------------

    event_comparison = pd.DataFrame(
        summaries
    )

    sample_comparison = pd.DataFrame(
        sample_summaries
    )

    event_comparison_path = (
        REPORT_DIR
        / "rigorous_event_level_model_comparison.csv"
    )

    sample_comparison_path = (
        REPORT_DIR
        / "sample_level_model_comparison.csv"
    )

    event_comparison.to_csv(
        event_comparison_path,
        index=False,
    )

    sample_comparison.to_csv(
        sample_comparison_path,
        index=False,
    )

    # --------------------------------------------------------
    # 10. Save evaluation metadata/events
    # --------------------------------------------------------

    metadata_path = (
        REPORT_DIR
        / "test_sequence_metadata.csv"
    )

    eligible_events_path = (
        REPORT_DIR
        / "eligible_test_events.csv"
    )

    metadata.to_csv(
        metadata_path,
        index=False,
    )

    eligible_events.to_csv(
        eligible_events_path,
        index=False,
    )

    # --------------------------------------------------------
    # 11. Final display
    # --------------------------------------------------------

    print_section(
        "FINAL SAMPLE-LEVEL MODEL COMPARISON",
        width=120,
    )

    print(
        sample_comparison[
            [
                "model",
                "threshold",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc",
                "false_positive",
                "false_negative",
                "true_positive",
            ]
        ].to_string(index=False)
    )

    print_section(
        "FINAL RIGOROUS EVENT-LEVEL MODEL COMPARISON",
        width=150,
    )

    print(
        event_comparison[
            [
                "model",
                "threshold",
                "eligible_events",
                "detected_events",
                "missed_events",
                "event_detection_rate",
                "event_detection_ci95_low",
                "event_detection_ci95_high",
                "raw_alerts",
                "valid_alerts",
                "false_alerts",
                "false_alarm_rate",
                "raw_alerts_per_event",
                "operational_alerts",
                "operational_valid_alerts",
                "operational_false_alerts",
                "operational_false_alarm_rate",
                "mean_lead_time_minutes",
                "median_lead_time_minutes",
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # 12. Final report paths
    # --------------------------------------------------------

    print_section("REPORTS SAVED")

    print(
        "Sample-level comparison:"
    )
    print(sample_comparison_path)

    print(
        "\nRigorous event-level comparison:"
    )
    print(event_comparison_path)

    print(
        "\nTest sequence metadata:"
    )
    print(metadata_path)

    print(
        "\nEligible test events:"
    )
    print(eligible_events_path)

    print("\nPer-model reports:")
    for model_name in predictions:
        slug = model_name.lower()

        print(
            f"  {slug}_sample_metrics.json"
        )
        print(
            f"  {slug}_event_results.csv"
        )
        print(
            f"  {slug}_alert_results.csv"
        )
        print(
            f"  {slug}_operational_alerts.csv"
        )

    print_section(
        "EVALUATION COMPLETED SUCCESSFULLY"
    )

    print("✓ No model was retrained.")
    print("✓ LSTM, GRU and XGBoost remained frozen.")
    print("✓ Validation thresholds remained frozen.")
    print("✓ No test-selected threshold was used.")
    print(
        "✓ Sequence/metadata alignment was validated."
    )
    print(
        "✓ Maintenance events were discovered from the full dataset."
    )
    print(
        "✓ Boundary-truncated warning windows were excluded."
    )
    print(
        "✓ Sample-level and event-level metrics were separated."
    )
    print(
        "✓ 95% Wilson intervals were added for event/alert rates."
    )
    print(
        "✓ Operational confirmation/cooldown logic was reported separately."
    )


if __name__ == "__main__":
    main()
