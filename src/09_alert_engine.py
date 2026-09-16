"""
09_alert_engine.py
Production-style operational alert engine for predictive maintenance.

UPDATED STATE-MACHINE VERSION

Purpose
-------
Convert frozen model risk scores from 08_inference.py into operational alerts.

IMPORTANT
---------
- No model training.
- No threshold tuning.
- Frozen validation thresholds are used.
- Confirmation and cooldown are operational rules, not ML parameters.
- Each model is evaluated independently.
- Designed for 1-minute sensor sampling.

Operational logic
-----------------
1. risk >= frozen threshold -> candidate alert.
2. Three NEW consecutive 1-minute candidate observations are required
   to confirm an alert.
3. When an alert is confirmed, the confirmation streak is RESET to zero.
4. The machine/model enters cooldown for 60 minutes.
5. Candidate observations during cooldown are ignored for confirmation.
6. When cooldown expires, a NEW 3-observation confirmation sequence is
   required. An old pre-cooldown streak is never reused.
7. A timestamp gap breaks the consecutive sequence.

This fixes the previous behavior where a continuously positive signal could
produce confirmation_count values such as 63 at the end of cooldown.
"""

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORTS_DIR = PROJECT_ROOT / "reports"
INPUT_FILE = REPORTS_DIR / "production_inference_test.csv"

OUTPUT_DIR = REPORTS_DIR / "alert_engine"

CANDIDATE_OUTPUT = OUTPUT_DIR / "candidate_alerts.csv"
CONFIRMED_OUTPUT = OUTPUT_DIR / "confirmed_alerts.csv"
STATE_OUTPUT = OUTPUT_DIR / "machine_alert_state.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "alert_engine_summary.csv"

# Frozen validation thresholds.
# DO NOT tune these on the test set.
FROZEN_THRESHOLDS = {
    "lstm": 0.050773,
    "gru": 0.060276,
    "xgboost": 0.033000,
}

# Operational policy.
CONFIRMATION_COUNT = 3
COOLDOWN_MINUTES = 60

# Expected telemetry frequency.
EXPECTED_INTERVAL_MINUTES = 1


# ============================================================
# VALIDATION / INPUT
# ============================================================

def check_input_file():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input inference file was not found:\n{INPUT_FILE}\n\n"
            "Run first:\n"
            "python src/08_inference.py"
        )


def load_inference_results():
    check_input_file()

    df = pd.read_csv(INPUT_FILE)

    required = [
        "machine_id",
        "timestamp",
        "lstm_risk",
        "lstm_prediction",
        "gru_risk",
        "gru_prediction",
        "xgboost_risk",
        "xgboost_prediction",
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            "production_inference_test.csv is missing required columns:\n"
            + "\n".join(f"  - {col}" for col in missing)
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    if df["timestamp"].isna().any():
        bad = int(df["timestamp"].isna().sum())
        raise ValueError(
            f"{bad} rows contain invalid timestamps."
        )

    if df["machine_id"].isna().any():
        raise ValueError("machine_id contains missing values.")

    # One row per machine/timestamp is expected.
    duplicate_mask = df.duplicated(
        subset=["machine_id", "timestamp"],
        keep=False
    )

    if duplicate_mask.any():
        count = int(duplicate_mask.sum())
        raise ValueError(
            f"Found {count} duplicate machine/timestamp rows."
        )

    df = df.sort_values(
        ["machine_id", "timestamp"]
    ).reset_index(drop=True)

    return df


# ============================================================
# MODEL LONG FORM
# ============================================================

def create_model_long_form(df):
    """
    Convert wide model predictions into one row per
    machine/timestamp/model.
    """

    common_cols = ["machine_id", "timestamp"]

    if "sequence_index" in df.columns:
        common_cols.append("sequence_index")

    if "actual_failure_2_6h" in df.columns:
        common_cols.append("actual_failure_2_6h")

    frames = []

    for model_name in FROZEN_THRESHOLDS:
        risk_col = f"{model_name}_risk"
        prediction_col = f"{model_name}_prediction"

        part = df[
            common_cols + [risk_col, prediction_col]
        ].copy()

        part = part.rename(
            columns={
                risk_col: "risk_score",
                prediction_col: "model_prediction",
            }
        )

        part["model"] = model_name
        part["threshold"] = FROZEN_THRESHOLDS[model_name]

        # Recompute candidate from the frozen threshold.
        part["candidate_alert"] = (
            part["risk_score"] >= part["threshold"]
        ).astype(int)

        # Protect against inconsistent/stale prediction columns.
        if not np.array_equal(
            part["candidate_alert"].to_numpy(),
            part["model_prediction"].astype(int).to_numpy()
        ):
            raise ValueError(
                f"{model_name}: model_prediction does not match "
                f"the frozen threshold."
            )

        frames.append(part)

    long_df = pd.concat(
        frames,
        ignore_index=True
    )

    long_df = long_df.sort_values(
        ["model", "machine_id", "timestamp"]
    ).reset_index(drop=True)

    return long_df


# ============================================================
# CANDIDATE ALERTS
# ============================================================

def build_candidate_alerts(long_df):
    candidates = long_df[
        long_df["candidate_alert"] == 1
    ].copy()

    if candidates.empty:
        return candidates

    candidates["candidate_id"] = np.arange(
        1,
        len(candidates) + 1
    )

    candidates["alert_type"] = "CANDIDATE"

    return candidates


# ============================================================
# OPERATIONAL ALERT STATE MACHINE
# ============================================================

def process_alert_state(long_df):
    """
    Apply the production alert state machine independently for each
    model + machine_id pair.

    State:

        NORMAL
           |
           | candidate
           v
        CONFIRMING
           |
           | 3 consecutive observations
           v
        ALERT CONFIRMED
           |
           v
        COOLDOWN
           |
           | cooldown expires
           v
        NORMAL / NEW CONFIRMATION

    Critical rule:
        Confirmation count is RESET immediately after an alert is
        confirmed. Candidates during cooldown do not build a streak.

    Therefore, a long continuous high-risk period cannot create
    confirmation_count=63. A new alert requires three NEW observations
    after cooldown.
    """

    long_df = long_df.sort_values(
        ["model", "machine_id", "timestamp"]
    ).reset_index(drop=True)

    confirmed_rows = []
    state_rows = []

    for (model, machine_id), group in long_df.groupby(
        ["model", "machine_id"],
        sort=False
    ):
        group = group.sort_values(
            "timestamp"
        ).reset_index(drop=True)

        consecutive_hits = 0
        last_timestamp = None

        cooldown_until = pd.NaT
        last_confirmed_timestamp = pd.NaT

        total_confirmed = 0
        current_state = "NORMAL"

        for _, row in group.iterrows():

            timestamp = row["timestamp"]
            is_candidate = int(row["candidate_alert"])

            # ------------------------------------------------
            # 1. Check timestamp continuity.
            # ------------------------------------------------

            is_consecutive = False

            if last_timestamp is not None:
                gap_minutes = (
                    timestamp - last_timestamp
                ).total_seconds() / 60.0

                is_consecutive = (
                    abs(
                        gap_minutes
                        - EXPECTED_INTERVAL_MINUTES
                    ) < 1e-9
                )

            # Any gap breaks a confirmation sequence.
            if not is_consecutive:
                consecutive_hits = 0

                # If the cooldown has also expired, return to normal.
                if (
                    pd.isna(cooldown_until)
                    or timestamp >= cooldown_until
                ):
                    current_state = "NORMAL"

            # ------------------------------------------------
            # 2. Check cooldown.
            # ------------------------------------------------

            in_cooldown = (
                pd.notna(cooldown_until)
                and timestamp < cooldown_until
            )

            if in_cooldown:
                # CRITICAL:
                # Do not accumulate evidence during cooldown.
                consecutive_hits = 0
                current_state = "COOLDOWN"

            else:
                # Cooldown has expired.
                if (
                    pd.notna(cooldown_until)
                    and timestamp >= cooldown_until
                ):
                    cooldown_until = pd.NaT
                    consecutive_hits = 0
                    current_state = "NORMAL"

                # ------------------------------------------------
                # 3. Build a NEW confirmation streak.
                # ------------------------------------------------

                if is_candidate:
                    if is_consecutive:
                        consecutive_hits += 1
                    else:
                        # This row starts a fresh streak.
                        consecutive_hits = 1

                    current_state = "CONFIRMING"

                else:
                    consecutive_hits = 0
                    current_state = "NORMAL"

                # ------------------------------------------------
                # 4. Confirm operational alert.
                # ------------------------------------------------

                if (
                    is_candidate
                    and consecutive_hits >= CONFIRMATION_COUNT
                ):
                    total_confirmed += 1
                    last_confirmed_timestamp = timestamp

                    new_cooldown_until = (
                        timestamp
                        + pd.Timedelta(
                            minutes=COOLDOWN_MINUTES
                        )
                    )

                    confirmed_rows.append({
                        "alert_id": (
                            f"{model.upper()}_"
                            f"{machine_id}_"
                            f"{timestamp.strftime('%Y%m%d%H%M%S')}"
                        ),
                        "machine_id": machine_id,
                        "model": model,
                        "alert_timestamp": timestamp,
                        "risk_score": float(
                            row["risk_score"]
                        ),
                        "threshold": float(
                            row["threshold"]
                        ),
                        "confirmation_count": int(
                            consecutive_hits
                        ),
                        "alert_status": "CONFIRMED",
                        "cooldown_minutes": (
                            COOLDOWN_MINUTES
                        ),
                        "cooldown_until": (
                            new_cooldown_until
                        ),
                        "sequence_index": row.get(
                            "sequence_index",
                            np.nan
                        ),
                    })

                    # ------------------------------------------------
                    # CRITICAL FIX:
                    # Reset confirmation immediately after alert.
                    # The next alert must start a completely NEW
                    # confirmation sequence after cooldown.
                    # ------------------------------------------------
                    consecutive_hits = 0
                    cooldown_until = new_cooldown_until
                    current_state = "COOLDOWN"

            last_timestamp = timestamp

        # --------------------------------------------------------
        # Final machine/model state.
        # --------------------------------------------------------

        state_rows.append({
            "machine_id": machine_id,
            "model": model,
            "last_timestamp": last_timestamp,
            "last_risk_score": (
                float(group.iloc[-1]["risk_score"])
                if len(group)
                else np.nan
            ),
            "last_threshold": float(
                FROZEN_THRESHOLDS[model]
            ),
            "last_prediction": int(
                group.iloc[-1]["candidate_alert"]
            ),
            "current_state": current_state,
            "confirmation_streak": int(
                consecutive_hits
            ),
            "last_confirmed_alert": (
                last_confirmed_timestamp
            ),
            "cooldown_until": cooldown_until,
            "total_confirmed_alerts": int(
                total_confirmed
            ),
        })

    confirmed_df = pd.DataFrame(
        confirmed_rows
    )

    if not confirmed_df.empty:
        confirmed_df = confirmed_df.sort_values(
            [
                "machine_id",
                "alert_timestamp",
                "model",
            ]
        ).reset_index(drop=True)

    state_df = pd.DataFrame(
        state_rows
    )

    return confirmed_df, state_df


# ============================================================
# SUMMARY
# ============================================================

def build_summary(
    long_df,
    candidate_df,
    confirmed_df
):
    rows = []

    for model, threshold in FROZEN_THRESHOLDS.items():

        model_data = long_df[
            long_df["model"] == model
        ]

        candidate_count = int(
            model_data["candidate_alert"].sum()
        )

        if not confirmed_df.empty:
            confirmed_count = int(
                (
                    confirmed_df["model"] == model
                ).sum()
            )
        else:
            confirmed_count = 0

        rows.append({
            "model": model,
            "frozen_threshold": threshold,
            "rows_processed": len(model_data),
            "machines_processed": int(
                model_data["machine_id"].nunique()
            ),
            "candidate_alerts": candidate_count,
            "confirmed_alerts": confirmed_count,
            "confirmation_count_required": (
                CONFIRMATION_COUNT
            ),
            "cooldown_minutes": (
                COOLDOWN_MINUTES
            ),
            "candidate_to_confirmed_ratio": (
                confirmed_count / candidate_count
                if candidate_count > 0
                else 0.0
            ),
        })

    total_candidates = int(
        len(candidate_df)
    )
    total_confirmed = int(
        len(confirmed_df)
    )

    rows.append({
        "model": "ALL_MODELS",
        "frozen_threshold": np.nan,
        "rows_processed": len(long_df),
        "machines_processed": int(
            long_df["machine_id"].nunique()
        ),
        "candidate_alerts": total_candidates,
        "confirmed_alerts": total_confirmed,
        "confirmation_count_required": (
            CONFIRMATION_COUNT
        ),
        "cooldown_minutes": (
            COOLDOWN_MINUTES
        ),
        "candidate_to_confirmed_ratio": (
            total_confirmed / total_candidates
            if total_candidates > 0
            else 0.0
        ),
    })

    return pd.DataFrame(rows)


# ============================================================
# PRINT
# ============================================================

def print_summary(
    df,
    candidate_df,
    confirmed_df,
    state_df,
    summary_df
):
    print("\n" + "=" * 75)
    print("09 - OPERATIONAL ALERT ENGINE")
    print("=" * 75)

    print(f"\nInput rows: {len(df)}")
    print(
        f"Machines: "
        f"{df['machine_id'].nunique()}"
    )

    print("\nOperational configuration:")
    print(
        f"  Confirmation count : "
        f"{CONFIRMATION_COUNT} NEW consecutive observations"
    )
    print(
        f"  Expected interval  : "
        f"{EXPECTED_INTERVAL_MINUTES} minute"
    )
    print(
        f"  Cooldown           : "
        f"{COOLDOWN_MINUTES} minutes"
    )

    print("\nFrozen model thresholds:")
    for model, threshold in FROZEN_THRESHOLDS.items():
        print(
            f"  {model:10s}: "
            f"{threshold:.6f}"
        )

    print("\nAlert summary:")

    for _, row in summary_df[
        summary_df["model"] != "ALL_MODELS"
    ].iterrows():

        print(
            f"  {row['model']:10s}: "
            f"{int(row['candidate_alerts'])} candidate → "
            f"{int(row['confirmed_alerts'])} confirmed"
        )

    print(
        f"\nALL MODELS: "
        f"{len(candidate_df)} candidate alerts → "
        f"{len(confirmed_df)} confirmed alerts"
    )

    if not confirmed_df.empty:

        print("\nConfirmed alerts:")

        display_cols = [
            "alert_id",
            "machine_id",
            "model",
            "alert_timestamp",
            "risk_score",
            "threshold",
            "confirmation_count",
            "cooldown_until",
        ]

        print(
            confirmed_df[
                display_cols
            ].to_string(index=False)
        )

    else:
        print(
            "\nNo operational alerts were confirmed."
        )

    print("\nFinal machine/model states:")

    state_display_cols = [
        "machine_id",
        "model",
        "current_state",
        "confirmation_streak",
        "last_confirmed_alert",
        "cooldown_until",
        "total_confirmed_alerts",
    ]

    print(
        state_df[
            state_display_cols
        ].to_string(index=False)
    )

    print("=" * 75)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("09 - OPERATIONAL ALERT ENGINE")
    print("=" * 75)

    print(
        f"\nProject root: "
        f"{PROJECT_ROOT}"
    )

    print(
        f"Input: "
        f"{INPUT_FILE}"
    )

    # 1. Load frozen inference output.
    df = load_inference_results()

    # 2. Convert three models to long form.
    long_df = create_model_long_form(df)

    # 3. Extract threshold-crossing candidates.
    candidate_df = build_candidate_alerts(
        long_df
    )

    # 4. Apply corrected state machine.
    confirmed_df, state_df = process_alert_state(
        long_df
    )

    # 5. Summary.
    summary_df = build_summary(
        long_df,
        candidate_df,
        confirmed_df
    )

    # 6. Save.
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    candidate_df.to_csv(
        CANDIDATE_OUTPUT,
        index=False
    )

    confirmed_df.to_csv(
        CONFIRMED_OUTPUT,
        index=False
    )

    state_df.to_csv(
        STATE_OUTPUT,
        index=False
    )

    summary_df.to_csv(
        SUMMARY_OUTPUT,
        index=False
    )

    # 7. Print.
    print_summary(
        df,
        candidate_df,
        confirmed_df,
        state_df,
        summary_df
    )

    print("\nSaved files:")
    print(
        f"  Candidate alerts : "
        f"{CANDIDATE_OUTPUT}"
    )
    print(
        f"  Confirmed alerts : "
        f"{CONFIRMED_OUTPUT}"
    )
    print(
        f"  Machine state    : "
        f"{STATE_OUTPUT}"
    )
    print(
        f"  Summary          : "
        f"{SUMMARY_OUTPUT}"
    )

    print("\nSUCCESS")
    print("No model was retrained.")
    print("No threshold was tuned.")
    print(
        "Confirmation streaks are reset after each alert."
    )
    print(
        "Cooldown observations cannot accumulate "
        "confirmation evidence."
    )
    print(
        "A new alert requires a fresh confirmation sequence."
    )


if __name__ == "__main__":
    main()
