"""
03_create_sequences.py

Creates leakage-safe train/validation/test sequences
for the LSTM model.

Run:

    python src/03_create_sequences.py
"""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features.csv"
)

SEQUENCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sequences.npz"
)

SCALER_PATH = (
    PROJECT_ROOT
    / "models"
    / "scaler.pkl"
)

FEATURE_LIST_PATH = (
    PROJECT_ROOT
    / "models"
    / "feature_columns.json"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "models"
    / "data_metadata.json"
)

SCALER_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

SEQUENCE_LENGTH = 120

TRAIN_PERCENTILE = 0.70

VALIDATION_PERCENTILE = 0.85

EXCLUDED_COLUMNS = [
    "timestamp",
    "machine_id",
    "is_maintenance_event",
    "failure_2_6h",
]


# ---------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------

def load_features():

    if not FEATURE_PATH.exists():

        raise FileNotFoundError(
            "features.csv not found.\n"
            "Run:\n"
            "python src/02_feature_engineering.py"
        )

    df = pd.read_csv(
        FEATURE_PATH
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df = df.sort_values(
        ["timestamp", "machine_id"]
    ).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------
# TIME SPLIT
# ---------------------------------------------------------------------

def time_split(df):

    train_cutoff = df[
        "timestamp"
    ].quantile(
        TRAIN_PERCENTILE
    )

    validation_cutoff = df[
        "timestamp"
    ].quantile(
        VALIDATION_PERCENTILE
    )

    train = df[
        df["timestamp"] <= train_cutoff
    ].copy()

    validation = df[
        (
            df["timestamp"] > train_cutoff
        )
        &
        (
            df["timestamp"] <= validation_cutoff
        )
    ].copy()

    test = df[
        df["timestamp"] > validation_cutoff
    ].copy()

    print("\nTime split:")
    print(
        f"Train      : {train['timestamp'].min()} "
        f"→ {train['timestamp'].max()}"
    )

    print(
        f"Validation : {validation['timestamp'].min()} "
        f"→ {validation['timestamp'].max()}"
    )

    print(
        f"Test       : {test['timestamp'].min()} "
        f"→ {test['timestamp'].max()}"
    )

    return train, validation, test


# ---------------------------------------------------------------------
# FEATURE LIST
# ---------------------------------------------------------------------

def get_feature_columns(df):

    feature_columns = [
        column
        for column in df.columns
        if column not in EXCLUDED_COLUMNS
    ]

    return feature_columns


# ---------------------------------------------------------------------
# SEQUENCE CREATION
# ---------------------------------------------------------------------

def create_sequences(
    X,
    y,
    sequence_length
):

    sequences = []
    targets = []

    for i in range(
        sequence_length,
        len(X)
    ):

        sequence = X[
            i - sequence_length:i
        ]

        target = y[i]

        sequences.append(
            sequence
        )

        targets.append(
            target
        )

    return (
        np.asarray(
            sequences,
            dtype=np.float32
        ),
        np.asarray(
            targets,
            dtype=np.int8
        )
    )


# ---------------------------------------------------------------------
# PROCESS EACH MACHINE
# ---------------------------------------------------------------------

def create_machine_sequences(
    df,
    feature_columns,
    scaler,
    fit_scaler=False
):

    all_X = []
    all_y = []

    for machine_id, group in df.groupby(
        "machine_id",
        sort=False
    ):

        group = group.sort_values(
            "timestamp"
        )

        X = group[
            feature_columns
        ].astype(float)

        y = group[
            "failure_2_6h"
        ].astype(int).values

        if fit_scaler:

            X_scaled = scaler.fit_transform(
                X
            )

        else:

            X_scaled = scaler.transform(
                X
            )

        if len(X_scaled) <= SEQUENCE_LENGTH:
            continue

        X_seq, y_seq = create_sequences(
            X_scaled,
            y,
            SEQUENCE_LENGTH
        )

        if len(X_seq) > 0:

            all_X.append(X_seq)
            all_y.append(y_seq)

    if not all_X:

        raise ValueError(
            "No sequences were created. "
            "Check dataset size."
        )

    return (
        np.concatenate(all_X),
        np.concatenate(all_y)
    )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    print("=" * 80)
    print("CREATING LSTM SEQUENCES")
    print("=" * 80)

    df = load_features()

    print(
        f"Feature dataset: {df.shape}"
    )

    train, validation, test = time_split(
        df
    )

    feature_columns = get_feature_columns(
        df
    )

    print(
        f"\nNumber of model features: "
        f"{len(feature_columns)}"
    )

    print("\nFitting scaler on training data only...")

    scaler = StandardScaler()

    # Fit only on training rows.
    scaler.fit(
        train[feature_columns]
    )

    joblib.dump(
        scaler,
        SCALER_PATH
    )

    # -------------------------------------------------------------
    # Create sequences
    # -------------------------------------------------------------

    X_train, y_train = create_machine_sequences(
        train,
        feature_columns,
        scaler,
        fit_scaler=False
    )

    X_validation, y_validation = create_machine_sequences(
        validation,
        feature_columns,
        scaler,
        fit_scaler=False
    )

    X_test, y_test = create_machine_sequences(
        test,
        feature_columns,
        scaler,
        fit_scaler=False
    )

    print("\nSequence shapes:")

    print(
        f"X_train      : {X_train.shape}"
    )

    print(
        f"y_train      : {y_train.shape}"
    )

    print(
        f"X_validation : {X_validation.shape}"
    )

    print(
        f"y_validation : {y_validation.shape}"
    )

    print(
        f"X_test       : {X_test.shape}"
    )

    print(
        f"y_test       : {y_test.shape}"
    )

    # -------------------------------------------------------------
    # Save arrays
    # -------------------------------------------------------------

    np.savez_compressed(
        SEQUENCE_PATH,
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        X_test=X_test,
        y_test=y_test
    )

    with open(
        FEATURE_LIST_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            feature_columns,
            f,
            indent=4
        )

    metadata = {

        "sequence_length": SEQUENCE_LENGTH,

        "train_start": str(
            train["timestamp"].min()
        ),

        "train_end": str(
            train["timestamp"].max()
        ),

        "validation_start": str(
            validation["timestamp"].min()
        ),

        "validation_end": str(
            validation["timestamp"].max()
        ),

        "test_start": str(
            test["timestamp"].min()
        ),

        "test_end": str(
            test["timestamp"].max()
        ),

        "number_of_features": len(
            feature_columns
        ),

        "feature_columns": feature_columns,
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )

    print(
        f"\nSaved sequences:\n{SEQUENCE_PATH}"
    )

    print(
        f"Saved scaler:\n{SCALER_PATH}"
    )

    print(
        f"Saved feature list:\n{FEATURE_LIST_PATH}"
    )


if __name__ == "__main__":
    main()