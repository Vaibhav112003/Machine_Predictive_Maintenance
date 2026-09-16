"""
02_feature_engineering.py

Creates leakage-safe time-series features and the
2-6 hour forward-looking prediction target.

Run:

    python src/02_feature_engineering.py
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "data.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features.csv"
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

SENSOR_COLUMNS = [
    "vibration_x",
    "vibration_y",
    "temperature_bearing",
    "ambient_temp",
    "rotational_speed_rpm",
    "pressure_psi",
]

ROLLING_WINDOWS = [
    5,
    15,
    30,
    60,
]


# ---------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------

def load_data() -> pd.DataFrame:

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    required = (
        [
            "timestamp",
            "machine_id",
            "is_maintenance_event",
        ]
        + SENSOR_COLUMNS
    )

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing columns:\n"
            + "\n".join(missing)
        )

    return df


# ---------------------------------------------------------------------
# BASIC CLEANING
# ---------------------------------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["timestamp", "machine_id"]
    )

    df = df.sort_values(
        ["machine_id", "timestamp"]
    )

    df = df.drop_duplicates(
        subset=["machine_id", "timestamp"],
        keep="last"
    )

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------
# VIBRATION MAGNITUDE
# ---------------------------------------------------------------------

def create_vibration_features(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = df.copy()

    df["vibration_magnitude"] = np.sqrt(
        df["vibration_x"] ** 2
        +
        df["vibration_y"] ** 2
    )

    return df


# ---------------------------------------------------------------------
# ROLLING FEATURES
# ---------------------------------------------------------------------

def create_rolling_features(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = df.copy()

    rolling_columns = SENSOR_COLUMNS + [
        "vibration_magnitude"
    ]

    grouped = df.groupby(
        "machine_id",
        sort=False
    )

    for column in rolling_columns:

        for window in ROLLING_WINDOWS:

            df[
                f"{column}_mean_{window}m"
            ] = (
                grouped[column]
                .transform(
                    lambda x: x.rolling(
                        window=window,
                        min_periods=window
                    ).mean()
                )
            )

            df[
                f"{column}_std_{window}m"
            ] = (
                grouped[column]
                .transform(
                    lambda x: x.rolling(
                        window=window,
                        min_periods=window
                    ).std()
                )
            )

            df[
                f"{column}_min_{window}m"
            ] = (
                grouped[column]
                .transform(
                    lambda x: x.rolling(
                        window=window,
                        min_periods=window
                    ).min()
                )
            )

            df[
                f"{column}_max_{window}m"
            ] = (
                grouped[column]
                .transform(
                    lambda x: x.rolling(
                        window=window,
                        min_periods=window
                    ).max()
                )
            )

    return df


# ---------------------------------------------------------------------
# DIFFERENCE FEATURES
# ---------------------------------------------------------------------

def create_change_features(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = df.copy()

    grouped = df.groupby(
        "machine_id",
        sort=False
    )

    for column in SENSOR_COLUMNS:

        df[
            f"{column}_diff_1m"
        ] = grouped[column].diff(1)

        df[
            f"{column}_diff_5m"
        ] = grouped[column].diff(5)

        df[
            f"{column}_diff_15m"
        ] = grouped[column].diff(15)

        df[
            f"{column}_pct_change_15m"
        ] = grouped[column].pct_change(
            periods=15
        )

    return df


# ---------------------------------------------------------------------
# SLOPE FEATURES
# ---------------------------------------------------------------------

def calculate_slope(values):

    values = np.asarray(
        values,
        dtype=float
    )

    if len(values) < 2:
        return np.nan

    if np.isnan(values).any():
        return np.nan

    x = np.arange(len(values))

    slope = np.polyfit(
        x,
        values,
        1
    )[0]

    return slope


def create_slope_features(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = df.copy()

    slope_columns = SENSOR_COLUMNS + [
        "vibration_magnitude"
    ]

    for column in slope_columns:

        for window in [15, 30, 60]:

            df[
                f"{column}_slope_{window}m"
            ] = (
                df.groupby("machine_id")[column]
                .transform(
                    lambda x: x.rolling(
                        window,
                        min_periods=window
                    ).apply(
                        calculate_slope,
                        raw=True
                    )
                )
            )

    return df


# ---------------------------------------------------------------------
# TEMP / AMBIENT DIFFERENCE
# ---------------------------------------------------------------------

def create_domain_features(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = df.copy()

    df["bearing_ambient_delta"] = (
        df["temperature_bearing"]
        -
        df["ambient_temp"]
    )

    df["rpm_pressure_ratio"] = (
        df["rotational_speed_rpm"]
        /
        (df["pressure_psi"].abs() + 1e-6)
    )

    return df


# ---------------------------------------------------------------------
# TIME FEATURES
# ---------------------------------------------------------------------

def create_time_features(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = df.copy()

    df["hour"] = df["timestamp"].dt.hour

    df["day_of_week"] = (
        df["timestamp"].dt.dayofweek
    )

    df["day_of_month"] = (
        df["timestamp"].dt.day
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    df["dow_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["dow_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    return df


# ---------------------------------------------------------------------
# 2-6 HOUR TARGET
# ---------------------------------------------------------------------

def create_forward_target(
    group: pd.DataFrame
) -> pd.DataFrame:

    group = group.copy()

    event = (
        group["is_maintenance_event"]
        .astype(int)
        .to_numpy()
    )

    n = len(group)

    target = np.full(
        n,
        np.nan
    )

    # We require the complete future
    # 2-6 hour window.
    for i in range(n):

        start = i + 120
        end = i + 361

        if end > n:
            continue

        future_window = event[
            start:end
        ]

        target[i] = int(
            future_window.max() == 1
        )

    group["failure_2_6h"] = target

    return group


def create_target(
    df: pd.DataFrame
) -> pd.DataFrame:

    return (
        df.groupby(
            "machine_id",
            group_keys=False
        )
        .apply(create_forward_target)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------
# FINAL CLEANUP
# ---------------------------------------------------------------------

def final_cleanup(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = df.copy()

    # Infinite values created by ratios / pct change
    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Remove rows where target cannot be determined.
    df = df.dropna(
        subset=["failure_2_6h"]
    )

    # Remove rows where engineered historical
    # features are unavailable.
    df = df.dropna()

    df["failure_2_6h"] = (
        df["failure_2_6h"]
        .astype(int)
    )

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    print("=" * 80)
    print("FEATURE ENGINEERING")
    print("=" * 80)

    df = load_data()

    print(
        f"Raw shape: {df.shape}"
    )

    df = clean_data(df)

    df = create_vibration_features(df)

    print("Created vibration features.")

    df = create_rolling_features(df)

    print("Created rolling features.")

    df = create_change_features(df)

    print("Created change features.")

    df = create_slope_features(df)

    print("Created slope features.")

    df = create_domain_features(df)

    print("Created domain features.")

    df = create_time_features(df)

    print("Created time features.")

    df = create_target(df)

    print(
        "Created 2-6 hour forward target."
    )

    df = final_cleanup(df)

    print(
        f"\nFinal feature shape: {df.shape}"
    )

    print(
        "\nTarget distribution:"
    )

    print(
        df["failure_2_6h"]
        .value_counts()
        .sort_index()
    )

    print(
        "\nTarget percentage:"
    )

    print(
        df["failure_2_6h"]
        .value_counts(
            normalize=True
        )
        .sort_index()
        * 100
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"\nSaved:\n{OUTPUT_PATH}"
    )

    print(
        f"\nTotal features: {len(df.columns)}"
    )


if __name__ == "__main__":
    main()