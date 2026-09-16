"""
01_eda.py

Exploratory Data Analysis for Industrial Predictive Maintenance.

Run from project root:

    python src/01_eda.py
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "data.csv"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# EXPECTED SCHEMA
# ---------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "timestamp",
    "machine_id",
    "vibration_x",
    "vibration_y",
    "temperature_bearing",
    "ambient_temp",
    "rotational_speed_rpm",
    "pressure_psi",
    "is_maintenance_event",
]

SENSOR_COLUMNS = [
    "vibration_x",
    "vibration_y",
    "temperature_bearing",
    "ambient_temp",
    "rotational_speed_rpm",
    "pressure_psi",
]


# ---------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------

def load_data() -> pd.DataFrame:

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"\nDataset not found:\n{DATA_PATH}\n\n"
            "Place your data.csv inside:\n"
            "data/raw/data.csv"
        )

    df = pd.read_csv(DATA_PATH)

    return df


# ---------------------------------------------------------------------
# VALIDATE SCHEMA
# ---------------------------------------------------------------------

def validate_schema(df: pd.DataFrame) -> None:

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing_columns)
        )


# ---------------------------------------------------------------------
# BASIC REPORT
# ---------------------------------------------------------------------

def basic_report(df: pd.DataFrame) -> None:

    print("\n" + "=" * 80)
    print("DATASET OVERVIEW")
    print("=" * 80)

    print(f"Rows       : {len(df):,}")
    print(f"Columns    : {len(df.columns):,}")
    print(f"Memory     : {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")


# ---------------------------------------------------------------------
# DATA QUALITY
# ---------------------------------------------------------------------

def data_quality_report(df: pd.DataFrame) -> None:

    print("\n" + "=" * 80)
    print("DATA QUALITY")
    print("=" * 80)

    missing = df.isnull().sum()

    missing = missing[missing > 0]

    if len(missing) == 0:
        print("No missing values found.")
    else:
        print("\nMissing values:")
        print(missing.sort_values(ascending=False))

    duplicates = df.duplicated().sum()

    print(f"\nDuplicate rows: {duplicates:,}")


# ---------------------------------------------------------------------
# TIMESTAMP ANALYSIS
# ---------------------------------------------------------------------

def timestamp_analysis(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    invalid_timestamp = df["timestamp"].isna().sum()

    print("\n" + "=" * 80)
    print("TIMESTAMP ANALYSIS")
    print("=" * 80)

    print(f"Invalid timestamps: {invalid_timestamp}")

    if invalid_timestamp == 0:

        print(f"Start: {df['timestamp'].min()}")
        print(f"End  : {df['timestamp'].max()}")

    return df


# ---------------------------------------------------------------------
# MACHINE ANALYSIS
# ---------------------------------------------------------------------

def machine_analysis(df: pd.DataFrame) -> None:

    print("\n" + "=" * 80)
    print("MACHINE ANALYSIS")
    print("=" * 80)

    machines = df["machine_id"].nunique()

    print(f"Number of machines: {machines}")

    print("\nRecords per machine:")

    print(
        df["machine_id"]
        .value_counts()
        .sort_index()
    )


# ---------------------------------------------------------------------
# EVENT ANALYSIS
# ---------------------------------------------------------------------

def event_analysis(df: pd.DataFrame) -> None:

    print("\n" + "=" * 80)
    print("MAINTENANCE EVENT ANALYSIS")
    print("=" * 80)

    counts = (
        df["is_maintenance_event"]
        .value_counts()
        .sort_index()
    )

    print(counts)

    event_percentage = (
        df["is_maintenance_event"].mean() * 100
    )

    print(
        f"\nMaintenance-event records: "
        f"{event_percentage:.4f}%"
    )

    print("\nEvents by machine:")

    print(
        df.loc[
            df["is_maintenance_event"] == 1,
            "machine_id"
        ]
        .value_counts()
        .sort_index()
    )


# ---------------------------------------------------------------------
# SENSOR STATISTICS
# ---------------------------------------------------------------------

def sensor_statistics(df: pd.DataFrame) -> None:

    print("\n" + "=" * 80)
    print("SENSOR STATISTICS")
    print("=" * 80)

    print(
        df[SENSOR_COLUMNS]
        .describe()
        .T
        .round(4)
    )


# ---------------------------------------------------------------------
# SENSOR DISTRIBUTIONS
# ---------------------------------------------------------------------

def plot_distributions(df: pd.DataFrame) -> None:

    print("\nCreating sensor distribution plots...")

    for column in SENSOR_COLUMNS:

        plt.figure(figsize=(10, 5))

        sns.histplot(
            df[column].dropna(),
            bins=60,
            kde=True
        )

        plt.title(
            f"Distribution - {column}"
        )

        plt.xlabel(column)
        plt.ylabel("Frequency")

        plt.tight_layout()

        plt.savefig(
            FIGURES_DIR / f"distribution_{column}.png",
            dpi=150
        )

        plt.close()


# ---------------------------------------------------------------------
# CORRELATION
# ---------------------------------------------------------------------

def plot_correlation(df: pd.DataFrame) -> None:

    print("Creating correlation matrix...")

    correlation = df[SENSOR_COLUMNS].corr()

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        square=True
    )

    plt.title("Sensor Correlation Matrix")

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "sensor_correlation.png",
        dpi=150
    )

    plt.close()


# ---------------------------------------------------------------------
# NORMAL VS MAINTENANCE
# ---------------------------------------------------------------------

def plot_event_comparison(df: pd.DataFrame) -> None:

    print("Creating normal vs maintenance plots...")

    for column in SENSOR_COLUMNS:

        plt.figure(figsize=(9, 5))

        sns.boxplot(
            data=df,
            x="is_maintenance_event",
            y=column
        )

        plt.title(
            f"{column}: Normal vs Maintenance"
        )

        plt.xlabel(
            "Maintenance Event (0 = Normal, 1 = Maintenance)"
        )

        plt.tight_layout()

        plt.savefig(
            FIGURES_DIR / f"event_comparison_{column}.png",
            dpi=150
        )

        plt.close()


# ---------------------------------------------------------------------
# TIME SERIES
# ---------------------------------------------------------------------

def plot_time_series(df: pd.DataFrame) -> None:

    print("Creating time-series plots...")

    machines = (
        df["machine_id"]
        .dropna()
        .unique()
    )

    for machine in machines:

        machine_df = (
            df[df["machine_id"] == machine]
            .sort_values("timestamp")
        )

        for column in SENSOR_COLUMNS:

            plt.figure(figsize=(14, 5))

            plt.plot(
                machine_df["timestamp"],
                machine_df[column],
                linewidth=0.7
            )

            events = machine_df[
                machine_df["is_maintenance_event"] == 1
            ]

            if not events.empty:

                plt.scatter(
                    events["timestamp"],
                    events[column],
                    s=12,
                    label="Maintenance Event"
                )

            plt.title(
                f"{machine} - {column}"
            )

            plt.xlabel("Timestamp")
            plt.ylabel(column)

            plt.tight_layout()

            safe_machine = str(machine).replace(
                "/", "_"
            )

            plt.savefig(
                FIGURES_DIR
                / f"{safe_machine}_{column}.png",
                dpi=120
            )

            plt.close()


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    print("\nStarting EDA...")

    df = load_data()

    validate_schema(df)

    df = timestamp_analysis(df)

    df = df.dropna(
        subset=["timestamp"]
    )

    df = df.sort_values(
        ["machine_id", "timestamp"]
    )

    basic_report(df)

    data_quality_report(df)

    machine_analysis(df)

    event_analysis(df)

    sensor_statistics(df)

    plot_distributions(df)

    plot_correlation(df)

    plot_event_comparison(df)

    plot_time_series(df)

    print("\n" + "=" * 80)
    print("EDA COMPLETE")
    print("=" * 80)

    print(
        f"Figures saved to:\n{FIGURES_DIR}"
    )


if __name__ == "__main__":
    main()
