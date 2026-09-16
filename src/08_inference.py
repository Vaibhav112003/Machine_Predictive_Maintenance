"""
08_inference.py
Production-style inference for the frozen predictive-maintenance models.

FIX:
The scaler artifact is loaded with joblib because the saved scaler was
serialized in joblib format. Do not use pickle.load() directly.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from tensorflow.keras.models import load_model


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

SEQUENCES_FILE = PROCESSED_DIR / "sequences.npz"
SCALER_FILE = MODELS_DIR / "scaler.pkl"
FEATURE_COLUMNS_FILE = MODELS_DIR / "feature_columns.json"

LSTM_FILE = MODELS_DIR / "best_lstm.keras"
GRU_FILE = MODELS_DIR / "best_gru.keras"
XGB_FILE = MODELS_DIR / "xgboost_model.json"

METADATA_FILE = REPORTS_DIR / "test_sequence_metadata.csv"
OUTPUT_FILE = REPORTS_DIR / "production_inference_test.csv"

LOOKBACK = 120
EXPECTED_FEATURES = 174

# FROZEN VALIDATION THRESHOLDS
THRESHOLDS = {
    "lstm": 0.050773,
    "gru": 0.060276,
    "xgboost": 0.033000,
}


# ============================================================
# HELPERS
# ============================================================

def check_file(path: Path, description: str):
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found:\n{path}"
        )


def load_feature_columns():
    check_file(FEATURE_COLUMNS_FILE, "Feature column file")

    with open(FEATURE_COLUMNS_FILE, "r", encoding="utf-8") as f:
        columns = json.load(f)

    if not isinstance(columns, list):
        raise ValueError("feature_columns.json must contain a list.")

    if len(columns) != EXPECTED_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_FEATURES} model features, "
            f"but feature_columns.json contains {len(columns)}."
        )

    return columns


def load_test_sequences():
    check_file(SEQUENCES_FILE, "Sequence dataset")

    data = np.load(SEQUENCES_FILE)

    if "X_test" not in data:
        raise KeyError("X_test was not found in sequences.npz.")

    X_test = data["X_test"]
    y_test = data["y_test"] if "y_test" in data else None

    expected_shape = (LOOKBACK, EXPECTED_FEATURES)

    if X_test.ndim != 3:
        raise ValueError(
            f"X_test must be 3-dimensional. Got shape {X_test.shape}"
        )

    if X_test.shape[1:] != expected_shape:
        raise ValueError(
            f"Expected each sequence to have shape {expected_shape}, "
            f"but got {X_test.shape[1:]}"
        )

    return X_test, y_test


def load_metadata():
    if not METADATA_FILE.exists():
        print(
            "\nWARNING: test_sequence_metadata.csv was not found."
            "\nPredictions will still be generated, but timestamps/"
            "machine IDs will not be attached."
        )
        return None

    metadata = pd.read_csv(METADATA_FILE)

    if len(metadata) == 0:
        raise ValueError("Metadata file is empty.")

    return metadata


def load_scaler():
    """
    Load the scaler using joblib.

    The .pkl extension does not guarantee that the artifact was created
    with pickle.dump(). The previous error:
        _pickle.UnpicklingError: STACK_GLOBAL requires str
    indicates pickle.load() is not the correct loader for this artifact.
    """
    check_file(SCALER_FILE, "Scaler")

    try:
        scaler = joblib.load(SCALER_FILE)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load scaler with joblib:\n{SCALER_FILE}\n\n"
            f"Original error: {exc}"
        ) from exc

    return scaler


def load_models():
    check_file(LSTM_FILE, "LSTM model")
    check_file(GRU_FILE, "GRU model")
    check_file(XGB_FILE, "XGBoost model")

    print("\nLoading frozen models...")

    lstm_model = load_model(LSTM_FILE, compile=False)
    print("  LSTM     : loaded")

    gru_model = load_model(GRU_FILE, compile=False)
    print("  GRU      : loaded")

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(XGB_FILE))
    print("  XGBoost  : loaded")

    return lstm_model, gru_model, xgb_model


# ============================================================
# XGBOOST SEQUENCE TRANSFORMATION
# ============================================================

def make_xgb_features(X):
    """
    Convert each 120 x 174 sequence into the same 1740-feature
    representation used during XGBoost training.

    174 original features x 10 statistics = 1740 features.
    """

    print("\nCreating XGBoost sequence features...")

    n_samples, lookback, n_features = X.shape

    if lookback != LOOKBACK or n_features != EXPECTED_FEATURES:
        raise ValueError(f"Unexpected sequence shape: {X.shape}")

    feature_blocks = [
        np.mean(X, axis=1),
        np.std(X, axis=1),
        np.min(X, axis=1),
        np.max(X, axis=1),
        np.median(X, axis=1),
        X[:, 0, :],
        X[:, -1, :],
        X[:, -1, :] - X[:, 0, :],
        np.abs(X[:, -1, :] - X[:, 0, :]),
        np.max(X, axis=1) - np.min(X, axis=1),
    ]

    X_xgb = np.concatenate(feature_blocks, axis=1)

    expected = EXPECTED_FEATURES * 10

    if X_xgb.shape[1] != expected:
        raise ValueError(
            f"Expected {expected} XGBoost features, "
            f"got {X_xgb.shape[1]}"
        )

    print(f"  XGBoost feature shape: {X_xgb.shape}")

    return X_xgb


# ============================================================
# INFERENCE
# ============================================================

def run_inference(X_test):
    lstm_model, gru_model, xgb_model = load_models()

    print("\nRunning LSTM inference...")
    lstm_risk = lstm_model.predict(
        X_test,
        batch_size=256,
        verbose=1
    ).reshape(-1)

    print("\nRunning GRU inference...")
    gru_risk = gru_model.predict(
        X_test,
        batch_size=256,
        verbose=1
    ).reshape(-1)

    X_xgb = make_xgb_features(X_test)

    print("\nRunning XGBoost inference...")
    xgb_risk = xgb_model.predict_proba(X_xgb)[:, 1]

    return lstm_risk, gru_risk, xgb_risk


def build_results(
    lstm_risk,
    gru_risk,
    xgb_risk,
    y_test=None,
    metadata=None
):
    n = len(lstm_risk)

    results = pd.DataFrame({
        "sequence_index": np.arange(n),
        "lstm_risk": lstm_risk,
        "gru_risk": gru_risk,
        "xgboost_risk": xgb_risk,
    })

    results["lstm_prediction"] = (
        results["lstm_risk"] >= THRESHOLDS["lstm"]
    ).astype(int)

    results["gru_prediction"] = (
        results["gru_risk"] >= THRESHOLDS["gru"]
    ).astype(int)

    results["xgboost_prediction"] = (
        results["xgboost_risk"] >= THRESHOLDS["xgboost"]
    ).astype(int)

    if metadata is not None:
        if len(metadata) != n:
            raise ValueError(
                f"Metadata/inference mismatch: "
                f"metadata={len(metadata)}, predictions={n}"
            )

        preferred = [
            "machine_id",
            "timestamp",
            "failure_2_6h",
        ]

        for col in preferred:
            if col in metadata.columns:
                results[col] = metadata[col].values

        front = [
            col for col in preferred if col in results.columns
        ] + [
            "sequence_index",
            "lstm_risk",
            "lstm_prediction",
            "gru_risk",
            "gru_prediction",
            "xgboost_risk",
            "xgboost_prediction",
        ]

        remaining = [
            col for col in results.columns if col not in front
        ]

        results = results[front + remaining]

    if y_test is not None:
        if len(y_test) != n:
            raise ValueError(
                f"y_test/inference mismatch: "
                f"y_test={len(y_test)}, predictions={n}"
            )

        results["actual_failure_2_6h"] = np.asarray(
            y_test
        ).astype(int)

    return results


def print_summary(results):
    print("\n" + "=" * 70)
    print("PRODUCTION INFERENCE TEST SUMMARY")
    print("=" * 70)

    print(f"Sequences processed : {len(results)}")
    print(f"Expected shape      : ({LOOKBACK}, {EXPECTED_FEATURES})")

    print("\nFrozen thresholds:")
    for model_name, threshold in THRESHOLDS.items():
        print(f"  {model_name:10s}: {threshold:.6f}")

    print("\nPositive predictions:")
    for model_name in ["lstm", "gru", "xgboost"]:
        col = model_name + "_prediction"
        print(
            f"  {model_name:10s}: "
            f"{results[col].sum():5d} "
            f"({results[col].mean() * 100:.2f}%)"
        )

    print("\nRisk-score ranges:")
    for name in ["lstm", "gru", "xgboost"]:
        col = name + "_risk"
        print(
            f"  {name:10s}: "
            f"min={results[col].min():.6f}, "
            f"max={results[col].max():.6f}, "
            f"mean={results[col].mean():.6f}"
        )

    if "actual_failure_2_6h" in results.columns:
        print("\nActual target:")
        print(
            f"  Positive sequences: "
            f"{results['actual_failure_2_6h'].sum()}"
        )
        print(
            f"  Negative sequences: "
            f"{(results['actual_failure_2_6h'] == 0).sum()}"
        )

    print("=" * 70)


def main():
    print("=" * 70)
    print("08 - PRODUCTION INFERENCE")
    print("=" * 70)

    print(f"\nProject root: {PROJECT_ROOT}")

    feature_columns = load_feature_columns()
    print(
        f"\nFeature configuration: "
        f"{len(feature_columns)} model features"
    )

    X_test, y_test = load_test_sequences()

    print("\nLoaded test sequences:")
    print(f"  X_test: {X_test.shape}")

    if y_test is not None:
        print(f"  y_test: {y_test.shape}")

    metadata = load_metadata()

    if metadata is not None:
        print(f"  metadata: {metadata.shape}")

    # Verify the production scaler artifact.
    # X_test is already scaled by the sequence-generation pipeline.
    # DO NOT scale X_test again.
    scaler = load_scaler()

    if hasattr(scaler, "n_features_in_"):
        print(
            f"  scaler expects: "
            f"{scaler.n_features_in_} features"
        )

    lstm_risk, gru_risk, xgb_risk = run_inference(X_test)

    results = build_results(
        lstm_risk,
        gru_risk,
        xgb_risk,
        y_test=y_test,
        metadata=metadata,
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)

    print_summary(results)

    print("\nSaved inference results:")
    print(f"  {OUTPUT_FILE}")

    print("\nSUCCESS")
    print("No model was retrained.")
    print("No threshold was tuned.")
    print("All three frozen models were used for inference.")


if __name__ == "__main__":
    main()
