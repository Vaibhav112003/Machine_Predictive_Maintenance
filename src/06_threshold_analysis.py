import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from xgboost import XGBClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
)


# ============================================================
# CONFIG
# ============================================================

SEQUENCE_PATH = "data/processed/sequences.npz"

LSTM_MODEL_PATH = "models/best_lstm.keras"
GRU_MODEL_PATH = "models/best_gru.keras"
XGB_MODEL_PATH = "models/xgboost_model.json"

REPORT_DIR = "reports"

os.makedirs(REPORT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING TEST DATA")
print("=" * 70)

data = np.load(SEQUENCE_PATH)

X_test = data["X_test"]
y_test = data["y_test"]

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# ============================================================
# CREATE XGBOOST FEATURES
# ============================================================

def create_tabular_features(X):

    mean_features = np.mean(X, axis=1)

    std_features = np.std(X, axis=1)

    min_features = np.min(X, axis=1)

    max_features = np.max(X, axis=1)

    median_features = np.median(X, axis=1)

    first_features = X[:, 0, :]

    last_features = X[:, -1, :]

    change_features = (
        X[:, -1, :] - X[:, 0, :]
    )

    abs_change_features = np.abs(
        X[:, -1, :] - X[:, 0, :]
    )

    range_features = (
        max_features - min_features
    )

    return np.concatenate(
        [
            mean_features,
            std_features,
            min_features,
            max_features,
            median_features,
            first_features,
            last_features,
            change_features,
            abs_change_features,
            range_features,
        ],
        axis=1,
    )


# ============================================================
# LOAD MODELS
# ============================================================

print("\n" + "=" * 70)
print("LOADING MODELS")
print("=" * 70)

print("\nLoading LSTM...")

lstm = tf.keras.models.load_model(
    LSTM_MODEL_PATH
)

print("LSTM loaded.")

print("\nLoading GRU...")

gru = tf.keras.models.load_model(
    GRU_MODEL_PATH
)

print("GRU loaded.")

print("\nLoading XGBoost...")

xgb = XGBClassifier()

xgb.load_model(
    XGB_MODEL_PATH
)

print("XGBoost loaded.")


# ============================================================
# PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("GENERATING TEST PREDICTIONS")
print("=" * 70)


# ------------------------------------------------------------
# LSTM
# ------------------------------------------------------------

print("\nLSTM predictions...")

lstm_prob = lstm.predict(
    X_test,
    batch_size=256,
    verbose=1
).ravel()


# ------------------------------------------------------------
# GRU
# ------------------------------------------------------------

print("\nGRU predictions...")

gru_prob = gru.predict(
    X_test,
    batch_size=256,
    verbose=1
).ravel()


# ------------------------------------------------------------
# XGBoost
# ------------------------------------------------------------

print("\nCreating XGBoost tabular features...")

X_test_tabular = create_tabular_features(
    X_test
)

print(
    "X_test_tabular:",
    X_test_tabular.shape
)

print("\nXGBoost predictions...")

xgb_prob = xgb.predict_proba(
    X_test_tabular
)[:, 1]


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

thresholds = np.arange(
    0.01,
    0.51,
    0.01
)


results = []


models = {

    "LSTM": lstm_prob,

    "GRU": gru_prob,

    "XGBoost": xgb_prob,
}


for model_name, probabilities in models.items():

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities
    )

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        false_positives = np.sum(
            (y_test == 0) &
            (predictions == 1)
        )

        false_negatives = np.sum(
            (y_test == 1) &
            (predictions == 0)
        )

        true_positives = np.sum(
            (y_test == 1) &
            (predictions == 1)
        )

        true_negatives = np.sum(
            (y_test == 0) &
            (predictions == 0)
        )

        results.append({

            "model": model_name,

            "threshold": float(
                threshold
            ),

            "precision": float(
                precision
            ),

            "recall": float(
                recall
            ),

            "f1": float(
                f1
            ),

            "roc_auc": float(
                roc_auc
            ),

            "pr_auc": float(
                pr_auc
            ),

            "true_positive": int(
                true_positives
            ),

            "false_positive": int(
                false_positives
            ),

            "true_negative": int(
                true_negatives
            ),

            "false_negative": int(
                false_negatives
            ),
        })


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

output_path = (
    "reports/"
    "threshold_analysis.csv"
)

results_df.to_csv(
    output_path,
    index=False
)


print("\n" + "=" * 70)
print("THRESHOLD ANALYSIS COMPLETE")
print("=" * 70)

print(
    "\nSaved:",
    output_path
)


# ============================================================
# SHOW BEST F1 PER MODEL
# ============================================================

print("\n" + "=" * 70)
print("BEST F1 THRESHOLD")
print("=" * 70)


for model_name in models.keys():

    model_results = results_df[
        results_df["model"] == model_name
    ]

    best_row = model_results.loc[
        model_results["f1"].idxmax()
    ]

    print(f"\n{model_name}")

    print(
        f"Threshold : "
        f"{best_row['threshold']:.2f}"
    )

    print(
        f"Precision : "
        f"{best_row['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{best_row['recall']:.4f}"
    )

    print(
        f"F1        : "
        f"{best_row['f1']:.4f}"
    )

    print(
        f"FP        : "
        f"{best_row['false_positive']}"
    )

    print(
        f"FN        : "
        f"{best_row['false_negative']}"
    )