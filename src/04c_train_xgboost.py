import os
import json
import numpy as np

from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

SEQUENCE_PATH = "data/processed/sequences.npz"

MODEL_DIR = "models"
REPORT_DIR = "reports"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# ============================================================
# LOAD SEQUENCE DATA
# ============================================================

print("=" * 70)
print("LOADING SEQUENCE DATA")
print("=" * 70)

data = np.load(SEQUENCE_PATH)

X_train = data["X_train"]
y_train = data["y_train"]

X_val = data["X_validation"]
y_val = data["y_validation"]

X_test = data["X_test"]
y_test = data["y_test"]


print("\nOriginal sequence shapes:")

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_val  :", X_val.shape)
print("y_val  :", y_val.shape)

print("X_test :", X_test.shape)
print("y_test :", y_test.shape)


# ============================================================
# SEQUENCE → TABULAR FEATURES
# ============================================================

def create_tabular_features(X):
    """
    Convert 3D time-series sequences into 2D tabular features.

    Input:
        X = samples × time_steps × features

    Output:
        samples × engineered_features

    For every original feature we calculate:

        mean
        std
        min
        max
        median
        first value
        last value
        change
        absolute change
        range
    """

    print("\nCreating tabular features...")

    # --------------------------------------------------------
    # Statistical features over 120-minute window
    # --------------------------------------------------------

    mean_features = np.mean(X, axis=1)

    std_features = np.std(X, axis=1)

    min_features = np.min(X, axis=1)

    max_features = np.max(X, axis=1)

    median_features = np.median(X, axis=1)

    # --------------------------------------------------------
    # Temporal features
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Combine everything
    # --------------------------------------------------------

    tabular = np.concatenate(
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

    return tabular


# ============================================================
# CREATE TABULAR DATA
# ============================================================

print("\n" + "=" * 70)
print("FEATURE TRANSFORMATION")
print("=" * 70)

X_train_tabular = create_tabular_features(X_train)

X_val_tabular = create_tabular_features(X_val)

X_test_tabular = create_tabular_features(X_test)


print("\nTabular shapes:")

print(
    "X_train_tabular:",
    X_train_tabular.shape
)

print(
    "X_val_tabular  :",
    X_val_tabular.shape
)

print(
    "X_test_tabular :",
    X_test_tabular.shape
)


# ============================================================
# CLASS IMBALANCE
# ============================================================

negative_count = np.sum(y_train == 0)

positive_count = np.sum(y_train == 1)

scale_pos_weight = (
    negative_count / positive_count
)


print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

print("Negative samples:", negative_count)

print("Positive samples:", positive_count)

print(
    "Scale positive weight:",
    scale_pos_weight
)


# ============================================================
# CREATE XGBOOST MODEL
# ============================================================

print("\n" + "=" * 70)
print("CREATING XGBOOST MODEL")
print("=" * 70)


model = XGBClassifier(

    objective="binary:logistic",

    eval_metric="aucpr",

    # Number of trees
    n_estimators=500,

    # Small learning rate
    learning_rate=0.03,

    # Tree complexity
    max_depth=6,

    min_child_weight=5,

    # Random sampling
    subsample=0.8,

    colsample_bytree=0.8,

    # Regularization
    gamma=0,

    reg_alpha=0.1,

    reg_lambda=1.0,

    # Handle class imbalance
    scale_pos_weight=scale_pos_weight,

    random_state=SEED,

    # Use all CPU cores
    n_jobs=-1,

    # Efficient tree construction
    tree_method="hist",
)


print("\nXGBoost model created.")


# ============================================================
# TRAIN MODEL
# ============================================================

print("\n" + "=" * 70)
print("STARTING XGBOOST TRAINING")
print("=" * 70)

model.fit(

    X_train_tabular,

    y_train,

    eval_set=[
        (X_train_tabular, y_train),
        (X_val_tabular, y_val),
    ],

    verbose=25,
)


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION EVALUATION")
print("=" * 70)


val_prob = model.predict_proba(
    X_val_tabular
)[:, 1]


val_pr_auc = average_precision_score(
    y_val,
    val_prob
)


val_roc_auc = roc_auc_score(
    y_val,
    val_prob
)


print(
    f"\nValidation PR-AUC : {val_pr_auc:.6f}"
)

print(
    f"Validation ROC-AUC: {val_roc_auc:.6f}"
)


# ============================================================
# THRESHOLD TUNING
# ============================================================

print("\n" + "=" * 70)
print("THRESHOLD TUNING")
print("=" * 70)


thresholds = np.arange(
    0.01,
    0.51,
    0.001
)


best_threshold = 0.5

best_f1 = -1.0

best_precision = 0.0

best_recall = 0.0


for threshold in thresholds:

    val_pred = (
        val_prob >= threshold
    ).astype(int)

    precision = precision_score(
        y_val,
        val_pred,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        val_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        val_pred,
        zero_division=0
    )

    if f1 > best_f1:

        best_f1 = f1

        best_threshold = threshold

        best_precision = precision

        best_recall = recall


print("\nBest validation threshold:")

print(
    f"Threshold : {best_threshold:.6f}"
)

print(
    f"Precision : {best_precision:.6f}"
)

print(
    f"Recall    : {best_recall:.6f}"
)

print(
    f"F1        : {best_f1:.6f}"
)


# ============================================================
# TEST PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("TEST EVALUATION")
print("=" * 70)


test_prob = model.predict_proba(
    X_test_tabular
)[:, 1]


# Apply threshold selected ONLY using validation data
test_pred = (
    test_prob >= best_threshold
).astype(int)


# ============================================================
# TEST METRICS
# ============================================================

test_precision = precision_score(
    y_test,
    test_pred,
    zero_division=0
)


test_recall = recall_score(
    y_test,
    test_pred,
    zero_division=0
)


test_f1 = f1_score(
    y_test,
    test_pred,
    zero_division=0
)


test_roc_auc = roc_auc_score(
    y_test,
    test_prob
)


test_pr_auc = average_precision_score(
    y_test,
    test_prob
)


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("XGBOOST TEST RESULTS")
print("=" * 70)

print(
    f"Threshold : {best_threshold:.6f}"
)

print(
    f"Precision : {test_precision:.6f}"
)

print(
    f"Recall    : {test_recall:.6f}"
)

print(
    f"F1        : {test_f1:.6f}"
)

print(
    f"ROC-AUC   : {test_roc_auc:.6f}"
)

print(
    f"PR-AUC    : {test_pr_auc:.6f}"
)


# ============================================================
# CONFUSION MATRIX VALUES
# ============================================================

true_negative = np.sum(
    (y_test == 0) & (test_pred == 0)
)

false_positive = np.sum(
    (y_test == 0) & (test_pred == 1)
)

false_negative = np.sum(
    (y_test == 1) & (test_pred == 0)
)

true_positive = np.sum(
    (y_test == 1) & (test_pred == 1)
)


print("\nConfusion Matrix:")

print(
    "[[{} {}]".format(
        true_negative,
        false_positive
    )
)

print(
    " [{} {}]]".format(
        false_negative,
        true_positive
    )
)


print("\nDetailed counts:")

print("True Negatives :", true_negative)

print("False Positives:", false_positive)

print("False Negatives:", false_negative)

print("True Positives :", true_positive)


# ============================================================
# SAVE MODEL
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "xgboost_model.json"
)


model.save_model(
    model_path
)


print("\nSaved XGBoost model:")

print(model_path)


# ============================================================
# SAVE THRESHOLD
# ============================================================

threshold_path = os.path.join(
    MODEL_DIR,
    "xgboost_threshold.json"
)


threshold_data = {

    "model": "XGBoost",

    "threshold": float(
        best_threshold
    ),

    "selection_metric": "F1",

    "validation_precision": float(
        best_precision
    ),

    "validation_recall": float(
        best_recall
    ),

    "validation_f1": float(
        best_f1
    ),
}


with open(
    threshold_path,
    "w"
) as f:

    json.dump(
        threshold_data,
        f,
        indent=4
    )


print("\nSaved threshold:")

print(threshold_path)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {

    "model": "XGBoost",

    "threshold": float(
        best_threshold
    ),

    "validation": {

        "precision": float(
            best_precision
        ),

        "recall": float(
            best_recall
        ),

        "f1": float(
            best_f1
        ),

        "pr_auc": float(
            val_pr_auc
        ),

        "roc_auc": float(
            val_roc_auc
        ),
    },

    "test": {

        "precision": float(
            test_precision
        ),

        "recall": float(
            test_recall
        ),

        "f1": float(
            test_f1
        ),

        "pr_auc": float(
            test_pr_auc
        ),

        "roc_auc": float(
            test_roc_auc
        ),

        "true_negative": int(
            true_negative
        ),

        "false_positive": int(
            false_positive
        ),

        "false_negative": int(
            false_negative
        ),

        "true_positive": int(
            true_positive
        ),
    },
}


metrics_path = os.path.join(
    REPORT_DIR,
    "xgboost_metrics.json"
)


with open(
    metrics_path,
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )


print("\nSaved metrics:")

print(metrics_path)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)

print("XGBOOST TRAINING AND EVALUATION COMPLETE")

print("=" * 70)