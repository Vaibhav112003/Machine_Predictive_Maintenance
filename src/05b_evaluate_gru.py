import os
import json
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# CONFIG
# ============================================================

SEQUENCE_PATH = "data/processed/sequences.npz"
MODEL_PATH = "models/best_gru.keras"

OUTPUT_DIR = "reports"
MODEL_DIR = "models"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

data = np.load(SEQUENCE_PATH)

X_val = data["X_validation"]
y_val = data["y_validation"]

X_test = data["X_test"]
y_test = data["y_test"]

print("Validation:", X_val.shape)
print("Test      :", X_test.shape)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading GRU model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Loaded:", MODEL_PATH)


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION PREDICTIONS")
print("=" * 70)

val_prob = model.predict(
    X_val,
    batch_size=256,
    verbose=1
).ravel()


# ============================================================
# THRESHOLD TUNING
# ============================================================

precision, recall, thresholds = precision_recall_curve(
    y_val,
    val_prob
)

# precision and recall have one extra element
f1_scores = (
    2 * precision[:-1] * recall[:-1]
    / (precision[:-1] + recall[:-1] + 1e-12)
)

best_idx = np.argmax(f1_scores)

best_threshold = float(thresholds[best_idx])

best_precision = float(precision[best_idx])
best_recall = float(recall[best_idx])
best_f1 = float(f1_scores[best_idx])


print("\nBest validation threshold:")
print(f"Threshold : {best_threshold:.6f}")
print(f"Precision : {best_precision:.4f}")
print(f"Recall    : {best_recall:.4f}")
print(f"F1        : {best_f1:.4f}")


# ============================================================
# SAVE THRESHOLD
# ============================================================

threshold_path = os.path.join(
    MODEL_DIR,
    "gru_threshold.json"
)

with open(threshold_path, "w") as f:
    json.dump(
        {
            "threshold": best_threshold,
            "selection_metric": "F1",
            "validation_precision": best_precision,
            "validation_recall": best_recall,
            "validation_f1": best_f1
        },
        f,
        indent=2
    )

print("\nSaved threshold:")
print(threshold_path)


# ============================================================
# TEST PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("TEST PREDICTIONS")
print("=" * 70)

test_prob = model.predict(
    X_test,
    batch_size=256,
    verbose=1
).ravel()


# ============================================================
# APPLY THRESHOLD
# ============================================================

test_pred = (
    test_prob >= best_threshold
).astype(int)


# ============================================================
# METRICS
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
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    test_pred
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("GRU TEST RESULTS")
print("=" * 70)

print(f"Threshold : {best_threshold:.6f}")
print(f"Precision : {test_precision:.6f}")
print(f"Recall    : {test_recall:.6f}")
print(f"F1        : {test_f1:.6f}")
print(f"ROC-AUC   : {test_roc_auc:.6f}")
print(f"PR-AUC    : {test_pr_auc:.6f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        test_pred,
        target_names=[
            "Normal",
            "Failure Warning"
        ],
        zero_division=0
    )
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {
    "model": "GRU",
    "threshold": best_threshold,
    "precision": float(test_precision),
    "recall": float(test_recall),
    "f1": float(test_f1),
    "roc_auc": float(test_roc_auc),
    "pr_auc": float(test_pr_auc),
    "confusion_matrix": cm.tolist()
}

metrics_path = os.path.join(
    OUTPUT_DIR,
    "gru_metrics.json"
)

with open(metrics_path, "w") as f:
    json.dump(
        metrics,
        f,
        indent=2
    )


print("\nSaved metrics:")
print(metrics_path)

print("\n" + "=" * 70)
print("GRU EVALUATION COMPLETE")
print("=" * 70)