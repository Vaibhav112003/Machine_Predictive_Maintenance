"""
05_evaluate.py

Evaluate the trained LSTM model.

Threshold is selected using VALIDATION data.
The frozen threshold is then applied to TEST data.

Run:

    python src/05_evaluate.py
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEQUENCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sequences.npz"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "best_lstm.keras"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "reports"
)

FIGURES_DIR = (
    RESULTS_DIR
    / "figures"
)

THRESHOLD_PATH = (
    PROJECT_ROOT
    / "models"
    / "threshold.json"
)

METRICS_PATH = (
    RESULTS_DIR
    / "final_metrics.json"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------

def load_data():

    data = np.load(
        SEQUENCE_PATH
    )

    return (
        data["X_validation"],
        data["y_validation"],
        data["X_test"],
        data["y_test"]
    )


# ---------------------------------------------------------------------
# THRESHOLD OPTIMIZATION
# ---------------------------------------------------------------------

def find_best_threshold(
    y_true,
    probabilities
):

    precision, recall, thresholds = (
        precision_recall_curve(
            y_true,
            probabilities
        )
    )

    f1 = (
        2
        * precision
        * recall
        /
        (
            precision
            + recall
            + 1e-12
        )
    )

    # precision_recall_curve returns
    # one more precision/recall value than
    # thresholds.
    valid_f1 = f1[:-1]

    best_index = np.argmax(
        valid_f1
    )

    threshold = thresholds[
        best_index
    ]

    return float(threshold)


# ---------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------

def calculate_metrics(
    y_true,
    probabilities,
    threshold
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    metrics = {

        "threshold": float(
            threshold
        ),

        "accuracy": float(
            accuracy_score(
                y_true,
                predictions
            )
        ),

        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),

        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),

        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities
            )
        ),

        "pr_auc": float(
            average_precision_score(
                y_true,
                probabilities
            )
        )
    }

    return (
        metrics,
        predictions
    )


# ---------------------------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------------------------

def save_confusion_matrix(
    y_true,
    predictions
):

    cm = confusion_matrix(
        y_true,
        predictions
    )

    plt.figure(
        figsize=(7, 6)
    )

    plt.imshow(
        cm
    )

    plt.title(
        "LSTM Confusion Matrix"
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    plt.xticks(
        [0, 1],
        ["Normal", "Failure Warning"]
    )

    plt.yticks(
        [0, 1],
        ["Normal", "Failure Warning"]
    )

    for i in range(2):

        for j in range(2):

            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    plt.colorbar()

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "confusion_matrix.png",
        dpi=150
    )

    plt.close()

    return cm


# ---------------------------------------------------------------------
# PR CURVE
# ---------------------------------------------------------------------

def save_pr_curve(
    y_true,
    probabilities
):

    precision, recall, _ = (
        precision_recall_curve(
            y_true,
            probabilities
        )
    )

    pr_auc = average_precision_score(
        y_true,
        probabilities
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        recall,
        precision,
        label=f"PR-AUC = {pr_auc:.4f}"
    )

    plt.xlabel(
        "Recall"
    )

    plt.ylabel(
        "Precision"
    )

    plt.title(
        "Precision-Recall Curve"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "precision_recall_curve.png",
        dpi=150
    )

    plt.close()


# ---------------------------------------------------------------------
# ROC CURVE
# ---------------------------------------------------------------------

def save_roc_curve(
    y_true,
    probabilities
):

    fpr, tpr, _ = roc_curve(
        y_true,
        probabilities
    )

    roc_auc = roc_auc_score(
        y_true,
        probabilities
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        fpr,
        tpr,
        label=f"ROC-AUC = {roc_auc:.4f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "ROC Curve"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "roc_curve.png",
        dpi=150
    )

    plt.close()


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    print("=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "Trained model not found.\n"
            "Run:\n"
            "python src/04_train_lstm.py"
        )

    (
        X_validation,
        y_validation,
        X_test,
        y_test
    ) = load_data()

    print(
        "\nLoading model..."
    )

    model = load_model(
        MODEL_PATH
    )

    # -------------------------------------------------------------
    # Validation predictions
    # -------------------------------------------------------------

    print(
        "\nGenerating validation predictions..."
    )

    validation_probabilities = (
        model.predict(
            X_validation,
            batch_size=512,
            verbose=1
        )
        .ravel()
    )

    # -------------------------------------------------------------
    # Find threshold using validation only
    # -------------------------------------------------------------

    threshold = find_best_threshold(
        y_validation,
        validation_probabilities
    )

    print(
        f"\nSelected threshold: "
        f"{threshold:.6f}"
    )

    # Save threshold
    with open(
        THRESHOLD_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "threshold": threshold
            },
            f,
            indent=4
        )

    # -------------------------------------------------------------
    # Test predictions
    # -------------------------------------------------------------

    print(
        "\nGenerating test predictions..."
    )

    test_probabilities = (
        model.predict(
            X_test,
            batch_size=512,
            verbose=1
        )
        .ravel()
    )

    metrics, predictions = calculate_metrics(
        y_test,
        test_probabilities,
        threshold
    )

    # -------------------------------------------------------------
    # Print results
    # -------------------------------------------------------------

    print("\n" + "=" * 80)
    print("FINAL TEST RESULTS")
    print("=" * 80)

    for key, value in metrics.items():

        print(
            f"{key:15s}: {value:.6f}"
        )

    print("\nClassification report:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Normal",
                "Failure Warning"
            ],
            digits=4,
            zero_division=0
        )
    )

    # -------------------------------------------------------------
    # Confusion matrix
    # -------------------------------------------------------------

    cm = save_confusion_matrix(
        y_test,
        predictions
    )

    print(
        "\nConfusion matrix:"
    )

    print(cm)

    # -------------------------------------------------------------
    # Curves
    # -------------------------------------------------------------

    save_pr_curve(
        y_test,
        test_probabilities
    )

    save_roc_curve(
        y_test,
        test_probabilities
    )

    # -------------------------------------------------------------
    # Save metrics
    # -------------------------------------------------------------

    metrics["true_negative"] = int(
        cm[0, 0]
    )

    metrics["false_positive"] = int(
        cm[0, 1]
    )

    metrics["false_negative"] = int(
        cm[1, 0]
    )

    metrics["true_positive"] = int(
        cm[1, 1]
    )

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    print(
        f"\nMetrics saved to:\n"
        f"{METRICS_PATH}"
    )

    print(
        f"\nThreshold saved to:\n"
        f"{THRESHOLD_PATH}"
    )


if __name__ == "__main__":
    main()