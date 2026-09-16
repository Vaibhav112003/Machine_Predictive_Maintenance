"""
04_train_lstm.py

Train deep-learning LSTM model for 2-6 hour
industrial maintenance early warning.

Run:

    python src/04_train_lstm.py
"""

from pathlib import Path
import json
import random
import warnings

import numpy as np
import tensorflow as tf

from sklearn.utils.class_weight import (
    compute_class_weight
)

from tensorflow.keras import (
    Sequential
)

from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Dropout,
    BatchNormalization
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    CSVLogger
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# REPRODUCIBILITY
# ---------------------------------------------------------------------

SEED = 42

random.seed(SEED)

np.random.seed(SEED)

tf.random.set_seed(SEED)

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

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

BEST_MODEL_PATH = (
    MODEL_DIR
    / "best_lstm.keras"
)

FINAL_MODEL_PATH = (
    MODEL_DIR
    / "predictive_maintenance_lstm.keras"
)

HISTORY_PATH = (
    MODEL_DIR
    / "training_history.json"
)

LOG_PATH = (
    MODEL_DIR
    / "training_log.csv"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ---------------------------------------------------------------------
# TRAINING CONFIG
# ---------------------------------------------------------------------

EPOCHS = 60

BATCH_SIZE = 256

LEARNING_RATE = 0.001


# ---------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------

def load_sequences():

    if not SEQUENCE_PATH.exists():

        raise FileNotFoundError(
            "sequences.npz not found.\n"
            "Run:\n"
            "python src/03_create_sequences.py"
        )

    data = np.load(
        SEQUENCE_PATH
    )

    return (
        data["X_train"],
        data["y_train"],
        data["X_validation"],
        data["y_validation"],
        data["X_test"],
        data["y_test"]
    )


# ---------------------------------------------------------------------
# CLASS WEIGHTS
# ---------------------------------------------------------------------

def calculate_class_weights(
    y_train
):

    classes = np.unique(
        y_train
    )

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )

    class_weights = {
        int(cls): float(weight)
        for cls, weight
        in zip(classes, weights)
    }

    return class_weights


# ---------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------

def build_model(
    sequence_length,
    number_of_features
):

    model = Sequential(
        [

            Input(
                shape=(
                    sequence_length,
                    number_of_features
                )
            ),

            LSTM(
                128,
                return_sequences=True
            ),

            BatchNormalization(),

            Dropout(0.30),

            LSTM(
                64,
                return_sequences=True
            ),

            Dropout(0.30),

            LSTM(
                32,
                return_sequences=False
            ),

            Dropout(0.20),

            Dense(
                64,
                activation="relu"
            ),

            BatchNormalization(),

            Dropout(0.20),

            Dense(
                32,
                activation="relu"
            ),

            Dropout(0.15),

            Dense(
                1,
                activation="sigmoid"
            )
        ]
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    )

    model.compile(
        optimizer=optimizer,

        loss="binary_crossentropy",

        metrics=[

            tf.keras.metrics.BinaryAccuracy(
                name="accuracy"
            ),

            tf.keras.metrics.Precision(
                name="precision"
            ),

            tf.keras.metrics.Recall(
                name="recall"
            ),

            tf.keras.metrics.AUC(
                name="roc_auc"
            ),

            tf.keras.metrics.AUC(
                name="pr_auc",
                curve="PR"
            )
        ]
    )

    return model


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    print("=" * 80)
    print("LSTM TRAINING")
    print("=" * 80)

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test
    ) = load_sequences()

    print(
        f"X_train      : {X_train.shape}"
    )

    print(
        f"X_validation : {X_validation.shape}"
    )

    print(
        f"X_test       : {X_test.shape}"
    )

    print(
        "\nTraining target distribution:"
    )

    unique, counts = np.unique(
        y_train,
        return_counts=True
    )

    for label, count in zip(
        unique,
        counts
    ):

        print(
            f"  Class {label}: "
            f"{count:,} "
            f"({count / len(y_train) * 100:.2f}%)"
        )

    # -------------------------------------------------------------
    # Class weights
    # -------------------------------------------------------------

    class_weights = calculate_class_weights(
        y_train
    )

    print(
        "\nClass weights:"
    )

    print(class_weights)

    # -------------------------------------------------------------
    # Model
    # -------------------------------------------------------------

    model = build_model(
        sequence_length=X_train.shape[1],
        number_of_features=X_train.shape[2]
    )

    print("\nModel architecture:")

    model.summary()

    # -------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------

    callbacks = [

        EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
            patience=12,
            restore_best_weights=True,
            verbose=1
        ),

        ReduceLROnPlateau(
            monitor="val_pr_auc",
            mode="max",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),

        ModelCheckpoint(
            filepath=str(
                BEST_MODEL_PATH
            ),
            monitor="val_pr_auc",
            mode="max",
            save_best_only=True,
            verbose=1
        ),

        CSVLogger(
            filename=str(
                LOG_PATH
            )
        )
    ]

    # -------------------------------------------------------------
    # Train
    # -------------------------------------------------------------

    history = model.fit(

        X_train,
        y_train,

        validation_data=(
            X_validation,
            y_validation
        ),

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        class_weight=class_weights,

        callbacks=callbacks,

        verbose=1,

        shuffle=True
    )

    # -------------------------------------------------------------
    # Save final model
    # -------------------------------------------------------------

    model.save(
        FINAL_MODEL_PATH
    )

    # -------------------------------------------------------------
    # Save history
    # -------------------------------------------------------------

    history_dict = {
        key: [
            float(value)
            for value in values
        ]
        for key, values
        in history.history.items()
    }

    with open(
        HISTORY_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            history_dict,
            f,
            indent=4
        )

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)

    print(
        f"Best model : {BEST_MODEL_PATH}"
    )

    print(
        f"Final model: {FINAL_MODEL_PATH}"
    )

    print(
        f"Training log: {LOG_PATH}"
    )


if __name__ == "__main__":
    main()