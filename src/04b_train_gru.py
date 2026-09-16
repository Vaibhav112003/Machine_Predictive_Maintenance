import os
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    GRU,
    Dense,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# CONFIG
# ============================================================

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_PATH = "data/processed/sequences.npz"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading sequences...")

data = np.load(DATA_PATH)

X_train = data["X_train"]
y_train = data["y_train"]

X_val = data["X_validation"]
y_val = data["y_validation"]

X_test = data["X_test"]
y_test = data["y_test"]

print("\nShapes:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_val  :", X_val.shape)
print("y_val  :", y_val.shape)
print("X_test :", X_test.shape)
print("y_test :", y_test.shape)


# ============================================================
# CLASS WEIGHTS
# ============================================================

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = {
    int(cls): float(weight)
    for cls, weight in zip(classes, weights)
}

print("\nClass weights:")
print(class_weights)


# ============================================================
# MODEL
# ============================================================

n_features = X_train.shape[2]

model = Sequential([
    Input(shape=(X_train.shape[1], n_features)),

    GRU(128, return_sequences=True),
    BatchNormalization(),
    Dropout(0.30),

    GRU(64, return_sequences=True),
    Dropout(0.30),

    GRU(32),
    Dropout(0.20),

    Dense(64, activation="relu"),
    BatchNormalization(),
    Dropout(0.20),

    Dense(32, activation="relu"),
    Dropout(0.15),

    Dense(1, activation="sigmoid")
])


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(
            name="roc_auc",
            curve="ROC"
        ),
        tf.keras.metrics.AUC(
            name="pr_auc",
            curve="PR"
        )
    ]
)


# ============================================================
# SUMMARY
# ============================================================

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

checkpoint_path = os.path.join(
    MODEL_DIR,
    "best_gru.keras"
)

callbacks = [

    EarlyStopping(
        monitor="val_pr_auc",
        mode="max",
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),

    ModelCheckpoint(
        checkpoint_path,
        monitor="val_pr_auc",
        mode="max",
        save_best_only=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_pr_auc",
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]


# ============================================================
# TRAIN
# ============================================================

print("\nStarting GRU training...\n")

history = model.fit(

    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=15,

    batch_size=256,

    class_weight=class_weights,

    callbacks=callbacks,

    verbose=1
)


# ============================================================
# SAVE HISTORY
# ============================================================

history_path = os.path.join(
    MODEL_DIR,
    "gru_history.json"
)

with open(history_path, "w") as f:
    json.dump(
        {
            key: [float(x) for x in values]
            for key, values in history.history.items()
        },
        f,
        indent=2
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 60)
print("GRU TRAINING COMPLETE")
print("=" * 60)

print("Best model:")
print(checkpoint_path)

print("History:")
print(history_path)

print("=" * 60)