import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from gg_stnet_model import build_gg_stnet

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# -----------------------------
# Config
# -----------------------------
FEATURES = ["GR", "AC", "DEN", "RT", "SP"]
LABEL_COL = "Lithology_Label"
NUM_CLASSES = 7

WINDOW_SIZE = 32
STEP_SIZE = 16
LABEL_MODE = "center"   

BATCH_SIZE = 128
EPOCHS = 200
LEARNING_RATE = 1e-3
PATIENCE = 20

MODEL_DIR = Path("outputs")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Utilities
# -----------------------------
def read_split_file(txt_path: str) -> list[str]:
    with open(txt_path, "r", encoding="utf-8") as f:
        files = [line.strip() for line in f if line.strip()]
    return files


def load_single_well(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)

    missing_cols = [c for c in FEATURES + [LABEL_COL] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{csv_path} is missing columns: {missing_cols}")

    x = df[FEATURES].values.astype(np.float32)
    y = df[LABEL_COL].values.astype(np.int32)
    return x, y


def fit_scaler_on_training_wells(train_files: list[str]) -> StandardScaler:
    scaler = StandardScaler()

    for fp in train_files:
        x, _ = load_single_well(fp)
        scaler.partial_fit(x)

    return scaler


def make_windows(
    x: np.ndarray,
    y: np.ndarray,
    scaler: StandardScaler,
    window_size: int = WINDOW_SIZE,
    step_size: int = STEP_SIZE,
    label_mode: str = LABEL_MODE
) -> tuple[np.ndarray, np.ndarray]:
    x_scaled = scaler.transform(x).astype(np.float32)

    if label_mode == "center":
        target_offset = window_size // 2
    elif label_mode == "last":
        target_offset = window_size - 1
    else:
        raise ValueError("label_mode must be 'center' or 'last'")

    windows, labels = [], []

    for start in range(0, len(x_scaled) - window_size + 1, step_size):
        end = start + window_size
        windows.append(x_scaled[start:end])
        labels.append(y[start + target_offset])

    if not windows:
        return (
            np.empty((0, window_size, len(FEATURES)), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    return np.asarray(windows, dtype=np.float32), np.asarray(labels, dtype=np.int32)


def build_dataset(
    files: list[str],
    scaler: StandardScaler,
    window_size: int = WINDOW_SIZE,
    step_size: int = STEP_SIZE,
    label_mode: str = LABEL_MODE
) -> tuple[np.ndarray, np.ndarray]:
    all_x, all_y = [], []

    for fp in files:
        x, y = load_single_well(fp)
        x_win, y_win = make_windows(
            x, y, scaler,
            window_size=window_size,
            step_size=step_size,
            label_mode=label_mode
        )
        if len(x_win) > 0:
            all_x.append(x_win)
            all_y.append(y_win)

    if not all_x:
        raise ValueError("No windows were generated. Check input files and window settings.")

    return np.concatenate(all_x, axis=0), np.concatenate(all_y, axis=0)


def evaluate_split(model: tf.keras.Model, x: np.ndarray, y: np.ndarray, split_name: str) -> None:
    loss, acc = model.evaluate(x, y, verbose=0)
    prob = model.predict(x, verbose=0)
    pred = np.argmax(prob, axis=1)

    print(f"\n[{split_name}] Loss: {loss:.4f} | Accuracy: {acc:.4f}")
    print(classification_report(y, pred, digits=4))
    print(f"[{split_name}] Confusion Matrix:\n{confusion_matrix(y, pred)}")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    train_files = read_split_file("splits/train_wells.txt")
    val_files = read_split_file("splits/val_wells.txt")
    test_files = read_split_file("splits/test_wells.txt")
    blind_files = read_split_file("splits/blind_wells.txt")

    print("[*] Fitting scaler on training wells only ...")
    scaler = fit_scaler_on_training_wells(train_files)
    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")

    print("[*] Building datasets ...")
    x_train, y_train = build_dataset(train_files, scaler)
    x_val, y_val = build_dataset(val_files, scaler)
    x_test, y_test = build_dataset(test_files, scaler)
    x_blind, y_blind = build_dataset(blind_files, scaler)

    print(f"[*] Train   : X={x_train.shape}, y={y_train.shape}")
    print(f"[*] Val     : X={x_val.shape}, y={y_val.shape}")
    print(f"[*] Test    : X={x_test.shape}, y={y_test.shape}")
    print(f"[*] Blind   : X={x_blind.shape}, y={y_blind.shape}")

    print("[*] Building GG-STNet ...")
    model = build_gg_stnet(
        sequence_length=WINDOW_SIZE,
        num_features=len(FEATURES),
        num_classes=NUM_CLASSES
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_DIR / "best_gg_stnet.keras"),
            monitor="val_loss",
            save_best_only=True
        )
    ]

    print("[*] Starting training ...")
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    print("[*] Evaluating on in-area test set ...")
    evaluate_split(model, x_test, y_test, "Test")

    print("[*] Evaluating on external blind-test set ...")
    evaluate_split(model, x_blind, y_blind, "Blind-Test")

    print("\n[SUCCESS] Training and evaluation finished.")
