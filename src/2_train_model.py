"""
src/2_train_model.py
─────────────────────────────────────────────────────────────
Option 2 – LSTM model training.

Loads all .npy keypoint sequences from data/, builds a 3-layer
stacked LSTM network, and trains it with:
  • TensorBoard callback  – live loss / accuracy curves
  • EarlyStopping         – prevents over-fitting
  • ModelCheckpoint       – saves only the best val_accuracy weights

Outputs
-------
  models/gesture_model.h5   – best model weights (Keras HDF5 format)
  models/label_map.npy      – ordered list of gesture class names
  logs/train/               – TensorBoard event files

Usage (standalone):
  python src/2_train_model.py

Usage (via main.py):
  Select option 2 from the main menu.
"""

import sys
import os

# ── Make project root importable regardless of invocation method ───────────────
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)
# ── Suppress TF / oneDNN startup noise ────────────────────────────────────────
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL",  "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL",    "3")
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
warnings.filterwarnings("ignore", message=".*HDF5.*")
from src.config import (
    DATA_PATH,
    LOG_PATH,
    MODEL_PATH,
    MODELS_DIR,
    LABELS_PATH,
    NUM_SEQUENCES,
    SEQUENCE_LENGTH,
    NUM_KEYPOINT_FEATURES,
    TEST_SIZE,
    BATCH_SIZE,
    MAX_EPOCHS,
    LEARNING_RATE,
    EARLY_STOP_PAT,
    setup_gpu,
)

import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    TensorBoard,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


# ─── Data loading ─────────────────────────────────────────────────────────────

def _load_data():
    """
    Walk the data/ directory and load every complete gesture sequence.

    Returns
    -------
    X       : np.ndarray  shape (N, SEQUENCE_LENGTH, NUM_KEYPOINT_FEATURES)
    y       : np.ndarray  one-hot encoded labels  shape (N, num_classes)
    actions : list[str]   ordered gesture class names (index == label int)

    Returns (None, None, None) if no usable data is found.
    """
    if not os.path.isdir(DATA_PATH):
        print(f"[ERROR] Data directory not found: {DATA_PATH}")
        print("        Run Option 1 first to collect gesture data.")
        return None, None, None

    # Discover gesture classes (each sub-folder = one class)
    actions = sorted(
        d for d in os.listdir(DATA_PATH)
        if os.path.isdir(os.path.join(DATA_PATH, d))
    )

    if not actions:
        print("[ERROR] No gesture folders found inside data/.")
        print("        Run Option 1 first to collect gesture data.")
        return None, None, None

    print(f"[INFO] Found {len(actions)} gesture class(es): {actions}")
    label_map = {name: idx for idx, name in enumerate(actions)}

    sequences, labels = [], []

    for action in actions:
        action_path = os.path.join(DATA_PATH, action)

        # Sort sequence folders numerically (0, 1, …, 29)
        seq_dirs = sorted(
            d for d in os.listdir(action_path)
            if os.path.isdir(os.path.join(action_path, d))
        )
        seq_dirs.sort(key=lambda x: int(x))

        skipped = 0
        for seq_dir in tqdm(seq_dirs, desc=f"  Loading '{action}'", unit="seq", leave=False):
            seq_path = os.path.join(action_path, seq_dir)
            window = []

            for frame_num in range(SEQUENCE_LENGTH):
                npy_file = os.path.join(seq_path, f"{frame_num}.npy")
                if not os.path.exists(npy_file):
                    # Incomplete sequence – skip silently with a counter
                    skipped += 1
                    window = []
                    break
                window.append(np.load(npy_file).astype(np.float32))

            if len(window) == SEQUENCE_LENGTH:
                sequences.append(window)
                labels.append(label_map[action])

        print(f"  {action:<20s}  sequences loaded: "
              f"{len(seq_dirs) - skipped}/{len(seq_dirs)}")

    if not sequences:
        print("[ERROR] No complete sequences found.")
        return None, None, None

    X = np.array(sequences, dtype=np.float32)               # (N, 30, 1662)
    y_int = np.array(labels)
    y = to_categorical(y_int, num_classes=len(actions))      # (N, C)

    print(f"\n[INFO] Dataset shape → X: {X.shape}  |  y: {y.shape}")
    counts = {a: int(np.sum(y_int == i)) for i, a in enumerate(actions)}
    print(f"[INFO] Samples per class: { {k: v for k, v in counts.items()} }")
    return X, y, actions, y_int


# ─── Model architecture ───────────────────────────────────────────────────────

def _build_model(num_classes: int) -> tf.keras.Model:
    """
    Three-layer stacked LSTM + dense head for temporal gesture classification.

    Input  : (SEQUENCE_LENGTH, NUM_KEYPOINT_FEATURES) = (30, 1662)
    Output : softmax probability over num_classes gesture labels

    Architecture
    ────────────
      LSTM(64,  return_seq=True,  tanh) → BatchNorm → Dropout(0.50)
      LSTM(128, return_seq=True,  tanh) → BatchNorm → Dropout(0.50)
      LSTM(64,  return_seq=False, tanh) → BatchNorm → Dropout(0.40)
      Dense(64, relu)                   → Dropout(0.30)
      Dense(32, relu)
      Dense(num_classes, softmax)
    """
    model = Sequential(
        [
            # ── LSTM block 1 ──────────────────────────────────────────────
            LSTM(
                64,
                return_sequences=True,
                activation="tanh",
                input_shape=(SEQUENCE_LENGTH, NUM_KEYPOINT_FEATURES),
                kernel_regularizer=tf.keras.regularizers.l2(1e-3),
            ),
            BatchNormalization(),
            Dropout(0.50),

            # ── LSTM block 2 ──────────────────────────────────────────────
            LSTM(
                128,
                return_sequences=True,
                activation="tanh",
                kernel_regularizer=tf.keras.regularizers.l2(1e-3),
            ),
            BatchNormalization(),
            Dropout(0.50),

            # ── LSTM block 3 ──────────────────────────────────────────────
            LSTM(
                64,
                return_sequences=False,
                activation="tanh",
                kernel_regularizer=tf.keras.regularizers.l2(1e-3),
            ),
            BatchNormalization(),
            Dropout(0.40),

            # ── Dense classification head ─────────────────────────────────
            Dense(64, activation="relu"),
            Dropout(0.30),
            Dense(32, activation="relu"),
            Dense(num_classes, activation="softmax"),
        ],
        name="gesture_lstm",
    )
    return model


# ─── Main training routine ────────────────────────────────────────────────────

def train_model() -> None:
    """
    End-to-end training pipeline:
      1. Configure GPU
      2. Load & split dataset
      3. Build LSTM model (pushed explicitly to GPU)
      4. Fit with TensorBoard / EarlyStopping / ModelCheckpoint callbacks
      5. Evaluate on held-out test set
      6. Save label map alongside the model
    """
    # ── 1. GPU ───────────────────────────────────────────────────────────────
    gpus = setup_gpu()
    device = "/GPU:0" if gpus else "/CPU:0"
    print(f"[INFO] Training device: {device}")
    print(f"[INFO] TensorFlow version: {tf.__version__}\n")

    # ── 2. Load data ─────────────────────────────────────────────────────────
    X, y, actions, y_int = _load_data()
    if X is None:
        return

    num_classes = len(actions)

    # Stratified split keeps class proportions in both sets.
    # Fall back to non-stratified if any class has fewer than 2 samples.
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=42,
            stratify=y_int,
        )
    except ValueError:
        print("[WARNING] Too few samples for stratified split – using random split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=42,
        )

    print(
        f"\n[INFO] Train: {X_train.shape[0]} samples  |  "
        f"Test: {X_test.shape[0]} samples  (before augmentation)"
    )

    # ── 2b. Double the training set with Gaussian-noise copies ───────────────
    rng   = np.random.default_rng(seed=42)
    X_aug = (X_train + rng.normal(0.0, 0.01, X_train.shape)).astype(np.float32)
    y_train_int = np.argmax(y_train, axis=1)
    X_train = np.concatenate([X_train, X_aug], axis=0)
    y_train = np.concatenate([y_train, y_train], axis=0)
    y_train_int = np.concatenate([y_train_int, y_train_int], axis=0)
    idx = rng.permutation(len(X_train))
    X_train, y_train, y_train_int = X_train[idx], y_train[idx], y_train_int[idx]

    # Compute class weights to counter imbalance
    cw_values = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=y_train_int,
    )
    class_weight_dict = dict(enumerate(cw_values))
    print(f"[INFO] Class weights: { {actions[i]: round(v, 3) for i, v in enumerate(cw_values)} }")
    print(f"[INFO] Augmented train: {X_train.shape[0]} samples\n")

    # ── 3. Build model on the target device ──────────────────────────────────
    with tf.device(device):
        model = _build_model(num_classes)
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss="categorical_crossentropy",
            metrics=["categorical_accuracy"],
        )

    model.summary()

    # ── 4. Callbacks ─────────────────────────────────────────────────────────
    os.makedirs(LOG_PATH,   exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    callbacks = [
        # TensorBoard – run `tensorboard --logdir logs/train` to visualise
        TensorBoard(
            log_dir=LOG_PATH,
            histogram_freq=1,
            write_graph=True,
            update_freq="epoch",
        ),
        # Halve the LR whenever val_categorical_accuracy plateaus for 8 epochs
        ReduceLROnPlateau(
            monitor="val_categorical_accuracy",
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            mode="max",
            verbose=1,
        ),
        # Stop training when val_categorical_accuracy stops improving
        EarlyStopping(
            monitor="val_categorical_accuracy",
            patience=EARLY_STOP_PAT,
            restore_best_weights=True,
            mode="max",
            verbose=1,
        ),
        # Persist the epoch with the lowest validation loss
        # (val_loss is the right discriminator once val_accuracy saturates at 1.0)
        ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,
        ),
    ]

    print(f"\n[INFO] TensorBoard log:  {LOG_PATH}")
    print(f"[INFO] Best model → {MODEL_PATH}")
    print(f"[INFO] Max epochs: {MAX_EPOCHS}  |  Batch size: {BATCH_SIZE}")
    print(f"[INFO] Early stopping patience: {EARLY_STOP_PAT} epochs\n")

    # ── 5. Train ─────────────────────────────────────────────────────────────
    with tf.device(device):
        history = model.fit(  # noqa: F841  (unused – available for inspection)
            X_train, y_train,
            epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1,
        )

    # ── 6. Evaluate ──────────────────────────────────────────────────────────
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n[RESULT] Test loss: {loss:.4f}  |  Test accuracy: {accuracy * 100:.2f}%")

    # ── 7. Save label map ────────────────────────────────────────────────────
    np.save(LABELS_PATH, np.array(actions))
    print(f"\n[SUCCESS] Model saved  →  {MODEL_PATH}")
    print(f"[SUCCESS] Labels saved →  {LABELS_PATH}")
    print(
        f"\n[TIP] Launch TensorBoard with:\n"
        f"      tensorboard --logdir \"{LOG_PATH}\"\n"
    )


# ── Allow direct execution ────────────────────────────────────────────────────
if __name__ == "__main__":
    train_model()
