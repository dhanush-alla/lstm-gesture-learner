"""
app.py  –  Sign Language Recognition  –  GUI Entry Point
─────────────────────────────────────────────────────────
VS Code-inspired PyQt5 interface with five panels:
  Home  ·  Collect Data  ·  Train Model  ·  Run Inference  ·  Language Recognition

Run with:
  python app.py
"""

import sys
import os
import time
import threading
import shutil
from collections import deque

# ── Suppress TF noise before any TF import ────────────────────────────────────
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL",  "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL",    "3")
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning,  module="tensorflow")
warnings.filterwarnings("ignore", message=".*deprecated.*", module="tensorflow")
warnings.filterwarnings("ignore", message=".*HDF5.*")

import numpy as np
import cv2

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QStackedWidget, QLineEdit,
    QTextEdit, QProgressBar, QFrame, QSizePolicy,
    QComboBox, QMessageBox, QSlider, QScrollArea,
)
from PyQt5.QtCore  import Qt, QThread, pyqtSignal
from PyQt5.QtGui   import QImage, QPixmap, QFont, QIcon, QPainter, QColor, QBrush

# ── Project root on sys.path ──────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.config import (
    DATA_PATH, MODEL_PATH, LABELS_PATH, MODELS_DIR,
    NUM_SEQUENCES, SEQUENCE_LENGTH, BREAK_SECONDS,
    CONFIDENCE_THRESHOLD, STABILITY_WINDOW, STABILITY_MIN_VOTES,
    TTS_ENABLED, TTS_RATE,
    NUM_KEYPOINT_FEATURES,
    TEST_SIZE, BATCH_SIZE, MAX_EPOCHS, LEARNING_RATE, EARLY_STOP_PAT,
    setup_gpu,
)
from src.extract_keypoints import (
    mp_holistic, mediapipe_detection, draw_styled_landmarks, extract_keypoints,
)

# ── Language recognition (ASL / ISL) ──────────────────────────────────────────
_LANG_TASK_PATH  = os.path.join(_ROOT, 'hand_landmarker.task')
_ASL_CLASSES     = [chr(i) for i in range(ord('A'), ord('Z') + 1)] + ['del', 'nothing', 'space']
_ISL_CLASSES     = [chr(i) for i in range(ord('a'), ord('z') + 1)]
_LANG_CONFIDENCE = {'asl': 0.80, 'isl': 0.80}
_LANG_HOLD_SECS  = 1.5

# ═══════════════════════════════════════════════════════════════════════════════
#  VS Code Color Palette
# ═══════════════════════════════════════════════════════════════════════════════
C = {
    "bg":           "#1e1e1e",
    "sidebar":      "#252526",
    "sidebar_hover":"#2a2d2e",
    "nav_active":   "#094771",
    "nav_border":   "#007acc",
    "panel":        "#2d2d30",
    "input_bg":     "#3c3c3c",
    "input_border": "#555555",
    "focus_border": "#007acc",
    "btn_primary":  "#0e639c",
    "btn_hover":    "#1177bb",
    "btn_danger":   "#c72e2e",
    "btn_danger_h": "#e53535",
    "btn_success":  "#16825d",
    "btn_success_h":"#1a9469",
    "text":         "#d4d4d4",
    "text_muted":   "#858585",
    "text_bright":  "#ffffff",
    "accent_blue":  "#569cd6",
    "accent_teal":  "#4ec9b0",
    "accent_green": "#4caf50",
    "accent_red":   "#f44747",
    "accent_orange":"#ce9178",
    "border":       "#3c3c3c",
    "progress_bg":  "#2d2d30",
    "progress_fill":"#007acc",
    "terminal_bg":  "#1a1a1a",
    "cam_bg":       "#141414",
}

# ═══════════════════════════════════════════════════════════════════════════════
#  Global QSS Stylesheet  (VS Code dark theme)
# ═══════════════════════════════════════════════════════════════════════════════
QSS = f"""
/* ── Base ── */
QWidget {{
    background-color: {C['bg']};
    color: {C['text']};
    font-family: "Segoe UI", "DejaVu Sans", sans-serif;
    font-size: 13px;
    outline: none;
}}
QLabel {{ background: transparent; }}
QMainWindow {{ background-color: {C['bg']}; }}

/* ── Buttons ── */
QPushButton {{
    background-color: {C['btn_primary']};
    color: {C['text_bright']};
    border: none;
    border-radius: 3px;
    padding: 7px 18px;
    font-size: 13px;
    font-weight: 500;
}}
QPushButton:hover    {{ background-color: {C['btn_hover']}; }}
QPushButton:pressed  {{ background-color: #0c5383; }}
QPushButton:disabled {{ background-color: {C['input_bg']}; color: {C['text_muted']}; }}

QPushButton#danger   {{ background-color: {C['btn_danger']}; }}
QPushButton#danger:hover {{ background-color: {C['btn_danger_h']}; }}

QPushButton#success  {{ background-color: {C['btn_success']}; }}
QPushButton#success:hover {{ background-color: {C['btn_success_h']}; }}

QPushButton#nav {{
    background-color: transparent;
    color: {C['text_muted']};
    text-align: left;
    padding: 10px 20px;
    border-radius: 0;
    font-size: 13px;
    border: none;
    border-left: 3px solid transparent;
}}
QPushButton#nav:hover {{
    background-color: {C['sidebar_hover']};
    color: {C['text']};
}}
QPushButton#nav_active {{
    background-color: {C['nav_active']};
    color: {C['text_bright']};
    text-align: left;
    padding: 10px 20px;
    border-radius: 0;
    font-size: 13px;
    border: none;
    border-left: 3px solid {C['nav_border']};
}}

/* ── Inputs ── */
QLineEdit {{
    background-color: {C['input_bg']};
    border: 1px solid {C['input_border']};
    border-radius: 3px;
    padding: 7px 10px;
    color: {C['text']};
    selection-background-color: {C['nav_active']};
}}
QLineEdit:focus {{ border: 1px solid {C['focus_border']}; }}

/* ── Log / Console ── */
QTextEdit {{
    background-color: {C['terminal_bg']};
    border: 1px solid {C['border']};
    border-radius: 3px;
    color: {C['text']};
    font-family: "Consolas", "Courier New", monospace;
    font-size: 12px;
    selection-background-color: {C['nav_active']};
}}

/* ── Progress bars ── */
QProgressBar {{
    background-color: {C['progress_bg']};
    border: none;
    border-radius: 3px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{
    background-color: {C['progress_fill']};
    border-radius: 3px;
}}

/* ── Scrollbars ── */
QScrollBar:vertical {{
    background: {C['bg']};
    width: 10px;
    border: none;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {C['input_bg']};
    border-radius: 5px;
    min-height: 20px;
}}
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {{ height: 0; }}

/* ── Status bar ── */
QStatusBar {{
    background: {C['nav_border']};
    color: white;
    font-size: 12px;
}}
QStatusBar::item {{ border: none; }}
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _section_label(text: str) -> QLabel:
    """Small all-caps muted label used above UI sections."""
    lbl = QLabel(text.upper())
    lbl.setStyleSheet(
        f"color: {C['text_muted']}; font-size: 10px; "
        "letter-spacing: 0.8px; font-weight: 600;"
    )
    return lbl


def _hsep() -> QFrame:
    sep = QFrame()
    sep.setFrameShape(QFrame.HLine)
    sep.setStyleSheet(f"color: {C['border']}; background: {C['border']}; max-height: 1px;")
    return sep


def _make_app_icon() -> QIcon:
    """Draw a simple hand-shaped icon programmatically (no image file needed)."""
    pix = QPixmap(64, 64)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)
    # Blue circle background
    p.setBrush(QBrush(QColor("#007acc")))
    p.setPen(Qt.NoPen)
    p.drawEllipse(1, 1, 62, 62)
    # White hand
    p.setBrush(QBrush(QColor("#ffffff")))
    # Four fingers: index, middle, ring, pinky (middle two taller)
    for fx, fw, fy in [(17, 7, 11), (25, 7, 8), (33, 7, 10), (41, 6, 15)]:
        p.drawRoundedRect(fx, fy, fw, 22, 3, 3)
    # Thumb
    p.drawRoundedRect(8, 28, 9, 9, 4, 4)
    # Palm (covers all bases)
    p.drawRoundedRect(14, 30, 36, 22, 6, 6)
    p.end()
    return QIcon(pix)


# ═══════════════════════════════════════════════════════════════════════════════
#  Camera View Widget  (aspect-ratio-preserving, responsive)
# ═══════════════════════════════════════════════════════════════════════════════

class CameraView(QLabel):
    """QLabel that scales a live camera frame while preserving aspect ratio."""

    def __init__(self, placeholder: str = "Camera feed will appear here."):
        super().__init__()
        self._last_frame: np.ndarray | None = None
        self._placeholder = placeholder
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setStyleSheet(
            f"background: {C['cam_bg']}; border: 1px solid {C['border']}; "
            f"border-radius: 4px; color: {C['text_muted']}; font-size: 14px;"
        )
        self.setText(placeholder)

    def set_frame(self, frame: np.ndarray) -> None:
        self._last_frame = frame
        self._render()

    def clear_frame(self) -> None:
        self._last_frame = None
        self.clear()
        self.setText(self._placeholder)

    def _render(self) -> None:
        if self._last_frame is None:
            return
        w, h = self.width(), self.height()
        if w < 1 or h < 1:
            return
        fh, fw = self._last_frame.shape[:2]
        scale = min(w / fw, h / fh)
        nw, nh = int(fw * scale), int(fh * scale)
        resized = cv2.resize(self._last_frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = QImage(rgb.data, nw, nh, nw * 3, QImage.Format_RGB888).copy()
        self.setPixmap(QPixmap.fromImage(img))

    def resizeEvent(self, event):
        self._render()
        super().resizeEvent(event)


# ═══════════════════════════════════════════════════════════════════════════════
#  Collect Data Worker
# ═══════════════════════════════════════════════════════════════════════════════

class CollectWorker(QThread):
    frame_ready    = pyqtSignal(np.ndarray)   # annotated webcam frame
    status_update  = pyqtSignal(str)           # short status message
    seq_progress   = pyqtSignal(int, int)      # (completed_seqs, total_seqs)
    frame_progress = pyqtSignal(int, int)      # (current_frame, total_frames)
    finished_ok    = pyqtSignal(str)           # gesture name on success
    error          = pyqtSignal(str)

    def __init__(self, action: str):
        super().__init__()
        self.action            = action
        self._show_landmarks   = True
        self._space_pressed    = False
        self._stop             = False

    def toggle_landmarks(self) -> None:
        self._show_landmarks = not self._show_landmarks

    def press_space(self) -> None:
        self._space_pressed = True

    def request_stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        action = self.action

        # Create directory tree
        for seq in range(NUM_SEQUENCES):
            os.makedirs(os.path.join(DATA_PATH, action, str(seq)), exist_ok=True)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.error.emit("Cannot open webcam (device 0).")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

        with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as holistic:

            # ── Phase 1: wait for SPACE ──────────────────────────────────
            self.status_update.emit("Press  SPACE  to begin recording")
            while not self._space_pressed and not self._stop:
                ret, frame = cap.read()
                if not ret:
                    self.msleep(16)
                    continue
                frame = cv2.flip(frame, 1)
                image, results = mediapipe_detection(frame, holistic)
                if self._show_landmarks:
                    draw_styled_landmarks(image, results)
                self.frame_ready.emit(image.copy())
                self.msleep(16)

            if self._stop:
                cap.release()
                return

            # ── Phase 2: record sequences ────────────────────────────────
            for sequence in range(NUM_SEQUENCES):
                if self._stop:
                    break

                # Countdown between sequences (skip before the first)
                if sequence > 0:
                    deadline = time.time() + BREAK_SECONDS
                    while time.time() < deadline and not self._stop:
                        ret, frame = cap.read()
                        if not ret:
                            self.msleep(16)
                            continue
                        frame = cv2.flip(frame, 1)
                        image, results = mediapipe_detection(frame, holistic)
                        if self._show_landmarks:
                            draw_styled_landmarks(image, results)
                        remaining = max(1, int(deadline - time.time()) + 1)
                        self.status_update.emit(
                            f"Get ready…  Sequence {sequence + 1}/{NUM_SEQUENCES}"
                            f"  starts in  {remaining}s"
                        )
                        self.frame_ready.emit(image.copy())
                        self.msleep(16)
                else:
                    self.msleep(600)   # brief pause before sequence 0

                if self._stop:
                    break

                self.status_update.emit(
                    f"● Recording  sequence  {sequence + 1} / {NUM_SEQUENCES}"
                )

                # ── Collect SEQUENCE_LENGTH frames ───────────────────────
                for frame_num in range(SEQUENCE_LENGTH):
                    if self._stop:
                        break
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    frame = cv2.flip(frame, 1)
                    image, results = mediapipe_detection(frame, holistic)
                    if self._show_landmarks:
                        draw_styled_landmarks(image, results)

                    keypoints = extract_keypoints(results)
                    np.save(
                        os.path.join(DATA_PATH, action, str(sequence), str(frame_num)),
                        keypoints,
                    )
                    self.frame_ready.emit(image.copy())
                    self.frame_progress.emit(frame_num + 1, SEQUENCE_LENGTH)
                    self.msleep(16)

                self.seq_progress.emit(sequence + 1, NUM_SEQUENCES)

        cap.release()
        if not self._stop:
            self.status_update.emit(f"✓  Collection complete for  '{action}'")
            self.finished_ok.emit(action)


# ═══════════════════════════════════════════════════════════════════════════════
#  Training Worker
# ═══════════════════════════════════════════════════════════════════════════════

class TrainingWorker(QThread):
    log_line    = pyqtSignal(str)
    epoch_done  = pyqtSignal(int, dict)   # (epoch_number, logs_dict)
    finished_ok = pyqtSignal(float)       # final test accuracy
    error       = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._stop_flag = [False]         # mutable reference passed to callback
        self._keras_model = None

    def request_stop(self) -> None:
        self._stop_flag[0] = True
        if self._keras_model is not None:
            self._keras_model.stop_training = True

    def run(self) -> None:
        # Lazy TF import so the main thread starts fast
        import tensorflow as tf
        from tensorflow.keras.models   import Sequential
        from tensorflow.keras.layers   import LSTM, Dense, Dropout, BatchNormalization
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks  import (
            EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
        )
        from tensorflow.keras.utils      import to_categorical
        from sklearn.model_selection     import train_test_split
        from sklearn.utils.class_weight  import compute_class_weight

        stop_flag   = self._stop_flag
        emit_log    = self.log_line.emit
        emit_epoch  = self.epoch_done.emit

        # ── Custom Keras callback ─────────────────────────────────────────
        class _GUICallback(tf.keras.callbacks.Callback):
            def on_epoch_begin(cb, epoch, logs=None):  # noqa: N805
                emit_log(f"Epoch {epoch + 1} / {MAX_EPOCHS}")

            def on_epoch_end(cb, epoch, logs=None):    # noqa: N805
                if logs:
                    emit_epoch(epoch + 1, dict(logs))
                if stop_flag[0]:
                    cb.model.stop_training = True

        # ── Load data ─────────────────────────────────────────────────────
        if not os.path.isdir(DATA_PATH):
            self.error.emit("data/ directory not found — collect data first.")
            return

        actions = sorted(
            d for d in os.listdir(DATA_PATH)
            if os.path.isdir(os.path.join(DATA_PATH, d))
        )
        if not actions:
            self.error.emit("No gesture folders found — collect data first.")
            return

        emit_log(f"Found {len(actions)} class(es):  {',  '.join(actions)}")
        label_map = {name: idx for idx, name in enumerate(actions)}
        sequences, labels = [], []

        for action in actions:
            action_path = os.path.join(DATA_PATH, action)
            seq_dirs = sorted(
                [d for d in os.listdir(action_path)
                 if os.path.isdir(os.path.join(action_path, d))],
                key=int,
            )
            skipped = 0
            for seq_dir in seq_dirs:
                seq_path = os.path.join(action_path, seq_dir)
                window = []
                for frame_num in range(SEQUENCE_LENGTH):
                    npy_file = os.path.join(seq_path, f"{frame_num}.npy")
                    if not os.path.exists(npy_file):
                        skipped += 1
                        window = []
                        break
                    window.append(np.load(npy_file).astype(np.float32))
                if len(window) == SEQUENCE_LENGTH:
                    sequences.append(window)
                    labels.append(label_map[action])
            emit_log(
                f"  {action:<20s}  loaded: "
                f"{len(seq_dirs) - skipped} / {len(seq_dirs)}"
            )

        if not sequences:
            self.error.emit("No complete sequences found.")
            return

        X = np.array(sequences, dtype=np.float32)
        y_int = np.array(labels)
        y = to_categorical(y_int, num_classes=len(actions))

        # Log per-class counts so imbalance is visible
        counts = {a: int(np.sum(y_int == i)) for i, a in enumerate(actions)}
        count_str = "  ".join(f"{a}: {n}" for a, n in counts.items())
        emit_log(f"Samples per class:  {count_str}")
        emit_log(f"Dataset shape:  X{X.shape}  y{y.shape}")

        # ── Train / test split ────────────────────────────────────────────
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=42,
                stratify=y_int,
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=42,
            )

        # Gaussian-noise augmentation (doubles training set)
        rng   = np.random.default_rng(seed=42)
        X_aug = (X_train + rng.normal(0.0, 0.01, X_train.shape)).astype(np.float32)
        # Keep integer labels for augmented copies (to compute class weights)
        y_train_int = np.argmax(y_train, axis=1)
        X_train = np.concatenate([X_train, X_aug], axis=0)
        y_train = np.concatenate([y_train, y_train], axis=0)
        y_train_int = np.concatenate([y_train_int, y_train_int], axis=0)
        idx     = rng.permutation(len(X_train))
        X_train, y_train, y_train_int = X_train[idx], y_train[idx], y_train_int[idx]

        # ── Compute class weights to counter imbalance ────────────────────
        cw_values = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(len(actions)),
            y=y_train_int,
        )
        class_weight_dict = dict(enumerate(cw_values))
        cw_str = "  ".join(f"{actions[i]}: {v:.2f}" for i, v in enumerate(cw_values))
        emit_log(f"Class weights:  {cw_str}")

        emit_log(
            f"Train: {X_train.shape[0]} samples  |  "
            f"Test: {X_test.shape[0]} samples  (after augmentation)\n"
        )

        # ── Build model ───────────────────────────────────────────────────
        num_classes = len(actions)
        model = Sequential([
            LSTM(64, return_sequences=True, activation="tanh",
                 input_shape=(SEQUENCE_LENGTH, NUM_KEYPOINT_FEATURES),
                 kernel_regularizer=tf.keras.regularizers.l2(1e-3)),
            BatchNormalization(), Dropout(0.50),
            LSTM(128, return_sequences=True, activation="tanh",
                 kernel_regularizer=tf.keras.regularizers.l2(1e-3)),
            BatchNormalization(), Dropout(0.50),
            LSTM(64, return_sequences=False, activation="tanh",
                 kernel_regularizer=tf.keras.regularizers.l2(1e-3)),
            BatchNormalization(), Dropout(0.40),
            Dense(64, activation="relu"), Dropout(0.30),
            Dense(32, activation="relu"),
            Dense(num_classes, activation="softmax"),
        ], name="gesture_lstm")

        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss="categorical_crossentropy",
            metrics=["categorical_accuracy"],
        )
        self._keras_model = model
        emit_log("Model built.  Starting training…\n")

        os.makedirs(MODELS_DIR, exist_ok=True)
        callbacks = [
            _GUICallback(),
            ReduceLROnPlateau(
                monitor="val_categorical_accuracy", factor=0.5,
                patience=8, min_lr=1e-6, mode="max", verbose=0,
            ),
            EarlyStopping(
                monitor="val_categorical_accuracy", patience=EARLY_STOP_PAT,
                restore_best_weights=True, mode="max", verbose=0,
            ),
            ModelCheckpoint(
                filepath=MODEL_PATH, monitor="val_loss",
                save_best_only=True, mode="min", verbose=0,
            ),
        ]

        model.fit(
            X_train, y_train,
            epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=0,
        )

        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        np.save(LABELS_PATH, np.array(actions))
        emit_log(f"\n✓  Done.  Test loss: {loss:.4f}  |  Test accuracy: {acc * 100:.2f}%")
        self.finished_ok.emit(acc)


# ═══════════════════════════════════════════════════════════════════════════════
#  Inference Worker
# ═══════════════════════════════════════════════════════════════════════════════

class InferenceWorker(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    # (top_label, top_prob, buf_len) — always emitted regardless of threshold
    prediction  = pyqtSignal(str, float, int)
    error       = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._show_landmarks = True
        self._stop           = False
        self.threshold       = CONFIDENCE_THRESHOLD  # mutable at runtime

    def toggle_landmarks(self) -> None:
        self._show_landmarks = not self._show_landmarks

    def request_stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        import tensorflow as tf

        if not (os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH)):
            self.error.emit("Model or label map not found — train the model first.")
            return

        model   = tf.keras.models.load_model(MODEL_PATH)
        actions = list(np.load(LABELS_PATH, allow_pickle=True))

        # GPU warm-up
        dummy = np.zeros((1, SEQUENCE_LENGTH, model.input_shape[-1]), dtype=np.float32)
        model.predict(dummy, verbose=0)

        # Optional TTS
        try:
            import pyttsx3 as _pyttsx3
            _tts_ok = True
        except ImportError:
            _tts_ok = False

        def _speak(text: str) -> None:
            if not (_tts_ok and TTS_ENABLED):
                return
            def _run():
                try:
                    e = _pyttsx3.init()
                    e.setProperty("rate",   TTS_RATE)
                    e.setProperty("volume", 0.9)
                    e.say(text)
                    e.runAndWait()
                    e.stop()
                except Exception:
                    pass
            threading.Thread(target=_run, daemon=True).start()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.error.emit("Cannot open webcam (device 0).")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        sequence     = deque(maxlen=SEQUENCE_LENGTH)
        pred_history = deque(maxlen=STABILITY_WINDOW)
        last_spoken  = ""

        with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0,
        ) as holistic:
            while not self._stop:
                ret, frame = cap.read()
                if not ret:
                    self.msleep(16)
                    continue

                frame = cv2.flip(frame, 1)
                image, results = mediapipe_detection(frame, holistic)
                if self._show_landmarks:
                    draw_styled_landmarks(image, results)

                keypoints = extract_keypoints(results)
                sequence.append(keypoints)

                if len(sequence) == SEQUENCE_LENGTH:
                    inp   = np.expand_dims(np.array(sequence, dtype=np.float32), 0)
                    probs = model(tf.constant(inp), training=False).numpy()[0]

                    top_idx  = int(np.argmax(probs))
                    top_prob = float(probs[top_idx])
                    top_label = str(actions[top_idx])
                    pred_history.append(top_idx)

                    thresh = self.threshold
                    if top_prob >= thresh:
                        votes = list(pred_history).count(top_idx)
                        if votes >= STABILITY_MIN_VOTES:
                            if top_label != last_spoken:
                                last_spoken = top_label
                                _speak(top_label)
                    # Always emit raw top prediction so UI can display it
                    self.prediction.emit(top_label, top_prob, len(sequence))
                else:
                    self.prediction.emit("", 0.0, len(sequence))

                self.frame_ready.emit(image.copy())

        cap.release()


# ═══════════════════════════════════════════════════════════════════════════════
#  Language Inference Worker  (ASL / ISL — MediaPipe HandLandmarker)
# ═══════════════════════════════════════════════════════════════════════════════

class LanguageInferenceWorker(QThread):
    frame_ready     = pyqtSignal(np.ndarray)
    prediction      = pyqtSignal(str, float)   # (smoothed_label, confidence)
    word_update     = pyqtSignal(str)           # current partial word
    sentence_update = pyqtSignal(str)           # committed sentence so far
    error           = pyqtSignal(str)

    # Hand skeleton connections (MediaPipe 21-point topology)
    _HAND_CONNS = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17),
    ]

    def __init__(self, mode: str = 'asl'):
        super().__init__()
        self.mode            = mode
        self._stop           = False
        self._show_landmarks = True
        self._action_queue: list = []
        self._lock           = threading.Lock()

    def toggle_landmarks(self) -> None:
        self._show_landmarks = not self._show_landmarks

    def push_action(self, action: str) -> None:
        with self._lock:
            self._action_queue.append(action)

    def request_stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        import tensorflow as tf
        from collections import Counter as _Counter
        import mediapipe as _mp
        from mediapipe.tasks.python.vision import (
            HandLandmarker as _HL, HandLandmarkerOptions as _HLO,
        )
        from mediapipe.tasks.python import BaseOptions as _BO

        if not os.path.exists(_LANG_TASK_PATH):
            self.error.emit(
                f"hand_landmarker.task not found at:\n{_LANG_TASK_PATH}\n"
                "Download it from MediaPipe and place it in the project root."
            )
            return

        model_file = os.path.join(MODELS_DIR, f'model_{self.mode}_compat.h5')
        if not os.path.exists(model_file):
            self.error.emit(
                f"Pre-trained model not found: {model_file}\n"
                "Run  python convert_models.py  once to generate the Keras-2 "
                "compatible weights from the original .keras files."
            )
            return

        classes = _ASL_CLASSES if self.mode == 'asl' else _ISL_CLASSES
        thresh  = _LANG_CONFIDENCE[self.mode]

        model = tf.keras.models.load_model(model_file, compile=False)
        model(tf.zeros((1, 63), dtype=tf.float32), training=False)   # warm-up

        landmarker = _HL.create_from_options(_HLO(
            base_options=_BO(model_asset_path=_LANG_TASK_PATH),
            num_hands=2,
            min_hand_detection_confidence=0.1,
            min_hand_presence_confidence=0.1,
            min_tracking_confidence=0.1,
        ))

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.error.emit("Cannot open webcam (device 0).")
            landmarker.close()
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        # Inference state
        pred_buffer     = deque(maxlen=10)
        prob_buffer     = deque(maxlen=8)
        word_buffer: list = []
        sentence:    list = []
        last_pred         = None
        last_change_t     = time.time()
        last_committed    = None
        token_released    = True

        while not self._stop:
            ret, frame = cap.read()
            if not ret:
                self.msleep(16)
                continue

            frame  = cv2.flip(frame, 1)
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = _mp.Image(image_format=_mp.ImageFormat.SRGB, data=rgb)
            try:
                results = landmarker.detect(mp_img)
            except Exception:
                results = None

            # ── 63-D wrist-normalised hand coords ────────────────────────
            coords = None
            if results and results.hand_landmarks:
                lms = results.hand_landmarks[0]
                raw = np.array([[lm.x, lm.y, lm.z] for lm in lms],
                               dtype=np.float32).flatten()
                raw[::3]  -= raw[0]
                raw[1::3] -= raw[1]
                raw[2::3] -= raw[2]
                coords = raw

            # ── Landmark overlay ──────────────────────────────────────────
            if self._show_landmarks and results and results.hand_landmarks:
                h_px, w_px = frame.shape[:2]
                for hand_lms in results.hand_landmarks:
                    pts = [(int(lm.x * w_px), int(lm.y * h_px)) for lm in hand_lms]
                    for a, b in self._HAND_CONNS:
                        cv2.line(frame, pts[a], pts[b], (80, 200, 120), 1)
                    for pt in pts:
                        cv2.circle(frame, pt, 4, (245, 117, 66), -1)

            # ── Model inference ───────────────────────────────────────────
            confidence = 0.0
            if coords is not None:
                probs      = model(tf.constant(coords[np.newaxis]), training=False).numpy()[0]
                prob_buffer.append(probs)
                avg        = np.mean(np.array(prob_buffer), axis=0)
                top_idx    = int(np.argmax(avg))
                confidence = float(np.max(avg))
                pred_buffer.append(classes[top_idx] if confidence > thresh else None)
            else:
                prob_buffer.clear()
                pred_buffer.append(None)

            smoothed = _Counter(pred_buffer).most_common(1)[0][0] if pred_buffer else None
            self.prediction.emit(smoothed or '', confidence)

            # ── Hold-to-commit ────────────────────────────────────────────
            now = time.time()
            if smoothed in (None, 'nothing'):
                token_released = True
            if smoothed != last_pred:
                last_pred     = smoothed
                last_change_t = now
            elif (smoothed is not None
                  and smoothed not in ('nothing',)
                  and (now - last_change_t) > _LANG_HOLD_SECS):
                if not (smoothed == last_committed and not token_released):
                    if smoothed == 'space':
                        if word_buffer:
                            sentence.append(''.join(word_buffer))
                            word_buffer.clear()
                    elif smoothed == 'del':
                        if word_buffer:
                            word_buffer.pop()
                    else:
                        word_buffer.append(smoothed)
                    last_committed = smoothed
                    token_released = False
                    self.word_update.emit(''.join(word_buffer))
                    self.sentence_update.emit(' '.join(sentence))

            # ── Manual UI actions ─────────────────────────────────────────
            with self._lock:
                actions = self._action_queue[:]
                self._action_queue.clear()
            for action in actions:
                if action == 'space':
                    if word_buffer:
                        sentence.append(''.join(word_buffer))
                        word_buffer.clear()
                elif action == 'del':
                    if word_buffer:
                        word_buffer.pop()
                    elif sentence:
                        word_buffer[:] = list(sentence.pop())
                elif action == 'clear':
                    sentence.clear()
                    word_buffer.clear()
                    pred_buffer.clear()
                    prob_buffer.clear()
                self.word_update.emit(''.join(word_buffer))
                self.sentence_update.emit(' '.join(sentence))

            self.frame_ready.emit(frame.copy())

        cap.release()
        landmarker.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  Home Panel
# ═══════════════════════════════════════════════════════════════════════════════

class HomePanel(QWidget):
    navigate = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(52, 44, 52, 32)
        root.setSpacing(0)

        # ── Title ─────────────────────────────────────────────────────────
        title = QLabel("Sign Language Recognition")
        title.setFont(QFont("Segoe UI", 24, QFont.Light))
        title.setStyleSheet(f"color: {C['text_bright']}; margin-bottom: 4px;")
        root.addWidget(title)

        sub = QLabel(
            "MediaPipe Holistic  ·  TensorFlow LSTM  ·  "
            "ASL / ISL Static Recognition  ·  Real-time Inference"
        )
        sub.setStyleSheet(f"color: {C['text_muted']}; font-size: 12px; margin-bottom: 24px;")
        root.addWidget(sub)

        root.addWidget(_hsep())
        root.addSpacing(28)

        # ── Cards ─────────────────────────────────────────────────────────
        cards_row = QHBoxLayout()
        cards_row.setSpacing(16)

        card_defs = [
            ("Collect Data",
             "Record 30×30-frame webcam sequences for a new or existing gesture class.",
             1, C["accent_teal"]),
            ("Train Model",
             "Build and fit the 3-layer stacked LSTM network on all collected data.",
             2, C["accent_blue"]),
            ("Custom Gestures",
             "Run real-time inference on your custom-trained LSTM gesture model.",
             3, C["accent_green"]),
            ("Language Recognition",
             "Recognise ASL or ISL static alphabets with a pre-trained model.",
             4, C["accent_orange"]),
        ]
        for card_title, card_desc, page_idx, color in card_defs:
            cards_row.addWidget(self._card(card_title, card_desc, page_idx, color))

        root.addLayout(cards_row)
        root.addSpacing(32)
        root.addWidget(_hsep())
        root.addSpacing(16)

        # ── Stats ──────────────────────────────────────────────────────────
        self._stats = QLabel()
        self._stats.setStyleSheet(
            f"color: {C['text_muted']}; font-size: 12px; "
            f"font-family: Consolas, monospace;"
        )
        root.addWidget(self._stats)
        root.addSpacing(24)
        root.addWidget(_hsep())
        root.addSpacing(16)

        # ── Delete gesture class ───────────────────────────────────────────
        root.addWidget(_section_label("Delete gesture class"))
        root.addSpacing(8)

        del_row = QHBoxLayout()
        del_row.setSpacing(10)
        self._delete_combo = QComboBox()
        self._delete_combo.setStyleSheet(
            f"QComboBox {{ background: {C['input_bg']}; border: 1px solid {C['input_border']}; "
            f"border-radius: 3px; padding: 6px 10px; color: {C['text']}; min-width: 160px; }}"
            f"QComboBox:focus {{ border: 1px solid {C['focus_border']}; }}"
            f"QComboBox QAbstractItemView {{ background: {C['input_bg']}; color: {C['text']}; "
            f"selection-background-color: {C['nav_active']}; border: 1px solid {C['border']}; }}"
        )
        del_row.addWidget(self._delete_combo)

        del_btn = QPushButton("Delete Class")
        del_btn.setObjectName("danger")
        del_btn.setFixedWidth(130)
        del_btn.clicked.connect(self._on_delete)
        del_row.addWidget(del_btn)
        del_row.addStretch()
        root.addLayout(del_row)

        root.addStretch()
        self.refresh_stats()

    def _card(self, title: str, desc: str, page_idx: int, color: str) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet(
            f"QFrame {{ background: {C['panel']}; border: 1px solid {C['border']}; "
            f"border-radius: 6px; }}"
        )
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(10)

        # Colour stripe
        stripe = QFrame()
        stripe.setFixedHeight(3)
        stripe.setStyleSheet(f"background: {color}; border: none; border-radius: 2px;")
        layout.addWidget(stripe)

        lbl_title = QLabel(title)
        lbl_title.setFont(QFont("Segoe UI", 13, QFont.DemiBold))
        lbl_title.setStyleSheet(f"color: {C['text_bright']};")
        layout.addWidget(lbl_title)

        lbl_desc = QLabel(desc)
        lbl_desc.setWordWrap(True)
        lbl_desc.setStyleSheet(f"color: {C['text_muted']}; font-size: 12px;")
        layout.addWidget(lbl_desc)

        btn = QPushButton("Open →")
        btn.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {color}; border: none; "
            f"padding: 2px 0; text-align: left; font-size: 13px; }}"
            f"QPushButton:hover {{ color: {C['text_bright']}; }}"
        )
        btn.setCursor(Qt.PointingHandCursor)
        btn.clicked.connect(lambda _checked, p=page_idx: self.navigate.emit(p))
        layout.addWidget(btn)
        return frame

    def refresh_stats(self) -> None:
        gestures = total_seq = 0
        if os.path.isdir(DATA_PATH):
            for d in os.listdir(DATA_PATH):
                p = os.path.join(DATA_PATH, d)
                if os.path.isdir(p):
                    gestures += 1
                    total_seq += sum(
                        1 for s in os.listdir(p)
                        if os.path.isdir(os.path.join(p, s))
                    )
        model_ready = os.path.exists(MODEL_PATH)
        self._stats.setText(
            f"Gestures: {gestures}   ·   Sequences: {total_seq}   ·   "
            f"Model: {'✓ ready' if model_ready else '✗ not trained'}"
        )
        self._refresh_delete_combo()

    def _refresh_delete_combo(self) -> None:
        self._delete_combo.clear()
        if os.path.isdir(DATA_PATH):
            for d in sorted(os.listdir(DATA_PATH)):
                if os.path.isdir(os.path.join(DATA_PATH, d)):
                    self._delete_combo.addItem(d)
        if self._delete_combo.count() == 0:
            self._delete_combo.addItem("(no classes)")

    def _on_delete(self) -> None:
        cls = self._delete_combo.currentText()
        if not cls or cls == "(no classes)":
            return
        reply = QMessageBox.question(
            self, "Delete Gesture Class",
            f"Permanently delete all data for  '{cls}'?\n\n"
            "The trained model will also be removed. You will need to retrain.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        shutil.rmtree(os.path.join(DATA_PATH, cls), ignore_errors=True)
        for stale in (MODEL_PATH, LABELS_PATH):
            try:
                os.remove(stale)
            except FileNotFoundError:
                pass
        self.refresh_stats()


# ═══════════════════════════════════════════════════════════════════════════════
#  Collect Panel
# ═══════════════════════════════════════════════════════════════════════════════

class CollectPanel(QWidget):
    def __init__(self):
        super().__init__()
        self._worker: CollectWorker | None = None
        self._build()

    def _build(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Camera area (left, dominant) ──────────────────────────────────
        cam_wrap = QWidget()
        cam_wrap.setStyleSheet(f"background: {C['cam_bg']};")
        cam_lay = QVBoxLayout(cam_wrap)
        cam_lay.setContentsMargins(16, 16, 8, 16)
        self.cam_view = CameraView(
            "Webcam feed appears here.\nEnter a gesture name and click  ▶ Start."
        )
        cam_lay.addWidget(self.cam_view)
        root.addWidget(cam_wrap, stretch=3)

        # ── Controls sidebar (right) ──────────────────────────────────────
        # Inner content widget
        ctrl_wrap = QWidget()
        ctrl_wrap.setStyleSheet(f"background: {C['sidebar']};")
        ctrl = QVBoxLayout(ctrl_wrap)
        ctrl.setContentsMargins(20, 24, 20, 20)
        ctrl.setSpacing(14)

        # Wrap in a scroll area so nothing is cut off on small windows
        scroll = QScrollArea()
        scroll.setWidget(ctrl_wrap)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(252)
        scroll.setMaximumWidth(320)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            f"QScrollArea {{ background: {C['sidebar']}; border: none; "
            f"border-left: 1px solid {C['border']}; }}"
            f"QScrollBar:vertical {{ background: {C['sidebar']}; width: 6px; }}"
            f"QScrollBar::handle:vertical {{ background: {C['input_bg']}; border-radius: 3px; }}"
        )

        heading = QLabel("Data Collection")
        heading.setFont(QFont("Segoe UI", 14, QFont.DemiBold))
        heading.setStyleSheet(f"color: {C['text_bright']};")
        ctrl.addWidget(heading)

        hint = QLabel(f"Records {NUM_SEQUENCES} sequences × {SEQUENCE_LENGTH} frames per gesture.")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {C['text_muted']}; font-size: 12px;")
        ctrl.addWidget(hint)

        ctrl.addWidget(_hsep())

        ctrl.addWidget(_section_label("Gesture name"))
        self.gesture_input = QLineEdit()
        self.gesture_input.setPlaceholderText("e.g.  Hello")
        ctrl.addWidget(self.gesture_input)

        ctrl.addWidget(_section_label("Status"))
        self.status_lbl = QLabel("Idle — enter a gesture name and click Start.")
        self.status_lbl.setWordWrap(True)
        self.status_lbl.setStyleSheet(
            f"color: {C['accent_teal']}; font-family: Consolas, monospace; font-size: 12px;"
        )
        ctrl.addWidget(self.status_lbl)

        ctrl.addWidget(_section_label(f"Sequences  ( / {NUM_SEQUENCES} )"))
        self.seq_bar = QProgressBar()
        self.seq_bar.setMaximum(NUM_SEQUENCES)
        self.seq_bar.setValue(0)
        self.seq_bar.setFixedHeight(8)
        ctrl.addWidget(self.seq_bar)

        self.seq_count_lbl = QLabel(f"0 / {NUM_SEQUENCES}")
        self.seq_count_lbl.setStyleSheet(f"color: {C['text_muted']}; font-size: 12px;")
        ctrl.addWidget(self.seq_count_lbl)

        ctrl.addWidget(_section_label(f"Frame capture  ( / {SEQUENCE_LENGTH} )"))
        self.frame_bar = QProgressBar()
        self.frame_bar.setMaximum(SEQUENCE_LENGTH)
        self.frame_bar.setValue(0)
        self.frame_bar.setFixedHeight(6)
        self.frame_bar.setStyleSheet(
            f"QProgressBar::chunk {{ background: {C['accent_teal']}; border-radius: 3px; }}"
        )
        ctrl.addWidget(self.frame_bar)

        ctrl.addWidget(_hsep())
        ctrl.addWidget(_section_label("Delete gesture class"))
        self._delete_combo = QComboBox()
        self._delete_combo.setStyleSheet(
            f"QComboBox {{ background: {C['input_bg']}; border: 1px solid {C['input_border']}; "
            f"border-radius: 3px; padding: 5px 8px; color: {C['text']}; }}"
            f"QComboBox:focus {{ border: 1px solid {C['focus_border']}; }}"
            f"QComboBox QAbstractItemView {{ background: {C['input_bg']}; color: {C['text']}; "
            f"selection-background-color: {C['nav_active']}; border: 1px solid {C['border']}; }}"
        )
        ctrl.addWidget(self._delete_combo)
        del_btn = QPushButton("Delete Class")
        del_btn.setObjectName("danger")
        del_btn.clicked.connect(self._on_delete)
        ctrl.addWidget(del_btn)

        ctrl.addStretch()

        # Keyboard hints
        kb = QLabel("SPACE  –  begin recording\nW          –  toggle landmarks")
        kb.setStyleSheet(
            f"color: {C['text_muted']}; font-size: 11px; "
            "font-family: Consolas, monospace;"
        )
        ctrl.addWidget(kb)

        # Action buttons
        self.start_btn = QPushButton("▶  Start")
        self.start_btn.setObjectName("success")
        self.start_btn.clicked.connect(self._on_start)
        ctrl.addWidget(self.start_btn)

        self.stop_btn = QPushButton("■  Stop")
        self.stop_btn.setObjectName("danger")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop)
        ctrl.addWidget(self.stop_btn)

        root.addWidget(scroll, stretch=0)

    # ── Slots ─────────────────────────────────────────────────────────────

    def _on_start(self) -> None:
        action = self.gesture_input.text().strip()
        if not action:
            self.status_lbl.setText("⚠  Enter a gesture name first.")
            return

        self._reset_progress()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.gesture_input.setEnabled(False)
        self.cam_view.setFocus()

        self._worker = CollectWorker(action)
        self._worker.frame_ready.connect(self.cam_view.set_frame)
        self._worker.status_update.connect(self.status_lbl.setText)
        self._worker.seq_progress.connect(self._on_seq_progress)
        self._worker.frame_progress.connect(lambda f, _t: self.frame_bar.setValue(f))
        self._worker.finished_ok.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_stop(self) -> None:
        if self._worker:
            try:
                self._worker.frame_ready.disconnect(self.cam_view.set_frame)
            except TypeError:
                pass
            self._worker.request_stop()
        self._enable_controls()
        self.cam_view.clear_frame()
        self.status_lbl.setText("Idle -- enter a gesture name and click Start.")

    def _on_seq_progress(self, done: int, total: int) -> None:
        self.seq_bar.setValue(done)
        self.seq_count_lbl.setText(f"{done} / {total}")

    def _on_done(self, action: str) -> None:
        self._enable_controls()
        self.status_lbl.setText(f"✓  '{action}'  collected successfully.")
        self.cam_view.clear_frame()

    def _on_error(self, msg: str) -> None:
        self._enable_controls()
        self.status_lbl.setText(f"✗  {msg}")

    def _enable_controls(self) -> None:
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.gesture_input.setEnabled(True)

    def _reset_progress(self) -> None:
        self.seq_bar.setValue(0)
        self.seq_count_lbl.setText(f"0 / {NUM_SEQUENCES}")
        self.frame_bar.setValue(0)

    # ── Called from MainWindow ────────────────────────────────────────────

    def toggle_landmarks(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.toggle_landmarks()

    def press_space(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.press_space()

    def cleanup(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.request_stop()
            self._worker.wait(2000)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._refresh_delete_combo()

    def _refresh_delete_combo(self) -> None:
        self._delete_combo.clear()
        if os.path.isdir(DATA_PATH):
            for d in sorted(os.listdir(DATA_PATH)):
                if os.path.isdir(os.path.join(DATA_PATH, d)):
                    self._delete_combo.addItem(d)
        if self._delete_combo.count() == 0:
            self._delete_combo.addItem("(no classes)")

    def _on_delete(self) -> None:
        cls = self._delete_combo.currentText()
        if not cls or cls == "(no classes)":
            return
        reply = QMessageBox.question(
            self, "Delete Gesture Class",
            f"Permanently delete all data for  '{cls}'?\n\n"
            "The trained model will also be removed. You will need to retrain.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        shutil.rmtree(os.path.join(DATA_PATH, cls), ignore_errors=True)
        for stale in (MODEL_PATH, LABELS_PATH):
            try:
                os.remove(stale)
            except FileNotFoundError:
                pass
        self._refresh_delete_combo()
        self.status_lbl.setText(
            f"Done. '{cls}' deleted. Model removed -- please retrain."
        )

class TrainPanel(QWidget):
    def __init__(self):
        super().__init__()
        self._worker: TrainingWorker | None = None
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(36, 28, 36, 24)
        root.setSpacing(14)

        heading = QLabel("Train Model")
        heading.setFont(QFont("Segoe UI", 20, QFont.Light))
        heading.setStyleSheet(f"color: {C['text_bright']};")
        root.addWidget(heading)

        sub = QLabel(
            "Builds a 3-layer stacked LSTM on all collected gesture data. "
            "Best checkpoint is saved to  models/gesture_model.h5."
        )
        sub.setWordWrap(True)
        sub.setStyleSheet(f"color: {C['text_muted']}; font-size: 12px;")
        root.addWidget(sub)

        root.addWidget(_hsep())

        # ── Live metrics row ──────────────────────────────────────────────
        metrics_row = QHBoxLayout()
        metrics_row.setSpacing(12)

        self._epoch_lbl    , w1 = self._metric_card("Epoch",     "—")
        self._train_acc_lbl, w2 = self._metric_card("Train Acc", "—")
        self._val_acc_lbl  , w3 = self._metric_card("Val Acc",   "—")
        self._val_loss_lbl , w4 = self._metric_card("Val Loss",  "—")
        for w in (w1, w2, w3, w4):
            metrics_row.addWidget(w)
        root.addLayout(metrics_row)

        # Epoch progress bar
        self.epoch_bar = QProgressBar()
        self.epoch_bar.setMaximum(MAX_EPOCHS)
        self.epoch_bar.setValue(0)
        self.epoch_bar.setFixedHeight(6)
        root.addWidget(self.epoch_bar)

        # ── Training log ──────────────────────────────────────────────────
        root.addWidget(_section_label("Training log"))
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFont(QFont("Consolas", 11))
        self.log_box.setMinimumHeight(180)
        root.addWidget(self.log_box, stretch=1)

        # ── Buttons ───────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self.train_btn = QPushButton("▶  Start Training")
        self.train_btn.setObjectName("success")
        self.train_btn.clicked.connect(self._on_start)
        btn_row.addWidget(self.train_btn)

        self.stop_btn = QPushButton("■  Stop")
        self.stop_btn.setObjectName("danger")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop)
        btn_row.addWidget(self.stop_btn)

        btn_row.addStretch()
        root.addLayout(btn_row)

    def _metric_card(self, label: str, value: str):
        """Returns (value_label, card_frame)."""
        frame = QFrame()
        frame.setStyleSheet(
            f"QFrame {{ background: {C['panel']}; border: 1px solid {C['border']}; "
            f"border-radius: 4px; }}"
        )
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(4)

        lbl = QLabel(label.upper())
        lbl.setStyleSheet(f"color: {C['text_muted']}; font-size: 10px; letter-spacing: 0.5px;")
        lay.addWidget(lbl)

        val = QLabel(value)
        val.setFont(QFont("Consolas", 16, QFont.Bold))
        val.setStyleSheet(f"color: {C['accent_teal']};")
        lay.addWidget(val)

        return val, frame

    # ── Slots ─────────────────────────────────────────────────────────────

    def _on_start(self) -> None:
        self.log_box.clear()
        self.epoch_bar.setValue(0)
        for lbl in (self._epoch_lbl, self._train_acc_lbl,
                    self._val_acc_lbl, self._val_loss_lbl):
            lbl.setText("—")
            lbl.setStyleSheet(f"color: {C['accent_teal']};")

        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self._worker = TrainingWorker()
        self._worker.log_line.connect(self._append)
        self._worker.epoch_done.connect(self._on_epoch)
        self._worker.finished_ok.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_stop(self) -> None:
        if self._worker:
            self._worker.request_stop()
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _append(self, text: str) -> None:
        self.log_box.append(text)
        sb = self.log_box.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_epoch(self, epoch: int, logs: dict) -> None:
        self.epoch_bar.setValue(epoch)
        self._epoch_lbl.setText(str(epoch))

        train_acc = logs.get("categorical_accuracy", 0.0)
        val_acc   = logs.get("val_categorical_accuracy", 0.0)
        val_loss  = logs.get("val_loss", 0.0)

        self._train_acc_lbl.setText(f"{train_acc * 100:.1f}%")
        self._val_acc_lbl.setText(f"{val_acc   * 100:.1f}%")
        self._val_loss_lbl.setText(f"{val_loss:.4f}")

        # Colour val_acc by quality
        color = (
            C["accent_green"]  if val_acc >= 0.90
            else C["accent_teal"]   if val_acc >= 0.60
            else C["accent_orange"]
        )
        self._val_acc_lbl.setStyleSheet(f"color: {color};")

        self._append(
            f"   train {train_acc * 100:.1f}%  ·  "
            f"val {val_acc * 100:.1f}%  ·  "
            f"loss {val_loss:.4f}"
        )

    def _on_done(self, acc: float) -> None:
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._append(f"\n✓  Training complete.  Test accuracy: {acc * 100:.2f}%")

    def _on_error(self, msg: str) -> None:
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._append(f"\n✗  {msg}")

    def cleanup(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.request_stop()
            self._worker.wait(3000)


# ═══════════════════════════════════════════════════════════════════════════════
#  Inference Panel
# ═══════════════════════════════════════════════════════════════════════════════

class InferencePanel(QWidget):
    def __init__(self):
        super().__init__()
        self._worker: InferenceWorker | None = None
        self._build()

    def _build(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Camera (dominant) ─────────────────────────────────────────────
        cam_wrap = QWidget()
        cam_wrap.setStyleSheet(f"background: {C['cam_bg']};")
        cam_lay = QVBoxLayout(cam_wrap)
        cam_lay.setContentsMargins(16, 16, 8, 16)
        self.cam_view = CameraView(
            "Webcam feed appears here.\nTrain the model first, then click  ▶ Start."
        )
        cam_lay.addWidget(self.cam_view)
        root.addWidget(cam_wrap, stretch=3)

        # ── Info sidebar ──────────────────────────────────────────────────
        info_wrap = QWidget()
        info_wrap.setStyleSheet(
            f"background: {C['sidebar']}; border-left: 1px solid {C['border']};"
        )
        info_wrap.setMinimumWidth(240)
        info_wrap.setMaximumWidth(300)
        info = QVBoxLayout(info_wrap)
        info.setContentsMargins(20, 24, 20, 20)
        info.setSpacing(14)

        heading = QLabel("Live Inference")
        heading.setFont(QFont("Segoe UI", 14, QFont.DemiBold))
        heading.setStyleSheet(f"color: {C['text_bright']};")
        info.addWidget(heading)

        info.addWidget(_hsep())

        # Prediction display
        info.addWidget(_section_label("Prediction"))
        self.pred_lbl = QLabel("—")
        self.pred_lbl.setFont(QFont("Segoe UI", 30, QFont.Bold))
        self.pred_lbl.setAlignment(Qt.AlignCenter)
        self.pred_lbl.setWordWrap(True)
        self.pred_lbl.setMinimumHeight(80)
        self.pred_lbl.setStyleSheet(
            f"color: {C['accent_green']}; padding: 14px; "
            f"background: {C['bg']}; border: 1px solid {C['border']}; "
            "border-radius: 6px;"
        )
        info.addWidget(self.pred_lbl)

        # Confidence bar
        info.addWidget(_section_label("Confidence"))
        self.conf_bar = QProgressBar()
        self.conf_bar.setMaximum(100)
        self.conf_bar.setValue(0)
        self.conf_bar.setFixedHeight(10)
        info.addWidget(self.conf_bar)

        self.conf_pct = QLabel("0 %")
        self.conf_pct.setStyleSheet(
            f"color: {C['text_muted']}; font-size: 12px; font-family: Consolas, monospace;"
        )
        info.addWidget(self.conf_pct)

        # Buffer fill
        info.addWidget(_section_label(f"Frame buffer  ( / {SEQUENCE_LENGTH} )"))
        self.buf_bar = QProgressBar()
        self.buf_bar.setMaximum(SEQUENCE_LENGTH)
        self.buf_bar.setValue(0)
        self.buf_bar.setFixedHeight(6)
        self.buf_bar.setStyleSheet(
            f"QProgressBar::chunk {{ background: {C['accent_teal']}; border-radius: 3px; }}"
        )
        info.addWidget(self.buf_bar)

        # Threshold slider
        info.addWidget(_section_label("Confidence threshold"))
        thresh_row = QHBoxLayout()
        thresh_row.setSpacing(8)
        self._thresh_slider = QSlider(Qt.Horizontal)
        self._thresh_slider.setMinimum(10)
        self._thresh_slider.setMaximum(95)
        self._thresh_slider.setValue(int(CONFIDENCE_THRESHOLD * 100))
        self._thresh_slider.setStyleSheet(
            f"QSlider::groove:horizontal {{ background: {C['input_bg']}; height: 4px; border-radius: 2px; }}"
            f"QSlider::handle:horizontal {{ background: {C['accent_blue']}; width: 14px; height: 14px; "
            f"margin: -5px 0; border-radius: 7px; }}"
            f"QSlider::sub-page:horizontal {{ background: {C['accent_blue']}; border-radius: 2px; }}"
        )
        thresh_row.addWidget(self._thresh_slider, stretch=1)
        self._thresh_lbl = QLabel(f"{int(CONFIDENCE_THRESHOLD * 100)}%")
        self._thresh_lbl.setFixedWidth(36)
        self._thresh_lbl.setStyleSheet(
            f"color: {C['text_muted']}; font-size: 11px; font-family: Consolas, monospace;"
        )
        thresh_row.addWidget(self._thresh_lbl)
        info.addLayout(thresh_row)
        self._thresh_slider.valueChanged.connect(self._on_thresh_change)

        info.addStretch()

        # Keyboard hints
        kb = QLabel("W  –  toggle landmarks")
        kb.setStyleSheet(
            f"color: {C['text_muted']}; font-size: 11px; "
            "font-family: Consolas, monospace;"
        )
        info.addWidget(kb)

        # Buttons
        self.start_btn = QPushButton("▶  Start")
        self.start_btn.setObjectName("success")
        self.start_btn.clicked.connect(self._on_start)
        info.addWidget(self.start_btn)

        self.stop_btn = QPushButton("■  Stop")
        self.stop_btn.setObjectName("danger")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop)
        info.addWidget(self.stop_btn)

        root.addWidget(info_wrap, stretch=0)

    # ── Slots ─────────────────────────────────────────────────────────────

    def _on_thresh_change(self, val: int) -> None:
        self._thresh_lbl.setText(f"{val}%")
        if self._worker and self._worker.isRunning():
            self._worker.threshold = val / 100.0

    def _on_start(self) -> None:
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.pred_lbl.setText("—")
        self.conf_bar.setValue(0)
        self.buf_bar.setValue(0)

        self._worker = InferenceWorker()
        self._worker.threshold = self._thresh_slider.value() / 100.0
        self._worker.frame_ready.connect(self.cam_view.set_frame)
        self._worker.prediction.connect(self._on_prediction)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_stop(self) -> None:
        if self._worker:
            self._worker.request_stop()
            self._worker.wait(3000)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.pred_lbl.setText("—")
        self.cam_view.clear_frame()

    def _on_prediction(self, label: str, conf: float, buf_len: int) -> None:
        self.buf_bar.setValue(buf_len)
        thresh = self._thresh_slider.value() / 100.0
        if label:
            pct = int(conf * 100)
            self.conf_bar.setValue(pct)
            self.conf_pct.setText(f"{pct} %")
            above = conf >= thresh
            # Colour: green=above 90%, teal=above threshold, orange=below threshold (dim)
            color = (
                C["accent_green"]    if conf >= 0.90
                else C["accent_teal"]     if above
                else C["accent_orange"]
            )
            self.conf_bar.setStyleSheet(
                f"QProgressBar {{ background: {C['progress_bg']}; border: none; border-radius: 3px; }}"
                f"QProgressBar::chunk {{ background: {color}; border-radius: 3px; }}"
            )
            # Show label always; dim it when below threshold
            label_color = C["accent_green"] if above else C["text_muted"]
            self.pred_lbl.setText(label)
            self.pred_lbl.setStyleSheet(
                f"color: {label_color}; padding: 14px; "
                f"background: {C['bg']}; border: 1px solid {C['border']}; "
                "border-radius: 6px;"
            )
        else:
            self.pred_lbl.setText("—")
            self.pred_lbl.setStyleSheet(
                f"color: {C['accent_green']}; padding: 14px; "
                f"background: {C['bg']}; border: 1px solid {C['border']}; "
                "border-radius: 6px;"
            )
            self.conf_bar.setValue(0)
            self.conf_pct.setText("0 %")

    def _on_error(self, msg: str) -> None:
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.pred_lbl.setText("Error")
        self.pred_lbl.setStyleSheet(
            self.pred_lbl.styleSheet().replace(C["accent_green"], C["accent_red"])
        )
        self.cam_view.setText(f"✗  {msg}")

    def toggle_landmarks(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.toggle_landmarks()

    def cleanup(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.request_stop()
            self._worker.wait(3000)


# ═══════════════════════════════════════════════════════════════════════════════
#  Language Panel  (ASL / ISL — static alphabet recognition)
# ═══════════════════════════════════════════════════════════════════════════════

class LanguagePanel(QWidget):
    _translate_result = pyqtSignal(str)   # thread-safe label update

    def __init__(self):
        super().__init__()
        self._worker: LanguageInferenceWorker | None = None
        self._mode          = 'asl'
        self._sentence_text = ''
        self._word_text     = ''
        self._build()

    def _build(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Camera (dominant left) ────────────────────────────────────────
        cam_wrap = QWidget()
        cam_wrap.setStyleSheet(f"background: {C['cam_bg']};")
        cam_lay = QVBoxLayout(cam_wrap)
        cam_lay.setContentsMargins(16, 16, 8, 16)
        self.cam_view = CameraView(
            "Webcam feed appears here.\nSelect  ASL  or  ISL  and click  ▶ Start."
        )
        cam_lay.addWidget(self.cam_view)
        root.addWidget(cam_wrap, stretch=3)

        # ── Sidebar ───────────────────────────────────────────────────────
        inner = QWidget()
        inner.setStyleSheet(f"background: {C['sidebar']};")
        info = QVBoxLayout(inner)
        info.setContentsMargins(20, 24, 20, 20)
        info.setSpacing(12)

        scroll = QScrollArea()
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(260)
        scroll.setMaximumWidth(320)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            f"QScrollArea {{ background: {C['sidebar']}; border: none; "
            f"border-left: 1px solid {C['border']}; }}"
            f"QScrollBar:vertical {{ background: {C['sidebar']}; width: 6px; }}"
            f"QScrollBar::handle:vertical {{ background: {C['input_bg']}; border-radius: 3px; }}"
        )

        heading = QLabel("Language Recognition")
        heading.setFont(QFont("Segoe UI", 14, QFont.DemiBold))
        heading.setStyleSheet(f"color: {C['text_bright']};")
        info.addWidget(heading)

        hint = QLabel("Static alphabet recognition using pre-trained ASL / ISL models.")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {C['text_muted']}; font-size: 12px;")
        info.addWidget(hint)

        info.addWidget(_hsep())

        # ── Mode selector ─────────────────────────────────────────────────
        info.addWidget(_section_label("Mode"))
        mode_row = QHBoxLayout()
        mode_row.setSpacing(8)
        _ms = (
            f"QPushButton {{ background: {C['input_bg']}; color: {C['text_muted']}; "
            f"border: 1px solid {C['input_border']}; border-radius: 3px; padding: 6px 18px; }}"
            f"QPushButton:checked {{ background: {C['nav_active']}; color: {C['text_bright']}; "
            f"border: 1px solid {C['nav_border']}; }}"
            f"QPushButton:hover:!checked {{ background: {C['sidebar_hover']}; color: {C['text']}; }}"
        )
        self._asl_btn = QPushButton("ASL")
        self._asl_btn.setCheckable(True)
        self._asl_btn.setChecked(True)
        self._asl_btn.setStyleSheet(_ms)
        self._asl_btn.clicked.connect(lambda: self._set_mode('asl'))
        self._isl_btn = QPushButton("ISL")
        self._isl_btn.setCheckable(True)
        self._isl_btn.setStyleSheet(_ms)
        self._isl_btn.clicked.connect(lambda: self._set_mode('isl'))
        mode_row.addWidget(self._asl_btn)
        mode_row.addWidget(self._isl_btn)
        mode_row.addStretch()
        info.addLayout(mode_row)

        info.addWidget(_hsep())

        # ── Prediction letter ─────────────────────────────────────────────
        info.addWidget(_section_label("Prediction"))
        self.pred_lbl = QLabel("—")
        self.pred_lbl.setFont(QFont("Segoe UI", 52, QFont.Bold))
        self.pred_lbl.setAlignment(Qt.AlignCenter)
        self.pred_lbl.setFixedHeight(100)
        self.pred_lbl.setStyleSheet(
            f"color: {C['accent_teal']}; padding: 10px; "
            f"background: {C['bg']}; border: 1px solid {C['border']}; border-radius: 6px;"
        )
        info.addWidget(self.pred_lbl)

        # Confidence bar
        info.addWidget(_section_label("Confidence"))
        self.conf_bar = QProgressBar()
        self.conf_bar.setMaximum(100)
        self.conf_bar.setValue(0)
        self.conf_bar.setFixedHeight(8)
        info.addWidget(self.conf_bar)
        self.conf_pct = QLabel("0 %")
        self.conf_pct.setStyleSheet(
            f"color: {C['text_muted']}; font-size: 12px; font-family: Consolas, monospace;"
        )
        info.addWidget(self.conf_pct)

        info.addWidget(_hsep())

        # ── Word / Sentence ───────────────────────────────────────────────
        info.addWidget(_section_label("Current word"))
        self.word_lbl = QLabel("")
        self.word_lbl.setFont(QFont("Consolas", 14, QFont.Bold))
        self.word_lbl.setWordWrap(True)
        self.word_lbl.setMinimumHeight(36)
        self.word_lbl.setAlignment(Qt.AlignCenter)
        self.word_lbl.setStyleSheet(
            f"color: {C['accent_blue']}; padding: 6px 10px; "
            f"background: {C['bg']}; border: 1px solid {C['border']}; border-radius: 4px;"
        )
        info.addWidget(self.word_lbl)

        info.addWidget(_section_label("Sentence"))
        self.sentence_box = QTextEdit()
        self.sentence_box.setReadOnly(True)
        self.sentence_box.setFixedHeight(68)
        self.sentence_box.setFont(QFont("Consolas", 11))
        info.addWidget(self.sentence_box)

        # Action buttons
        action_row = QHBoxLayout()
        action_row.setSpacing(6)
        space_btn = QPushButton("Space")
        space_btn.clicked.connect(lambda: self._push_action('space'))
        del_btn = QPushButton("Del")
        del_btn.setObjectName("danger")
        del_btn.clicked.connect(lambda: self._push_action('del'))
        clear_btn = QPushButton("Clear")
        clear_btn.setObjectName("danger")
        clear_btn.clicked.connect(lambda: self._push_action('clear'))
        for b in (space_btn, del_btn, clear_btn):
            b.setFixedHeight(30)
            action_row.addWidget(b)
        info.addLayout(action_row)

        # Speak / Translate
        util_row = QHBoxLayout()
        util_row.setSpacing(6)
        speak_btn = QPushButton("Speak")
        speak_btn.clicked.connect(self._on_speak)
        translate_btn = QPushButton("Translate")
        translate_btn.clicked.connect(self._on_translate)
        for b in (speak_btn, translate_btn):
            b.setFixedHeight(30)
            util_row.addWidget(b)
        info.addLayout(util_row)

        # Target language selector
        lang_row = QHBoxLayout()
        lang_row.setSpacing(8)
        lang_lbl = QLabel("Target:")
        lang_lbl.setStyleSheet(f"color: {C['text_muted']}; font-size: 12px;")
        lang_row.addWidget(lang_lbl)
        self._lang_combo = QComboBox()
        self._lang_combo.addItems([
            "Hindi (hi)", "Telugu (te)", "Spanish (es)", "French (fr)", "German (de)",
        ])
        self._lang_combo.setStyleSheet(
            f"QComboBox {{ background: {C['input_bg']}; border: 1px solid {C['input_border']}; "
            f"border-radius: 3px; padding: 4px 8px; color: {C['text']}; }}"
            f"QComboBox QAbstractItemView {{ background: {C['input_bg']}; color: {C['text']}; "
            f"selection-background-color: {C['nav_active']}; border: 1px solid {C['border']}; }}"
        )
        lang_row.addWidget(self._lang_combo, 1)
        info.addLayout(lang_row)

        self.translate_lbl = QLabel("")
        self.translate_lbl.setWordWrap(True)
        self.translate_lbl.setStyleSheet(
            f"color: {C['accent_orange']}; font-size: 12px; font-family: Consolas, monospace;"
        )
        info.addWidget(self.translate_lbl)
        self._translate_result.connect(self.translate_lbl.setText)

        info.addWidget(_hsep())

        # Hints
        hold_hint = QLabel(
            "Hold a gesture 1.5 s to commit it.\n"
            "ASL:  space  /  del  gestures auto-trigger."
        )
        hold_hint.setWordWrap(True)
        hold_hint.setStyleSheet(f"color: {C['text_muted']}; font-size: 11px;")
        info.addWidget(hold_hint)

        kb = QLabel("W  –  toggle landmarks")
        kb.setStyleSheet(
            f"color: {C['text_muted']}; font-size: 11px; font-family: Consolas, monospace;"
        )
        info.addWidget(kb)

        info.addStretch()

        # Start / Stop
        self.start_btn = QPushButton("▶  Start")
        self.start_btn.setObjectName("success")
        self.start_btn.clicked.connect(self._on_start)
        info.addWidget(self.start_btn)

        self.stop_btn = QPushButton("■  Stop")
        self.stop_btn.setObjectName("danger")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop)
        info.addWidget(self.stop_btn)

        root.addWidget(scroll, stretch=0)

    # ── Mode ──────────────────────────────────────────────────────────────

    def _set_mode(self, mode: str) -> None:
        if mode == self._mode:
            return
        self._mode = mode
        self._asl_btn.setChecked(mode == 'asl')
        self._isl_btn.setChecked(mode == 'isl')
        if self._worker and self._worker.isRunning():
            self._on_stop()
            self._on_start()

    # ── Slots ─────────────────────────────────────────────────────────────

    def _on_start(self) -> None:
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._asl_btn.setEnabled(False)
        self._isl_btn.setEnabled(False)
        self.pred_lbl.setText("—")
        self.conf_bar.setValue(0)
        self.conf_pct.setText("0 %")
        self._worker = LanguageInferenceWorker(self._mode)
        self._worker.frame_ready.connect(self.cam_view.set_frame)
        self._worker.prediction.connect(self._on_prediction)
        self._worker.word_update.connect(self._on_word)
        self._worker.sentence_update.connect(self._on_sentence)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_stop(self) -> None:
        if self._worker:
            try:
                self._worker.frame_ready.disconnect(self.cam_view.set_frame)
            except TypeError:
                pass
            self._worker.request_stop()
            self._worker.wait(3000)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._asl_btn.setEnabled(True)
        self._isl_btn.setEnabled(True)
        self.cam_view.clear_frame()
        self.pred_lbl.setText("—")

    def _on_prediction(self, label: str, conf: float) -> None:
        pct = int(conf * 100)
        self.conf_bar.setValue(pct)
        self.conf_pct.setText(f"{pct} %")
        if label and label not in ('nothing',):
            color = (
                C['accent_green']  if conf >= 0.90
                else C['accent_teal'] if conf >= 0.70
                else C['accent_orange']
            )
            display = label.upper() if self._mode == 'asl' else label
            self.pred_lbl.setText(display)
            self.pred_lbl.setStyleSheet(
                f"color: {color}; padding: 10px; "
                f"background: {C['bg']}; border: 1px solid {C['border']}; border-radius: 6px;"
            )
        else:
            self.pred_lbl.setText("—")
            self.pred_lbl.setStyleSheet(
                f"color: {C['accent_teal']}; padding: 10px; "
                f"background: {C['bg']}; border: 1px solid {C['border']}; border-radius: 6px;"
            )

    def _on_word(self, text: str) -> None:
        self._word_text = text
        display = text.upper() if self._mode == 'asl' else text
        self.word_lbl.setText(display)

    def _on_sentence(self, text: str) -> None:
        self._sentence_text = text
        display = text.upper() if self._mode == 'asl' else text
        self.sentence_box.setText(display)

    def _on_error(self, msg: str) -> None:
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._asl_btn.setEnabled(True)
        self._isl_btn.setEnabled(True)
        self.cam_view.setText(f"✗  {msg}")

    def _push_action(self, action: str) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.push_action(action)

    def _get_full_text(self) -> str:
        parts = [self._sentence_text, self._word_text]
        text  = ' '.join(p for p in parts if p).strip()
        return text.upper() if self._mode == 'asl' else text

    def _on_speak(self) -> None:
        text = self._get_full_text()
        if not text:
            return
        import subprocess
        safe   = text.replace("'", "''")
        ps_cmd = (
            f"Add-Type -AssemblyName System.Speech; "
            f"([System.Speech.Synthesis.SpeechSynthesizer]::new()).Speak('{safe}')"
        )
        threading.Thread(
            target=lambda: subprocess.run(
                ['powershell', '-NoProfile', '-Command', ps_cmd],
                capture_output=True,
            ), daemon=True
        ).start()

    def _on_translate(self) -> None:
        text = self._get_full_text()
        if not text:
            return
        lang_code = self._lang_combo.currentText().split('(')[-1].rstrip(')')
        def _do():
            try:
                from deep_translator import GoogleTranslator
                result = GoogleTranslator(source='auto', target=lang_code).translate(text)
                self._translate_result.emit(result)
            except ImportError:
                self._translate_result.emit(
                    "[deep-translator not installed — run: pip install deep-translator]"
                )
            except Exception as exc:
                self._translate_result.emit(f"[Translation error: {exc}]")
        threading.Thread(target=_do, daemon=True).start()

    def toggle_landmarks(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.toggle_landmarks()

    def cleanup(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.request_stop()
            self._worker.wait(3000)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Window
# ═══════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sign Language Recognition")
        self.setMinimumSize(900, 580)
        self.resize(1280, 800)
        self._build()

    def _build(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        outer = QHBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Sidebar ───────────────────────────────────────────────────────
        sidebar = QWidget()
        sidebar.setFixedWidth(210)
        sidebar.setStyleSheet(f"background: {C['sidebar']};")
        sb = QVBoxLayout(sidebar)
        sb.setContentsMargins(0, 0, 0, 0)
        sb.setSpacing(0)

        # Logo bar
        logo_bar = QWidget()
        logo_bar.setFixedHeight(52)
        logo_bar.setStyleSheet(
            f"background: {C['sidebar']}; "
            f"border-bottom: 1px solid {C['border']};"
        )
        logo_lay = QHBoxLayout(logo_bar)
        logo_lay.setContentsMargins(12, 0, 16, 0)
        logo_lay.setSpacing(8)
        icon_lbl = QLabel()
        icon_lbl.setPixmap(_make_app_icon().pixmap(24, 24))
        icon_lbl.setFixedSize(24, 24)
        logo_lay.addWidget(icon_lbl)
        logo_lbl = QLabel("RSL Model")
        logo_lbl.setFont(QFont("Segoe UI", 12, QFont.DemiBold))
        logo_lbl.setStyleSheet(f"color: {C['text_bright']};")
        logo_lay.addWidget(logo_lbl)
        sb.addWidget(logo_bar)

        # Navigation buttons
        nav_defs = [
            ("⌂   Home",                  0),
            ("⦿   Collect Data",           1),
            ("⚙   Train Model",            2),
            ("◉   Custom Gestures",        3),
            ("⌨   Language Recognition",   4),
        ]
        self._nav_btns: list[QPushButton] = []
        for text, idx in nav_defs:
            btn = QPushButton(text)
            btn.setObjectName("nav")
            btn.clicked.connect(lambda _c, i=idx: self._navigate(i))
            sb.addWidget(btn)
            self._nav_btns.append(btn)

        sb.addStretch()

        # Version footer
        footer = QLabel("TF 2.15  ·  MP 0.10.14  ·  PyQt5")
        footer.setStyleSheet(
            f"color: {C['text_muted']}; font-size: 10px; padding: 8px 16px;"
        )
        sb.addWidget(footer)

        outer.addWidget(sidebar)

        # Vertical divider
        vdiv = QFrame()
        vdiv.setFrameShape(QFrame.VLine)
        vdiv.setStyleSheet(f"color: {C['border']}; background: {C['border']}; max-width: 1px;")
        outer.addWidget(vdiv)

        # ── Stacked content area ──────────────────────────────────────────
        self._stack = QStackedWidget()

        self._home    = HomePanel()
        self._collect = CollectPanel()
        self._train   = TrainPanel()
        self._infer   = InferencePanel()
        self._lang    = LanguagePanel()

        self._home.navigate.connect(self._navigate)

        for panel in (self._home, self._collect, self._train, self._infer, self._lang):
            self._stack.addWidget(panel)

        outer.addWidget(self._stack, stretch=1)

        # Status bar
        self.statusBar().showMessage(
            "  Ready  ·  Select a section from the sidebar to begin."
        )

        self._navigate(0)

    # ── Navigation ────────────────────────────────────────────────────────

    def _navigate(self, idx: int) -> None:
        # Clean up departing panels that own webcam resources
        prev = self._stack.currentIndex()
        if prev == 1 and idx != 1:
            self._collect.cleanup()
        elif prev == 3 and idx != 3:
            self._infer.cleanup()
        elif prev == 4 and idx != 4:
            self._lang.cleanup()

        # Update nav button styles
        for i, btn in enumerate(self._nav_btns):
            name = "nav_active" if i == idx else "nav"
            btn.setObjectName(name)
            btn.style().unpolish(btn)
            btn.style().polish(btn)

        self._stack.setCurrentIndex(idx)

        if idx == 0:
            self._home.refresh_stats()

        labels = ["Home", "Collect Data", "Train Model", "Custom Gestures", "Language Recognition"]
        self.statusBar().showMessage(f"  {labels[idx]}")

    # ── Global key handling ───────────────────────────────────────────────

    def keyPressEvent(self, event) -> None:
        key = event.key()
        idx = self._stack.currentIndex()

        if key == Qt.Key_W:
            if idx == 1:
                self._collect.toggle_landmarks()
            elif idx == 3:
                self._infer.toggle_landmarks()
            elif idx == 4:
                self._lang.toggle_landmarks()

        elif key == Qt.Key_Space and idx == 1:
            # Only forward Space if the gesture name input doesn't have focus
            if not self._collect.gesture_input.hasFocus():
                self._collect.press_space()

        else:
            super().keyPressEvent(event)

    # ── Close ─────────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        self._collect.cleanup()
        self._train.cleanup()
        self._infer.cleanup()
        self._lang.cleanup()
        event.accept()


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")
    app = QApplication(sys.argv)
    app.setStyleSheet(QSS)

    font = QFont("Segoe UI", 10)
    font.setStyleHint(QFont.SansSerif)
    app.setFont(font)

    _ico_path = os.path.join(_ROOT, 'app_icon.ico')
    icon = QIcon(_ico_path) if os.path.exists(_ico_path) else _make_app_icon()
    app.setWindowIcon(icon)
    win = MainWindow()
    win.setWindowIcon(icon)
    win.show()
    win.raise_()
    win.activateWindow()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
