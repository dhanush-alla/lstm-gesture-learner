"""
src/3_run_inference.py
─────────────────────────────────────────────────────────────
Option 3 – Real-time gesture inference.

Opens the webcam, extracts MediaPipe keypoints frame-by-frame and
feeds a rolling 30-frame window into the trained LSTM model.

Prediction rules
────────────────
  • A prediction is surfaced ONLY when the model's top-class probability
    exceeds CONFIDENCE_THRESHOLD (80 % by default).
  • A short stability filter requires the same class to win in at least
    STABILITY_MIN_VOTES of the last STABILITY_WINDOW predictions before
    the on-screen label is updated – this eliminates single-frame flickers.
  • The displayed label fades when raw confidence drops well below the
    threshold (< 60 % of the threshold).

Controls
────────
  Q – gracefully close the webcam and exit.

Usage (standalone):
  python src/3_run_inference.py

Usage (via main.py):
  Select option 3 from the main menu.
"""

import sys
import os
from collections import deque

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
from src.config import (
    MODEL_PATH,
    LABELS_PATH,
    SEQUENCE_LENGTH,
    CONFIDENCE_THRESHOLD,
    STABILITY_WINDOW,
    STABILITY_MIN_VOTES,
    TTS_ENABLED,
    TTS_RATE,
    TRANSLATE_ENABLED,
    TRANSLATE_TARGET_LANG,
    setup_gpu,
)
from src.extract_keypoints import (
    mp_holistic,
    mediapipe_detection,
    draw_styled_landmarks,
    extract_keypoints,
)

import cv2
import numpy as np
import tensorflow as tf
import threading

# ── Optional: text-to-speech ────────────────────────────────────────────────────
try:
    import pyttsx3 as _pyttsx3
    _TTS_AVAILABLE = True
except ImportError:
    _TTS_AVAILABLE = False
    print("[TTS] pyttsx3 not installed – speech output disabled.")

# ── Optional: translation ────────────────────────────────────────────────────
try:
    from deep_translator import GoogleTranslator as _GoogleTranslator
    _TRANSLATOR_AVAILABLE = True
except ImportError:
    _TRANSLATOR_AVAILABLE = False
    print("[TRANSLATE] deep-translator not installed – translation disabled.")

_WIN = "Sign Language – Real-time Inference  |  Q = Quit"


# ─── TTS helpers ────────────────────────────────────────────────────────────────

def _speak_async(text: str) -> None:
    """
    Speak *text* in a fire-and-forget background thread.

    A new pyttsx3 engine is created per utterance because pyttsx3's
    COM apartment on Windows is not safely shareable across threads.
    The thread is marked daemon so it never blocks program exit.
    """
    if not (_TTS_AVAILABLE and TTS_ENABLED):
        return

    def _run() -> None:
        try:
            engine = _pyttsx3.init()
            engine.setProperty("rate",   TTS_RATE)
            engine.setProperty("volume", 0.9)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as exc:  # noqa: BLE001
            # TTS failure must never crash the video loop
            print(f"[TTS WARNING] {exc}")

    threading.Thread(target=_run, daemon=True).start()


def _translate(text: str) -> str:
    """
    Translate *text* into TRANSLATE_TARGET_LANG using Google Translate.
    Returns the original text unchanged on any error.
    """
    if not (_TRANSLATOR_AVAILABLE and TRANSLATE_ENABLED):
        return text
    try:
        return _GoogleTranslator(
            source="auto", target=TRANSLATE_TARGET_LANG
        ).translate(text)
    except Exception as exc:  # noqa: BLE001
        print(f"[TRANSLATE WARNING] {exc}")
        return text


# ─── Model / label loading ────────────────────────────────────────────────────

def _load_model_and_labels():
    """
    Load the trained Keras model and gesture label array from disk.

    Returns
    -------
    model   : tf.keras.Model  (or None on failure)
    actions : list[str]       ordered list of gesture class names
    """
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file not found: {MODEL_PATH}")
        print("        Please run Option 2 to train the model first.")
        return None, None

    if not os.path.exists(LABELS_PATH):
        print(f"[ERROR] Label map not found: {LABELS_PATH}")
        print("        Please run Option 2 to train the model first.")
        return None, None

    print(f"[INFO] Loading model …  {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

    actions = list(np.load(LABELS_PATH, allow_pickle=True))
    print(f"[INFO] Model ready.  Gesture classes: {actions}\n")
    return model, actions


# ─── On-screen inference UI ───────────────────────────────────────────────────

def _draw_inference_ui(
    image: np.ndarray,
    prediction: str,
    confidence: float,
    buffer_fill: int,
    h: int,
    w: int,
) -> None:
    """
    Render a clean inference overlay on the live video frame.

    Overlay elements
    ────────────────
    Top bar     – predicted gesture name (large text)
    Buffer bar  – thin green bar at the top edge showing
                  how full the 30-frame rolling window is
    Conf bar    – colour-coded confidence percentage at bottom-left
    Help text   – "Press Q to quit" at bottom-right
    """
    # ── Top bar background ────────────────────────────────────────────────────
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.60, image, 0.40, 0, image)

    # ── Rolling-window fill indicator (thin bar at the very top edge) ─────────
    fill_px = int(w * buffer_fill / SEQUENCE_LENGTH)
    cv2.rectangle(image, (0, 0), (fill_px, 5), (0, 255, 100), -1)

    # ── Prediction text ───────────────────────────────────────────────────────
    if prediction:
        cv2.putText(
            image, prediction,
            (12, 52),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 60), 3, cv2.LINE_AA,
        )

        # ── Confidence bar (bottom-left) ──────────────────────────────────────
        BAR_X, BAR_Y, BAR_W, BAR_H = 10, h - 45, 300, 22

        # Background track
        cv2.rectangle(image, (BAR_X, BAR_Y), (BAR_X + BAR_W, BAR_Y + BAR_H),
                      (45, 45, 45), -1)

        # Filled portion
        filled_w = int(BAR_W * confidence)
        if confidence >= 0.90:
            bar_color = (0, 230, 0)        # green  – very confident
        elif confidence >= CONFIDENCE_THRESHOLD:
            bar_color = (0, 200, 255)      # yellow-orange – above threshold
        else:
            bar_color = (0, 80, 200)       # red-ish – marginal
        cv2.rectangle(image, (BAR_X, BAR_Y), (BAR_X + filled_w, BAR_Y + BAR_H),
                      bar_color, -1)

        # Percentage label
        cv2.putText(
            image, f"{confidence * 100:.1f}%",
            (BAR_X + BAR_W + 8, BAR_Y + 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 1, cv2.LINE_AA,
        )
        # "Confidence" caption
        cv2.putText(
            image, "Confidence",
            (BAR_X, BAR_Y - 7),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 190, 190), 1, cv2.LINE_AA,
        )

    else:
        # No prediction above threshold yet
        cv2.putText(
            image, "Waiting for gesture …",
            (12, 48),
            cv2.FONT_HERSHEY_SIMPLEX, 0.95, (160, 160, 160), 2, cv2.LINE_AA,
        )

    # ── Quit hint ─────────────────────────────────────────────────────────────
    cv2.putText(
        image, "Press Q to quit",
        (w - 165, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (190, 190, 190), 1, cv2.LINE_AA,
    )


# ─── Main inference loop ──────────────────────────────────────────────────────

def run_inference() -> None:
    """
    Real-time LSTM gesture inference pipeline.

    1. Configure GPU memory growth.
    2. Load trained model + label map.
    3. Warm-up the GPU inference path with a dummy batch.
    4. Open webcam; extract MediaPipe keypoints every frame.
    5. Maintain a rolling 30-frame deque; run model.predict() once full.
    6. Apply confidence threshold + stability filter before updating display.
    7. Gracefully release all resources when Q is pressed.
    """
    # ── GPU ───────────────────────────────────────────────────────────────────
    gpus = setup_gpu()
    print(f"[INFO] Inference device: {'GPU:0' if gpus else 'CPU'}")

    # ── Load model ────────────────────────────────────────────────────────────
    model, actions = _load_model_and_labels()
    if model is None:
        return

    num_features = model.input_shape[-1]   # typically 1662

    # ── GPU warm-up: eliminates the 1-2 s latency on the very first prediction
    dummy = np.zeros((1, SEQUENCE_LENGTH, num_features), dtype=np.float32)
    model.predict(dummy, verbose=0)
    print("[INFO] GPU warm-up complete.")

    # ── State ─────────────────────────────────────────────────────────────────
    # Rolling window: automatically discards frames older than SEQUENCE_LENGTH
    sequence: deque = deque(maxlen=SEQUENCE_LENGTH)

    # Recent prediction history for stability filtering
    pred_history: deque = deque(maxlen=STABILITY_WINDOW)

    # Currently displayed label and its confidence
    current_label      = ""
    current_confidence = 0.0
    last_spoken_label  = ""   # tracks what was last announced via TTS

    # ── Webcam ────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam (device index 0).")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    print(f"[INFO] Running live inference …  "
          f"Threshold: {CONFIDENCE_THRESHOLD * 100:.0f}%  |  "
          f"TTS: {'on' if _TTS_AVAILABLE and TTS_ENABLED else 'off'}  |  "
          f"Translate: {'on → ' + TRANSLATE_TARGET_LANG if _TRANSLATOR_AVAILABLE and TRANSLATE_ENABLED else 'off'}  |  "
          f"Press Q to quit.\n")

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Failed to grab frame – stopping.")
                break

            frame = cv2.flip(frame, 1)
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)

            h, w = image.shape[:2]

            # ── Extract & buffer keypoints ───────────────────────────────────
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)

            # ── Run inference when the window is fully populated ─────────────
            if len(sequence) == SEQUENCE_LENGTH:
                # Shape: (1, 30, num_features)
                input_data = np.expand_dims(
                    np.array(sequence, dtype=np.float32), axis=0
                )

                # model() call (no Python overhead of model.predict's data pipeline)
                probs = model(
                    tf.constant(input_data), training=False
                ).numpy()[0]

                top_idx    = int(np.argmax(probs))
                top_prob   = float(probs[top_idx])

                pred_history.append(top_idx)

                if top_prob >= CONFIDENCE_THRESHOLD:
                    # Stability check: require majority vote across recent preds
                    votes = list(pred_history).count(top_idx)
                    if votes >= STABILITY_MIN_VOTES:
                        new_label = str(actions[top_idx])
                        current_label      = new_label
                        current_confidence = top_prob

                        # ── TTS: only announce when the label changes ─────────
                        if new_label != last_spoken_label:
                            last_spoken_label = new_label
                            spoken_text = _translate(new_label)
                            _speak_async(spoken_text)
                else:
                    # Only wipe the label if confidence has dropped well below
                    # threshold (hysteresis prevents rapid label flicker)
                    if top_prob < CONFIDENCE_THRESHOLD * 0.60:
                        current_label      = ""
                        current_confidence = 0.0

            # ── Draw overlay ─────────────────────────────────────────────────
            _draw_inference_ui(
                image,
                current_label,
                current_confidence,
                len(sequence),
                h, w,
            )

            cv2.imshow(_WIN, image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # ── Release all resources ─────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam released.  Inference stopped.\n")


# ── Allow direct execution ────────────────────────────────────────────────────
if __name__ == "__main__":
    run_inference()
