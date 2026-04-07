"""
src/config.py
─────────────────────────────────────────────────────────────
Central configuration for the Sign Language Recognition
framework.  Every path, hyper-parameter, and shared constant
lives here so the rest of the codebase never hard-codes values.
"""

import os

# ─── Project Paths ────────────────────────────────────────────────────────────
# BASE_DIR always resolves to the project root regardless of how the
# script is invoked (directly or via main.py).
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH   = os.path.join(BASE_DIR, 'data')          # raw .npy gesture sequences
LOG_PATH    = os.path.join(BASE_DIR, 'logs', 'train') # TensorBoard event files
MODELS_DIR  = os.path.join(BASE_DIR, 'models')        # saved Keras model weights
MODEL_PATH  = os.path.join(MODELS_DIR, 'gesture_model.h5')
LABELS_PATH = os.path.join(MODELS_DIR, 'label_map.npy')

# ─── Data Collection Parameters ───────────────────────────────────────────────
NUM_SEQUENCES   = 30   # video sequences collected per gesture class
SEQUENCE_LENGTH = 30   # frames (time-steps) per sequence
BREAK_SECONDS   = 3    # countdown gap between consecutive sequences

# ─── MediaPipe Holistic Feature Dimensions ────────────────────────────────────
#
#   Component     Landmarks   Values/landmark   Total
#   ──────────────────────────────────────────────────
#   Pose              33        4 (x,y,z,vis)    132
#   Face mesh        468        3 (x,y,z)       1404
#   Left  hand        21        3 (x,y,z)         63
#   Right hand        21        3 (x,y,z)         63
#   ──────────────────────────────────────────────── 
#   TOTAL                                       1662
#
NUM_POSE_FEATURES     = 33  * 4   # 132
NUM_FACE_FEATURES     = 468 * 3   # 1404
NUM_HAND_FEATURES     = 21  * 3   # 63 (per hand)
NUM_KEYPOINT_FEATURES = (
    NUM_POSE_FEATURES + NUM_FACE_FEATURES + 2 * NUM_HAND_FEATURES
)  # 1662 total features per frame

# ─── Model Training Parameters ────────────────────────────────────────────────
TEST_SIZE       = 0.20   # proportion of data held out for validation
BATCH_SIZE      = 16     # small batches increase stochasticity on tiny datasets
MAX_EPOCHS      = 200
LEARNING_RATE   = 5e-4   # lower LR; ReduceLROnPlateau decays it further
EARLY_STOP_PAT  = 30     # EarlyStopping patience — give LR decay time to help

# ─── Inference Parameters ─────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.80  # minimum softmax probability to surface a prediction
STABILITY_WINDOW     = 5     # rolling history of recent predictions for smoothing
STABILITY_MIN_VOTES  = 3     # how many of the last N must agree to confirm gesture

# ─── Translation & Text-to-Speech (inference only) ────────────────────────────
TTS_ENABLED           = True   # announce predicted gestures aloud via pyttsx3
TTS_RATE              = 160    # speech rate in words per minute
TRANSLATE_ENABLED     = False  # translate label before speaking (uses deep-translator)
TRANSLATE_TARGET_LANG = "es"   # ISO 639-1 target language code  (e.g. "es", "fr", "de")


# ─── GPU Configuration ────────────────────────────────────────────────────────
def setup_gpu():
    """
    Enable TensorFlow GPU memory growth so the framework shares VRAM
    with the OS and other processes instead of pre-allocating everything.

    Call this ONCE at the very start of any script that uses TensorFlow,
    BEFORE any model is built or loaded (TF initialises the GPU device
    on first use; memory-growth flags must be set prior to that).

    Returns
    -------
    gpus : list
        Physical GPU devices found.  Empty list → no GPU available.
    """
    import tensorflow as tf  # lazy import – collect_data doesn't need TF

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical = tf.config.list_logical_devices('GPU')
            print(f"[GPU] {len(gpus)} Physical GPU(s) | "
                  f"{len(logical)} Logical GPU(s) – memory growth enabled.")
            for i, g in enumerate(gpus):
                print(f"      → Device {i}: {g.name}")
        except RuntimeError as exc:
            # Devices were already initialised elsewhere (harmless in most cases).
            print(f"[GPU WARNING] {exc}")
    else:
        print("[GPU] No CUDA-capable GPU detected – running on CPU.")

    return gpus
