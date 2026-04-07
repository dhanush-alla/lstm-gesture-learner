"""
main.py
─────────────────────────────────────────────────────────────
Sign Language Gesture Recognition – main entry point.

Presents a simple terminal menu and dispatches to the three
pipeline stages.  TensorFlow GPU memory-growth is configured
here (once, before any module that touches TF is imported) and
a brief system report is printed so the user can immediately
verify that the GPU is being utilised.

Usage
─────
  python main.py
"""

import os
import sys
import importlib.util
import warnings

# ── Silence noisy TF / oneDNN startup logs BEFORE anything imports TF ─────────
os.environ["TF_CPP_MIN_LOG_LEVEL"]   = "3"   # suppress C++ INFO / WARNING logs
os.environ["TF_ENABLE_ONEDNN_OPTS"]  = "0"   # disable oneDNN floating-point notices
os.environ["ABSL_MIN_LOG_LEVEL"]     = "3"   # suppress absl-py InitializeLog messages
warnings.filterwarnings("ignore", category=DeprecationWarning)  # Python-level TF warnings
warnings.filterwarnings("ignore", category=UserWarning,         module="tensorflow")
warnings.filterwarnings("ignore", message=".*deprecated.*",     module="tensorflow")

# Suppress the TF Python logger (e.g. "GPU support not available on native Windows")
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# ── Resolve project root so sub-modules can always be found ───────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


# ─── GPU configuration (must happen before any TF graph is created) ───────────
# Import config first – it does NOT import TensorFlow itself at module level,
# so this is safe even when only running data collection.
from src.config import setup_gpu


# ─── Dynamic script loader (handles filenames with numeric prefix) ────────────

def _load_script(relative_path: str):
    """
    Import a Python file by path, bypassing normal import machinery.

    This is needed for files whose names start with a digit (e.g.
    '1_collect_data.py'), which Python's regular import system cannot
    handle as package attributes.
    """
    full_path = os.path.join(BASE_DIR, relative_path)
    spec   = importlib.util.spec_from_file_location("_pipeline_module", full_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ─── UI helpers ───────────────────────────────────────────────────────────────

def _banner() -> None:
    print()
    print("=" * 62)
    print("      SIGN LANGUAGE GESTURE RECOGNITION FRAMEWORK")
    print("      Powered by MediaPipe  ·  TensorFlow  ·  OpenCV")
    print("=" * 62)


def _menu() -> None:
    print()
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │  [1]  Take Input     – collect webcam training data  │")
    print("  │  [2]  Train Model    – build & fit the LSTM network  │")
    print("  │  [3]  Run Inference  – live real-time recognition    │")
    print("  │  [0]  Exit                                           │")
    print("  └─────────────────────────────────────────────────────┘")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    _banner()

    # ── Configure GPU once for the entire process lifetime ────────────────────
    print("\n[SYSTEM] Checking GPU …")
    gpus = setup_gpu()

    # Report requires TF; only pulled in now (after memory-growth is set)
    try:
        import tensorflow as tf
        if gpus:
            name = tf.test.gpu_device_name() or "GPU:0"
            print(f"[SYSTEM] TensorFlow {tf.__version__}  →  GPU  {name}")
        else:
            print(f"[SYSTEM] TensorFlow {tf.__version__}  →  CPU")
    except Exception:
        pass  # TF not installed – will fail gracefully when chosen option runs

    # ── Ensure runtime directories exist ──────────────────────────────────────
    from src.config import DATA_PATH, LOG_PATH, MODELS_DIR
    for d in (DATA_PATH, LOG_PATH, MODELS_DIR):
        os.makedirs(d, exist_ok=True)

    # ── Main menu loop ────────────────────────────────────────────────────────
    while True:
        _menu()

        try:
            choice = input("  Enter choice [0-3]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n[INFO] Interrupted.  Goodbye!")
            sys.exit(0)

        if choice == "1":
            print("\n" + "─" * 62)
            print("  MODE: Data Collection")
            print("─" * 62)
            mod = _load_script(os.path.join("src", "1_collect_data.py"))
            mod.collect_data()

        elif choice == "2":
            print("\n" + "─" * 62)
            print("  MODE: Model Training")
            print("─" * 62)
            mod = _load_script(os.path.join("src", "2_train_model.py"))
            mod.train_model()

        elif choice == "3":
            print("\n" + "─" * 62)
            print("  MODE: Real-time Inference")
            print("─" * 62)
            mod = _load_script(os.path.join("src", "3_run_inference.py"))
            mod.run_inference()

        elif choice == "0":
            print("\n[INFO] Goodbye!\n")
            sys.exit(0)

        else:
            print("  [!] Invalid choice.  Please enter 1, 2, 3, or 0.")


if __name__ == "__main__":
    main()
