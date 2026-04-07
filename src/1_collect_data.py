"""
src/1_collect_data.py
─────────────────────────────────────────────────────────────
Option 1 – Webcam-based gesture data collection.

For each gesture name entered by the user, this script records:
  • NUM_SEQUENCES (30) video sequences
  • each sequence consisting of SEQUENCE_LENGTH (30) frames
  • each frame saved as a NumPy array of MediaPipe keypoints

Data is written to:
  data/<gesture_name>/<sequence_index>/<frame_index>.npy

Usage (standalone):
  python src/1_collect_data.py

Usage (via main.py):
  Select option 1 from the main menu.
"""

import sys
import os
import time
import cv2
import numpy as np
from tqdm import tqdm

# ── Make project root importable regardless of invocation method ───────────────
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from src.config import (
    DATA_PATH,
    NUM_SEQUENCES,
    SEQUENCE_LENGTH,
    BREAK_SECONDS,
)
from src.extract_keypoints import (
    mp_holistic,
    mediapipe_detection,
    draw_styled_landmarks,
    extract_keypoints,
)

# Window title (shared between helpers)
_WIN = "Sign Language – Data Collection  |  Q = Quit"


# ─── Directory helpers ────────────────────────────────────────────────────────

def _create_directories(action: str) -> None:
    """Create the full folder tree for a gesture before recording starts."""
    for seq in range(NUM_SEQUENCES):
        path = os.path.join(DATA_PATH, action, str(seq))
        os.makedirs(path, exist_ok=True)


# ─── On-screen UI helpers ─────────────────────────────────────────────────────

def _draw_recording_ui(
    image: np.ndarray,
    action: str,
    sequence: int,
    frame_num: int,
) -> None:
    """
    Overlay recording status onto the live webcam frame.

    Shows:
      • Top bar    – gesture name, sequence/frame counters, REC indicator
      • Bottom bar – overall progress bar across all sequences
    """
    h, w = image.shape[:2]

    # Semi-transparent top banner
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, 65), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, image, 0.45, 0, image)

    # Gesture name  (left)
    cv2.putText(
        image, f"Gesture: {action}",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 80), 2, cv2.LINE_AA,
    )

    # Sequence / frame counter  (left, second row)
    cv2.putText(
        image,
        f"Seq {sequence + 1:02d}/{NUM_SEQUENCES}   "
        f"Frame {frame_num + 1:02d}/{SEQUENCE_LENGTH}",
        (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 100), 1, cv2.LINE_AA,
    )

    # Blinking REC dot + label  (top-right)
    cv2.circle(image, (w - 30, 30), 11, (0, 0, 220), -1)
    cv2.putText(
        image, "REC",
        (w - 68, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 220), 1, cv2.LINE_AA,
    )

    # Bottom progress bar: sequences completed so far
    bar_total = w - 20
    filled = int(bar_total * sequence / NUM_SEQUENCES)
    cv2.rectangle(image, (10, h - 18), (10 + bar_total, h - 8), (50, 50, 50), -1)
    cv2.rectangle(image, (10, h - 18), (10 + filled,    h - 8), (0, 200, 80), -1)
    cv2.putText(
        image, "Progress",
        (10, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA,
    )


def _show_countdown(
    cap: cv2.VideoCapture,
    holistic,
    action: str,
    next_seq: int,
) -> bool:
    """
    Display an animated countdown overlay between recording sequences so the
    user can reposition their hands.  The live skeleton is still rendered
    during the break so the user can verify their hand placement.

    Parameters
    ----------
    cap        : open VideoCapture object
    holistic   : active Holistic context-manager instance
    action     : gesture name (displayed as reminder)
    next_seq   : 1-based index of the upcoming sequence

    Returns
    -------
    bool – True to continue, False if the user pressed Q to quit early.
    """
    deadline = time.time() + BREAK_SECONDS

    while time.time() < deadline:
        ret, frame = cap.read()
        if not ret:
            return True

        frame = cv2.flip(frame, 1)
        image, results = mediapipe_detection(frame, holistic)
        draw_styled_landmarks(image, results)

        remaining = int(deadline - time.time()) + 1
        h, w = image.shape[:2]

        # Dim background
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.40, image, 0.60, 0, image)

        # "Get Ready!" headline
        cv2.putText(
            image, "Get Ready!",
            (w // 2 - 105, h // 2 - 55),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA,
        )

        # Upcoming sequence info
        cv2.putText(
            image,
            f"Sequence  {next_seq} / {NUM_SEQUENCES}  incoming",
            (w // 2 - 175, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.80, (255, 255, 255), 2, cv2.LINE_AA,
        )

        # Large countdown digit
        cv2.putText(
            image, str(remaining),
            (w // 2 - 22, h // 2 + 85),
            cv2.FONT_HERSHEY_SIMPLEX, 2.8, (0, 230, 0), 5, cv2.LINE_AA,
        )

        # Gesture reminder (top-left)
        cv2.putText(
            image, f"Gesture: '{action}'",
            (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 80), 2, cv2.LINE_AA,
        )

        # Quit hint
        cv2.putText(
            image, "Press Q to quit",
            (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA,
        )

        cv2.imshow(_WIN, image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False

    return True


# ─── Main collection routine ──────────────────────────────────────────────────

def collect_data() -> None:
    """
    Interactive webcam-based data collection for a single gesture class.

    Flow
    ────
    1. Prompt for gesture name → create directory structure.
    2. Show live feed; wait for SPACE to begin.
    3. For each of the NUM_SEQUENCES sequences:
         a. Show BREAK_SECONDS countdown  (skip before first sequence).
         b. Record SEQUENCE_LENGTH frames; save each as an .npy file.
    4. Print completion summary.
    """
    # ── gesture name ─────────────────────────────────────────────────────────
    action = input("\n  Enter gesture name to record (e.g. 'Hello'): ").strip()
    if not action:
        print("[ERROR] Gesture name cannot be empty.")
        return

    _create_directories(action)
    print(f"\n[INFO] Directories ready for '{action}'")
    print(
        f"[INFO] Will record {NUM_SEQUENCES} sequences "
        f"× {SEQUENCE_LENGTH} frames each."
    )
    print("[INFO] Controls:  SPACE = start   |   Q = quit at any time\n")

    # ── open webcam ──────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam (device index 0).")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:

        # ── Wait-for-SPACE start screen ───────────────────────────────────
        print("[INFO] Webcam open.  Press SPACE when ready to begin recording.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)

            h, w = image.shape[:2]

            # Dark top banner
            overlay = image.copy()
            cv2.rectangle(overlay, (0, 0), (w, 70), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.55, image, 0.45, 0, image)

            cv2.putText(
                image, f"Gesture: '{action}'",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 80), 2, cv2.LINE_AA,
            )
            cv2.putText(
                image, "Press  SPACE  to start recording",
                (w // 2 - 230, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 255), 2, cv2.LINE_AA,
            )
            cv2.putText(
                image, "Press  Q  to go back",
                (w // 2 - 145, h // 2 + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, (180, 180, 180), 1, cv2.LINE_AA,
            )

            cv2.imshow(_WIN, image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                break
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

        # ── Record sequences ──────────────────────────────────────────────
        seq_pbar = tqdm(
            range(NUM_SEQUENCES),
            desc=f"  Collecting '{action}'",
            unit="seq",
            colour="green",
            leave=True,
        )
        for sequence in seq_pbar:
            seq_pbar.set_postfix({"seq": f"{sequence + 1}/{NUM_SEQUENCES}"})

            # Countdown between sequences (skip before the very first one)
            if sequence > 0:
                ok = _show_countdown(cap, holistic, action, sequence + 1)
                if not ok:
                    cap.release()
                    cv2.destroyAllWindows()
                    seq_pbar.close()
                    print(f"\n[INFO] Recording stopped early at sequence {sequence}.")
                    return
            else:
                # Brief pause so the user is ready for frame 0
                time.sleep(0.8)

            print(f"  ● Recording sequence {sequence + 1:2d}/{NUM_SEQUENCES} …", end="", flush=True)

            # ── Collect SEQUENCE_LENGTH frames ───────────────────────────
            for frame_num in range(SEQUENCE_LENGTH):
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)

                # Overlay recording UI
                _draw_recording_ui(image, action, sequence, frame_num)

                cv2.putText(
                    image, "Press Q to quit",
                    (10, image.shape[0] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA,
                )

                cv2.imshow(_WIN, image)

                # ── Extract and persist keypoints ─────────────────────────
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(
                    DATA_PATH, action, str(sequence), str(frame_num)
                )
                np.save(npy_path, keypoints)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    print(
                        f"\n[INFO] Recording halted.  "
                        f"Saved up to sequence {sequence}, frame {frame_num}."
                    )
                    return

            print("  ✓ done")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()

    print(f"\n[SUCCESS] Data collection complete for gesture '{action}'!")
    print(f"  Saved {NUM_SEQUENCES} sequences of {SEQUENCE_LENGTH} frames.")
    print(f"  Location: {os.path.join(DATA_PATH, action)}\n")


# ── Allow direct execution ────────────────────────────────────────────────────
if __name__ == "__main__":
    collect_data()
