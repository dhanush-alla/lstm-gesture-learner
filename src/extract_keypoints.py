"""
src/extract_keypoints.py
─────────────────────────────────────────────────────────────
Shared MediaPipe utilities used by both the data-collection and
real-time inference pipelines:

  • mediapipe_detection()   – run Holistic on a BGR webcam frame
  • draw_styled_landmarks() – render colour-coded skeleton overlay
  • extract_keypoints()     – flatten all landmarks → fixed 1-D array
"""

import warnings
import cv2
import numpy as np
import mediapipe as mp

# In mediapipe 0.10.x the Solutions API is still present but deprecated.
# Suppress the deprecation notices so the terminal stays clean.
warnings.filterwarnings("ignore", category=UserWarning,        module="mediapipe")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="mediapipe")

import sys
import os
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from src.config import (
    NUM_POSE_FEATURES,
    NUM_FACE_FEATURES,
    NUM_HAND_FEATURES,
)

# ─── MediaPipe module handles ──────────────────────────────────────────────────
mp_holistic       = mp.solutions.holistic
mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# FACEMESH_CONTOURS may live on the holistic module or on face_mesh depending
# on the mediapipe version installed.  Resolve once at import time.
try:
    _FACEMESH_CONTOURS = mp_holistic.FACEMESH_CONTOURS
except AttributeError:
    _FACEMESH_CONTOURS = mp.solutions.face_mesh.FACEMESH_CONTOURS


# ─── Per-component drawing specs ─────────────────────────────────────────────
_FACE_LM   = mp_drawing.DrawingSpec(color=(80,  110,  10), thickness=1, circle_radius=1)
_FACE_CON  = mp_drawing.DrawingSpec(color=(80,  256, 121), thickness=1, circle_radius=1)

_POSE_LM   = mp_drawing.DrawingSpec(color=(80,   22,  10), thickness=2, circle_radius=4)
_POSE_CON  = mp_drawing.DrawingSpec(color=(80,   44, 121), thickness=2, circle_radius=2)

_LH_LM     = mp_drawing.DrawingSpec(color=(121,  22,  76), thickness=2, circle_radius=4)
_LH_CON    = mp_drawing.DrawingSpec(color=(121,  44, 250), thickness=2, circle_radius=2)

_RH_LM     = mp_drawing.DrawingSpec(color=(245, 117,  66), thickness=2, circle_radius=4)
_RH_CON    = mp_drawing.DrawingSpec(color=(245,  66, 230), thickness=2, circle_radius=2)


# ─── Public API ───────────────────────────────────────────────────────────────

def mediapipe_detection(image: np.ndarray, model):
    """
    Run MediaPipe Holistic on a single BGR webcam frame.

    MediaPipe expects RGB input.  This function handles the colour-space
    conversion in both directions and temporarily marks the array as
    non-writeable while processing to avoid an unnecessary copy inside
    the MediaPipe C++ back-end.

    Parameters
    ----------
    image : np.ndarray
        BGR frame from cv2.VideoCapture  (H × W × 3, uint8).
    model : mp.solutions.holistic.Holistic
        An initialised Holistic context-manager instance.

    Returns
    -------
    image   : np.ndarray   – BGR frame (same shape as input, writable again)
    results : MediaPipe results object with .pose_landmarks, .face_landmarks,
              .left_hand_landmarks, .right_hand_landmarks attributes.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False        # zero-copy path inside MediaPipe
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_styled_landmarks(image: np.ndarray, results) -> None:
    """
    Draw colour-coded skeletal overlays for every detected body component.

    Colour legend
    ─────────────
    Face mesh   → green tones        (thin lines, small dots)
    Pose        → dark blue/purple   (thick lines, large dots)
    Left hand   → magenta/purple     (thick lines, large dots)
    Right hand  → orange/pink        (thick lines, large dots)

    Parameters
    ----------
    image   : np.ndarray – BGR image; modified **in-place**.
    results : MediaPipe Holistic results object.
    """
    # Face mesh contour lines
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            _FACEMESH_CONTOURS,
            _FACE_LM,
            _FACE_CON,
        )

    # Full-body pose skeleton
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            _POSE_LM,
            _POSE_CON,
        )

    # Left hand skeleton
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            _LH_LM,
            _LH_CON,
        )

    # Right hand skeleton
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            _RH_LM,
            _RH_CON,
        )


def extract_keypoints(results) -> np.ndarray:
    """
    Flatten all detected MediaPipe landmarks into a single 1-D float32 array.

    Feature vector layout  (total = 1662 values)
    ─────────────────────────────────────────────
    Indices   [0   : 132 ]  Pose      33 lm × (x, y, z, visibility)
    Indices   [132 : 1536]  Face     468 lm × (x, y, z)
    Indices   [1536: 1599]  LeftHand  21 lm × (x, y, z)
    Indices   [1599: 1662]  RightHand 21 lm × (x, y, z)

    Any component NOT detected in the current frame is zero-padded so the
    vector always has a fixed length, making it safe to stack into NumPy
    arrays without pre-filtering incomplete frames.

    Parameters
    ----------
    results : MediaPipe Holistic results object.

    Returns
    -------
    np.ndarray of shape (1662,), dtype float32.
    """
    pose = (
        np.array(
            [[lm.x, lm.y, lm.z, lm.visibility]
             for lm in results.pose_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
        if results.pose_landmarks
        else np.zeros(NUM_POSE_FEATURES, dtype=np.float32)
    )

    face = (
        np.array(
            [[lm.x, lm.y, lm.z]
             for lm in results.face_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
        if results.face_landmarks
        else np.zeros(NUM_FACE_FEATURES, dtype=np.float32)
    )

    left_hand = (
        np.array(
            [[lm.x, lm.y, lm.z]
             for lm in results.left_hand_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
        if results.left_hand_landmarks
        else np.zeros(NUM_HAND_FEATURES, dtype=np.float32)
    )

    right_hand = (
        np.array(
            [[lm.x, lm.y, lm.z]
             for lm in results.right_hand_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
        if results.right_hand_landmarks
        else np.zeros(NUM_HAND_FEATURES, dtype=np.float32)
    )

    return np.concatenate([pose, face, left_hand, right_hand])
