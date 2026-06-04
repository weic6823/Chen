"""Global configuration for AirPresenter."""

from __future__ import annotations

from typing import Final


# Camera and window.
CAMERA_INDEX: Final[int] = 0
FRAME_WIDTH: Final[int] = 1280
FRAME_HEIGHT: Final[int] = 720
WINDOW_NAME: Final[str] = "AirPresenter"
MIRROR_IMAGE: Final[bool] = True
SHOW_LANDMARKS: Final[bool] = True

# MediaPipe detection.
MAX_NUM_HANDS: Final[int] = 1
MIN_DETECTION_CONFIDENCE: Final[float] = 0.65
MIN_TRACKING_CONFIDENCE: Final[float] = 0.65

# Gesture thresholds. MediaPipe hand coordinates are normalized.
FINGER_EXTEND_Y_GAP: Final[float] = 0.05
THUMB_EXTEND_X_GAP: Final[float] = 0.08
DRAW_FINGERS_CLOSE_GAP: Final[float] = 0.055
OK_CIRCLE_DISTANCE: Final[float] = 0.065
FIST_TIP_PALM_DISTANCE: Final[float] = 0.16

# Debounce and hold timing.
STABLE_FRAMES: Final[int] = 8
ACTION_COOLDOWN_SECONDS: Final[float] = 1.0
DRAW_TOGGLE_COOLDOWN_SECONDS: Final[float] = 1.2
CLEAR_HOLD_SECONDS: Final[float] = 0.5
EXIT_HOLD_SECONDS: Final[float] = 1.0

# Stillness and forward-push detection.
STILL_HISTORY_SIZE: Final[int] = 8
STILL_MOVEMENT_THRESHOLD: Final[float] = 0.025
PUSH_HISTORY_SIZE: Final[int] = 10
PUSH_SCALE_DELTA: Final[float] = 0.06

# Annotation rendering.
ANNOTATION_COLOR: Final[tuple[int, int, int]] = (0, 0, 255)
ANNOTATION_THICKNESS: Final[int] = 5
ANNOTATION_ALPHA: Final[float] = 0.72
SMOOTHING_WINDOW: Final[int] = 5
MIN_POINT_DISTANCE: Final[float] = 3.0

# On-screen text.
FONT_SCALE: Final[float] = 0.8
FONT_THICKNESS: Final[int] = 2
