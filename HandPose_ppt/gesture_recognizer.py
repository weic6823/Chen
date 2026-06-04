"""Gesture recognition and presenter state machine."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Optional

import numpy as np

import config


class PresenterState(str, Enum):
    """Presenter interaction states."""

    NORMAL = "NORMAL"
    DRAWING = "DRAWING"
    PAUSED = "PAUSED"


class GestureAction(str, Enum):
    """Actions emitted by the gesture recognizer."""

    NONE = "none"
    NEXT = "next"
    PREVIOUS = "previous"
    TOGGLE_DRAW = "toggle_draw"
    CLEAR = "clear"
    EXIT = "exit"


@dataclass(slots=True)
class GestureResult:
    """Gesture recognition result for a single frame."""

    state: PresenterState
    action: GestureAction = GestureAction.NONE
    gesture_name: str = "none"
    is_drawing: bool = False
    should_exit: bool = False
    debug: dict[str, bool] = field(default_factory=dict)


class GestureRecognizer:
    """Recognize README-defined hand gestures and maintain presenter state."""

    def __init__(self) -> None:
        self.state = PresenterState.NORMAL
        self.last_action_time = 0.0
        self.last_toggle_time = 0.0
        self.last_stable_gesture = "none"
        self.stable_count = 0
        self.wrist_history: Deque[np.ndarray] = deque(maxlen=config.STILL_HISTORY_SIZE)
        self.scale_history: Deque[float] = deque(maxlen=config.PUSH_HISTORY_SIZE)
        self.clear_start_time: Optional[float] = None
        self.exit_start_time: Optional[float] = None
        self.fist_start_time: Optional[float] = None

    def reset_when_no_hand(self) -> None:
        """Reset short gesture buffers when no hand is detected."""
        self.last_stable_gesture = "none"
        self.stable_count = 0
        self.wrist_history.clear()
        self.scale_history.clear()
        self.clear_start_time = None
        self.exit_start_time = None
        self.fist_start_time = None

    def recognize(self, landmarks: np.ndarray, handedness: str) -> GestureResult:
        """Recognize the current frame and emit presenter actions."""
        now = time.monotonic()
        flags = self._compute_flags(landmarks, handedness)
        raw_gesture = self._classify_gesture(flags)
        stable_gesture = self._update_stability(raw_gesture)
        action = GestureAction.NONE

        if flags["ok"]:
            if self.exit_start_time is None:
                self.exit_start_time = now
            if now - self.exit_start_time >= config.EXIT_HOLD_SECONDS:
                return GestureResult(
                    state=self.state,
                    action=GestureAction.EXIT,
                    gesture_name="ok",
                    is_drawing=self.state == PresenterState.DRAWING,
                    should_exit=True,
                    debug=flags,
                )
        else:
            self.exit_start_time = None

        if flags["clear"]:
            if self.clear_start_time is None:
                self.clear_start_time = now
            if now - self.clear_start_time >= config.CLEAR_HOLD_SECONDS:
                action = GestureAction.CLEAR
                self.clear_start_time = None
        else:
            self.clear_start_time = None

        if self.state in (PresenterState.DRAWING, PresenterState.PAUSED) and flags["fist"]:
            if self.fist_start_time is None:
                self.fist_start_time = now
            if now - self.fist_start_time >= 0.35 and self._toggle_ready(now):
                self.state = (
                    PresenterState.PAUSED
                    if self.state == PresenterState.DRAWING
                    else PresenterState.DRAWING
                )
                self.last_toggle_time = now
                self.fist_start_time = None
        elif not flags["fist"]:
            self.fist_start_time = None

        if stable_gesture == "draw_mode" and self._toggle_ready(now):
            self.state = (
                PresenterState.DRAWING
                if self.state == PresenterState.NORMAL
                else PresenterState.NORMAL
            )
            self.last_toggle_time = now
            action = GestureAction.TOGGLE_DRAW

        if self.state == PresenterState.NORMAL and action == GestureAction.NONE:
            if stable_gesture == "next" and self._action_ready(now):
                action = GestureAction.NEXT
                self.last_action_time = now
            elif stable_gesture == "previous" and self._action_ready(now):
                action = GestureAction.PREVIOUS
                self.last_action_time = now

        is_drawing = self.state == PresenterState.DRAWING
        return GestureResult(
            state=self.state,
            action=action,
            gesture_name=stable_gesture,
            is_drawing=is_drawing,
            should_exit=False,
            debug=flags,
        )

    def get_index_tip_pixel(self, landmarks: np.ndarray, width: int, height: int) -> tuple[int, int]:
        """Convert the index fingertip to pixel coordinates."""
        index_tip = landmarks[8]
        return int(index_tip[0] * width), int(index_tip[1] * height)

    def _compute_flags(self, landmarks: np.ndarray, handedness: str) -> dict[str, bool]:
        fingers = self._finger_states(landmarks, handedness)
        palm_front = self._is_palm_facing_camera(landmarks)
        still = self._is_hand_still(landmarks)
        pushed = self._is_pushing_forward(landmarks)
        index_middle_close = self._distance(landmarks[8], landmarks[12]) < config.DRAW_FINGERS_CLOSE_GAP
        ok_circle = self._distance(landmarks[4], landmarks[8]) < config.OK_CIRCLE_DISTANCE

        return {
            "thumb": fingers["thumb"],
            "index": fingers["index"],
            "middle": fingers["middle"],
            "ring": fingers["ring"],
            "pinky": fingers["pinky"],
            "palm_front": palm_front,
            "still": still,
            "pushed": pushed,
            "next": fingers["index"]
            and not fingers["middle"]
            and not fingers["ring"]
            and not fingers["pinky"]
            and not fingers["thumb"]
            and palm_front
            and still,
            "previous": fingers["thumb"]
            and not fingers["index"]
            and not fingers["middle"]
            and not fingers["ring"]
            and not fingers["pinky"]
            and palm_front
            and still,
            "draw_mode": fingers["index"]
            and fingers["middle"]
            and index_middle_close
            and not fingers["ring"]
            and not fingers["pinky"]
            and not fingers["thumb"]
            and palm_front,
            "clear": all(fingers.values()) and palm_front and pushed,
            "ok": ok_circle and fingers["middle"] and fingers["ring"] and fingers["pinky"],
            "fist": self._is_fist(landmarks),
        }

    def _classify_gesture(self, flags: dict[str, bool]) -> str:
        for name in ("draw_mode", "next", "previous", "clear", "ok", "fist"):
            if flags[name]:
                return name
        return "none"

    def _update_stability(self, gesture_name: str) -> str:
        if gesture_name == self.last_stable_gesture:
            self.stable_count += 1
        else:
            self.last_stable_gesture = gesture_name
            self.stable_count = 1
        if self.stable_count >= config.STABLE_FRAMES:
            return gesture_name
        return "none"

    def _finger_states(self, landmarks: np.ndarray, handedness: str) -> dict[str, bool]:
        index = landmarks[8][1] < landmarks[6][1] - config.FINGER_EXTEND_Y_GAP
        middle = landmarks[12][1] < landmarks[10][1] - config.FINGER_EXTEND_Y_GAP
        ring = landmarks[16][1] < landmarks[14][1] - config.FINGER_EXTEND_Y_GAP
        pinky = landmarks[20][1] < landmarks[18][1] - config.FINGER_EXTEND_Y_GAP

        thumb_tip_x = landmarks[4][0]
        thumb_ip_x = landmarks[3][0]
        thumb = abs(thumb_tip_x - thumb_ip_x) > config.THUMB_EXTEND_X_GAP
        if handedness in {"Left", "Right"}:
            thumb = thumb and abs(landmarks[4][0] - landmarks[2][0]) > config.THUMB_EXTEND_X_GAP

        return {
            "thumb": bool(thumb),
            "index": bool(index),
            "middle": bool(middle),
            "ring": bool(ring),
            "pinky": bool(pinky),
        }

    def _is_palm_facing_camera(self, landmarks: np.ndarray) -> bool:
        palm_width = self._distance(landmarks[5], landmarks[17])
        wrist_to_middle = self._distance(landmarks[0], landmarks[9])
        if wrist_to_middle == 0:
            return False
        return palm_width / wrist_to_middle > 0.55

    def _is_hand_still(self, landmarks: np.ndarray) -> bool:
        self.wrist_history.append(landmarks[0].copy())
        if len(self.wrist_history) < self.wrist_history.maxlen:
            return False
        points = np.array(self.wrist_history)
        movement = float(np.max(np.linalg.norm(points[:, :2] - points[-1, :2], axis=1)))
        return movement < config.STILL_MOVEMENT_THRESHOLD

    def _is_pushing_forward(self, landmarks: np.ndarray) -> bool:
        scale = self._palm_scale(landmarks)
        self.scale_history.append(scale)
        if len(self.scale_history) < self.scale_history.maxlen:
            return False
        return self.scale_history[-1] - self.scale_history[0] > config.PUSH_SCALE_DELTA

    def _is_fist(self, landmarks: np.ndarray) -> bool:
        palm_center = np.mean(landmarks[[0, 5, 9, 13, 17], :2], axis=0)
        tip_indices = [8, 12, 16, 20]
        distances = [float(np.linalg.norm(landmarks[idx, :2] - palm_center)) for idx in tip_indices]
        return max(distances) < config.FIST_TIP_PALM_DISTANCE

    def _palm_scale(self, landmarks: np.ndarray) -> float:
        return float(
            (
                self._distance(landmarks[0], landmarks[9])
                + self._distance(landmarks[5], landmarks[17])
                + self._distance(landmarks[0], landmarks[13])
            )
            / 3.0
        )

    def _distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        return float(np.linalg.norm(p1[:2] - p2[:2]))

    def _action_ready(self, now: float) -> bool:
        return now - self.last_action_time >= config.ACTION_COOLDOWN_SECONDS

    def _toggle_ready(self, now: float) -> bool:
        return now - self.last_toggle_time >= config.DRAW_TOGGLE_COOLDOWN_SECONDS
