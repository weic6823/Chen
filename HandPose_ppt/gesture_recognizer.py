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
    """Recognize hand gestures and maintain NORMAL/DRAWING/PAUSED state."""

    def __init__(self) -> None:
        self.state = PresenterState.NORMAL
        self.last_action_time = 0.0
        self.last_toggle_time = 0.0
        self.last_stable_gesture = "none"
        self.stable_count = 0
        self.wrist_history: Deque[np.ndarray] = deque(maxlen=config.STILL_HISTORY_SIZE)
        self.swipe_history: Deque[np.ndarray] = deque(maxlen=config.SWIPE_HISTORY_SIZE)
        self.scale_history: Deque[float] = deque(maxlen=config.PUSH_HISTORY_SIZE)
        self.exit_start_time: Optional[float] = None
        self.pause_armed = False
        self.pause_armed_time: Optional[float] = None

    def reset_when_no_hand(self) -> None:
        """Reset short gesture buffers when no hand is detected."""
        self.last_stable_gesture = "none"
        self.stable_count = 0
        self.wrist_history.clear()
        self.swipe_history.clear()
        self.scale_history.clear()
        self.exit_start_time = None
        self.pause_armed = False
        self.pause_armed_time = None

    def recognize(self, landmarks: np.ndarray, handedness: str) -> GestureResult:
        """Recognize the current frame and emit presenter actions."""
        now = time.monotonic()
        flags = self._compute_flags(landmarks, handedness)
        raw_gesture = self._classify_gesture(flags)
        stable_gesture = self._update_stability(raw_gesture)
        action = GestureAction.NONE
        gesture_name = stable_gesture

        # Keep OK as an emergency quit gesture.
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

        if self.state == PresenterState.NORMAL:
            if stable_gesture == "draw_mode" and self._toggle_ready(now):
                self.state = PresenterState.DRAWING
                self.last_toggle_time = now
                action = GestureAction.TOGGLE_DRAW
                gesture_name = "draw_mode"
                self.swipe_history.clear()
            elif flags["swipe_down"] and self._action_ready(now):
                action = GestureAction.NEXT
                gesture_name = "swipe_down"
                self.last_action_time = now
                self.swipe_history.clear()
            elif flags["swipe_up"] and self._action_ready(now):
                action = GestureAction.PREVIOUS
                gesture_name = "swipe_up"
                self.last_action_time = now
                self.swipe_history.clear()

        else:
            if stable_gesture == "exit_draw" and self._toggle_ready(now):
                self.state = PresenterState.NORMAL
                self.last_toggle_time = now
                self.pause_armed = False
                action = GestureAction.TOGGLE_DRAW
                gesture_name = "exit_draw"
            elif flags["open_front"]:
                self.pause_armed = True
                self.pause_armed_time = now
                gesture_name = "pause_ready"
            elif flags["fist"] and self._pause_sequence_ready(now) and self._toggle_ready(now):
                self.state = PresenterState.PAUSED
                self.last_toggle_time = now
                self.pause_armed = False
                self.pause_armed_time = None
                action = GestureAction.TOGGLE_DRAW
                gesture_name = "pause"
            elif self.pause_armed_time is not None and now - self.pause_armed_time > config.PAUSE_SEQUENCE_TIMEOUT_SECONDS:
                self.pause_armed = False
                self.pause_armed_time = None

        draw_suppressed = (
            action == GestureAction.TOGGLE_DRAW
            or flags["open_front"]
            or flags["open_back"]
            or flags["fist"]
        )
        is_drawing = self.state == PresenterState.DRAWING and not draw_suppressed
        return GestureResult(
            state=self.state,
            action=action,
            gesture_name=gesture_name,
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
        open_hand = all(fingers.values())
        fist = self._is_fist(landmarks)
        palm_front = self._is_palm_facing_camera(landmarks, handedness)
        hand_back = self._is_back_of_hand_facing_camera(landmarks, handedness)
        open_front = open_hand and palm_front and self._fingers_pointing_up(landmarks)
        open_back = open_hand and hand_back and self._fingers_pointing_up(landmarks)
        swipe = self._vertical_swipe(landmarks, open_front)
        ok_circle = self._distance(landmarks[4], landmarks[8]) < config.OK_CIRCLE_DISTANCE

        return {
            "thumb": fingers["thumb"],
            "index": fingers["index"],
            "middle": fingers["middle"],
            "ring": fingers["ring"],
            "pinky": fingers["pinky"],
            "open_front": open_front,
            "open_back": open_back,
            "palm_front": palm_front,
            "hand_back": hand_back,
            "swipe_down": swipe == "down",
            "swipe_up": swipe == "up",
            "draw_mode": fingers["thumb"] and fingers["index"] and not fingers["middle"] and not fingers["ring"] and not fingers["pinky"],
            "exit_draw": open_back,
            "ok": ok_circle and fingers["middle"] and fingers["ring"] and fingers["pinky"],
            "fist": fist,
        }

    def _classify_gesture(self, flags: dict[str, bool]) -> str:
        for name in ("exit_draw", "draw_mode", "swipe_down", "swipe_up", "open_front", "ok", "fist"):
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

    def _fingers_pointing_up(self, landmarks: np.ndarray) -> bool:
        fingertips_y = np.mean(landmarks[[8, 12, 16, 20], 1])
        knuckles_y = np.mean(landmarks[[5, 9, 13, 17], 1])
        return bool(fingertips_y < knuckles_y - config.FINGER_EXTEND_Y_GAP)

    def _vertical_swipe(self, landmarks: np.ndarray, open_front: bool) -> str:
        if not open_front:
            self.swipe_history.clear()
            return "none"

        self.swipe_history.append(self._palm_center(landmarks))
        if len(self.swipe_history) < self.swipe_history.maxlen:
            return "none"

        points = np.array(self.swipe_history)
        delta_x = float(points[-1][0] - points[0][0])
        delta_y = float(points[-1][1] - points[0][1])
        if abs(delta_x) > config.SWIPE_MAX_X_DELTA:
            return "none"
        if delta_y > config.SWIPE_Y_DELTA:
            return "down"
        if delta_y < -config.SWIPE_Y_DELTA:
            return "up"
        return "none"

    def _is_palm_facing_camera(self, landmarks: np.ndarray, handedness: str) -> bool:
        return self._palm_orientation_score(landmarks, handedness) > config.PALM_NORMAL_Z_THRESHOLD

    def _is_back_of_hand_facing_camera(self, landmarks: np.ndarray, handedness: str) -> bool:
        return self._palm_orientation_score(landmarks, handedness) < -config.PALM_NORMAL_Z_THRESHOLD

    def _palm_orientation_score(self, landmarks: np.ndarray, handedness: str) -> float:
        wrist = landmarks[0]
        index_mcp = landmarks[5]
        pinky_mcp = landmarks[17]
        normal_z = float(np.cross(index_mcp - wrist, pinky_mcp - wrist)[2])
        hand_sign = -1.0 if handedness == "Left" else 1.0
        return normal_z * hand_sign * float(config.PALM_FRONT_RIGHT_HAND_NORMAL_SIGN)

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

    def _palm_center(self, landmarks: np.ndarray) -> np.ndarray:
        return np.mean(landmarks[[0, 5, 9, 13, 17], :2], axis=0)

    def _distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        return float(np.linalg.norm(p1[:2] - p2[:2]))

    def _action_ready(self, now: float) -> bool:
        return now - self.last_action_time >= config.ACTION_COOLDOWN_SECONDS

    def _toggle_ready(self, now: float) -> bool:
        return now - self.last_toggle_time >= config.DRAW_TOGGLE_COOLDOWN_SECONDS

    def _pause_sequence_ready(self, now: float) -> bool:
        return (
            self.pause_armed
            and self.pause_armed_time is not None
            and now - self.pause_armed_time <= config.PAUSE_SEQUENCE_TIMEOUT_SECONDS
        )
