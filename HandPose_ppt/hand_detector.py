"""基于 MediaPipe 的手部关键点检测模块。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

import config


@dataclass(slots=True)
class HandDetectionResult:
    """保存单只手的检测结果。"""

    landmarks: np.ndarray
    handedness: str
    score: float
    raw_landmarks: object


class HandDetector:
    """封装 MediaPipe Hands，输出归一化手部关键点。"""

    def __init__(self) -> None:
        """初始化 MediaPipe Hands 检测器。"""
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=config.MAX_NUM_HANDS,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )

    def detect(self, frame_bgr: np.ndarray) -> Optional[HandDetectionResult]:
        """检测图像中的手，并返回置信度最高的一只手。"""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.hands.process(frame_rgb)
        frame_rgb.flags.writeable = True

        if not results.multi_hand_landmarks:
            return None

        best_index = 0
        best_score = 0.0
        handedness = "Unknown"
        if results.multi_handedness:
            scores = [item.classification[0].score for item in results.multi_handedness]
            best_index = int(np.argmax(scores))
            best_score = float(scores[best_index])
            handedness = results.multi_handedness[best_index].classification[0].label

        raw_landmarks = results.multi_hand_landmarks[best_index]
        landmarks = np.array(
            [[point.x, point.y, point.z] for point in raw_landmarks.landmark],
            dtype=np.float32,
        )
        return HandDetectionResult(
            landmarks=landmarks,
            handedness=handedness,
            score=best_score,
            raw_landmarks=raw_landmarks,
        )

    def draw_landmarks(self, frame_bgr: np.ndarray, result: HandDetectionResult) -> None:
        """在图像上绘制手部骨架点。"""
        self.mp_drawing.draw_landmarks(
            frame_bgr,
            result.raw_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_styles.get_default_hand_landmarks_style(),
            self.mp_styles.get_default_hand_connections_style(),
        )

    def close(self) -> None:
        """释放 MediaPipe Hands 资源。"""
        self.hands.close()
