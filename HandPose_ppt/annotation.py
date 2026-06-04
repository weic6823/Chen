"""空中手指标注渲染模块。"""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional

import cv2
import numpy as np

import config


class AnnotationRenderer:
    """维护透明标注画布，并绘制平滑后的手指轨迹。"""

    def __init__(self, width: int, height: int) -> None:
        """创建指定尺寸的标注画布。"""
        self.width = width
        self.height = height
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.point_buffer: Deque[tuple[int, int]] = deque(maxlen=config.SMOOTHING_WINDOW)
        self.last_point: Optional[tuple[int, int]] = None
        self.is_drawing = False

    def set_drawing(self, is_drawing: bool) -> None:
        """设置当前是否记录轨迹。"""
        self.is_drawing = is_drawing
        if not is_drawing:
            self.reset_current_stroke()

    def add_point(self, point: tuple[int, int]) -> None:
        """加入一个食指指尖点，并用移动平均平滑轨迹。"""
        if not self.is_drawing:
            return

        x = int(np.clip(point[0], 0, self.width - 1))
        y = int(np.clip(point[1], 0, self.height - 1))
        self.point_buffer.append((x, y))

        # 移动平均能减轻 MediaPipe 单帧抖动带来的锯齿轨迹
        avg_x = int(sum(p[0] for p in self.point_buffer) / len(self.point_buffer))
        avg_y = int(sum(p[1] for p in self.point_buffer) / len(self.point_buffer))
        smooth_point = (avg_x, avg_y)

        if self.last_point is None:
            self.last_point = smooth_point
            return

        distance = float(np.linalg.norm(np.array(smooth_point) - np.array(self.last_point)))
        if distance < config.MIN_POINT_DISTANCE:
            return

        cv2.line(
            self.canvas,
            self.last_point,
            smooth_point,
            config.ANNOTATION_COLOR,
            config.ANNOTATION_THICKNESS,
            lineType=cv2.LINE_AA,
        )
        self.last_point = smooth_point

    def reset_current_stroke(self) -> None:
        """断开当前笔画，下一次绘制从新点开始。"""
        self.point_buffer.clear()
        self.last_point = None

    def clear(self) -> None:
        """清空全部标注轨迹。"""
        self.canvas[:] = 0
        self.reset_current_stroke()

    def get_overlay(self) -> np.ndarray:
        """返回当前标注画布。"""
        return self.canvas
