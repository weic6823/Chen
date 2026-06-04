"""AirPresenter 主程序入口。"""

from __future__ import annotations

import time

import cv2
import numpy as np

import config
from annotation import AnnotationRenderer
from gesture_recognizer import GestureAction, GestureRecognizer, PresenterState
from hand_detector import HandDetector
from ppt_controller import PPTController


def draw_status_panel(frame: np.ndarray, state: PresenterState, gesture_name: str, fps: float) -> None:
    """在画面左上角绘制 FPS、状态和当前手势。"""
    color = (40, 220, 80) if state == PresenterState.NORMAL else (0, 170, 255)
    if state == PresenterState.PAUSED:
        color = (0, 80, 255)

    cv2.putText(frame, f"FPS: {fps:.1f}", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, (255, 255, 255), config.FONT_THICKNESS)
    cv2.putText(frame, f"STATE: {state.value}", (18, 68), cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, color, config.FONT_THICKNESS)
    cv2.putText(frame, f"GESTURE: {gesture_name}", (18, 102), cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, (255, 255, 255), config.FONT_THICKNESS)


def create_camera() -> cv2.VideoCapture:
    """创建并配置摄像头对象。"""
    capture = cv2.VideoCapture(config.CAMERA_INDEX)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    return capture


def handle_action(action: GestureAction, annotation: AnnotationRenderer, controller: PPTController) -> None:
    """根据识别出的动作执行 PPT 控制或标注控制。"""
    if action == GestureAction.NEXT:
        controller.next_slide()
    elif action == GestureAction.PREVIOUS:
        controller.previous_slide()
    elif action == GestureAction.CLEAR:
        annotation.clear()
    elif action == GestureAction.EXIT:
        controller.quit_slideshow()


def main() -> None:
    """初始化模块并运行 AirPresenter 主循环。"""
    capture = create_camera()
    if not capture.isOpened():
        raise RuntimeError("无法打开摄像头，请检查 CAMERA_INDEX 或摄像头占用情况。")

    detector = HandDetector()
    recognizer = GestureRecognizer()
    controller = PPTController()
    annotation = AnnotationRenderer(config.FRAME_WIDTH, config.FRAME_HEIGHT)

    last_time = time.monotonic()
    fps = 0.0
    gesture_name = "none"
    state = PresenterState.NORMAL

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if config.MIRROR_IMAGE:
                frame = cv2.flip(frame, 1)

            height, width = frame.shape[:2]
            if annotation.width != width or annotation.height != height:
                annotation = AnnotationRenderer(width, height)

            detection = detector.detect(frame)
            if detection is None:
                recognizer.reset_when_no_hand()
                annotation.reset_current_stroke()
                annotation.set_drawing(False if state == PresenterState.NORMAL else annotation.is_drawing)
                gesture_name = "none"
            else:
                result = recognizer.recognize(detection.landmarks, detection.handedness)
                state = result.state
                gesture_name = result.gesture_name

                handle_action(result.action, annotation, controller)
                annotation.set_drawing(result.is_drawing)

                # DRAWING 状态记录食指轨迹；PAUSED 状态保持画布但不新增点
                if result.is_drawing:
                    point = recognizer.get_index_tip_pixel(detection.landmarks, width, height)
                    annotation.add_point(point)

                if config.SHOW_LANDMARKS:
                    detector.draw_landmarks(frame, detection)

                if result.should_exit:
                    break

            overlay = annotation.get_overlay()
            frame = cv2.addWeighted(frame, 1.0, overlay, config.ANNOTATION_ALPHA, 0)

            current_time = time.monotonic()
            delta = current_time - last_time
            last_time = current_time
            if delta > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / delta) if fps > 0 else 1.0 / delta

            draw_status_panel(frame, state, gesture_name, fps)
            cv2.imshow(config.WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                annotation.clear()

    finally:
        capture.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
