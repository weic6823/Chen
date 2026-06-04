"""PPT 键盘控制模块。"""

from __future__ import annotations

import pyautogui


class PPTController:
    """使用 pyautogui 模拟键盘事件控制当前活动的 PPT 窗口。"""

    def __init__(self) -> None:
        """初始化控制器并设置较短的按键间隔。"""
        pyautogui.PAUSE = 0.03

    def next_slide(self) -> None:
        """切换到下一页幻灯片。"""
        pyautogui.press("right")

    def previous_slide(self) -> None:
        """切换到上一页幻灯片。"""
        pyautogui.press("left")

    def quit_slideshow(self) -> None:
        """退出幻灯片放映或让主程序结束。"""
        pyautogui.press("esc")
