# AirPresenter：基于手部骨架姿态的隔空 PPT 演示与标注系统

AirPresenter 使用普通笔记本摄像头采集画面，通过 MediaPipe 检测手部 21 个关键点，再根据手指伸展、掌心朝向、静止状态和动作持续时间识别手势。系统可以隔空控制 PPT 翻页，并支持用食指在画面中进行空中标注。

## 项目结构

```text
HandPose_ppt/
├── main.py                 # 主程序入口
├── hand_detector.py        # MediaPipe 手部检测模块
├── gesture_recognizer.py   # 手势识别与状态机
├── annotation.py           # 标注轨迹渲染模块
├── ppt_controller.py       # PPT 键盘控制模块
├── config.py               # 阈值与参数配置
├── requirements.txt        # Python 依赖
└── README.md               # 项目说明
```

## 环境准备

建议使用新的 Python 虚拟环境：

```powershell
cd D:\dev\projects\AirPresenter\HandPose_ppt
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## 运行方式

先打开 PPT 并进入放映模式，然后让 PPT 窗口保持为当前活动窗口，再运行：

```powershell
python main.py
```

程序启动后会打开摄像头窗口。按 `q` 退出程序，按 `c` 清空所有标注。

## 手势说明

- 下一页：食指伸直，其余四指弯曲，掌心朝向摄像头，手保持静止。
- 上一页：拇指横向伸出，其余四指弯曲，掌心朝向摄像头，手保持静止。
- 标注模式开关：食指和中指同时伸直并拢，其余手指弯曲，掌心朝向摄像头。
- 暂停标注：在标注模式下握拳进入 `PAUSED`，再次握拳恢复 `DRAWING`。
- 清空标注：五指张开，并向摄像头方向前推，保持约 0.5 秒。
- 退出：OK 手势保持约 1 秒。

## 状态说明

- `NORMAL`：可触发上一页、下一页、进入标注模式。
- `DRAWING`：记录食指指尖轨迹，忽略翻页手势。
- `PAUSED`：保留已有标注，但暂停记录新的轨迹。

## 参数调整

常用参数集中在 `config.py`：

- `STABLE_FRAMES`：连续稳定帧数，数值越大越不容易误触发。
- `ACTION_COOLDOWN_SECONDS`：翻页动作冷却时间。
- `ANNOTATION_THICKNESS`：标注线条粗细。
- `SMOOTHING_WINDOW`：轨迹移动平均窗口，数值越大越平滑。
- `SHOW_LANDMARKS`：是否显示 MediaPipe 手部骨架。

如果摄像头打不开，可以修改 `CAMERA_INDEX`，常见值为 `0` 或 `1`。
