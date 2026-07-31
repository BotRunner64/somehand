# Python API 教程

[English](../../en/tutorials/api.md)

先完成[安装](../getting-started/installation.md)中的 API 路径，并下载 `mjcf` 资产：

```bash
somehand assets download --only mjcf
```

当程序已经能够提供手部 landmark，并且自己管理循环、可视化或硬件输出时，使用 `somehand.api`。API 不会自动启动摄像头或控制器。

## 单手重定向

engine 只创建一次，之后为每个 frame 重复调用：

```python
import numpy as np

from somehand.api import DEFAULT_CONFIG_PATH, HandFrame, RetargetingEngine

engine = RetargetingEngine.from_config_path(str(DEFAULT_CONFIG_PATH))


def retarget_right_hand(landmarks_3d: np.ndarray) -> np.ndarray:
    landmarks = np.asarray(landmarks_3d, dtype=np.float64)
    if landmarks.shape != (21, 3):
        raise ValueError("landmarks_3d must have shape (21, 3)")

    result = engine.process(
        HandFrame(
            landmarks_3d=landmarks,
            landmarks_2d=None,
            hand_side="right",
        )
    )
    return result.qpos
```

21 行数据必须使用 MediaPipe 手部 landmark 顺序。`result.qpos` 使用所选 MJCF 模型的关节顺序。frame 手别必须与配置手别一致。

使用其他已提交配置：

```python
from somehand.api import RetargetingEngine, resolve_config_path

config_path = resolve_config_path("left/omnihand_left.yaml")
left_engine = RetargetingEngine.from_config_path(str(config_path))
```

自定义配置可以传绝对路径或已存在的相对路径。

## 双手重定向

双手配置会组合左右两个 engine：

```python
import numpy as np

from somehand.api import (
    BiHandFrame,
    BiHandRetargetingEngine,
    DEFAULT_BIHAND_CONFIG_PATH,
    HandFrame,
)

engine = BiHandRetargetingEngine.from_config_path(
    str(DEFAULT_BIHAND_CONFIG_PATH)
)


def retarget_both(
    left_landmarks: np.ndarray | None,
    right_landmarks: np.ndarray | None,
):
    left_frame = None if left_landmarks is None else HandFrame(
        np.asarray(left_landmarks, dtype=np.float64), None, "left"
    )
    right_frame = None if right_landmarks is None else HandFrame(
        np.asarray(right_landmarks, dtype=np.float64), None, "right"
    )
    return engine.process(BiHandFrame(left=left_frame, right=right_frame))
```

某一侧未检测到时传 `None`。返回值的 `result.left.qpos` 和 `result.right.qpos` 是两侧目标，结果标记会说明当前 frame 检测到了哪一侧：

```python
result = retarget_both(left_landmarks, None)
print(result.left_detected, result.right_detected)
```

某侧第一次检测前返回模型中立姿态；有过检测后，该侧缺失时会保留上一次结果。

## 稳定公开接口

嵌入使用统一从 `somehand.api` 导入：

| 用途 | 名称 |
| --- | --- |
| 单手 | `HandFrame`、`RetargetingEngine`、`RetargetingStepResult` |
| 双手 | `BiHandFrame`、`BiHandRetargetingEngine`、`BiHandRetargetingResult` |
| 配置加载 | `load_retargeting_config`、`load_bihand_config` |
| 内置路径 | `DEFAULT_CONFIG_PATH`、`DEFAULT_BIHAND_CONFIG_PATH`、`resolve_config_path` |

如果外部资产不在默认数据目录，请在导入 somehand 前设置 `SOMEHAND_HOME`。配置和资产路径规则见[配置](../reference/configuration.md)与[资产与模型](../reference/assets.md)。
