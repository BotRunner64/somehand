# API 用法

当另一个 Python 程序自己负责输入采集、调度、可视化或硬件控制时，用 API。

## 安装

嵌入使用安装核心 release wheel 即可。只有使用内置 webcam/video/PICO 命令时，才需要 CLI extras。

```bash
pip install "somehand @ https://github.com/BotRunner64/somehand/releases/download/v0.3.0/somehand-0.3.0-py3-none-any.whl"
```

如果需要可编辑的源码安装：

```bash
pip install -e .
```

---

## 准备你需要的内容

release wheel 已包含仓库中提交的 retargeting 配置，只需下载匹配的外部 MJCF 资产：

```bash
somehand assets download --only mjcf
```

通过受支持的 API 获取 wheel 内置默认配置路径：

```python
from somehand.api import DEFAULT_BIHAND_CONFIG_PATH, DEFAULT_CONFIG_PATH, resolve_config_path

print(DEFAULT_CONFIG_PATH)
print(DEFAULT_BIHAND_CONFIG_PATH)
print(resolve_config_path("right/omnihand_right.yaml"))
```

如果资产不在默认用户数据目录，请在导入 somehand 前设置 `SOMEHAND_HOME`。自定义配置仍可作为普通文件系统路径传入。

---

## 稳定导入

嵌入使用从 `somehand.api` 导入。它是受支持的外部 API 面。

```python
from somehand.api import (
    BiHandFrame,
    BiHandRetargetingConfig,
    BiHandRetargetingEngine,
    BiHandRetargetingResult,
    DEFAULT_BIHAND_CONFIG_PATH,
    DEFAULT_CONFIG_PATH,
    HandFrame,
    RetargetingConfig,
    RetargetingEngine,
    RetargetingStepResult,
    load_bihand_config,
    load_retargeting_config,
    resolve_config_path,
)
```

---

## 默认双手示例

```python
import numpy as np

from somehand.api import BiHandFrame, BiHandRetargetingEngine, HandFrame

engine = BiHandRetargetingEngine.from_config_path(
    str(DEFAULT_BIHAND_CONFIG_PATH)
)

left_frame = HandFrame(
    landmarks_3d=np.zeros((21, 3), dtype=np.float64),
    landmarks_2d=None,
    hand_side="left",
)
right_frame = HandFrame(
    landmarks_3d=np.zeros((21, 3), dtype=np.float64),
    landmarks_2d=None,
    hand_side="right",
)

result = engine.process(BiHandFrame(left=left_frame, right=right_frame))
print(result.left.qpos)
print(result.right.qpos)
```

`landmarks_3d` 必须是 21x3 手部 landmark 数组。某一侧没有检测到时传 `None`；engine 会保留该侧上一次结果。

---

## 单步调用

如果你的程序已经有自己的循环，使用 `RetargetingEngine.process()` 或 `BiHandRetargetingEngine.process()`：

```python
engine = BiHandRetargetingEngine.from_config_path(str(DEFAULT_BIHAND_CONFIG_PATH))
result = engine.process(BiHandFrame(left=left_frame, right=right_frame))
```
