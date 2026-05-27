# API 用法

当另一个 Python 程序自己负责输入采集、调度、可视化或硬件控制时，用 API。

## 安装

嵌入使用从本仓库安装核心包即可。只有使用内置 webcam/video/PICO 命令时，才需要 CLI extras。

```bash
pip install -e .
pip install huggingface_hub
```

如果从 ModelScope 下载，而不是 HuggingFace，需要安装 ModelScope client：

```bash
pip install modelscope
```

---

## 准备你需要的内容

配置文件来自本仓库，模型资产从云端下载。下面以双手 LinkerHand L6 为例。

```bash
mkdir -p configs/retargeting/{base,left,right,bihand} assets/mjcf
cp -a /path/to/somehand/configs/retargeting/base/_universal_common.yaml configs/retargeting/base/
cp -a /path/to/somehand/configs/retargeting/base/linkerhand_l6.yaml configs/retargeting/base/
cp -a /path/to/somehand/configs/retargeting/left/linkerhand_l6_left.yaml configs/retargeting/left/
cp -a /path/to/somehand/configs/retargeting/right/linkerhand_l6_right.yaml configs/retargeting/right/
cp -a /path/to/somehand/configs/retargeting/bihand/linkerhand_l6_bihand.yaml configs/retargeting/bihand/
```

从 HuggingFace 下载对应模型资产：

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="12e21/somehand-assets",
    allow_patterns=[
        "assets/mjcf/linkerhand_l6_left/**",
        "assets/mjcf/linkerhand_l6_right/**",
    ],
    local_dir=".",
)
```

使用 ModelScope 时，使用同一组文件列表，调用 `modelscope.snapshot_download`，仓库为 `BingqianWu/somehand-assets`。

应用目录最终应有这些路径：

```text
configs/retargeting/bihand/linkerhand_l6_bihand.yaml
configs/retargeting/left/linkerhand_l6_left.yaml
configs/retargeting/right/linkerhand_l6_right.yaml
configs/retargeting/base/linkerhand_l6.yaml
configs/retargeting/base/_universal_common.yaml
assets/mjcf/linkerhand_l6_left/model.xml
assets/mjcf/linkerhand_l6_right/model.xml
```

---

## 稳定导入

嵌入使用优先从 `somehand.api` 导入：

```python
from somehand.api import BiHandFrame, BiHandRetargetingEngine, HandFrame
```

相关 namespace：

| Namespace | 用途 |
| --- | --- |
| `somehand.api` | 稳定嵌入入口：配置、frame、单步 engine。 |
| `somehand.core` | 纯 domain model 和转换函数。 |
| `somehand.app` | 需要 source -> engine -> sink 编排时使用的 application session。 |
| `somehand.runtime` / `somehand.infrastructure` | 运行时 adapter 和实现细节；只有需要具体 adapter 时再用。 |

---

## 双手 L6 示例

```python
import numpy as np

from somehand.api import BiHandFrame, BiHandRetargetingEngine, HandFrame

engine = BiHandRetargetingEngine.from_config_path(
    "configs/retargeting/bihand/linkerhand_l6_bihand.yaml"
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

## 什么时候用 Session

如果你的程序已经有自己的循环，用 `BiHandRetargetingEngine.process()`。只有想复用 somehand 的 source -> engine -> sink 循环时，才用 `somehand.app.BiHandRetargetingSession`：

```python
from somehand.app import BiHandRetargetingSession
from somehand.api import BiHandRetargetingEngine

engine = BiHandRetargetingEngine.from_config_path("configs/retargeting/bihand/linkerhand_l6_bihand.yaml")
session = BiHandRetargetingSession(engine, sinks=[my_sink])
summary = session.run(my_source, input_type="custom")
```

`my_source` 和 `my_sink` 需要实现内置 runtime source/sink 使用的同类方法。
