# API Usage

Use the API when another Python program owns input capture, scheduling, visualization, or hardware control.

## Install

For embedding, install the core package from this repository. The CLI extras are only needed when you use built-in webcam/video/PICO commands.

```bash
pip install -e .
pip install huggingface_hub
```

If you download from ModelScope instead of HuggingFace, install the ModelScope client:

```bash
pip install modelscope
```

---

## Prepare What You Need

Use the config files from this repository, and download the matching model assets. This example uses bi-hand LinkerHand L6.

```bash
mkdir -p configs/retargeting/{base,left,right,bihand} assets/mjcf
cp -a /path/to/somehand/configs/retargeting/base/_universal_common.yaml configs/retargeting/base/
cp -a /path/to/somehand/configs/retargeting/base/linkerhand_l6.yaml configs/retargeting/base/
cp -a /path/to/somehand/configs/retargeting/left/linkerhand_l6_left.yaml configs/retargeting/left/
cp -a /path/to/somehand/configs/retargeting/right/linkerhand_l6_right.yaml configs/retargeting/right/
cp -a /path/to/somehand/configs/retargeting/bihand/linkerhand_l6_bihand.yaml configs/retargeting/bihand/
```

Download the matching model assets from HuggingFace:

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

From ModelScope, use the same file list with `modelscope.snapshot_download` and repo `BingqianWu/somehand-assets`.

The application directory should then include:

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

## Stable Imports

Prefer `somehand.api` for embedding:

```python
from somehand.api import BiHandFrame, BiHandRetargetingEngine, HandFrame
```

Related namespaces:

| Namespace | Use |
| --- | --- |
| `somehand.api` | Stable embedding surface: configs, frames, one-step engines. |
| `somehand.core` | Pure domain models and transformations. |
| `somehand.app` | Application sessions when you want source -> engine -> sink orchestration. |
| `somehand.runtime` / `somehand.infrastructure` | Runtime adapters and implementation details; use only when you need those concrete adapters. |

---

## Bi-Hand L6 Example

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

`landmarks_3d` must be a 21x3 hand-landmark array. If one side is missing, pass `None`; the engine keeps the last result for that side.

---

## When to Use Sessions

Use `BiHandRetargetingEngine.process()` when your program already owns the loop. Use `somehand.app.BiHandRetargetingSession` only when you want somehand's source -> engine -> sink loop:

```python
from somehand.app import BiHandRetargetingSession
from somehand.api import BiHandRetargetingEngine

engine = BiHandRetargetingEngine.from_config_path("configs/retargeting/bihand/linkerhand_l6_bihand.yaml")
session = BiHandRetargetingSession(engine, sinks=[my_sink])
summary = session.run(my_source, input_type="custom")
```

`my_source` and `my_sink` must implement the same methods used by the built-in runtime sources and sinks.
