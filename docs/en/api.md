# API Usage

Use the API when another Python program owns input capture, scheduling, visualization, or hardware control.

## Install

For embedding, install the core release wheel. CLI extras are only needed for built-in webcam/video/PICO commands.

```bash
pip install "somehand @ https://github.com/BotRunner64/somehand/releases/download/v0.3.0/somehand-0.3.0-py3-none-any.whl"
```

For an editable source checkout instead:

```bash
pip install -e .
```

---

## Prepare What You Need

Release wheels already contain the checked-in retargeting configs. Download the matching external MJCF assets:

```bash
somehand assets download --only mjcf
```

Use the bundled default paths from the supported API surface:

```python
from somehand.api import DEFAULT_BIHAND_CONFIG_PATH, DEFAULT_CONFIG_PATH, resolve_config_path

print(DEFAULT_CONFIG_PATH)
print(DEFAULT_BIHAND_CONFIG_PATH)
print(resolve_config_path("right/omnihand_right.yaml"))
```

Set `SOMEHAND_HOME` before importing somehand when assets live outside the default user-data directory. Custom configs can still be passed as ordinary filesystem paths.

---

## Stable Imports

Use `somehand.api` for embedding. It is the supported external API surface.

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

## Default Bi-Hand Example

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

`landmarks_3d` must be a 21x3 hand-landmark array. If one side is missing, pass `None`; the engine keeps the last result for that side.

---

## One-Step Use

Use `RetargetingEngine.process()` or `BiHandRetargetingEngine.process()` when your program already owns the loop:

```python
engine = BiHandRetargetingEngine.from_config_path(str(DEFAULT_BIHAND_CONFIG_PATH))
result = engine.process(BiHandFrame(left=left_frame, right=right_frame))
```
