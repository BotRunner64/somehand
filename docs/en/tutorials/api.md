# Python API Tutorial

[中文版](../../zh/tutorials/api.md)

Complete the API path in [Installation](../getting-started/installation.md) and download the `mjcf` asset group first:

```bash
somehand assets download --only mjcf
```

Use `somehand.api` when your program already provides hand landmarks and owns the loop, visualization, or hardware output. The API does not start a camera or controller for you.

## Single-Hand Retargeting

Create the engine once and reuse it for every frame:

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

The 21 rows must use MediaPipe hand-landmark order. `result.qpos` follows the selected MJCF model's joint order. The frame side must match the config side.

To use another checked-in config:

```python
from somehand.api import RetargetingEngine, resolve_config_path

config_path = resolve_config_path("left/omnihand_left.yaml")
left_engine = RetargetingEngine.from_config_path(str(config_path))
```

An absolute or existing relative path can be used for a custom config.

## Two-Hand Retargeting

Use a bi-hand config to pair one left and one right engine:

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

Pass `None` for a side that is not detected. The returned `result.left.qpos` and `result.right.qpos` contain the two targets; the flags report which sides were present:

```python
result = retarget_both(left_landmarks, None)
print(result.left_detected, result.right_detected)
```

Before a side's first detection, its result is the model's neutral pose. Afterward, a missing side keeps its previous result.

## Supported Public Surface

Import embedding APIs from `somehand.api`:

| Need | Names |
| --- | --- |
| Single hand | `HandFrame`, `RetargetingEngine`, `RetargetingStepResult` |
| Two hands | `BiHandFrame`, `BiHandRetargetingEngine`, `BiHandRetargetingResult` |
| Config loading | `load_retargeting_config`, `load_bihand_config` |
| Built-in paths | `DEFAULT_CONFIG_PATH`, `DEFAULT_BIHAND_CONFIG_PATH`, `resolve_config_path` |

Set `SOMEHAND_HOME` before importing somehand if the external assets are outside the default data directory. Config and asset path rules are described in [Configuration](../reference/configuration.md) and [Assets and Models](../reference/assets.md).
