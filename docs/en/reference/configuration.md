# Configuration

[中文版](../../zh/reference/configuration.md)

Retargeting configs select the robot model, hand side, constraints, and optional controller metadata.

## Config Tree

```text
configs/retargeting/
├── base/       # Shared constraints for each model
├── left/       # Left-hand model and asset bindings
├── right/      # Right-hand model and asset bindings
└── bihand/     # Pairs of left and right configs
```

The defaults are:

- Single hand: `right/linkerhand_l20_right.yaml`
- Two hands: `bihand/linkerhand_l20_bihand.yaml`

Select another checked-in config from the CLI:

```bash
somehand webcam \
    --hand right \
    --config right/omnihand_right.yaml
```

Or resolve the same portable name from Python:

```python
from somehand.api import resolve_config_path

config_path = resolve_config_path("right/omnihand_right.yaml")
```

## How Configs Compose

A side config normally extends a base model and supplies the side-specific MJCF:

```yaml
extends: "../base/linkerhand_l20.yaml"

hand:
  name: "linkerhand_l20_right"
  side: "right"
  mjcf_path: "../../../assets/mjcf/linkerhand_l20_right/model.xml"
```

A bi-hand config pairs two side configs:

```yaml
left:
  config: "../left/linkerhand_l20_left.yaml"

right:
  config: "../right/linkerhand_l20_right.yaml"
```

`extends` recursively merges mappings; an overriding list replaces the base list. Each `extends` target is relative to the file that declares it. `hand.mjcf_path` is relative to the selected side config, and bi-hand `left` / `right` paths are relative to the bi-hand config.

## Main Fields

| Section | Purpose |
| --- | --- |
| `hand` | `name`, `side`, `mjcf_path`, and optional `urdf_source` metadata. |
| `retargeting.constraint_defaults` | Default weights and activation settings for constraints. |
| `retargeting.vector_constraints` | Match directions between two human landmarks and two robot bodies/sites. |
| `retargeting.distance_constraints` | Match distances, normally between fingertips. |
| `retargeting.frame_constraints` | Align a frame built from three human and three robot points. |
| `retargeting.preprocess` | Input smoothing through `temporal_filter_alpha`. |
| `retargeting.solver` | Iteration, tolerance, activation smoothing, and output smoothing settings. |
| `controller` | Real-backend model family and default speed/torque metadata. CLI flags select the runtime backend and transport. |
| `viewer` | Bi-hand window size, model poses, and camera target. |

Robot names in constraints must exist as `body` or `site` names in the selected MJCF. Human indices use MediaPipe's 21-point hand-landmark order.

## Path Rules

- Source checkouts read the committed `configs/retargeting/` tree.
- Release wheels contain a copy of that tree. Treat bundled configs as read-only.
- In a bundled config, an `assets/...` MJCF reference resolves below `SOMEHAND_HOME` or the default data directory.
- In a custom config, ordinary relative paths resolve from that config file.
- Set `SOMEHAND_CONFIG_ROOT` before importing somehand only when replacing the entire built-in config root.

## Add or Customize a Model

1. Start from the closest config family under `base/`, `left/`, or `right/`.
2. Point `hand.mjcf_path` at the model and set the correct `hand.side`.
3. Use robot body/site names from that MJCF in each constraint.
4. Add a `bihand/` config only after both side configs exist.
5. Load the config and verify it in normal and diagnostic viewer modes.

```bash
somehand replay \
    --recording recordings/pico_right.pkl \
    --config /path/to/custom_right.yaml \
    --viewer-mode diagnostic
```
