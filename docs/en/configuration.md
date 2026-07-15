# Configuration

Use this page when you need a different hand model, asset path, or hardware default.

## Pick a Config

```
configs/retargeting/
├── base/       # shared model constraints
├── left/       # left-hand runtime configs
├── right/      # right-hand runtime configs
└── bihand/     # bi-hand viewer/replay configs
```

| Mode | Default config |
| --- | --- |
| Single hand | `configs/retargeting/right/linkerhand_l20_right.yaml` |
| Bi-hand | `configs/retargeting/bihand/linkerhand_l20_bihand.yaml` |

---

## What to Edit

| Goal | Edit |
| --- | --- |
| Use another checked-in hand | Pass the source/wheel-portable `--config <side>/<model>_<side>.yaml` |
| Change MJCF path or hand-side binding | `left/` or `right/` config |
| Change shared retargeting constraints | `base/` config |
| Change bi-hand viewer/replay pairing | `bihand/` config |
| Change hardware defaults | `controller` section in the runtime config |

---

## Runtime Shape

Single-hand configs usually extend a base config and bind it to one side:

```yaml
extends: "../base/linkerhand_l20.yaml"

hand:
  name: "linkerhand_l20_right"
  side: "right"
  mjcf_path: "../../../assets/mjcf/linkerhand_l20_right/model.xml"
```

Bi-hand configs compose one left config and one right config:

```yaml
left:
  config: "../left/linkerhand_l20_left.yaml"

right:
  config: "../right/linkerhand_l20_right.yaml"
```

Relative paths resolve from the YAML file location. `extends` can be chained.

Release wheels bundle the checked-in configs and use them for CLI defaults. Treat bundled files as read-only: copy the matching config family to your project before customizing it. Bundled `assets/...` references resolve under `SOMEHAND_HOME`; custom config paths retain normal filesystem-relative behavior.

For example, `--config right/omnihand_right.yaml` resolves against the checked-in tree in a source checkout and the bundled tree in a wheel install.

---

## Fields That Usually Matter

| Section | Use |
| --- | --- |
| `hand` | Model name, side, MJCF path, optional URDF source metadata. |
| `controller` | Backend defaults, rates, transport, SDK path, hardware model family. |
| `retargeting` | Common solver/preprocess settings plus hand-specific explicit constraints. |
| `viewer` | Bi-hand panel, camera, and pose settings. |

---

## Validation Notes

- `retargeting.preset` is rejected; vector, distance, frame, and angle constraints belong in the hand config
- `retargeting.vector_loss` and per-vector `loss_type` / `loss_scale` are rejected
- Legacy vector keys are rejected: `human_vector_pairs`, `origin_link_names`, `task_link_names`, `vector_weights`
- Removed sections are rejected: `position_constraints`, `pinch`
- Runtime validation checks backend names, transport names, and positive control/sim rates

---

## Upgrade From 0.2

The simplest migration is to start from the matching checked-in 0.3 config and reapply only model paths or controller overrides. A 0.2 preset such as:

```yaml
retargeting:
  preset: universal
  vector_loss:
    type: direction
```

must become explicit hand-specific constraints:

```yaml
extends: "./_universal_common.yaml"

retargeting:
  vector_constraints:
    - human: [1, 2]
      robot: ["thumb_metacarpals", "thumb_proximal"]
  distance_constraints:
    - human: [4, 8]
      robot: ["thumb_tip", "index_tip"]
      robot_types: ["site", "site"]
```

Use the actual names and full constraint set from the matching file under `configs/retargeting/base/`; the abbreviated example only shows the schema change.
