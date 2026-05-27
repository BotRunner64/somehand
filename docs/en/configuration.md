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
| Use another checked-in hand | Pass `--config configs/retargeting/<side>/<model>_<side>.yaml` |
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

---

## Fields That Usually Matter

| Section | Use |
| --- | --- |
| `hand` | Model name, side, MJCF path, optional URDF source metadata. |
| `controller` | Backend defaults, rates, transport, SDK path, hardware model family. |
| `retargeting` | `preset: universal` for standard configs, or explicit constraints for custom models. |
| `viewer` | Bi-hand panel, camera, and pose settings. |

---

## Validation Notes

- `retargeting.preset` only accepts `universal` when set
- Legacy vector keys are rejected: `human_vector_pairs`, `origin_link_names`, `task_link_names`, `vector_weights`
- Removed sections are rejected: `position_constraints`, `pinch`
- Runtime validation checks backend names, transport names, and positive control/sim rates
