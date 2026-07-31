# CLI Tutorial

[中文版](../../zh/tutorials/cli.md)

Complete [Installation](../getting-started/installation.md) first. The CLI owns input capture, retargeting, visualization, recording, and the optional controller backend.

## 1. Choose an Input

| Input | Command |
| --- | --- |
| Webcam | `somehand webcam --camera 0` |
| Video file | `somehand video --video input.mp4` |
| Saved recording | `somehand replay --recording input.pkl` |
| PICO Bridge | `somehand pico --hand right` |
| hc_mocap UDP | `somehand hc-mocap --hand right --udp-port 1118` |

The default is the right LinkerHand L20 config with the viewer backend. For the shortest first run:

```bash
somehand webcam
```

Use `--swap-hands` with `webcam` or `video` if MediaPipe reports the opposite side. Add `--loop` to replay a recording continuously.

## 2. Choose the Hand Model

Pass a config relative to `configs/retargeting/`, or pass an ordinary filesystem path:

```bash
# Left LinkerHand L20
somehand webcam \
    --hand left \
    --config left/linkerhand_l20_left.yaml

# Right OmniHand
somehand webcam \
    --hand right \
    --config right/omnihand_right.yaml

# Default two-hand pair
somehand webcam --hand both
```

For two hands with another model, pass the matching `bihand/<model>_bihand.yaml` config. See [Configuration](../reference/configuration.md) for the available files and YAML layout.

## 3. Choose the Output

| Backend | Purpose | Example |
| --- | --- | --- |
| `viewer` | Show retargeted targets without controlling a device. This is the default. | `somehand webcam` |
| `sim` | Send targets to a MuJoCo simulation. | `somehand webcam --backend sim` |
| `real` | Send targets to supported LinkerHand hardware. | `somehand webcam --backend real --can-interface can0` |

Simulation and real hardware are single-hand only. Before using `real`, verify the same input and config with `viewer` or `sim`, prepare the LinkerHand SDK, and set the required transport flags:

```bash
somehand webcam \
    --hand right \
    --backend real \
    --transport can \
    --can-interface can0 \
    --sdk-root /path/to/linkerhand-python-sdk
```

Use `--transport modbus --modbus-port <port>` for a MODBUS device. `--model-family` overrides automatic LinkerHand model-family detection.

## 4. Record, Replay, and Export

Save detected frames while running a live input:

```bash
somehand webcam --record-output recordings/webcam_right.pkl
```

For PICO recording, press `r` in the terminal or viewer to start, then `s` to stop, save, and exit. Other live inputs record until the command ends.

Replay or render the result:

```bash
somehand replay \
    --recording recordings/webcam_right.pkl

somehand dump-video \
    --recording recordings/webcam_right.pkl \
    --output recordings/webcam_right.mp4
```

`dump-video` renders as fast as possible and does not read a live input.

## 5. Inspect Retargeting Constraints

Diagnostic mode draws the configured vector, distance, and frame constraints on the landmark and robot-hand views:

```bash
somehand replay \
    --recording recordings/webcam_right.pkl \
    --viewer-mode diagnostic
```

## Common Flags

| Flag | Meaning |
| --- | --- |
| `--hand left|right|both` | Select the tracked side. `both` selects the default bi-hand config unless `--config` is provided. |
| `--config <path>` | Select a retargeting YAML config. |
| `--backend viewer|sim|real` | Select visualization, simulation, or hardware output. |
| `--record-output <path>` | Save input landmarks to a replayable pickle. |
| `--viewer-mode normal|diagnostic` | Hide or show constraint overlays. |
| `--control-rate`, `--sim-rate` | Set controller and simulation rates. |
| `--signal-fps` | Resample live PICO or hc_mocap input to a fixed rate. |

Run `somehand <command> --help` for input-specific networking, camera, and hardware flags.

## Limits

- Two-hand mode supports the `viewer` backend and recording-based `dump-video`; controller backends are single-hand only.
- `pico` starts its PC receiver in the same process. Do not run another receiver on the same port.
- Use `--reference-bvh` with hc_mocap only when its joint order differs from the built-in order.
