# CLI Usage

Use the CLI when you want somehand to own the input loop, viewer, recorder, or hardware backend. Use [API Usage](api.md) when another Python program owns those pieces.

| Task | Command | Notes |
| --- | --- | --- |
| Live camera retargeting | `somehand webcam --camera 0` | Add `--swap-hands` if MediaPipe reports the opposite side. |
| Retarget an existing video | `somehand video --video input.mp4` | Uses the same backends as live camera input. |
| Replay a saved recording | `somehand replay --recording recordings/webcam_hand.pkl` | Add `--loop` for continuous replay. |
| Render a recording to MP4 | `somehand dump-video --recording input.pkl --output output.mp4` | Renders from a recording, not from a live stream. |
| Receive live PICO hand tracking | `somehand pico --hand right` | Requires the PICO Bridge package and headset app. |
| Receive live hc_mocap UDP | `somehand hc-mocap --hand right --udp-port 1118` | Use `--reference-bvh` only when the joint order differs. |
| Drive a real hand | `somehand webcam --backend real --hand right` | Currently single-hand only; configure transport and SDK flags as needed. |

---

## Common Flags

| Flag | Use |
| --- | --- |
| `--config <path>` | Select a retargeting YAML config. |
| `--hand left|right|both` | Select hand side; `both` switches to the default bi-hand config if no config is passed. |
| `--backend viewer|sim|real` | Choose MuJoCo viewer, MuJoCo sim, or real hardware. |
| `--record-output <path>` | Save tracked input as a replayable `.pkl` file. |
| `--control-rate`, `--sim-rate` | Tune control/simulation rates. |
| `--transport`, `--can-interface`, `--modbus-port`, `--sdk-root`, `--model-family` | Real-hardware settings. |

---

## Input-Specific Flags

| Input | Useful flags |
| --- | --- |
| `webcam` | `--camera`, `--swap-hands` |
| `video` | `--video`, `--swap-hands` |
| `pico` | `--signal-fps`, `--pico-host`, `--pico-port`, `--pico-advertise-ip`, `--no-pico-discovery`, `--pico-timeout` |
| `hc-mocap` | `--signal-fps`, `--reference-bvh`, `--udp-host`, `--udp-port`, `--udp-timeout`, `--udp-stats-every` |

---

## Limits

- `--hand both` is supported with `--backend viewer` for live and replay commands.
- `dump-video` can render bi-hand recordings, but it is recording-based.
- `real` backend is currently single-hand only.
- `pico` starts the PC receiver in-process; do not run another receiver on the same port.
- LinkerHand real control requires the LinkerHand SDK and the correct `model_family`.
