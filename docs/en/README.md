# somehand Documentation

**somehand** maps human hand motion to configurable robot dexterous-hand models — from webcam input all the way to real hardware control.

---

## Supported Inputs

| Source | Description |
| --- | --- |
| **MediaPipe webcam** | Live hand tracking from a camera |
| **MediaPipe video** | Offline tracking from a video file |
| **PICO VR** | Live hand tracking via PICO Bridge |
| **hc_mocap UDP** | Live hand data over UDP |
| **Saved recordings** | Replay `.pkl` recordings from any source above |

## Supported Backends

| Backend | Description |
| --- | --- |
| **viewer** | MuJoCo visualization (single or bi-hand) |
| **sim** | MuJoCo simulation with physics |
| **real** | Real-hand hardware control (single-hand only) |

---

## Documentation Map

| Doc | What it covers |
| --- | --- |
| [Getting Started](getting-started.md) | Install, asset download, optional SDK setup, first run |
| [Runtime Modes](runtime-modes.md) | What each CLI mode does, when to use it, all options |
| [Configuration](configuration.md) | YAML config layout, schema, and which file to edit |
| [Assets & Models](assets-and-models.md) | Asset groups, external repos, 20+ supported hand models |
| [Troubleshooting](troubleshooting.md) | Common setup and runtime issues with fixes |
| [Maintainer Guide](maintainer-guide.md) | Update workflow, verification, maintenance rules |

---

## Scope

This repository is optimized for:

- **Single-hand retargeting** with configurable YAML models
- **Bi-hand visualization** and replay in `viewer` mode
- **Asset-light** source control — large runtime assets hosted externally

> Read [Runtime Modes](runtime-modes.md) before assuming a specific input + backend combination is supported.
