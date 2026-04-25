<p align="center">
  <img src="docs/images/somehand_logo.png" width="180" alt="somehand">
</p>

<h1 align="center">somehand</h1>

<p align="center">
  Universal dexterous-hand retargeting with MediaPipe, MuJoCo, and configurable YAML hand models.
  <br/>
  Turn human hand motion into robot-hand targets — visualize in MuJoCo, replay offline, or drive real hardware.
</p>

<p align="center">
  <a href="docs/en/README.md">English Docs</a> •
  <a href="docs/zh/README.md">中文文档</a> •
  <a href="docs/en/getting-started.md">Getting Started</a> •
  <a href="docs/en/runtime-modes.md">Runtime Modes</a> •
  <a href="docs/en/configuration.md">Configuration</a>
</p>

---

## Highlights

- **20+ hand models** — LinkerHand, Inspire, OmniHand, RoHand, Dex5, and more, all driven by YAML configs
- **Multiple input sources** — webcam (MediaPipe), video file, PICO VR (XRoboToolkit), hc_mocap UDP, saved recordings
- **Flexible backends** — MuJoCo viewer, MuJoCo sim, or real-hand hardware control
- **Bi-hand visualization** — side-by-side dual-hand replay and rendering
- **Asset-light** — large runtime assets hosted externally (ModelScope / HuggingFace), not in Git

---

## Supported Hand Models

| Company | Model | DoF | Joints |
| --- | --- | ---: | ---: |
| LinkerHand | L6 | 6 | 11 |
| LinkerHand | L10 | 10 | 20 |
| LinkerHand | L20 | 16 | 21 |
| LinkerHand | L20 Pro | 17 | 21 |
| LinkerHand | L21 | 17 | 17 |
| LinkerHand | L25 | 21 | 21 |
| LinkerHand | L30 | 20 | 20 |
| LinkerHand | LHG20 | 16 | 21 |
| LinkerHand | O6 | 6 | 11 |
| LinkerHand | O7 | 7 | 17 |
| LinkerHand | T12 | 14 | 19 |
| DexRobot | DexHand021 | 20 | 20 |
| Unitree | Dex5 | 20 | 20 |
| Inspire | DFQ | 6 | 12 |
| Inspire | FTP | 6 | 12 |
| AGIBOT | OmniHand | 10 | 16 |
| BrainCo | Revo2 | 6 | 11 |
| OYMotion | RoHand | 6 | 25 |
| Sharpa | Wave 01 | 22 | 22 |
| Wuji | Wuji Hand | 20 | 20 |

---

## Quick Start

**1. Install**

```bash
git submodule update --init --recursive
pip install -e .
```

**2. Download assets**

```bash
python scripts/setup/download_assets.py
```

**3. Run**

```bash
somehand replay --recording recordings/pico_right.pkl
```

You should see a MuJoCo viewer replaying the sample recording.

---

## More Examples

**Live webcam retargeting**

```bash
somehand webcam

# On macOS, use the following command
/opt/homebrew/anaconda3/envs/somehand/bin/mjpython \
/opt/homebrew/anaconda3/envs/somehand/bin/somehand webcam --hand both
```

**Replay in MuJoCo sim**

```bash
somehand replay \
    --backend sim \
    --hand right \
    --config configs/retargeting/right/linkerhand_o6_right.yaml \
    --recording recordings/pico_right.pkl
```

**Render a recording to MP4**

```bash
somehand dump-video \
    --recording recordings/webcam_hand.pkl \
    --output recordings/webcam_hand_replay.mp4
```

**Offline acceptance check**

```bash
python scripts/acceptance_check.py \
    --config configs/retargeting/right/linkerhand_l20_right.yaml \
    --video recordings/acceptance.mp4 \
    --hand right
```

---

## Documentation

Full docs covering installation, runtime modes, configuration, assets, and troubleshooting:

| Topic | English | 中文 |
| --- | --- | --- |
| Getting Started | [getting-started.md](docs/en/getting-started.md) | [快速开始](docs/zh/getting-started.md) |
| Runtime Modes | [runtime-modes.md](docs/en/runtime-modes.md) | [运行模式](docs/zh/runtime-modes.md) |
| Configuration | [configuration.md](docs/en/configuration.md) | [配置说明](docs/zh/configuration.md) |
| Assets & Models | [assets-and-models.md](docs/en/assets-and-models.md) | [资产与模型](docs/zh/assets-and-models.md) |
| Troubleshooting | [troubleshooting.md](docs/en/troubleshooting.md) | [常见问题](docs/zh/troubleshooting.md) |
| Maintainer Guide | [maintainer-guide.md](docs/en/maintainer-guide.md) | [维护指南](docs/zh/maintainer-guide.md) |

## License

[Apache 2.0](LICENSE)
