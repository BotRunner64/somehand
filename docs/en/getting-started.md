# Getting Started

## Requirements

- **Python >=3.10**
- Git submodules initialized
- MuJoCo-compatible runtime environment
- External runtime assets (downloaded separately — not in Git)

---

## 1. Install

```bash
git submodule update --init --recursive
pip install -e .
```

Verify:

```bash
somehand --help
```

## 2. Download Runtime Assets

```bash
python scripts/setup/download_assets.py --only mjcf mediapipe
```

Other useful variants:

| Command | What it downloads |
| --- | --- |
| `python scripts/setup/download_assets.py` | Everything |
| `python scripts/setup/download_assets.py --only examples` | Sample recordings and reference assets |
| `python scripts/setup/download_assets.py --source huggingface --repo-id 12e21/somehand-assets` | From HuggingFace instead of ModelScope |

Default asset repositories:

- **ModelScope**: `BingqianWu/somehand-assets`
- **HuggingFace**: `12e21/somehand-assets`

## 3. (Optional) SDK Setup

Only needed for specific input/backend modes:

| Integration | Setup command | When needed |
| --- | --- | --- |
| **LinkerHand** real backend | `bash scripts/setup_linkerhand_sdk.sh` | Controlling real LinkerHand hardware |
| **PICO** / XRoboToolkit input | `bash scripts/setup_xrobotoolkit.sh` | Live PICO hand tracking |

---

## First Run

**Webcam input** — the simplest way to verify your setup:

```bash
somehand webcam
```

On macOS, run MuJoCo viewers through `mjpython`:

```bash
mjpython "$(command -v somehand)" webcam --hand both
```

**Replay a saved recording:**

```bash
somehand replay --recording recordings/webcam_hand.pkl
```

**Render a recording to video:**

```bash
somehand dump-video \
    --recording recordings/webcam_hand.pkl \
    --output recordings/webcam_hand_replay.mp4
```

---

## Next Steps

- Missing assets? → [Troubleshooting](troubleshooting.md)
- Need another hand model? → [Configuration](configuration.md)
- Need real hardware or PICO input? → [Runtime Modes](runtime-modes.md)
