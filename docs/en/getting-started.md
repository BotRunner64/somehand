# Getting Started

## Requirements

- **Python >=3.10**
- MuJoCo-compatible runtime environment
- External runtime assets (downloaded separately — not in Git)

---

## 1. Install

```bash
pip install "somehand[cli] @ https://github.com/BotRunner64/somehand/releases/download/v0.3.0/somehand-0.3.0-py3-none-any.whl"
```

For an editable source checkout instead:

```bash
git clone --recurse-submodules https://github.com/BotRunner64/somehand.git
cd somehand
pip install -e ".[cli]"
```

Verify:

```bash
somehand --help
```

## 2. Download Runtime Assets

```bash
somehand assets download --only mjcf mediapipe
```

Other useful variants:

| Command | What it downloads |
| --- | --- |
| `somehand assets download` | Everything |
| `somehand assets download --only examples` | Sample recordings and reference assets |
| `somehand assets download --source huggingface --repo-id 12e21/somehand-assets` | From HuggingFace instead of ModelScope |

Default asset repositories:

- **ModelScope**: `BingqianWu/somehand-assets`
- **HuggingFace**: `12e21/somehand-assets`

In a source checkout, assets default to the repository root. A wheel install uses the platform user-data directory. Set `SOMEHAND_HOME` to keep a stable, explicit location:

```bash
export SOMEHAND_HOME="$HOME/somehand-data"
somehand assets download --only mjcf mediapipe examples
```

## 3. (Optional) SDK Setup

Only needed for specific input/backend modes:

| Integration | Setup command | When needed |
| --- | --- | --- |
| **LinkerHand** real backend | Source checkout: `bash scripts/setup_linkerhand_sdk.sh`; wheel: pass a separately installed SDK with `--sdk-root` | Controlling real LinkerHand hardware |
| **PICO Bridge** input | Installed with the `somehand[cli]` extra | Live PICO hand tracking |

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
EXAMPLE_ROOT="${SOMEHAND_HOME:-$HOME/somehand-data}"
somehand assets download --only examples --data-root "$EXAMPLE_ROOT"
somehand replay --recording "$EXAMPLE_ROOT/recordings/pico_right.pkl"
```

**Render a recording to video:**

```bash
somehand dump-video \
    --recording "$EXAMPLE_ROOT/recordings/pico_right.pkl" \
    --output recordings/pico_right_replay.mp4
```

---

## Next Steps

- Need assets or models? → [Assets & Models](assets-and-models.md)
- Need another hand model? → [Configuration](configuration.md)
- Need terminal commands? → [CLI Usage](runtime-modes.md)
- Embedding in Python? → [API Usage](api.md)
