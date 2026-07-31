# Installation

[中文版](../../zh/getting-started/installation.md)

## Requirements

- Python 3.10 or newer
- A desktop environment that can run MuJoCo viewers
- A camera only if you plan to use the `webcam` command

Runtime MJCF files and tracking models are not included in Git or release wheels. Download them after installing the package.

## 1. Install somehand

Choose the command for your use case.

### CLI

This installs the built-in webcam, video, and PICO inputs:

```bash
pip install "somehand[cli] @ https://github.com/BotRunner64/somehand/releases/download/v0.3.0/somehand-0.3.0-py3-none-any.whl"
```

### Python API only

```bash
pip install "somehand @ https://github.com/BotRunner64/somehand/releases/download/v0.3.0/somehand-0.3.0-py3-none-any.whl"
```

### Source checkout

```bash
git clone --recurse-submodules https://github.com/BotRunner64/somehand.git
cd somehand
pip install -e ".[cli]"
```

Use `pip install -e ".[dev]"` instead when developing somehand.

## 2. Choose the Asset Location

A source checkout stores assets in the repository root. A wheel uses the platform user-data directory; on Linux this is normally `~/.local/share/somehand`.

To use one explicit location, set `SOMEHAND_HOME` before downloading assets or importing somehand:

```bash
export SOMEHAND_HOME="$HOME/somehand-data"
```

## 3. Download Assets

Download only what your workflow needs:

| Workflow | Command |
| --- | --- |
| Webcam or video | `somehand assets download --only mjcf mediapipe` |
| PICO, hc_mocap, or landmark API | `somehand assets download --only mjcf` |
| Replay the provided examples | `somehand assets download --only mjcf examples` |
| Everything | `somehand assets download` |

ModelScope is the default source. See [Assets and Models](../reference/assets.md) for HuggingFace and custom-location options.

## 4. Optional Hardware Setup

PICO Bridge support is already included in the CLI install, but the headset app must also be running.

The LinkerHand real backend needs its SDK. In a source checkout, prepare the bundled submodule with:

```bash
bash scripts/setup_linkerhand_sdk.sh
```

With a wheel install, install the SDK separately and pass its directory with `--sdk-root`.

## 5. Verify

Check the command is installed:

```bash
somehand --help
```

After downloading `mjcf` and `mediapipe`, run:

```bash
somehand webcam
```

The setup is working when the camera and MuJoCo windows open and the robot hand follows the detected right hand. Press `q` in the camera window to stop. If the detected side is reversed, rerun with `--swap-hands`.

On macOS, launch the viewer through `mjpython`:

```bash
mjpython "$(command -v somehand)" webcam
```

Continue with the [CLI tutorial](../tutorials/cli.md) or [Python API tutorial](../tutorials/api.md).
