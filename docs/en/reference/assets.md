# Assets and Models

[中文版](../../zh/reference/assets.md)

Runtime assets are downloaded separately. They are not committed to Git and are not included in release wheels.

## Asset Groups

| Group | Installed path | Needed for |
| --- | --- | --- |
| `mjcf` | `assets/mjcf/` | Every retargeting engine and viewer |
| `mediapipe` | `assets/models/hand_landmarker.task` | `webcam` and `video` inputs |
| `examples` | `assets/` and `recordings/` | Sample replay and reference files |

Download one or more groups:

```bash
somehand assets download --only mjcf mediapipe
somehand assets download --only examples
somehand assets download
```

The last command downloads all groups.

## Download Source

The defaults are:

- ModelScope: `BingqianWu/somehand-assets`
- HuggingFace: `12e21/somehand-assets`

ModelScope is used unless another source is selected:

```bash
somehand assets download \
    --source huggingface \
    --only mjcf examples
```

Use `--repo-id <owner/repo>` to override the repository or `--cache-dir <path>` to override the download cache.

## Local Data Root

The default depends on how somehand is installed:

| Install | Data root |
| --- | --- |
| Source checkout | Repository root |
| Linux wheel | `$XDG_DATA_HOME/somehand` or `~/.local/share/somehand` |
| macOS wheel | `~/Library/Application Support/somehand` |
| Windows wheel | `%LOCALAPPDATA%\somehand` |

Set one stable location before both downloading and running:

```bash
export SOMEHAND_HOME="$HOME/somehand-data"
somehand assets download --only mjcf mediapipe
somehand webcam
```

`--data-root <path>` changes one download destination. Runtime lookup does not automatically inherit that flag, so set `SOMEHAND_HOME` to the same path when running.

## Supported Config Families

Current checked-in configs cover:

- LinkerHand L6, L10, L20, L20 Pro, L21, L25, L30, LHG20, O6, O7, and T12
- DexRobot DexHand021 and Unitree Dex5
- Inspire DFQ and FTP
- AGIBOT OmniHand, BrainCo Revo2, OYMotion RoHand, Sharpa Wave 01, and Wuji Hand

Use the actual files under `configs/retargeting/{left,right,bihand}` as the source of truth. Side coverage is not perfectly symmetric: L20 Pro, L30, and T12 currently have right-hand configs only; LHG20 currently has a left-hand config only. Bi-hand configs exist only where both sides are checked in.

Config coverage means the model can be loaded for retargeting and visualization. The real backend currently targets supported LinkerHand hardware and has narrower coverage.

## Convert a URDF

From a source checkout:

```bash
PYTHONPATH=src python scripts/convert_urdf_to_mjcf.py \
    --urdf path/to/model.urdf \
    --output assets/mjcf/my_hand
```

Keep generated MJCF and mesh files in the external asset store, not Git. Then add the matching config under `configs/retargeting/` and verify it before listing the model as supported.
