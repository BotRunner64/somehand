# Assets and Models

Runtime assets are downloaded on demand. They are not stored in Git.

## Download What You Need

```bash
# Minimum runtime assets
somehand assets download --only mjcf mediapipe

# Sample recordings and reference assets
somehand assets download --only examples

# Everything
somehand assets download
```

| Group | Local path |
| --- | --- |
| `mjcf` | `assets/mjcf/` |
| `mediapipe` | `assets/models/hand_landmarker.task` |
| `examples` | `assets/` and `recordings/` |

Default source is ModelScope repo `BingqianWu/somehand-assets`. To use HuggingFace:

```bash
somehand assets download --source huggingface --repo-id 12e21/somehand-assets
```

Source checkouts store assets under the repository root by default. Wheel installs use the platform user-data directory (`$XDG_DATA_HOME/somehand` or `~/.local/share/somehand` on Linux). Set `SOMEHAND_HOME` for a stable override, or pass `--data-root` to one download command. Runtime commands and downloads should use the same `SOMEHAND_HOME`.

---

## Check Model Availability

Use `configs/retargeting/{left,right,bihand}` as the source of truth. Current config families include LinkerHand, Inspire, Dex5, DexHand021, OmniHand, Revo2, RoHand, Sharpa Wave, and Wuji Hand.

Coverage notes:

- `left/` and `right/` are not perfectly symmetric.
- `bihand/` contains only paired configs that have been checked in.
- Real-backend support is narrower than config-file coverage.

---

## Convert URDF to MJCF

```bash
PYTHONPATH=src python scripts/convert_urdf_to_mjcf.py --urdf path/to/model.urdf --output assets/mjcf/my_hand
```

After conversion, store the generated runtime asset externally, add or update the matching config under `configs/retargeting/`, then verify it before documenting it as supported.
