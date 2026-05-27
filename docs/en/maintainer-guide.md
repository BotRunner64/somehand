# Maintainer Guide

## Rules

- Keep `README.md` short: project summary, quick run, docs links.
- Keep user docs task-first. Put reference details after the task path.
- Update `docs/en` and `docs/zh` together with mirrored filenames and equivalent meaning.
- Document only current, verified behavior.

## Before Editing Docs

Check the current CLI and paths:

```bash
PYTHONPATH=src python -m somehand.cli --help
PYTHONPATH=src python -m somehand.cli webcam --help
PYTHONPATH=src python -m somehand.cli pico --help
PYTHONPATH=src python -m somehand.cli hc-mocap --help
python - <<'PY'
from somehand.api import RetargetingEngine
print(RetargetingEngine)
PY
python scripts/setup/download_assets.py --help
PYTHONPATH=src python scripts/convert_urdf_to_mjcf.py --help
```

Useful code locations:

| Area | Location |
| --- | --- |
| CLI parsing | `src/somehand/cli/` |
| Default paths and asset metadata | `src/somehand/paths.py`, `src/somehand/external_assets.py` |
| Config loading | `src/somehand/infrastructure/config_loader.py` |
| Runtime glue | `src/somehand/runtime/`, `src/somehand/application/` |

## When Adding a Public Hand Model

1. Add or update the runtime asset outside Git.
2. Add or update configs under `configs/retargeting/`.
3. Verify the config loads and the intended runtime path works.
4. Update both language docs only after verification.

Run focused doc-related tests when relevant:

```bash
pytest -q tests/test_docs_structure.py tests/test_config_model.py tests/test_download_assets.py tests/test_paths.py
```
