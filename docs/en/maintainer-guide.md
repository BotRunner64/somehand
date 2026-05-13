# Maintainer Guide

## Documentation Rules

- **README.md** is a landing page only — keep it concise
- Detailed docs go under `docs/`
- **Always update `docs/en` and `docs/zh` together** — mirrored filenames, equivalent meaning
- Never document planned features as if they already exist
- Verify commands and paths against current code before editing docs

---

## Update Workflow

1. Confirm current CLI help and default paths
2. Update English and Chinese docs together
3. Keep examples runnable against checked-in configs and scripts
4. Run verification before finishing

### Verification Commands

```bash
PYTHONPATH=src python -m somehand.cli --help
PYTHONPATH=src python -m somehand.cli webcam --help
PYTHONPATH=src python -m somehand.cli pico --help
PYTHONPATH=src python -m somehand.cli hc-mocap --help
python scripts/setup/download_assets.py --help
PYTHONPATH=src python scripts/convert_urdf_to_mjcf.py --help
PYTHONPATH=src python scripts/acceptance_check.py --help
pytest -q tests/test_cli.py tests/test_config_model.py tests/test_download_assets.py tests/test_paths.py
```

---

## Code Areas Relevant to Docs

| Area | Location |
| --- | --- |
| CLI parsing & dispatch | `src/somehand/cli/` |
| Default paths & asset metadata | `src/somehand/paths.py`, `src/somehand/external_assets.py` |
| Config loading & validation | `src/somehand/infrastructure/config_loader.py` |
| Runtime & backend glue | `src/somehand/runtime/`, `src/somehand/application/` |
| Integration scripts | `scripts/` |

---

## Adding or Updating a Hand Model

1. Prepare or convert the MJCF asset
2. Add or update the config in `configs/retargeting/`
3. Verify the config loads and the runtime path works
4. Update both language docs if public capability changed

---

## Third-Party Integrations

| Integration | Bootstrap script | Notes |
| --- | --- | --- |
| **PICO Bridge** | `pico-bridge @ https://github.com/BotRunner64/pico-bridge/releases/download/v0.2.0/pico_bridge-0.2.0-py3-none-any.whl` | Required for `pico` mode |
| **LinkerHand SDK** | `scripts/setup_linkerhand_sdk.sh` | Required for `real` backend |

> Upstream SDK docs remain upstream — this repo only documents the integration path used here.
