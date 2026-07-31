# Code Structure

[中文版](../../zh/reference/code-structure.md)

## Runtime Data Flow

```text
webcam / video / PICO / hc_mocap / recording
                       │
                       ▼
              HandFrame or BiHandFrame
                       │
                       ▼
       RetargetingEngine or BiHandRetargetingEngine
                       │
                       ▼
    RetargetingStepResult or BiHandRetargetingResult
                       │
                       ▼
          viewer / recorder / sim / real hand
```

Input adapters normalize data into domain frames. The application engine preprocesses landmarks, updates solver targets, and returns robot joint positions. Runtime sinks then display, save, simulate, or send those results to hardware.

See [Retargeting Algorithm](retargeting.md) for the objective and per-frame solve.

## Python Package

Core code lives under `src/somehand/`:

| Path | Responsibility |
| --- | --- |
| `domain/` | Data models, config dataclasses, hand-side rules, and pure landmark preprocessing. No device I/O. |
| `application/` | Single-hand and bi-hand engines plus session orchestration. |
| `infrastructure/` | YAML loading, the MuJoCo hand model, retargeting solver, input/output adapters, artifacts, and hardware controllers. |
| `runtime/` | Runtime validation, source transforms/sampling, viewer behavior, and public runtime adapter exports. |
| `cli/` | Argument parsing, command dispatch, and assembly of sources, engines, sinks, and controllers. |
| `api.py` | Stable imports for applications embedding somehand. |
| `paths.py`, `external_assets.py` | Config roots, data roots, asset manifests, and path resolution. |

New features should keep these boundaries: domain types stay independent of I/O; adapters belong in infrastructure; use-case coordination belongs in application; CLI-only behavior stays in `cli/` or `runtime/`.

## Repository Layout

| Path | Contents |
| --- | --- |
| `configs/retargeting/` | The checked-in source of truth for model configs. |
| `tests/` | Pytest coverage, usually named after the feature under test. |
| `scripts/` | Setup, conversion, acceptance, recording, and rendering utilities. |
| `docs/en/`, `docs/zh/` | Mirrored English and Chinese documentation. |
| `third_party/` | Vendor SDKs and Git submodules. |
| `assets/`, `recordings/` | Downloaded or generated local data; large files are not committed. |

Release builds copy `configs/retargeting/` into the wheel. They do not copy runtime assets or recordings.

## Where to Make a Change

| Change | Start here |
| --- | --- |
| Add a domain type or validation rule | `src/somehand/domain/` |
| Change one-step retargeting behavior | `src/somehand/application/engine.py` or `bihand_engine.py` |
| Add an input, viewer, controller, or file adapter | `src/somehand/infrastructure/` and its runtime export |
| Add a CLI command or flag | `src/somehand/cli/parser.py`, then `commands.py` |
| Add a hand model | `configs/retargeting/` plus external MJCF assets |
| Change stable embedding imports | `src/somehand/api.py` |

## Verification

Run the focused tests for the changed area, then the full suite:

```bash
pytest -q
ruff check src tests
```

For config, asset, path, or documentation changes, the useful focused set is:

```bash
pytest -q \
    tests/test_docs_structure.py \
    tests/test_config_model.py \
    tests/test_download_assets.py \
    tests/test_paths.py
```

Documentation changes must update `docs/en` and `docs/zh` together with mirrored filenames and equivalent meaning.
