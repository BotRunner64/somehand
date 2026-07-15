# Changelog

All notable changes to this project will be documented in this file.

## 0.3.0 - 2026-07-15

### Added

- Added `--viewer-mode diagnostic` for human-landmark and robot-hand overlays covering vector, distance, frame, and angle constraints.
- Added installable-wheel support with bundled retargeting configs and `somehand assets download` for external runtime assets.

### Changed

- Materialized hand-specific retargeting constraints in YAML while keeping shared weights and solver settings in `constraint_defaults`.
- Simplified vector constraints to direction loss and removed world-anchor and scaled keyvector residual objectives.
- Kept fingertip sites hidden in normal rendering and exposed them only in diagnostic mode.
- Stored downloaded assets under `SOMEHAND_HOME` or the platform user-data directory for wheel installs; source checkouts continue to use the repository root by default.

### Fixed

- Corrected finger-segment mappings across supported hand configs, including DexHand021 and OmniHand middle-finger vectors.
- Refined single-hand and bi-hand landmark overlays, camera framing, and target-direction transforms.
- Migrated the three short cloud sample recordings from legacy `dex_mujoco` format identifiers to `somehand` identifiers.

### Breaking changes

- Removed `retargeting.preset`; configs must define explicit vector, distance, frame, and angle constraints.
- Removed `retargeting.vector_loss` and per-vector `loss_type` / `loss_scale` fields.
- Retargeting output may change for existing hand models because constraint mappings and objective terms were corrected.

## 0.2.0 - 2026-06-09

- Replaced the PICO XRoboToolkit integration with the PICO Bridge receiver and added receiver host, port, advertised-IP, and discovery CLI options.
- Added `somehand.api` as the supported public import surface for embedding retargeting in Python applications.
- Moved MediaPipe, OpenCV, and PICO Bridge into optional CLI/dev dependencies so core library installs stay lighter.
- Split CLI and API documentation across English and Chinese docs, with README reduced to a landing page.
- Switched the LinkerHand SDK submodule to the BotRunner64 fork and removed the XRoboToolkit submodule and setup scripts.
- Added regression coverage for PICO Bridge input, optional dependency boundaries, lazy imports, and public API exports.

## 0.1.0 - 2026-05-04

Initial release of somehand.

- Universal dexterous-hand retargeting based on MediaPipe, MuJoCo, Mink, and YAML hand model configs.
- Support for webcam, video file, PICO VR, hc_mocap UDP, and saved-recording inputs.
- MuJoCo viewer, MuJoCo sim, real-hand control, replay, and video-dump workflows.
- Retargeting presets for left-hand, right-hand, and bi-hand setups across supported robot hand models.
- External asset download workflow for runtime models and large generated data.
