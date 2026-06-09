# Changelog

All notable changes to this project will be documented in this file.

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
