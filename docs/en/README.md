# somehand Documentation

[中文文档](../zh/README.md)

somehand converts 21-point human-hand landmarks into joint targets for configurable robot hands. Use the CLI for a complete input-to-output workflow, or use the Python API when your application already owns the input loop.

New users should complete [Installation](getting-started/installation.md), then follow either the CLI or API tutorial.

## Installation

- [Installation](getting-started/installation.md) — install the package, download the required assets, and verify one run.

## Tutorials

- [CLI](tutorials/cli.md) — webcam, video, recordings, PICO, hc_mocap, simulation, and real hardware.
- [Python API](tutorials/api.md) — retarget one or two landmark streams from Python.

## Reference

- [Configuration](reference/configuration.md) — choose a hand model and edit retargeting YAML.
- [Assets and Models](reference/assets.md) — asset groups, download locations, and model coverage.
- [Code Structure](reference/code-structure.md) — runtime data flow and repository layout.

## Current Scope

The CLI supports one or two hands with the viewer backend. Simulation and real-hardware backends are single-hand only. The stable embedding surface is `somehand.api`.
