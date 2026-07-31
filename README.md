<p align="center">
  <img src="docs/images/supported_hand_models.png" width="270" alt="Supported dexterous hand models">
</p>

<h1 align="center">somehand</h1>

<p align="center">
  Universal dexterous-hand retargeting with MediaPipe, MuJoCo, and YAML-configured robot hands.
  <br/>
  Use live tracking or recorded motion to visualize, simulate, or control a supported hand.
</p>

<p align="center">
  <a href="docs/en/README.md">English documentation</a> •
  <a href="docs/zh/README.md">中文文档</a>
</p>

somehand provides a command-line tool for complete tracking workflows and a small Python API for embedding retargeting in another application. Inputs include webcam, video, PICO Bridge, hc_mocap UDP, and saved recordings. Robot models and retargeting constraints are selected with YAML configs; large runtime assets are downloaded separately.

## Supported Hand Models

| Company | Model | DoF | Joints |
| --- | --- | ---: | ---: |
| LinkerHand | L6 | 6 | 11 |
| LinkerHand | L10 | 10 | 20 |
| LinkerHand | L20 | 16 | 21 |
| LinkerHand | L20 Pro | 17 | 21 |
| LinkerHand | L21 | 17 | 17 |
| LinkerHand | L25 | 21 | 21 |
| LinkerHand | L30 | 20 | 20 |
| LinkerHand | LHG20 | 16 | 21 |
| LinkerHand | O6 | 6 | 11 |
| LinkerHand | O7 | 7 | 17 |
| LinkerHand | T12 | 14 | 19 |
| DexRobot | DexHand021 | 20 | 20 |
| Unitree | Dex5 | 20 | 20 |
| Inspire | DFQ | 6 | 12 |
| Inspire | FTP | 6 | 12 |
| AGIBOT | OmniHand | 10 | 16 |
| BrainCo | Revo2 | 6 | 11 |
| OYMotion | RoHand | 6 | 25 |
| Sharpa | Wave 01 | 22 | 22 |
| Wuji | Wuji Hand | 20 | 20 |

See `configs/retargeting/{left,right,bihand}` for current single-hand and bi-hand availability.

Start with [Installation](docs/en/getting-started/installation.md) or [安装](docs/zh/getting-started/installation.md), then choose the [CLI tutorial](docs/en/tutorials/cli.md) or [Python API tutorial](docs/en/tutorials/api.md).

## License

[Apache 2.0](LICENSE)
