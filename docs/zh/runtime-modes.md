# CLI 用法

当你希望 somehand 接管输入循环、viewer、录制或硬件 backend 时，用 CLI。当这些部分由另一个 Python 程序接管时，看 [API 用法](api.md)。

| 用途 | 命令 | 备注 |
| --- | --- | --- |
| 实时摄像头 retargeting | `somehand webcam --camera 0` | MediaPipe 左右手判断反了时加 `--swap-hands`。 |
| 对已有视频 retargeting | `somehand video --video input.mp4` | 可使用与摄像头相同的 backend。 |
| 回放已保存的录制 | `somehand replay --recording recordings/webcam_hand.pkl` | 连续回放加 `--loop`。 |
| 把录制渲染成 MP4 | `somehand dump-video --recording input.pkl --output output.mp4` | 输入是录制文件，不是实时流。 |
| 接 PICO 实时手部追踪 | `somehand pico --hand right` | 需要 PICO Bridge 包和头显端 app。 |
| 接 hc_mocap UDP | `somehand hc-mocap --hand right --udp-port 1118` | joint 顺序不一致时才用 `--reference-bvh`。 |
| 控制真机 | `somehand webcam --backend real --hand right` | 当前仅支持单手；按硬件配置 transport 和 SDK 参数。 |

---

## 通用参数

| 参数 | 用途 |
| --- | --- |
| `--config <path>` | 选择 retargeting YAML 配置。 |
| `--hand left|right|both` | 选择手别；未显式传配置时，`both` 会切到默认双手配置。 |
| `--backend viewer|sim|real` | 选择 MuJoCo viewer、MuJoCo sim 或真机硬件。 |
| `--record-output <path>` | 把追踪输入保存成可回放的 `.pkl` 文件。 |
| `--control-rate`, `--sim-rate` | 调整控制/仿真频率。 |
| `--transport`, `--can-interface`, `--modbus-port`, `--sdk-root`, `--model-family` | 真机硬件参数。 |

---

## 输入专用参数

| 输入 | 常用参数 |
| --- | --- |
| `webcam` | `--camera`, `--swap-hands` |
| `video` | `--video`, `--swap-hands` |
| `pico` | `--signal-fps`, `--pico-host`, `--pico-port`, `--pico-advertise-ip`, `--no-pico-discovery`, `--pico-timeout` |
| `hc-mocap` | `--signal-fps`, `--reference-bvh`, `--udp-host`, `--udp-port`, `--udp-timeout`, `--udp-stats-every` |

---

## 限制

- `--hand both` 只支持 live/replay 命令配合 `--backend viewer`。
- `dump-video` 可以渲染双手录制，但它基于录制文件。
- `real` backend 当前仅支持单手。
- `pico` 会在进程内启动 PC receiver；不要在同一端口再启动另一个 receiver。
- LinkerHand 真机控制需要 LinkerHand SDK 和正确的 `model_family`。
