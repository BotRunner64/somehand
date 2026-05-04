# 运行模式

## CLI 总览

| 命令 | 用途 | 输入 | 输出 |
| --- | --- | --- | --- |
| **`webcam`** | 从摄像头实时手部追踪 | 摄像头设备 | viewer / sim / real |
| **`video`** | 对视频文件离线追踪 | MP4 等 | viewer / sim / real |
| **`replay`** | 回放已保存的录制 | `.pkl` 文件 | viewer / sim / real |
| **`dump-video`** | 把录制渲染成 MP4 | `.pkl` 文件 | MP4 文件 |
| **`pico`** | 通过 PICO Bridge 实时追踪 | PICO Bridge 数据流 | viewer / sim / real |
| **`hc-mocap`** | 从 hc_mocap UDP 接数据 | UDP 数据包 | viewer / sim / real |

---

## 通用参数

多数命令共享：

| 参数 | 说明 |
| --- | --- |
| `--config` | retargeting 配置 YAML 路径 |
| `--hand {left,right,both}` | 选择手别 |
| `--backend {viewer,sim,real}` | 输出 backend |
| `--record-output` | 保存当前输入为可回放的 `.pkl` 录制 |

实时命令的附加控制参数：

`--control-rate` · `--sim-rate` · `--transport` · `--can-interface` · `--modbus-port` · `--sdk-root` · `--model-family`

---

## 各模式详细说明

### `webcam`

从摄像头实时做人手 retargeting。

```bash
somehand webcam --camera 0
```

| 参数 | 说明 |
| --- | --- |
| `--swap-hands` | 当 MediaPipe 左右手判断反了时使用 |
| `--record-output <path>` | 把当前输入录成回放文件 |

### `video`

对已有操作视频做离线 retargeting。

```bash
somehand video --video input.mp4
```

视频镜像或左右手判断错误时，可加 `--swap-hands`。

### `replay`

按真实时间回放已保存的手部录制。

```bash
somehand replay --recording recordings/webcam_hand.pkl
```

使用 `--loop` 循环回放。

### `dump-video`

把录制尽快渲染成 MP4（非实时）。

```bash
somehand dump-video \
    --recording recordings/webcam_hand.pkl \
    --output recordings/webcam_hand_replay.mp4
```

### `pico`

通过 PICO Bridge 接 PICO 实时手部追踪。

```bash
somehand pico --hand right
```

| 参数 | 说明 |
| --- | --- |
| `--signal-fps` | 对实时输入做固定频率重采样 |
| `--pico-host` / `--pico-port` | PICO Bridge PC receiver 监听地址 |
| `--pico-advertise-ip` | 向头显广播的 PC IPv4 地址 |
| `--no-pico-discovery` | 关闭 PICO Bridge UDP discovery |
| `--pico-timeout` | 控制启动与帧等待时间 |

> 需要安装 PICO Bridge PC receiver 包，并在头显端启动 PICO Bridge app。
> `somehand pico` 会在进程内启动 PC receiver；不要在同一端口同时运行 standalone `pico-bridge-receiver`。

### `hc-mocap`

从 `hc_mocap` UDP 接实时手部数据。

```bash
somehand hc-mocap --hand right --udp-port 1118
```

| 参数 | 说明 |
| --- | --- |
| `--signal-fps` | 对实时输入做固定频率重采样 |
| `--reference-bvh` | 覆盖内置 joint 顺序 |
| `--udp-host` / `--udp-port` | 网络设置 |
| `--udp-timeout` / `--udp-stats-every` | 连接调优 |

---

## 重要限制

- **双手模式（`--hand both`）** 仅在 `--backend viewer` 下支持
- **`dump-video`** 支持双手渲染，但基于录制文件，不是实时双手控制
- **真机 backend** 当前仅支持单手
- **`pico`** 依赖 PICO Bridge receiver 和头显端 app
- **LinkerHand 真机控制** 依赖 LinkerHand SDK 与正确的 `model_family` 映射
