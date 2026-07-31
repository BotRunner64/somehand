# CLI 教程

[English](../../en/tutorials/cli.md)

请先完成[安装](../getting-started/installation.md)。CLI 负责输入采集、重定向、可视化、录制和可选的控制 backend。

## 1. 选择输入

| 输入 | 命令 |
| --- | --- |
| 摄像头 | `somehand webcam --camera 0` |
| 视频文件 | `somehand video --video input.mp4` |
| 已保存录制 | `somehand replay --recording input.pkl` |
| PICO Bridge | `somehand pico --hand right` |
| hc_mocap UDP | `somehand hc-mocap --hand right --udp-port 1118` |

默认使用右手 LinkerHand L20 配置和 viewer backend。最短的首次运行命令是：

```bash
somehand webcam
```

如果 MediaPipe 识别的左右手相反，在 `webcam` 或 `video` 后添加 `--swap-hands`。回放时添加 `--loop` 可循环播放。

## 2. 选择手模型

可以传相对于 `configs/retargeting/` 的配置，也可以传普通文件路径：

```bash
# 左手 LinkerHand L20
somehand webcam \
    --hand left \
    --config left/linkerhand_l20_left.yaml

# 右手 OmniHand
somehand webcam \
    --hand right \
    --config right/omnihand_right.yaml

# 默认双手组合
somehand webcam --hand both
```

双手使用其他模型时，传入对应的 `bihand/<model>_bihand.yaml` 配置。可用文件和 YAML 结构见[配置](../reference/configuration.md)。

## 3. 选择输出

| Backend | 用途 | 示例 |
| --- | --- | --- |
| `viewer` | 只显示重定向目标，不控制设备；这是默认值。 | `somehand webcam` |
| `sim` | 把目标发送到 MuJoCo 仿真。 | `somehand webcam --backend sim` |
| `real` | 把目标发送到支持的 LinkerHand 真机。 | `somehand webcam --backend real --can-interface can0` |

仿真和真机只支持单手。使用 `real` 前，应先用 `viewer` 或 `sim` 验证相同输入和配置，准备好 LinkerHand SDK，再填写所需 transport 参数：

```bash
somehand webcam \
    --hand right \
    --backend real \
    --transport can \
    --can-interface can0 \
    --sdk-root /path/to/linkerhand-python-sdk
```

MODBUS 设备使用 `--transport modbus --modbus-port <port>`。`--model-family` 可覆盖自动识别的 LinkerHand 型号族。

## 4. 录制、回放和导出

运行实时输入时保存检测到的 frame：

```bash
somehand webcam --record-output recordings/webcam_right.pkl
```

PICO 录制需要在终端或 viewer 中按 `r` 开始，再按 `s` 停止、保存并退出。其他实时输入会一直录制到命令结束。

回放或渲染录制结果：

```bash
somehand replay \
    --recording recordings/webcam_right.pkl

somehand dump-video \
    --recording recordings/webcam_right.pkl \
    --output recordings/webcam_right.mp4
```

`dump-video` 会尽快完成离线渲染，不读取实时输入。

## 5. 检查重定向约束

诊断模式会在 landmark 和机器人手视图中绘制配置的 vector、distance 和 frame 约束：

```bash
somehand replay \
    --recording recordings/webcam_right.pkl \
    --viewer-mode diagnostic
```

## 常用参数

| 参数 | 含义 |
| --- | --- |
| `--hand left|right|both` | 选择追踪手别。未传 `--config` 时，`both` 会选择默认双手配置。 |
| `--config <path>` | 选择 retargeting YAML 配置。 |
| `--backend viewer|sim|real` | 选择可视化、仿真或硬件输出。 |
| `--record-output <path>` | 把输入 landmark 保存为可回放的 pickle。 |
| `--viewer-mode normal|diagnostic` | 隐藏或显示约束叠层。 |
| `--control-rate`, `--sim-rate` | 设置控制和仿真频率。 |
| `--signal-fps` | 把实时 PICO 或 hc_mocap 输入重采样到固定频率。 |

输入专用的网络、摄像头和硬件参数以 `somehand <command> --help` 为准。

## 限制

- 双手模式支持 `viewer` backend 和基于录制的 `dump-video`；控制 backend 只支持单手。
- `pico` 会在同一进程启动 PC receiver，不要在相同端口再运行另一个 receiver。
- 只有 hc_mocap 的关节顺序与内置顺序不同时才需要 `--reference-bvh`。
