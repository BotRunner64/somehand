# dex-mujoco

基于 MediaPipe + Mink (MuJoCo 微分逆运动学) 的通用灵巧手 Retargeting 系统。

通过摄像头或视频捕捉人手姿态，使用向量 retargeting 策略将人手动作映射到任意机器人灵巧手模型上。

## 特性

- **通用手部模型支持**：通过 YAML 配置文件适配任意 MJCF/URDF 灵巧手模型
- **向量 Retargeting**：匹配手指间的相对方向向量，对人手与机器人手的尺寸差异天然鲁棒
- **实时性能**：基于 Mink 的 QP 微分逆运动学求解，支持 30+ FPS 实时运行
- **多输入源**：支持摄像头实时检测和视频文件离线处理
- **URDF 自动转换**：内置 URDF → MJCF 转换工具，自动添加 actuator 和指尖 site

## 核心算法

向量 retargeting 的核心思想：**方向取自人手，长度取自机器人**。

对于每对映射的关节向量：
1. 从 MediaPipe 关键点计算人手方向向量并归一化
2. 获取机器人对应关节间的当前距离
3. 目标位置 = 起点位置 + 人手方向 × 机器人距离
4. 通过 Mink FrameTask 驱动 IK 求解器达到目标

## 安装

```bash
pip install -e .
```

主要依赖：mujoco, mink, mediapipe, opencv-python, numpy, pyyaml, daqp

安装后主要入口是：

```bash
dex-retarget --help
```

## 快速开始

### 1. 转换手部模型（以 Linkerhand L20 为例）

```bash
python scripts/convert_urdf_to_mjcf.py \
    --urdf ../linkerhand-urdf/l20/right/linkerhand_l20_right.urdf \
    --output assets/mjcf/linkerhand_l20_right
```

### 2. 查看模型

```bash
python scripts/visualize_hand.py --mjcf assets/mjcf/linkerhand_l20_right/model.xml
```

### 3. 实时 Retargeting（摄像头）

```bash
dex-retarget webcam --visualize
```

`--visualize` 现在会打开两个独立的 `MuJoCo` viewer：一个看输入手势 / mocap landmarks，一个看 retarget 后的机器人手。为避免双 viewer 在同一进程下闪退，输入手势窗口会由单独子进程承载。

按 `q` 退出。

### 4. 自动录制验收视频（摄像头）

```bash
python scripts/record_webcam.py \
    --camera 0 \
    --duration 12 \
    --countdown 3 \
    --output recordings/acceptance.mp4
```

默认会先预热相机，再倒计时开始，录满固定时长后自动保存。录好之后可以直接跑离线验收：

```bash
python scripts/acceptance_check.py \
    --config configs/retargeting/linkerhand_l20.yaml \
    --video recordings/acceptance.mp4 \
    --hand Right
```

### 5. 离线 Retargeting（视频）

```bash
dex-retarget video \
    --video input.mp4 \
    --output trajectory.pickle \
    --visualize
```

### 5.1 hc_mocap 手部输入

如果你已经有 Teleopit 的 `hc_mocap` BVH 或 UDP 流，可以跳过 MediaPipe，直接把手骨架转成 21 点后喂给当前 retargeting：

```bash
dex-retarget hc-mocap bvh \
    --bvh assets/ref_with_toe.bvh \
    --hand Right \
    --visualize
```

实时 UDP 模式：

```bash
dex-retarget hc-mocap udp \
    --hand Right \
    --visualize \
    --udp-stats-every 120
```

你当前这类高频命令现在可以缩到：

```bash
dex-retarget hc-mocap udp \
    --hand Left \
    --visualize
```

UDP 模式不依赖 `Teleopit` Python 包；只要求你的 SDK 发送的每个 UDP 包都是一行 BVH motion floats，并且 joint 顺序与 `--reference-bvh` 一致。只有离线 `--bvh` 模式在未安装 `teleopit` 时才需要 `--teleopit-root /path/to/Teleopit`。
`hc_mocap` 输入会自动使用 wrist 真局部坐标做 retarget，因此即使配置文件里是 `wrist_local`，脚本也会切到更适合 `hc_mocap` 的处理方式。
如果要检查 UDP 是否正常进入，可以看终端的 `UDP stats` 输出，确认 `recv` / `valid` 是否持续增长，以及 `floats` 是否等于参考 BVH 的通道数。
输入手势窗口里显示的是参与 retarget 的输入 landmarks，因此看到的是已经对齐到机器人坐标系的手部骨架；机器人窗口单独显示 retarget 后的手模型。

### 5.2 PICO 4 手部输入

先确保 `xrobotoolkit_sdk` 可导入；如果没装，可以运行：

```bash
bash scripts/setup_xrobotoolkit.sh
```

然后先探测 PICO / XRoboToolkit 链路是否正常：

```bash
python scripts/probe_pico_xrobotoolkit.py --hand Right
```

探测正常后，直接运行实时 retargeting：

```bash
dex-retarget pico \
    --hand Right \
    --visualize
```

常见前提：

- 头显里已打开 Hand Tracking 权限
- 当前在 gesture / hand tracking 模式，而不是 controller mode
- XRoboToolkit / RobotLinuxDemo 在头显前台，并已连接 PC service

如果长时间收不到手数据，可以把等待时间调长：

```bash
dex-retarget pico \
    --hand Right \
    --pico-timeout 90 \
    --visualize
```

### 6. 跑验收脚本

先跑合成基线，确认当前配置的旋转不变性、镜像一致性、静态抖动、求解误差和吞吐量：

```bash
python scripts/acceptance_check.py --config configs/retargeting/linkerhand_l20.yaml
```

如果有真实视频，再追加离线验收：

```bash
python scripts/acceptance_check.py \
    --config configs/retargeting/linkerhand_l20.yaml \
    --video input.mp4 \
    --hand Right
```

## 适配新的机器人手

1. **转换模型**：用 `convert_urdf_to_mjcf.py` 将 URDF 转换为 MJCF
2. **编写配置**：在 `configs/retargeting/` 下创建 YAML 配置文件，定义 MediaPipe 关键点到机器人 body 的映射关系
3. **运行测试**：指定新配置运行 `dex-retarget webcam --visualize` 验证效果

配置文件格式参考 `configs/retargeting/linkerhand_l20.yaml`，核心字段：

```yaml
hand:
  name: "your_hand"
  mjcf_path: "path/to/model.xml"

retargeting:
  human_vector_pairs:    # MediaPipe 关键点索引对
    - [0, 5]             # wrist -> index_mcp
  origin_link_names:     # 机器人向量起点 body 名称
    - "base_link"
  task_link_names:       # 机器人向量终点 body 名称
    - "index_proximal"
  preprocess:
    frame: "wrist_local"   # 推荐：先对齐到手腕局部坐标系
    temporal_filter_alpha: 0.35
  solver:
    backend: "daqp"
    max_iterations: 20
```

## 项目结构

```
dex-mujoco/
├── configs/retargeting/    # Retargeting 配置文件
├── assets/mjcf/            # 转换后的 MJCF 模型（gitignored）
├── scripts/                # 可执行脚本
│   ├── acceptance_check.py
│   ├── convert_urdf_to_mjcf.py
│   ├── record_webcam.py
│   └── visualize_hand.py
└── src/dex_mujoco/         # 核心库
    ├── cli.py                  # 统一 dex-retarget CLI
    ├── constants.py            # MediaPipe 关键点定义
    ├── hand_detector.py        # MediaPipe 手部检测
    ├── hand_model.py           # MuJoCo 模型封装
    ├── ik_solver.py            # Mink IK 求解器
    ├── input_sources.py        # 输入源适配
    ├── retargeting_config.py   # 配置加载
    ├── runtime.py              # 通用 retarget 运行时
    ├── urdf_converter.py       # URDF → MJCF 转换
    ├── vector_retargeting.py   # 向量 retargeting 核心
    └── visualization.py        # MuJoCo 可视化
```

## 参考项目

- [dex-retargeting](https://github.com/dexsuite/dex-retargeting) — 手部 retargeting 算法
- [mink](https://github.com/kevinzakka/mink) — MuJoCo 微分逆运动学库
- [GMR](https://github.com/YanjieZe/GMR) — 全身运动 retargeting
