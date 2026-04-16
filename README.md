# somehand

基于 MediaPipe + MuJoCo 的通用灵巧手 Retargeting 系统。

通过摄像头或视频捕捉人手姿态，使用向量 retargeting 策略将人手动作映射到任意机器人灵巧手模型上。

## 特性

- **通用手部模型支持**：通过 YAML 配置文件适配任意 MJCF/URDF 灵巧手模型
- **向量 Retargeting**：匹配手指间的相对方向向量，对人手与机器人手的尺寸差异天然鲁棒
- **可扩展架构**：核心库已按 `domain / application / infrastructure / interfaces` 分层重构
- **实时性能**：基于 MuJoCo Jacobian + `scipy` SLSQP 优化，支持实时运行
- **多输入源**：支持摄像头实时检测和视频文件离线处理
- **URDF 自动转换**：内置 URDF → MJCF 转换工具，自动添加 actuator 和指尖 site

## 已支持的 Dex Hand

下表按当前仓库中的 retargeting 配置整理，新增机型时建议同步更新。

| 公司 | 型号 | 自由度 | 关节数量 |
| --- | --- | ---: | ---: |
| DexRobot | DexHand021 | 20 | 20 |
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
| BrainCo | Revo2 | 6 | 11 |
| Wuji | Wuji Hand | 20 | 20 |

说明：

- 这张表描述的是 **retargeting 配置支持**，不等同于所有机型都具备真机 `real` backend 支持。
- 自由度按 `URDF` 中“非固定、非 `mimic`”关节统计；关节数量按 `URDF` 中非固定关节统计。
- 云端默认分发的是运行所需的 `MJCF` 资产，不是 `URDF` 源文件。
- 本仓库不提交 `assets/` 实际内容；本地仅保留占位目录，所有资产统一放在云端资产仓。

当前仓库只保留一套 `universal` retargeting 约束实现。它使用核心向量约束，并保留拇指必要的 frame / residual / pinch distance 约束。可直接运行的示例配置：

```bash
somehand replay --backend sim --hand right --config configs/retargeting/right/linkerhand_o6_right.yaml --recording recordings/pico_right.pkl
```

## 核心算法

向量 retargeting 的核心思想：**方向取自人手，长度取自机器人**。

对于每对映射的关节向量：
1. 从 MediaPipe 关键点计算人手方向向量并归一化
2. 获取机器人对应关节间的当前距离
3. 目标位置 = 起点位置 + 人手方向 × 机器人距离
4. 通过 MuJoCo 运动学与约束优化求解器驱动机器人手达到目标

## 安装

```bash
git submodule update --init --recursive
pip install -e .
```

项目里的大体积 `assets` / 样例数据统一走外部仓库，不直接塞进 Git。现在支持和 `Teleopit` 类似的下载流程：

```bash
# ModelScope（推荐）
python scripts/setup/download_assets.py

# 或者按组下载
python scripts/setup/download_assets.py --only mjcf mediapipe examples

# HuggingFace 也支持
python scripts/setup/download_assets.py --source huggingface --repo-id 12e21/somehand-assets
```

脚本当前约定的远端布局写在 `src/somehand/external_assets.py`，当前包含三组资源：

- `mjcf`：`archives/mjcf_assets.tar.gz` → `assets/mjcf/`
- `mediapipe`：`models/hand_landmarker.task` → `assets/models/hand_landmarker.task`
- `examples`：
  - `archives/reference_assets.tar.gz` → `assets/`
  - `archives/sample_recordings.tar.gz` → `recordings/`

当前默认资产模型仓：

- `ModelScope`：`BingqianWu/somehand-assets`
- `HuggingFace`：`12e21/somehand-assets`

如果你重新生成了 `MJCF` 或其它资源，请上传到上述云端资产仓，再由本地执行下载脚本同步；不要把生成结果直接提交到 `assets/`。

主要依赖：mujoco, mink, mediapipe, opencv-python, numpy, pyyaml, daqp

安装后主要入口是：

```bash
somehand --help
```

如果你是新 clone 下来的，也可以直接：

```bash
git clone --recurse-submodules <repo-url>
cd somehand
pip install -e .
```

可选的第三方 SDK 现在统一放在 `third_party/`：

- `third_party/xrobotoolkit/XRoboToolkit-PC-Service-Pybind`：PICO / XRoboToolkit Python binding 子仓库
- `third_party/linkerhand-python-sdk`：LinkerHand Python SDK 子仓库（real backend 默认从这里找）

## 快速开始

### 1. 准备 assets

如果你已经有外部 asset 仓库，先把默认运行资源拉下来：

```bash
python scripts/setup/download_assets.py --only mjcf mediapipe
```

`hc-mocap` 默认使用代码内置的 joint 顺序和骨架定义，不再依赖仓库里的默认 `BVH` 文件；只有你想覆盖默认 UDP 解析格式时，才需要额外传 `--reference-bvh <path>`.

如果你想下载仓库里不再直接存放的参考 `BVH` 和样例录制数据：

```bash
python scripts/setup/download_assets.py --only examples
```

如果你不想维护远端资产，也可以继续手动转换/准备本地文件。

### 2. 转换手部模型（以 Linkerhand L20 为例）

```bash
python scripts/convert_urdf_to_mjcf.py \
    --urdf ../linkerhand-urdf/l20/right/linkerhand_l20_right.urdf \
    --output assets/mjcf/linkerhand_l20_right
```

### 3. 查看模型

```bash
python scripts/visualize_hand.py --mjcf assets/mjcf/linkerhand_l20_right/model.xml
```

### 4. 实时 Retargeting（摄像头）

```bash
somehand webcam
```

默认会打开两个独立的 `MuJoCo` viewer：一个看输入手势 / mocap landmarks，一个看 retarget 后的机器人手。为避免双 viewer 在同一进程下闪退，输入手势窗口会由单独子进程承载。

按 `q` 退出。

如果想把当前输入直接录成可回放的离线数据：

```bash
somehand webcam \
    --record-output recordings/webcam_hand.pkl
```

### 5. 自动录制验收视频（摄像头）

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
    --config configs/retargeting/right/linkerhand_l20_right.yaml \
    --video recordings/acceptance.mp4 \
    --hand right
```

### 6. 离线 Retargeting（视频）

```bash
somehand video \
    --video input.mp4
```

### 6.1 离线 Retargeting（已录制手部数据）

如果你想绕过摄像头 / PICO / UDP 输入，直接复现某次采集到的手部 landmarks：

```bash
somehand replay \
    --recording recordings/webcam_hand.pkl
```

`replay` 默认就会按录制时的 fps 实时回放。
如果想离线导出机器人手的 MuJoCo 视频，使用单独命令：

```bash
somehand dump-video \
    --recording recordings/webcam_hand.pkl \
    --output recordings/webcam_hand_replay.mp4
```

### 6.2 hc_mocap 手部输入

如果你已经有 Teleopit 的 `hc_mocap` UDP 流，可以跳过 MediaPipe，直接把手骨架转成 21 点后喂给当前 retargeting：

```bash
somehand hc-mocap \
    --hand right \
    --udp-stats-every 120
```

你当前这类高频命令现在可以缩到：

```bash
somehand hc-mocap \
    --hand left
```

`hc-mocap` 现在只保留 UDP 模式，不再支持离线 `BVH` 输入。它也不依赖 `Teleopit` Python 包；默认直接使用代码内置的 joint 顺序解析每个 UDP 包里的一行 BVH motion floats。只有当你想覆盖默认解析格式时，才需要额外传 `--reference-bvh`。
`hc_mocap` 输入会自动使用 wrist 真局部坐标做 retarget，因此即使配置文件里是 `wrist_local`，脚本也会切到更适合 `hc_mocap` 的处理方式。
如果要检查 UDP 是否正常进入，可以看终端的 `UDP stats` 输出，确认 `recv` / `valid` 是否持续增长，以及 `floats` 是否等于内置格式的通道数（默认是 `159`）。
输入手势窗口里显示的是参与 retarget 的输入 landmarks，因此看到的是已经对齐到机器人坐标系的手部骨架；机器人窗口单独显示 retarget 后的手模型。

### 6.3 PICO 4 手部输入

先确保 `xrobotoolkit_sdk` 可导入；如果没装，可以运行：

```bash
git submodule update --init --recursive
bash scripts/setup_xrobotoolkit.sh
```

然后先探测 PICO / XRoboToolkit 链路是否正常：

```bash
python scripts/probe_pico_xrobotoolkit.py --hand right
```

探测正常后，直接运行实时 retargeting：

```bash
somehand pico \
    --hand right
```

如果想把 PICO 输入录下来，供之后离线复现：

```bash
somehand pico \
    --hand right \
    --record-output recordings/pico_hand.pkl
```

运行后：

- 在终端或 robot-hand MuJoCo 窗口按 `r` 开始录制
- 在终端或 robot-hand MuJoCo 窗口按 `s` 结束录制，并自动保存到 `--record-output` 指定路径
- 如果中途直接 `Ctrl+C`，也会保存已经录到的内容

常见前提：

- 头显里已打开 Hand Tracking 权限
- 当前在 gesture / hand tracking 模式，而不是 controller mode
- XRoboToolkit / RobotLinuxDemo 在头显前台，并已连接 PC service

如果长时间收不到手数据，可以把等待时间调长：

```bash
somehand pico \
    --hand right \
    --pico-timeout 90
```

### 6.4 LinkerHand 真机 backend

如果你想直接把 retarget 结果发到 LinkerHand 真手，先初始化 SDK 子仓库：

```bash
git submodule update --init --recursive
bash scripts/setup_linkerhand_sdk.sh
```

这个脚本现在也会顺手执行：

```bash
python3 -m pip install -r third_party/linkerhand-python-sdk/requirements.txt
```

如果你想指定解释器，可以这样运行：

```bash
PYTHON_BIN=python bash scripts/setup_linkerhand_sdk.sh
```

现在默认 `sdk_root` 就是仓库内的：

```text
third_party/linkerhand-python-sdk
```

因此通常不需要再手动传 `--sdk-root`；只有当你想切到自定义 SDK 路径时才需要覆盖。

### 6. 跑验收脚本

先跑合成基线，确认当前配置的旋转不变性、双侧预处理一致性、静态抖动、求解误差和吞吐量：

```bash
python scripts/acceptance_check.py --config configs/retargeting/right/linkerhand_l20_right.yaml
```

如果有真实视频，再追加离线验收：

```bash
python scripts/acceptance_check.py \
    --config configs/retargeting/right/linkerhand_l20_right.yaml \
    --video input.mp4 \
    --hand right
```

## 适配新的机器人手

1. **转换模型**：用 `convert_urdf_to_mjcf.py` 将 URDF 转换为 MJCF
2. **编写配置**：在 `configs/retargeting/base/` 写共享模板，在 `configs/retargeting/left/` 或 `configs/retargeting/right/` 放可直接运行的侧别配置
3. **运行测试**：指定新配置运行 `somehand webcam --visualize` 验证效果

仓库里现在已经接入这些五指手型示例：

- `dexhand021`：来自 `dexrobot_urdf`，建议使用 `*_simplified.urdf`
- `dex5`：来自 `unitree_ros/robots/dexterous_hand_description/dex5_1`
- `inspire_dfq`：来自 `unitree_ros/robots/g1_description/inspire_hand/DFQ_*`
- `inspire_ftp`：来自 `unitree_ros/robots/g1_description/inspire_hand/FTP_*`
- `omnihand`：来自 `omnihand_description-omnihandT2_1`
- `revo2`：来自 `revo2_description`
- `rohand`：来自 `rohand_urdf_ros2`
- `sharpa_wave`：来自 `sharpa-urdf-usd-xml/wave_01`
- `wujihand`：来自 `wuji-hand-description`，仓库内已带左右手 MJCF 与配置

例如直接跑 `Wuji` 右手配置：

```bash
somehand webcam \
    --config configs/retargeting/right/wujihand_right.yaml \
    --hand right
```

配置文件格式参考 `configs/retargeting/right/linkerhand_l20_right.yaml`，核心字段：

```yaml
hand:
  name: "your_hand"
  side: "right"
  mjcf_path: "path/to/model.xml"

retargeting:
  vector_constraints:
    - human: [0, 5]      # wrist -> index_mcp
      robot: ["base_link", "index_proximal"]
      robot_types: ["body", "body"]
      weight: 1.0
  frame_constraints:      # 可选：例如给 thumb CMC 增加局部朝向目标
    - name: "thumb_cmc_frame"
      human_origin: 1
      human_primary: 2
      human_secondary: 5
      robot_origin: "thumb_base"
      robot_primary: "thumb_proximal"
      robot_secondary: "index_proximal"
      primary_weight: 1.5
      secondary_weight: 1.0
  preprocess:
    temporal_filter_alpha: 0.35
  solver:
    max_iterations: 20
```

## 架构概览

当前核心库已经按分层接口重构：

- `domain`：纯配置模型、手部帧模型、landmark 预处理、方向目标计算
- `application`：会话编排与 retargeting engine
- `infrastructure`：MuJoCo/MediaPipe/PICO/hc_mocap/文件输出/预览窗口等适配器
- `interfaces`：CLI 和面向外部的薄入口

顶层模块统一位于 `somehand` 包下；命令行入口、文档示例和脚本导入路径都已切换到 `somehand`。

## 项目结构

```
somehand/
├── src/somehand/
│   ├── domain/               # 纯领域模型、配置、预处理
│   ├── application/          # pipeline / session / engine 编排
│   ├── infrastructure/       # MuJoCo / 输入源 / sink / 持久化
│   ├── interfaces/           # CLI 等外部接口
│   ├── acceptance.py         # 验收指标
│   ├── hand_detector.py      # MediaPipe 检测封装
│   ├── hc_mocap_input.py     # hc_mocap provider
│   ├── pico_input.py         # PICO provider
│   ├── urdf_converter.py     # URDF → MJCF 转换
│   └── visualization.py      # MuJoCo viewer
├── configs/retargeting/    # Retargeting 配置文件
│   ├── base/              # 按型号复用的共享模板
│   ├── left/              # 可直接运行的左手配置
│   └── right/             # 可直接运行的右手配置
├── assets/                # 仅保留占位目录；实际资源从云端资产仓下载
│   ├── mjcf/
│   └── models/
├── scripts/                # 薄工具脚本 / 诊断脚本
│   ├── acceptance_check.py
│   ├── convert_urdf_to_mjcf.py
│   ├── record_webcam.py
│   └── visualize_hand.py
└── tests/                  # 单元测试与会话编排测试
```

## 参考项目

- [mink](https://github.com/kevinzakka/mink) — MuJoCo 微分逆运动学库
- [GMR](https://github.com/YanjieZe/GMR) — 全身运动 retargeting
