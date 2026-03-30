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
python scripts/retarget_webcam.py --config configs/retargeting/linkerhand_l20.yaml
```

如果想同时看 MediaPipe 的 3D 关键点浏览器视图：

```bash
python scripts/retarget_webcam.py \
    --config configs/retargeting/linkerhand_l20.yaml \
    --viser
```

默认 `viser` 显示的是 `preprocess_landmarks()` 之后的手局部坐标系 3D 点。如果想看原始 MediaPipe 世界坐标，可以显式指定：

```bash
python scripts/retarget_webcam.py \
    --config configs/retargeting/linkerhand_l20.yaml \
    --viser \
    --viser-space raw
```

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
python scripts/retarget_video.py \
    --video input.mp4 \
    --config configs/retargeting/linkerhand_l20.yaml \
    --output trajectory.pickle \
    --visualize
```

也可以只打开 `viser` 里的 3D landmarks 调试视图：

```bash
python scripts/retarget_video.py \
    --video input.mp4 \
    --config configs/retargeting/linkerhand_l20.yaml \
    --viser
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
3. **运行测试**：指定新配置运行 `retarget_webcam.py` 验证效果

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
│   ├── convert_urdf_to_mjcf.py
│   ├── record_webcam.py
│   ├── retarget_webcam.py
│   ├── retarget_video.py
│   └── visualize_hand.py
└── src/dex_mujoco/         # 核心库
    ├── constants.py            # MediaPipe 关键点定义
    ├── hand_detector.py        # MediaPipe 手部检测
    ├── hand_model.py           # MuJoCo 模型封装
    ├── ik_solver.py            # Mink IK 求解器
    ├── retargeting_config.py   # 配置加载
    ├── urdf_converter.py       # URDF → MJCF 转换
    ├── vector_retargeting.py   # 向量 retargeting 核心
    └── visualization.py        # MuJoCo 可视化
```

## 参考项目

- [dex-retargeting](https://github.com/dexsuite/dex-retargeting) — 手部 retargeting 算法
- [mink](https://github.com/kevinzakka/mink) — MuJoCo 微分逆运动学库
- [GMR](https://github.com/YanjieZe/GMR) — 全身运动 retargeting
