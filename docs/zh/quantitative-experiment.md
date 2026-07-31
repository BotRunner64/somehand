# 定量实验

这套流程用于无 viewer、无硬件控制器地运行冻结后的跨手型比较以及 distance/frame 消融。原始录制和生成结果保存在 `recordings/` 下，不提交到仓库。

## 1. 预检

安装开发依赖、下载 MJCF 资产并检查完整实验矩阵：

```bash
pip install -e ".[dev]"
python scripts/setup/download_assets.py --only mjcf
python scripts/experiments/preflight_quantitative.py
```

v1 manifest 覆盖 19 个右手配置。每只手必须解析出 4 个 distance constraints、1 个 thumb-frame constraint、5 个公共 base-to-tip 评测方向、5 个 fingertips 以及 fingertip collision geometry。运行矩阵包含 19 个 Full，以及 `linkerhand_l20_right` 的 `no_distance` 和 `no_frame`。

## 2. 直接使用已有 PICO 录制

探索性实验不要求重新录制固定协议。已有数据 manifest 直接使用
`recordings/pico_right.pkl`，覆盖 6 个代表手型和 L20 的两组消融：

```bash
python scripts/experiments/run_quantitative.py \
  --manifest configs/experiments/quantitative_existing_pico.yaml
python scripts/experiments/summarize_available_quantitative.py
```

由于这份录制没有动作标签，探索性汇总对所有保存帧计算 Direction Error、
Normalized Semantic Keypoint Error、site 距离跟踪误差和 thumb-frame error。
Site 距离跟踪误差是机器人各组 thumb-to-finger site 距离与对应 human
fingertip 距离之差的绝对值；它是连续量，不使用接触阈值，也不受 collision
mesh 厚度影响。生成的 CSV、JSON 和图片文件名均包含 `available`，并明确记录
全帧统计口径。

## 3. 录制固定 PICO 序列

连接 PICO Bridge，保持右手处于 active 状态，然后运行：

```bash
python scripts/experiments/record_quantitative_pico.py
```

终端会依次提示七种动作：握拳、四种双指 pinch、tripod pinch 和 thumb opposition。每种动作重复五次。默认协议以 80 Hz 录制 12,600 帧、持续 157.5 秒，不包含倒计时。

录制程序生成：

- `recordings/quantitative/pico_right_v1.pkl`
- `recordings/quantitative/pico_right_v1.annotations.json`

只要出现采样帧缺失，本次录制就会中止且不会保存正式产物。已有文件默认不会覆盖，只有显式传入 `--force` 才允许覆盖。

在运行任何机器人输出前，先检查动作完成情况与采样连续性：

```bash
python scripts/experiments/validate_quantitative_recording.py
```

## 4. 运行正式实验矩阵

运行全部 21 个 job：

```bash
python scripts/experiments/run_quantitative.py
```

常用的局部运行方式：

```bash
python scripts/experiments/run_quantitative.py --dry-run
python scripts/experiments/run_quantitative.py \
  --hand linkerhand_l20_right \
  --method full
```

仅用于无标注 smoke check：

```bash
python scripts/experiments/run_quantitative.py \
  --recording recordings/samples/pico_right_short.pkl \
  --allow-unannotated \
  --hand linkerhand_l20_right \
  --method full \
  --max-frames 20 \
  --output-dir /tmp/somehand-quantitative-smoke
```

每个 `hand × method` 的正式结果由压缩 NumPy 数组和 JSON metadata 组成。metadata 包含输入/配置哈希、解析后的完整配置、语义映射、模型耦合数量、环境版本以及 SLSQP 状态。

## 5. 生成正式表格与图片

全部正式运行结束后执行：

```bash
python scripts/experiments/summarize_quantitative.py
python scripts/experiments/render_ablation_snapshots.py
```

summary 目录会包含全部手型表、六手型主表、distance 消融表与曲线、frame 消融表与柱图，以及由同一个 opposition 输入帧生成的 Full/w-o-frame 姿态图。

## 冻结的指标口径

- 主 Direction Error 使用 5 个公共 finger-base-to-tip 射线，不随各手 objective topology 改变；同时保存实际 objective segments 的误差供补充诊断。
- Normalized Semantic Keypoint Error 使用 5 个 fingertips、wrist/world 原点以及 wrist-to-middle-base palm scale。
- Site Distance Tracking MAE 是 robot fingertip-site 距离与对应 human fingertip 距离之差的平均绝对值。正式 distance 消融使用已标注 pinch 的 hold 帧；无标签探索性评测使用所有保存帧。
- Collision-surface distance 仍作为逐帧诊断数据保存，但不再进入跨手得分、排名或消融结论。
- Mean Solve Time 只计量一次 SciPy SLSQP 调用，并在正式运行前执行 throwaway warm-up。
- Thumb-Frame Error 即使在 `no_frame` 下也沿用 Full 的语义 frame mapping。

查看正式机器人输出后，不得修改录制、标注、阈值、映射或 manifest。优化失败和无效 episode 必须报告，不能丢弃。
