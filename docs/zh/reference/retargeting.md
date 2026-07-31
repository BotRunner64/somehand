# 重定向算法

[English](../../en/reference/retargeting.md)

somehand 独立求解每只手。双手重定向会分别对左手和右手运行同一套单手流程。

## 处理流程

```text
21 点手部 landmark
        │
        ▼
以手腕为原点的手掌坐标系 + 手别变换
        │
        ▼
landmark 时序滤波
        │
        ▼
vector、frame 和 distance 目标
        │
        ▼
带关节限位的关节角优化
        │
        ▼
关节输出平滑 + mimic joint
```

首先平移 landmark，使手腕点成为原点。点 `0`、`5`、`9` 用于建立手掌坐标系，再通过手别相关变换映射到机器人坐标系。如果手掌点退化而无法建立坐标系，则回退到固定的 MediaPipe 到 MuJoCo 坐标轴映射。随后使用指数移动平均平滑变换后的 landmark。

## 优化目标

令 `q` 表示机器人关节位置。MuJoCo 正向运动学提供机器人 body/site 的位置和旋转。求解器最小化：

```text
loss(q) = 方向项
        + 坐标轴项
        + 已激活的距离项
        + norm_delta * ||q - q_previous||²
```

### Vector 方向

一条 vector 约束把两个人手 landmark 映射到两个机器人 body/site。两侧向量都会归一化，因此该项匹配方向而不是长度：

```text
direction_loss = weight * (1 - dot(robot_direction, human_direction))
```

方向完全一致时损失为零，方向差异越大，损失越大。

### Frame 朝向

一条 frame 约束包含原点、主方向点和次方向点。Gram-Schmidt 正交化会建立两个人手目标轴；对应的机器人局部轴从模型中确定，再随当前原点 body/site 的旋转变化。主轴和次轴分别使用相同的余弦方向损失，并拥有独立权重。

当单个向量不能确定绕该向量的旋转时，应使用 frame 约束，例如拇指根部朝向。

### 指尖距离

Distance 约束先计算机器人目标距离：

```text
raw:         target = scale * human_distance
hand_scaled: target = scale * human_distance
                      * robot_middle_finger_length
                      / human_middle_finger_length
```

仓库内置的 retargeting 配置使用 `raw`，直接应用输入 landmark 距离，不做手部尺寸补偿。需要补偿时仍可使用 `hand_scaled`；人手中指长度为 `9 → 10 → 11 → 12` 三段之和，机器人长度使用解析出的中指关节链。

距离激活值由人手距离 `d` 和 `threshold` 决定：

```text
linear:   activation = max(0, 1 - d / threshold)
gaussian: activation = exp(-(d / (threshold / 2))²)
```

当 `threshold <= 0` 时，激活值恒为 `1`。激活值经过时序平滑后再参与计算。距离损失是单向的：

```text
distance_loss = weight * activation
                * max(robot_distance - target_distance, 0)²
```

机器人点距离过大时，该项会把它们拉近；距离小于目标时不会受到惩罚，因此适合表达捏合闭合。

### 时序正则项

`norm_delta * ||q - q_previous||²` 抑制相对上一帧输出的剧烈变化。虽然名称中有 `delta`，但 `norm_delta` 是损失权重，不是收敛容差。

## 求解器

优化器使用 SciPy SLSQP、MuJoCo 位置/旋转 Jacobian 和解析目标梯度。它会：

- 从当前关节状态开始求解；
- 只优化独立关节，再重建 mimic joint；
- 使用 MJCF 关节范围作为上下界；
- 达到 `max_iterations` 或 SLSQP 收敛时停止；
- 使用内部固定的 SLSQP `ftol=1e-6`。

优化完成后，还会用一次指数移动平均混合当前结果和上一帧输出。

## 求解与平滑参数

仓库内所有模型配置都继承 `base/_universal_common.yaml`。“内置生效值”是经过继承和代码默认值补全后的结果；自定义配置没有填写或继承字段时使用“代码默认值”。

| 参数 | 内置生效值 | 代码默认值 | 影响 |
| --- | ---: | ---: | --- |
| `preprocess.temporal_filter_alpha` | `0.65` | `0.35` | Landmark EMA 系数。越小越平滑，但输入延迟越大；越大响应越快，但保留更多噪声。有效范围为 `(0, 1]`。 |
| `solver.max_iterations` | `60` | `30` | 每帧 SLSQP 最大迭代数。增大可能改善困难姿态的收敛，但会增加计算量。 |
| `solver.norm_delta` | `0.001` | `0.01` | 上一帧输出惩罚权重。增大可抑制关节跳变，但会阻碍快速动作。应保持非负。 |
| `solver.output_alpha` | `0.92` | `0.70` | 关节输出 EMA 系数。越小越平滑，但输出延迟越大；越大越接近优化器结果。有效范围为 `(0, 1]`。 |
| `solver.activation_alpha` | `0.30` | `0.30` | 距离激活值 EMA 系数。越小，捏合约束的启用和停用变化越慢。应保持在 `(0, 1]`。 |

本页所有 EMA 都使用：

```text
filtered_t = alpha * current_t + (1 - alpha) * filtered_(t-1)
```

第一帧直接使用当前值。

## 约束参数

| 参数 | 影响 |
| --- | --- |
| `weight` | 乘在单条 vector 或 distance 损失上。数值越大，该约束相对其他项的影响越大。 |
| `primary_weight`、`secondary_weight` | 分别控制 frame 约束的两条坐标轴。 |
| `scale` | 把人手距离转换为机器人目标距离时使用的倍率。 |
| `threshold` | 控制人手两点相距多远时 distance 约束停止激活，单位与输入 landmark 距离一致。值越大，捏合约束越早启用。 |
| `activation_type` | 选择 `linear` 或 `gaussian` 距离激活函数。 |
| `scale_mode` | `raw`（默认）直接使用输入距离；`hand_scaled` 会补偿人手与机器人手的尺寸差异。 |

单条约束中填写的值会覆盖 `constraint_defaults`。Vector 的 `terminal_weight` 用于机器人第二个点为 site 的约束；distance 的 `weights_by_human` 根据 landmark 对选择默认权重。

## 推荐调参顺序

1. 先用 `--viewer-mode diagnostic` 检查 landmark 对和机器人 body/site 名称。
2. 调整 `temporal_filter_alpha` 和 `output_alpha`，平衡噪声与延迟。
3. 只有仍有明显关节跳变或动作阻力过大时，再调整 `norm_delta`。
4. 针对姿态误差调整约束权重、distance `scale` 和 `threshold`。
5. 只有优化器确实需要更多计算才能得到合理姿态时，才增大 `max_iterations`。

| 现象 | 优先检查的参数 |
| --- | --- |
| Landmark 噪声或关节抖动 | 降低 `temporal_filter_alpha` 或 `output_alpha`，再考虑增大 `norm_delta`。 |
| 动作响应太慢 | 增大两个 alpha，或减小 `norm_delta`。 |
| 捏合闭合太晚或不充分 | 先检查指尖 site，再增大 `threshold`、`weight` 或调整 `scale`。 |
| 捏合闭合太早 | 减小 `threshold` 或 distance `weight`。 |
| 手指弯曲正确，但根部朝向错误 | 检查 frame 约束及其两项权重。 |
| 单帧求解过慢 | 减小 `max_iterations`；改变约束覆盖前先做性能分析。 |

每次只调整一组参数，并用多个代表性姿态验证。YAML 结构和路径规则见[配置](configuration.md)。
