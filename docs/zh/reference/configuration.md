# 配置

[English](../../en/reference/configuration.md)

Retargeting 配置用于选择机器人模型、手别、约束和可选控制器元数据。

本页说明 YAML 结构；目标函数、求解参数和调参方法见[重定向算法](retargeting.md)。

## 配置目录

```text
configs/retargeting/
├── base/       # 每个模型的共享约束
├── left/       # 左手模型和资产绑定
├── right/      # 右手模型和资产绑定
└── bihand/     # 左右手配置组合
```

默认配置是：

- 单手：`right/linkerhand_l20_right.yaml`
- 双手：`bihand/linkerhand_l20_bihand.yaml`

CLI 选择其他已提交配置：

```bash
somehand webcam \
    --hand right \
    --config right/omnihand_right.yaml
```

Python 中可解析同一个可移植名称：

```python
from somehand.api import resolve_config_path

config_path = resolve_config_path("right/omnihand_right.yaml")
```

## 配置如何组合

单侧配置通常继承一个基础模型，并指定该侧 MJCF：

```yaml
extends: "../base/linkerhand_l20.yaml"

hand:
  name: "linkerhand_l20_right"
  side: "right"
  mjcf_path: "../../../assets/mjcf/linkerhand_l20_right/model.xml"
```

双手配置组合左右两个单侧配置：

```yaml
left:
  config: "../left/linkerhand_l20_left.yaml"

right:
  config: "../right/linkerhand_l20_right.yaml"
```

`extends` 会递归合并 mapping；新列表会替换基础列表。每个 `extends` 目标相对于声明它的文件解析；`hand.mjcf_path` 相对于最终选择的单侧配置解析，双手配置中的 `left` / `right` 路径相对于该双手配置解析。

## 主要字段

| 配置段 | 用途 |
| --- | --- |
| `hand` | `name`、`side`、`mjcf_path` 和可选的 `urdf_source` 元数据。 |
| `retargeting.constraint_defaults` | 约束的默认权重和激活设置。 |
| `retargeting.vector_constraints` | 匹配两个人手 landmark 与两个机器人物体/site 之间的方向。 |
| `retargeting.distance_constraints` | 匹配距离，通常用于指尖。 |
| `retargeting.frame_constraints` | 对齐由三个人手点和三个机器人点构成的坐标系。 |
| `retargeting.preprocess` | 通过 `temporal_filter_alpha` 平滑输入。 |
| `retargeting.solver` | 迭代、时序正则、激活平滑和输出平滑设置。 |
| `controller` | 真机 backend 的型号族和默认速度/力矩元数据。运行 backend 和 transport 由 CLI 参数选择。 |
| `viewer` | 双手窗口尺寸、模型 pose 和相机目标。 |

约束里的机器人名称必须是所选 MJCF 中存在的 `body` 或 `site` 名称。人手索引使用 MediaPipe 的 21 点手部 landmark 顺序。

## 路径规则

- 源码安装读取仓库中的 `configs/retargeting/`。
- release wheel 包含该目录的副本，内置配置应视为只读。
- 内置配置里的 `assets/...` MJCF 引用会解析到 `SOMEHAND_HOME` 或默认数据目录下。
- 自定义配置的普通相对路径从该配置文件所在目录解析。
- 只有替换整个内置配置根目录时，才需要在导入 somehand 前设置 `SOMEHAND_CONFIG_ROOT`。

## 新增或定制模型

1. 从 `base/`、`left/` 或 `right/` 中最接近的配置族开始。
2. 让 `hand.mjcf_path` 指向模型，并设置正确的 `hand.side`。
3. 每条约束都使用该 MJCF 中存在的 body/site 名称。
4. 左右配置都存在后，再添加 `bihand/` 配置。
5. 加载配置，并用普通和诊断 viewer 验证。

```bash
somehand replay \
    --recording recordings/pico_right.pkl \
    --config /path/to/custom_right.yaml \
    --viewer-mode diagnostic
```
