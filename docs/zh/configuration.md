# 配置说明

需要换手模型、改资产路径或调整硬件默认值时，看这一页。

## 选择配置

```
configs/retargeting/
├── base/       # 共享模型约束
├── left/       # 左手运行配置
├── right/      # 右手运行配置
└── bihand/     # 双手 viewer/replay 配置
```

| 模式 | 默认配置 |
| --- | --- |
| 单手 | `configs/retargeting/right/linkerhand_l20_right.yaml` |
| 双手 | `configs/retargeting/bihand/linkerhand_l20_bihand.yaml` |

---

## 该改哪里

| 目的 | 修改位置 |
| --- | --- |
| 使用另一个已提交的手模型 | 传 `--config configs/retargeting/<side>/<model>_<side>.yaml` |
| 修改 MJCF 路径或手别绑定 | `left/` 或 `right/` 配置 |
| 修改共享 retargeting 约束 | `base/` 配置 |
| 修改双手 viewer/replay 组合 | `bihand/` 配置 |
| 修改硬件默认值 | 运行配置里的 `controller` 段 |

---

## 运行配置形状

单手配置通常继承一个 base 配置，并绑定到某一侧手：

```yaml
extends: "../base/linkerhand_l20.yaml"

hand:
  name: "linkerhand_l20_right"
  side: "right"
  mjcf_path: "../../../assets/mjcf/linkerhand_l20_right/model.xml"
```

双手配置组合一份左手配置和一份右手配置：

```yaml
left:
  config: "../left/linkerhand_l20_left.yaml"

right:
  config: "../right/linkerhand_l20_right.yaml"
```

相对路径从 YAML 文件所在目录解析。`extends` 支持链式继承。

---

## 通常需要关注的字段

| 段 | 用途 |
| --- | --- |
| `hand` | 模型名、手别、MJCF 路径、可选 URDF 来源元信息。 |
| `controller` | backend 默认值、频率、transport、SDK 路径、硬件型号族。 |
| `retargeting` | 标准配置用 `preset: universal`；自定义模型可显式写约束。 |
| `viewer` | 双手面板、相机、pose 设置。 |

---

## 校验规则

- 设置 `retargeting.preset` 时只能是 `universal`
- 旧 vector 字段会被拒绝：`human_vector_pairs`、`origin_link_names`、`task_link_names`、`vector_weights`
- 已移除段会被拒绝：`position_constraints`、`pinch`
- 运行时校验会检查 backend 名称、transport 名称，以及正数控制/仿真频率
