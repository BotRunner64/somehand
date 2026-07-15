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
| 使用另一个已提交的手模型 | 传源码/wheel 通用的 `--config <side>/<model>_<side>.yaml` |
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

release wheel 会内置仓库中提交的配置并作为 CLI 默认值。内置文件应视为只读；需要定制时，先把对应配置族复制到项目中。内置配置里的 `assets/...` 引用会解析到 `SOMEHAND_HOME`；自定义配置仍按普通文件系统相对路径处理。

例如，`--config right/omnihand_right.yaml` 在源码检出中解析到已提交配置树，在 wheel 安装中解析到内置配置树。

---

## 通常需要关注的字段

| 段 | 用途 |
| --- | --- |
| `hand` | 模型名、手别、MJCF 路径、可选 URDF 来源元信息。 |
| `controller` | backend 默认值、频率、transport、SDK 路径、硬件型号族。 |
| `retargeting` | 公共 solver/preprocess 设置，以及每个手型自己的显式约束。 |
| `viewer` | 双手面板、相机、pose 设置。 |

---

## 校验规则

- `retargeting.preset` 会被拒绝；vector、distance、frame 和 angle 约束应写在手型配置中
- `retargeting.vector_loss` 以及每条 vector 的 `loss_type` / `loss_scale` 会被拒绝
- 旧 vector 字段会被拒绝：`human_vector_pairs`、`origin_link_names`、`task_link_names`、`vector_weights`
- 已移除段会被拒绝：`position_constraints`、`pinch`
- 运行时校验会检查 backend 名称、transport 名称，以及正数控制/仿真频率

---

## 从 0.2 升级

最稳妥的迁移方式是从 0.3 中对应的已提交配置开始，只重新应用模型路径或 controller 覆盖。0.2 中这样的 preset：

```yaml
retargeting:
  preset: universal
  vector_loss:
    type: direction
```

需要改为手型专属的显式约束：

```yaml
extends: "./_universal_common.yaml"

retargeting:
  vector_constraints:
    - human: [1, 2]
      robot: ["thumb_metacarpals", "thumb_proximal"]
  distance_constraints:
    - human: [4, 8]
      robot: ["thumb_tip", "index_tip"]
      robot_types: ["site", "site"]
```

实际名称和完整约束集合应以 `configs/retargeting/base/` 下匹配的文件为准；上面的缩略示例只展示 schema 变化。
