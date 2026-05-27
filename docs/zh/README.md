# somehand 文档

按入口找文档。CLI 用于运行打包好的工具，API 用于在 Python 里嵌入 retargeting。

---

## 从这里开始

| 我想做什么 | 阅读 |
| --- | --- |
| 安装依赖并验证一次运行 | [快速开始](getting-started.md) |
| 从终端运行 | [CLI 用法](runtime-modes.md) |
| 在 Python 中嵌入 retargeting | [API 用法](api.md) |
| 选择手模型或修改 YAML 配置 | [配置说明](configuration.md) |
| 下载运行时资产或查看模型覆盖 | [资产与模型](assets-and-models.md) |

---

## 参考

| 文档 | 内容 |
| --- | --- |
| [快速开始](getting-started.md) | 安装、资产、首次运行、可选 SDK 配置 |
| [CLI 用法](runtime-modes.md) | 直播、回放、导出和硬件用途的终端命令 |
| [API 用法](api.md) | 稳定 Python import、单步 engine、session hook |
| [配置说明](configuration.md) | 该改哪个配置文件，以及通常需要关注的字段 |
| [资产与模型](assets-and-models.md) | 下载命令、本地路径、已覆盖的配置家族 |
| [维护指南](maintainer-guide.md) | 文档规则、验证命令、模型更新流程 |

---

## 项目范围

CLI 常用路径是单手 retargeting。双手能力主要用于 `viewer` 回放/渲染，真机控制当前仅支持单手。API 用户通常自己提供 landmark frame，并直接调用 engine。
