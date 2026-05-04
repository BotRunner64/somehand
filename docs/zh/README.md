# somehand 文档

**somehand** 将人手动作映射到可配置的机器人灵巧手模型 —— 从摄像头输入一路到真机控制。

---

## 支持的输入源

| 来源 | 说明 |
| --- | --- |
| **MediaPipe 摄像头** | 从摄像头实时手部追踪 |
| **MediaPipe 视频** | 对视频文件离线追踪 |
| **PICO VR** | 通过 PICO Bridge 接实时手部输入 |
| **hc_mocap UDP** | 通过 UDP 接实时手部数据 |
| **已录制数据** | 回放以上任意来源生成的 `.pkl` 录制 |

## 支持的 Backend

| Backend | 说明 |
| --- | --- |
| **viewer** | MuJoCo 可视化（单手或双手） |
| **sim** | MuJoCo 物理仿真 |
| **real** | 真机硬件控制（仅支持单手） |

---

## 文档导航

| 文档 | 内容 |
| --- | --- |
| [快速开始](getting-started.md) | 安装、资产下载、可选 SDK 配置、首次运行 |
| [运行模式](runtime-modes.md) | 各 CLI 模式的用途、适用场景、全部参数 |
| [配置说明](configuration.md) | YAML 配置结构、schema、该改哪个文件 |
| [资产与模型](assets-and-models.md) | 资产分组、外部仓库、20+ 支持手模型 |
| [常见问题](troubleshooting.md) | 常见安装与运行问题及解决方式 |
| [维护指南](maintainer-guide.md) | 更新流程、验证方式、维护规则 |

---

## 项目范围

本仓库当前主要面向：

- **单手 retargeting** —— 基于 YAML 配置驱动
- **双手可视化** —— `viewer` 模式下的双手回放与渲染
- **轻量仓库** —— 大体积运行时资产托管在外部

> 在假设某个输入源 + backend 组合可用之前，请先阅读 [运行模式](runtime-modes.md)。
