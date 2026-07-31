# somehand 文档

[English documentation](../en/README.md)

somehand 把 21 点人手 landmark 转换为可配置机器人手的关节目标。需要完整的输入到输出流程时使用 CLI；已有程序自己管理输入循环时使用 Python API。

第一次使用请先完成[安装](getting-started/installation.md)，再选择 CLI 或 API 教程。

## 安装

- [安装](getting-started/installation.md) — 安装软件包、下载所需资产并验证一次运行。

## 教程

- [CLI](tutorials/cli.md) — 摄像头、视频、录制、PICO、hc_mocap、仿真和真机。
- [Python API](tutorials/api.md) — 在 Python 中重定向单手或双手 landmark 流。

## 参考资料

- [配置](reference/configuration.md) — 选择手模型并修改 retargeting YAML。
- [重定向算法](reference/retargeting.md) — 目标函数、求解过程和参数调节。
- [资产与模型](reference/assets.md) — 资产分组、下载位置和模型覆盖。
- [代码结构](reference/code-structure.md) — 运行时数据流和仓库布局。

## 当前范围

CLI 的 viewer backend 支持单手和双手；仿真和真机 backend 仅支持单手。稳定的嵌入入口是 `somehand.api`。
