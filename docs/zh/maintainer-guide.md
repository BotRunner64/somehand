# 维护指南

## 文档规则

- **README.md** 只保留 landing page 内容，保持简洁
- 详细文档放到 `docs/`
- **更新时必须同时维护 `docs/en` 与 `docs/zh`** —— 文件名镜像，语义一致
- 不要把计划中的功能写成已支持
- 写文档前先根据当前代码核对命令和路径

---

## 更新流程

1. 确认当前 CLI help 和默认路径
2. 英文与中文文档同步更新
3. 示例命令必须能对上仓库里的现有配置和脚本
4. 收尾前做验证

### 验证命令

```bash
PYTHONPATH=src python -m somehand.cli --help
PYTHONPATH=src python -m somehand.cli webcam --help
PYTHONPATH=src python -m somehand.cli pico --help
PYTHONPATH=src python -m somehand.cli hc-mocap --help
python scripts/setup/download_assets.py --help
PYTHONPATH=src python scripts/convert_urdf_to_mjcf.py --help
PYTHONPATH=src python scripts/acceptance_check.py --help
pytest -q tests/test_cli.py tests/test_config_model.py tests/test_download_assets.py tests/test_paths.py
```

---

## 与文档相关的代码位置

| 模块 | 位置 |
| --- | --- |
| CLI 解析与分发 | `src/somehand/cli/` |
| 默认路径与资产元数据 | `src/somehand/paths.py`、`src/somehand/external_assets.py` |
| 配置加载与校验 | `src/somehand/infrastructure/config_loader.py` |
| 运行时与 backend 对接 | `src/somehand/runtime/`、`src/somehand/application/` |
| 集成脚本 | `scripts/` |

---

## 新增或更新手模型

1. 准备或转换 MJCF 资产
2. 在 `configs/retargeting/` 下新增或更新配置
3. 验证配置可加载且运行路径正常
4. 如果对外能力有变化，同步更新中英文文档

---

## 第三方集成

| 集成 | 引导脚本 | 说明 |
| --- | --- | --- |
| **PICO Bridge** | `pico-bridge @ https://github.com/BotRunner64/pico-bridge/releases/download/v0.1.0/pico_bridge-0.1.0-py3-none-any.whl` | `pico` 模式所需 |
| **LinkerHand SDK** | `scripts/setup_linkerhand_sdk.sh` | `real` backend 所需 |

> 上游 SDK 的详细文档仍以上游为准 —— 本仓库只说明这里实际使用的集成路径。
