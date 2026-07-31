# 代码结构

[English](../../en/reference/code-structure.md)

## 运行时数据流

```text
摄像头 / 视频 / PICO / hc_mocap / 录制
                  │
                  ▼
         HandFrame 或 BiHandFrame
                  │
                  ▼
  RetargetingEngine 或 BiHandRetargetingEngine
                  │
                  ▼
RetargetingStepResult 或 BiHandRetargetingResult
                  │
                  ▼
       viewer / 录制 / 仿真 / 真机
```

输入适配器先把数据统一为 domain frame。application engine 预处理 landmark、更新求解目标并返回机器人关节位置；runtime sink 再显示、保存、仿真或把结果发送到硬件。

## Python 包

核心代码位于 `src/somehand/`：

| 路径 | 职责 |
| --- | --- |
| `domain/` | 数据模型、配置 dataclass、手别规则和纯 landmark 预处理，不做设备 I/O。 |
| `application/` | 单手/双手 engine 和 session 编排。 |
| `infrastructure/` | YAML 加载、MuJoCo 手模型、重定向求解器、输入输出适配器、artifact 和硬件控制器。 |
| `runtime/` | 运行时校验、输入转换/采样、viewer 行为和公开 runtime 适配器导出。 |
| `cli/` | 参数解析、命令分发，以及 source、engine、sink、controller 的组装。 |
| `api.py` | 供外部程序嵌入 somehand 的稳定导入入口。 |
| `paths.py`、`external_assets.py` | 配置根目录、数据根目录、资产清单和路径解析。 |

新增功能应保持这些边界：domain 类型不依赖 I/O；适配器放在 infrastructure；用例编排放在 application；CLI 专用行为放在 `cli/` 或 `runtime/`。

## 仓库布局

| 路径 | 内容 |
| --- | --- |
| `configs/retargeting/` | 已提交模型配置的唯一事实来源。 |
| `tests/` | Pytest 测试，文件名通常与功能对应。 |
| `scripts/` | 配置、转换、验收、录制和渲染工具。 |
| `docs/en/`、`docs/zh/` | 文件名镜像的中英文文档。 |
| `third_party/` | 第三方 SDK 和 Git submodule。 |
| `assets/`、`recordings/` | 下载或生成的本地数据，不提交大文件。 |

release 构建会把 `configs/retargeting/` 复制进 wheel，但不会复制运行时资产或录制。

## 修改位置

| 修改内容 | 从这里开始 |
| --- | --- |
| 新增 domain 类型或校验规则 | `src/somehand/domain/` |
| 修改单步重定向行为 | `src/somehand/application/engine.py` 或 `bihand_engine.py` |
| 新增输入、viewer、controller 或文件适配器 | `src/somehand/infrastructure/` 及对应 runtime 导出 |
| 新增 CLI 命令或参数 | `src/somehand/cli/parser.py`，然后是 `commands.py` |
| 新增手模型 | `configs/retargeting/` 和外部 MJCF 资产 |
| 修改稳定嵌入接口 | `src/somehand/api.py` |

## 验证

先运行修改区域的测试，再运行完整测试和 lint：

```bash
pytest -q
ruff check src tests
```

配置、资产、路径或文档变更可先运行：

```bash
pytest -q \
    tests/test_docs_structure.py \
    tests/test_config_model.py \
    tests/test_download_assets.py \
    tests/test_paths.py
```

文档修改必须同步更新 `docs/en` 和 `docs/zh`，保持文件名镜像、含义一致。
