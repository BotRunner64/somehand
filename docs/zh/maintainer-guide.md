# 维护指南

## 规则

- `README.md` 只保留项目概览、快速运行和文档入口。
- 用户文档按任务组织，参考细节放在任务路径后面。
- `docs/en` 和 `docs/zh` 必须同步更新，文件名镜像，含义对等。
- 只记录当前已验证的行为。

## 编辑文档前

先确认当前 CLI 和路径：

```bash
PYTHONPATH=src python -m somehand.cli --help
PYTHONPATH=src python -m somehand.cli webcam --help
PYTHONPATH=src python -m somehand.cli pico --help
PYTHONPATH=src python -m somehand.cli hc-mocap --help
python - <<'PY'
from somehand.api import RetargetingEngine
print(RetargetingEngine)
PY
python scripts/setup/download_assets.py --help
PYTHONPATH=src python scripts/convert_urdf_to_mjcf.py --help
```

相关代码位置：

| 区域 | 位置 |
| --- | --- |
| CLI 解析 | `src/somehand/cli/` |
| 默认路径和资产元数据 | `src/somehand/paths.py`, `src/somehand/external_assets.py` |
| 配置加载 | `src/somehand/infrastructure/config_loader.py` |
| 运行时 glue | `src/somehand/runtime/`, `src/somehand/application/` |

## 新增公开手模型时

1. 在 Git 外新增或更新运行时资产。
2. 在 `configs/retargeting/` 下新增或更新配置。
3. 验证配置可加载，且对应运行路径可用。
4. 只有验证通过后，才更新双语文档。

相关测试可以跑：

```bash
pytest -q tests/test_docs_structure.py tests/test_config_model.py tests/test_download_assets.py tests/test_paths.py
```
