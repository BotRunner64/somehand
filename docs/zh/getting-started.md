# 快速开始

## 环境要求

- **Python >=3.10**
- 已初始化 Git submodule
- 能运行 MuJoCo 的环境
- 运行时资产需要单独下载（不在 Git 仓库中）

---

## 1. 安装

```bash
git submodule update --init --recursive
pip install -e ".[cli]"
```

验证：

```bash
somehand --help
```

## 2. 下载运行时资产

```bash
python scripts/setup/download_assets.py --only mjcf mediapipe
```

其他常见变体：

| 命令 | 下载内容 |
| --- | --- |
| `python scripts/setup/download_assets.py` | 全部 |
| `python scripts/setup/download_assets.py --only examples` | 样例录制和参考资产 |
| `python scripts/setup/download_assets.py --source huggingface --repo-id 12e21/somehand-assets` | 从 HuggingFace 下载 |

默认资产仓：

- **ModelScope**：`BingqianWu/somehand-assets`
- **HuggingFace**：`12e21/somehand-assets`

## 3.（可选）SDK 配置

仅在特定输入/backend 模式下需要：

| 集成 | 配置命令 | 使用场景 |
| --- | --- | --- |
| **LinkerHand** 真机 backend | `bash scripts/setup_linkerhand_sdk.sh` | 控制 LinkerHand 真机硬件 |
| **PICO Bridge** 输入 | 由 `pip install -e ".[cli]"` 通过 release wheel 依赖安装 | PICO 实时手部追踪 |

---

## 首次运行

**摄像头输入** —— 最简单的验证方式：

```bash
somehand webcam
```

在 macOS 上，请通过 `mjpython` 启动 MuJoCo viewer：

```bash
mjpython "$(command -v somehand)" webcam --hand both
```

**回放已有录制：**

```bash
somehand replay --recording recordings/webcam_hand.pkl
```

**导出录制为视频：**

```bash
somehand dump-video \
    --recording recordings/webcam_hand.pkl \
    --output recordings/webcam_hand_replay.mp4
```

---

## 下一步

- 需要资产或模型？→ [资产与模型](assets-and-models.md)
- 要接新手模型？→ [配置说明](configuration.md)
- 需要终端命令？→ [CLI 用法](runtime-modes.md)
- 要在 Python 中嵌入？→ [API 用法](api.md)
