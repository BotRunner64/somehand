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
pip install -e .
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
| **PICO** / XRoboToolkit 输入 | `bash scripts/setup_xrobotoolkit.sh` | PICO 实时手部追踪 |

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

- 缺少资产？→ [常见问题](troubleshooting.md)
- 要接新手模型？→ [配置说明](configuration.md)
- 要接真机或 PICO？→ [运行模式](runtime-modes.md)
