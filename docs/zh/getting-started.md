# 快速开始

## 环境要求

- **Python >=3.10**
- 能运行 MuJoCo 的环境
- 运行时资产需要单独下载（不在 Git 仓库中）

---

## 1. 安装

```bash
pip install "somehand[cli] @ https://github.com/BotRunner64/somehand/releases/download/v0.3.0/somehand-0.3.0-py3-none-any.whl"
```

如果需要可编辑的源码安装：

```bash
git clone --recurse-submodules https://github.com/BotRunner64/somehand.git
cd somehand
pip install -e ".[cli]"
```

验证：

```bash
somehand --help
```

## 2. 下载运行时资产

```bash
somehand assets download --only mjcf mediapipe
```

其他常见变体：

| 命令 | 下载内容 |
| --- | --- |
| `somehand assets download` | 全部 |
| `somehand assets download --only examples` | 样例录制和参考资产 |
| `somehand assets download --source huggingface --repo-id 12e21/somehand-assets` | 从 HuggingFace 下载 |

默认资产仓：

- **ModelScope**：`BingqianWu/somehand-assets`
- **HuggingFace**：`12e21/somehand-assets`

源码检出默认把资产放在仓库根目录；wheel 安装默认使用系统的用户数据目录。可以设置 `SOMEHAND_HOME` 固定位置：

```bash
export SOMEHAND_HOME="$HOME/somehand-data"
somehand assets download --only mjcf mediapipe examples
```

## 3.（可选）SDK 配置

仅在特定输入/backend 模式下需要：

| 集成 | 配置命令 | 使用场景 |
| --- | --- | --- |
| **LinkerHand** 真机 backend | 源码检出运行 `bash scripts/setup_linkerhand_sdk.sh`；wheel 安装用 `--sdk-root` 指向单独安装的 SDK | 控制 LinkerHand 真机硬件 |
| **PICO Bridge** 输入 | 通过 `somehand[cli]` extra 安装 | PICO 实时手部追踪 |

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
EXAMPLE_ROOT="${SOMEHAND_HOME:-$HOME/somehand-data}"
somehand assets download --only examples --data-root "$EXAMPLE_ROOT"
somehand replay --recording "$EXAMPLE_ROOT/recordings/pico_right.pkl"
```

**导出录制为视频：**

```bash
somehand dump-video \
    --recording "$EXAMPLE_ROOT/recordings/pico_right.pkl" \
    --output recordings/pico_right_replay.mp4
```

---

## 下一步

- 需要资产或模型？→ [资产与模型](assets-and-models.md)
- 要接新手模型？→ [配置说明](configuration.md)
- 需要终端命令？→ [CLI 用法](runtime-modes.md)
- 要在 Python 中嵌入？→ [API 用法](api.md)
