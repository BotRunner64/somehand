# 安装

[English](../../en/getting-started/installation.md)

## 环境要求

- Python 3.10 或更高版本
- 能运行 MuJoCo viewer 的桌面环境
- 只有使用 `webcam` 命令时才需要摄像头

Git 仓库和 release wheel 都不包含运行时 MJCF 与追踪模型。安装软件包后需要单独下载。

## 1. 安装 somehand

按使用方式选择一条命令。

### CLI

该方式会安装内置摄像头、视频和 PICO 输入所需依赖：

```bash
pip install "somehand[cli] @ https://github.com/BotRunner64/somehand/releases/download/v0.3.0/somehand-0.3.0-py3-none-any.whl"
```

### 仅使用 Python API

```bash
pip install "somehand @ https://github.com/BotRunner64/somehand/releases/download/v0.3.0/somehand-0.3.0-py3-none-any.whl"
```

### 源码安装

```bash
git clone --recurse-submodules https://github.com/BotRunner64/somehand.git
cd somehand
pip install -e ".[cli]"
```

开发 somehand 时改用 `pip install -e ".[dev]"`。

## 2. 选择资产目录

源码安装默认把资产放在仓库根目录；wheel 安装使用系统用户数据目录，Linux 通常是 `~/.local/share/somehand`。

如需固定位置，请在下载资产或导入 somehand 前设置 `SOMEHAND_HOME`：

```bash
export SOMEHAND_HOME="$HOME/somehand-data"
```

## 3. 下载资产

只下载当前流程需要的内容：

| 使用方式 | 命令 |
| --- | --- |
| 摄像头或视频 | `somehand assets download --only mjcf mediapipe` |
| PICO、hc_mocap 或 landmark API | `somehand assets download --only mjcf` |
| 回放提供的样例 | `somehand assets download --only mjcf examples` |
| 全部资产 | `somehand assets download` |

默认从 ModelScope 下载。HuggingFace 和自定义目录用法见[资产与模型](../reference/assets.md)。

## 4. 可选硬件配置

CLI 安装已经包含 PICO Bridge 支持，但头显端 app 也必须处于运行状态。

LinkerHand 真机 backend 需要它的 SDK。源码安装可运行：

```bash
bash scripts/setup_linkerhand_sdk.sh
```

wheel 安装需要单独安装 SDK，再通过 `--sdk-root` 传入 SDK 目录。

## 5. 验证

先检查命令是否安装成功：

```bash
somehand --help
```

下载 `mjcf` 和 `mediapipe` 后运行：

```bash
somehand webcam
```

摄像头和 MuJoCo 窗口能够打开，机器人手能跟随检测到的右手，就说明安装成功。在摄像头窗口按 `q` 退出。如果左右手识别相反，重新运行并添加 `--swap-hands`。

macOS 需要通过 `mjpython` 启动 viewer：

```bash
mjpython "$(command -v somehand)" webcam
```

接下来阅读 [CLI 教程](../tutorials/cli.md)或 [Python API 教程](../tutorials/api.md)。
