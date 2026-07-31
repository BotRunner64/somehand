# 资产与模型

[English](../../en/reference/assets.md)

运行时资产需要单独下载，不提交到 Git，也不包含在 release wheel 中。

## 资产分组

| 分组 | 安装路径 | 用途 |
| --- | --- | --- |
| `mjcf` | `assets/mjcf/` | 所有 retargeting engine 和 viewer |
| `mediapipe` | `assets/models/hand_landmarker.task` | `webcam` 和 `video` 输入 |
| `examples` | `assets/` 和 `recordings/` | 样例回放与参考文件 |

下载一个或多个分组：

```bash
somehand assets download --only mjcf mediapipe
somehand assets download --only examples
somehand assets download
```

最后一条命令会下载全部分组。

## 下载来源

默认仓库是：

- ModelScope：`BingqianWu/somehand-assets`
- HuggingFace：`12e21/somehand-assets`

未指定时使用 ModelScope，切换来源的方法是：

```bash
somehand assets download \
    --source huggingface \
    --only mjcf examples
```

使用 `--repo-id <owner/repo>` 覆盖仓库，使用 `--cache-dir <path>` 覆盖下载缓存目录。

## 本地数据根目录

默认位置取决于安装方式：

| 安装方式 | 数据根目录 |
| --- | --- |
| 源码安装 | 仓库根目录 |
| Linux wheel | `$XDG_DATA_HOME/somehand` 或 `~/.local/share/somehand` |
| macOS wheel | `~/Library/Application Support/somehand` |
| Windows wheel | `%LOCALAPPDATA%\somehand` |

请在下载和运行前设置同一个固定位置：

```bash
export SOMEHAND_HOME="$HOME/somehand-data"
somehand assets download --only mjcf mediapipe
somehand webcam
```

`--data-root <path>` 只改变单次下载目的地，运行时不会自动继承该参数。因此运行时应把 `SOMEHAND_HOME` 设置为相同路径。

## 已支持配置族

当前已提交配置覆盖：

- LinkerHand L6、L10、L20、L20 Pro、L21、L25、L30、LHG20、O6、O7 和 T12
- DexRobot DexHand021 与 Unitree Dex5
- Inspire DFQ 与 FTP
- AGIBOT OmniHand、BrainCo Revo2、OYMotion RoHand、Sharpa Wave 01 与 Wuji Hand

准确范围以 `configs/retargeting/{left,right,bihand}` 中实际存在的文件为准。左右覆盖并不完全对称：L20 Pro、L30 和 T12 当前只有右手配置，LHG20 当前只有左手配置；只有左右配置都存在的模型才提供双手配置。

配置覆盖表示该模型可以用于重定向和可视化。真机 backend 当前面向支持的 LinkerHand 硬件，覆盖范围更窄。

## URDF 转换

在源码仓库中运行：

```bash
PYTHONPATH=src python scripts/convert_urdf_to_mjcf.py \
    --urdf path/to/model.urdf \
    --output assets/mjcf/my_hand
```

生成的 MJCF 和 mesh 应放在外部资产仓，不要提交到 Git。然后在 `configs/retargeting/` 添加对应配置，验证通过后再把模型列为支持。
