# 资产与模型

运行时资产按需下载，不存放在 Git 中。

## 先下载需要的内容

```bash
# 运行所需的最小集合
python scripts/setup/download_assets.py --only mjcf mediapipe

# 样例录制和参考资产
python scripts/setup/download_assets.py --only examples

# 全部下载
python scripts/setup/download_assets.py
```

| 分组 | 本地路径 |
| --- | --- |
| `mjcf` | `assets/mjcf/` |
| `mediapipe` | `assets/models/hand_landmarker.task` |
| `examples` | `assets/` 和 `recordings/` |

默认来源是 ModelScope 仓库 `BingqianWu/somehand-assets`。如需 HuggingFace：

```bash
python scripts/setup/download_assets.py --source huggingface --repo-id 12e21/somehand-assets
```

---

## 查看模型覆盖

以 `configs/retargeting/{left,right,bihand}` 为准。当前配置族包括 LinkerHand、Inspire、Dex5、DexHand021、OmniHand、Revo2、RoHand、Sharpa Wave 和 Wuji Hand。

覆盖说明：

- `left/` 和 `right/` 不完全对称
- `bihand/` 只包含已提交的成对配置
- 真机 backend 的实际支持范围比配置文件覆盖更窄

---

## URDF 转 MJCF

```bash
PYTHONPATH=src python scripts/convert_urdf_to_mjcf.py --urdf path/to/model.urdf --output assets/mjcf/my_hand
```

转换完成后，把生成的运行时资产放到外部资产仓，再在 `configs/retargeting/` 下新增或更新匹配配置，验证通过后再写入文档。
