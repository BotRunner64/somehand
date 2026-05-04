# 常见问题

## 缺少运行时资产

**典型现象：** 找不到 `assets/mjcf/.../model.xml`、`hand_landmarker.task` 不存在、缺少录制文件

**处理方式：**

```bash
python scripts/setup/download_assets.py --only mjcf mediapipe
python scripts/setup/download_assets.py --only examples
```

---

## `--hand both` 配合控制 backend 失败

**原因：** 双手执行仅在 `viewer` 模式下支持。

**处理方式：**

- `--hand both` 时使用 `--backend viewer`
- 目标是 `sim` 或 `real` 时，改用单手配置

---

## `pico` 收不到数据

**检查清单：**

- [ ] 是否已安装 PICO Bridge PC receiver 包
- [ ] 头显端是否已启动 PICO Bridge app 并连接到 PC receiver
- [ ] `--pico-host`、`--pico-port`、`--pico-advertise-ip` 是否匹配当前网络
- [ ] timeout 是否足够长（`--pico-timeout`）

**配置命令：**

```bash
pip install -e .
```

---

## 真机 backend 启动失败

**检查清单：**

- [ ] `third_party/linkerhand-python-sdk` 下是否存在 LinkerHand SDK
- [ ] SDK 的 Python 依赖是否已安装
- [ ] transport 配置是否与设备一致
- [ ] 所选手模的 `model_family` 是否能正确解析

**配置命令：**

```bash
bash scripts/setup_linkerhand_sdk.sh
```

---

## 配置 YAML 被加载器拒绝

**常见原因：**

- 仍然保留了旧版向量 schema 字段（`human_vector_pairs`、`origin_link_names` 等）
- 仍然保留了已移除的 `position_constraints` 或 `pinch`
- `backend` 或 `transport` 名称不合法
- 控制频率或仿真频率不是正数

> 可以对照 `configs/retargeting/` 下的现有文件，以及 [配置说明](configuration.md) 中的 schema 约束。
