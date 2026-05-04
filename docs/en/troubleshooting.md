# Troubleshooting

## Missing Runtime Assets

**Symptoms:** `assets/mjcf/.../model.xml` not found, `hand_landmarker.task` not found, missing recordings

**Fix:**

```bash
python scripts/setup/download_assets.py --only mjcf mediapipe
python scripts/setup/download_assets.py --only examples
```

---

## `--hand both` Fails with Control Backend

**Cause:** Bi-hand execution is only supported in `viewer` mode.

**Fix:**

- Use `--backend viewer` for `--hand both`
- Use single-hand configs when targeting `sim` or `real`

---

## `pico` Does Not Receive Frames

**Checklist:**

- [ ] The PICO Bridge PC receiver package is installed
- [ ] The PICO Bridge headset app is running and connected to the PC receiver
- [ ] `--pico-host`, `--pico-port`, and `--pico-advertise-ip` match your network setup
- [ ] Timeout is long enough (`--pico-timeout`)

**Setup command:**

```bash
pip install -e .
```

---

## Real Backend Cannot Start

**Checklist:**

- [ ] LinkerHand SDK exists at `third_party/linkerhand-python-sdk`
- [ ] SDK Python dependencies are installed
- [ ] Transport settings match your device setup
- [ ] `model_family` resolves correctly for the selected hand

**Setup command:**

```bash
bash scripts/setup_linkerhand_sdk.sh
```

---

## Config Loader Rejects a YAML File

**Common causes:**

- Legacy vector schema keys still present (`human_vector_pairs`, `origin_link_names`, etc.)
- Removed sections still present (`position_constraints`, `pinch`)
- Invalid `backend` or `transport` values
- Non-positive control or simulation rate

> Compare against existing files under `configs/retargeting/` and see [Configuration](configuration.md) for schema details.
