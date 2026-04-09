#!/usr/bin/env python3
"""Diagnostic script: print thumb body positions and direction vectors at various poses.

Usage:
    python scripts/debug_thumb_direction.py [--config configs/retargeting/right/linkerhand_l20_right.yaml]
"""

import argparse
import numpy as np
import mujoco
from pathlib import Path

np.set_printoptions(precision=4, suppress=True)


def load_model(mjcf_path: str):
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)
    return model, data


def get_body_pos(model, data, name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return data.xpos[bid].copy()


def get_site_pos(model, data, name):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    return data.site_xpos[sid].copy()


def get_body_xmat(model, data, name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return data.xmat[bid].reshape(3, 3).copy()


def direction(a, b):
    v = b - a
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


def print_thumb_state(model, data, prefix=""):
    """Print positions and directions of thumb chain."""
    # Detect body name prefix (some models use "rh_" prefix)
    has_prefix = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rh_thumb_distal") >= 0
    p = "rh_" if has_prefix else ""

    # Try to find thumb bodies
    thumb_bodies = []
    for name in [f"{p}thumb_metacarpals_base1", f"{p}thumb_metacarpals_base2",
                 f"{p}thumb_metacarpals", f"{p}thumb_proximal", f"{p}thumb_distal"]:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            thumb_bodies.append(name)

    tip_name = f"{p}thumb_distal_tip"

    print(f"\n{'='*60}")
    print(f"{prefix}Thumb body world positions:")
    print(f"{'='*60}")
    for name in thumb_bodies:
        pos = get_body_pos(model, data, name)
        xmat = get_body_xmat(model, data, name)
        local_y = xmat[:, 1]  # joint axis direction in world frame
        print(f"  {name:35s} pos={pos}  joint_axis(Y)={local_y}")

    tip_pos = get_site_pos(model, data, tip_name)
    print(f"  {tip_name:35s} pos={tip_pos}")

    # Print finger body positions for comparison
    print(f"\n  Finger reference positions:")
    for finger in ["index", "middle", "ring", "pinky"]:
        for seg in ["proximal"]:
            name = f"{p}{finger}_{seg}"
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                print(f"  {name:35s} pos={data.xpos[bid]}")

    # Direction vectors
    print(f"\n{prefix}Thumb direction vectors (world frame):")
    print(f"-"*60)

    world_pos = np.zeros(3)  # world origin
    base1_pos = get_body_pos(model, data, thumb_bodies[0])
    meta_pos = get_body_pos(model, data, f"{p}thumb_metacarpals")
    prox_pos = get_body_pos(model, data, f"{p}thumb_proximal")
    dist_pos = get_body_pos(model, data, f"{p}thumb_distal")

    print(f"  world -> tip:        {direction(world_pos, tip_pos)}  (opposition vector [0,4])")
    print(f"  base1 -> tip:        {direction(base1_pos, tip_pos)}  ([1,4] redundant)")
    print(f"  base1 -> metacarpals:{direction(base1_pos, meta_pos)}  ([1,2])")
    print(f"  base1 -> proximal:   {direction(base1_pos, prox_pos)}  ([1,3])")
    print(f"  metacarpals -> tip:  {direction(meta_pos, tip_pos)}  ([2,4])")
    print(f"  proximal -> tip:     {direction(prox_pos, tip_pos)}  ([3,4])")
    print(f"  distal -> tip:       {direction(dist_pos, tip_pos)}  (link direction)")

    # Key diagnostic: what are the main axes?
    print(f"\n{prefix}Axis analysis:")
    thumb_dir = direction(base1_pos, tip_pos)
    link_dir = direction(dist_pos, tip_pos)
    print(f"  Overall thumb direction (base1->tip): {thumb_dir}")
    print(f"  Dominant axis: {'X' if abs(thumb_dir[0]) > max(abs(thumb_dir[1]), abs(thumb_dir[2])) else 'Y' if abs(thumb_dir[1]) > abs(thumb_dir[2]) else 'Z'}")
    print(f"  Distal link direction (distal->tip):  {link_dir}")
    print(f"  Dominant axis: {'X' if abs(link_dir[0]) > max(abs(link_dir[1]), abs(link_dir[2])) else 'Y' if abs(link_dir[1]) > abs(link_dir[2]) else 'Z'}")


def test_human_transform():
    """Show what human thumb landmarks map to after transformation."""
    from dex_mujoco.vector_retargeting import preprocess_landmarks, _OPERATOR2ROBOT_RIGHT

    print(f"\n{'='*60}")
    print("Human landmark transformation test")
    print(f"{'='*60}")

    print(f"\n_OPERATOR2ROBOT_RIGHT matrix:")
    print(_OPERATOR2ROBOT_RIGHT)

    # Simulate a right hand with thumb pointing inward (toward other fingers)
    # MediaPipe convention: x=right, y=down, z=toward camera
    # For a right hand palm facing camera:
    landmarks = np.array([
        [0.0, 0.0, 0.0],      # 0: wrist
        [-0.03, -0.01, 0.0],   # 1: thumb_cmc (to the right and slightly up)
        [-0.05, -0.03, 0.0],   # 2: thumb_mcp
        [-0.04, -0.06, 0.0],   # 3: thumb_ip (curling inward)
        [-0.02, -0.08, 0.0],   # 4: thumb_tip (pointing toward middle finger)
        [0.02, -0.06, 0.0],    # 5: index_mcp
        [0.02, -0.10, 0.0],    # 6: index_pip
        [0.02, -0.13, 0.0],    # 7: index_dip
        [0.02, -0.15, 0.0],    # 8: index_tip
        [0.0, -0.07, 0.0],     # 9: middle_mcp
        [0.0, -0.11, 0.0],     # 10: middle_pip
        [0.0, -0.14, 0.0],     # 11: middle_dip
        [0.0, -0.16, 0.0],     # 12: middle_tip
        [-0.02, -0.06, 0.0],   # 13: ring_mcp
        [-0.02, -0.10, 0.0],   # 14: ring_pip
        [-0.02, -0.13, 0.0],   # 15: ring_dip
        [-0.02, -0.15, 0.0],   # 16: ring_tip
        [-0.04, -0.05, 0.0],   # 17: pinky_mcp
        [-0.04, -0.08, 0.0],   # 18: pinky_pip
        [-0.04, -0.10, 0.0],   # 19: pinky_dip
        [-0.04, -0.12, 0.0],   # 20: pinky_tip
    ], dtype=np.float64)

    print("\n--- Thumb pointing inward (toward middle finger) ---")
    transformed = preprocess_landmarks(landmarks, hand_side="right")

    # Key directions
    def dir_vec(a, b):
        v = b - a
        n = np.linalg.norm(v)
        return v / n if n > 1e-8 else v

    print(f"\nTransformed landmark positions (robot frame):")
    names = ["wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
             "index_mcp", "", "", "", "middle_mcp"]
    for i in [0, 1, 2, 3, 4, 5, 9]:
        print(f"  [{i:2d}] {names[i] if i < len(names) else '':15s} {transformed[i]}")

    print(f"\nHuman thumb direction vectors (in robot frame):")
    print(f"  [0,4] wrist -> tip:     {dir_vec(transformed[0], transformed[4])}")
    print(f"  [1,4] cmc -> tip:       {dir_vec(transformed[1], transformed[4])}")
    print(f"  [1,2] cmc -> mcp:       {dir_vec(transformed[1], transformed[2])}")
    print(f"  [3,4] ip -> tip:        {dir_vec(transformed[3], transformed[4])}")
    print(f"  [0,5] wrist -> idx_mcp: {dir_vec(transformed[0], transformed[5])}")
    print(f"  [0,9] wrist -> mid_mcp: {dir_vec(transformed[0], transformed[9])}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/retargeting/right/linkerhand_l20_right.yaml")
    args = parser.parse_args()

    # Load config to get MJCF path
    import yaml
    config_path = Path(args.config)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    mjcf_path = str(config_path.parent / cfg["hand"]["mjcf_path"])
    print(f"Loading model: {mjcf_path}")
    model, data = load_model(mjcf_path)

    # Test 1: Default pose (qpos=0)
    data.qpos[:] = 0
    mujoco.mj_fwdPosition(model, data)
    print_thumb_state(model, data, prefix="[qpos=0] ")

    # Test 2: Thumb cmc_yaw at max (most inward rotation)
    data.qpos[:] = 0
    for i in range(model.nq):
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if jname and "thumb_cmc_yaw" in jname:
            data.qpos[i] = model.jnt_range[i][1]  # max value
            print(f"\n>>> Setting {jname} = {model.jnt_range[i][1]:.3f} (max)")
    mujoco.mj_fwdPosition(model, data)
    print_thumb_state(model, data, prefix="[cmc_yaw=max] ")

    # Test 3: All thumb joints at mid-range
    data.qpos[:] = 0
    for i in range(model.nq):
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if jname and "thumb" in jname:
            mid = (model.jnt_range[i][0] + model.jnt_range[i][1]) / 2
            data.qpos[i] = mid
            print(f"  Setting {jname} = {mid:.3f}")
    mujoco.mj_fwdPosition(model, data)
    print_thumb_state(model, data, prefix="[thumb mid-range] ")

    # Test 4: Human landmark transformation
    try:
        test_human_transform()
    except ImportError:
        print("\n[WARN] Could not import dex_mujoco, skipping human transform test")
        print("       Run with: python -m scripts.debug_thumb_direction")


if __name__ == "__main__":
    main()
