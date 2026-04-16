"""Convert URDF hand models to MJCF for use with MuJoCo/Mink."""

import os
import re
import shutil
import tempfile
import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def _find_leaf_bodies(worldbody: ET.Element) -> list[str]:
    """Find leaf bodies (no child bodies) in MJCF XML - these are fingertips."""
    leaf_bodies = []

    def _walk(elem):
        child_bodies = elem.findall("body")
        if not child_bodies and elem.tag == "body":
            name = elem.get("name", "")
            if name:
                leaf_bodies.append(name)
        for child in child_bodies:
            _walk(child)

    _walk(worldbody)
    return leaf_bodies


def _body_world_rotation(data, body_id: int) -> np.ndarray:
    return np.array(data.xmat[body_id], copy=True).reshape(3, 3)


def _quat_to_matrix(mujoco_module, quat: np.ndarray) -> np.ndarray:
    matrix = np.zeros(9, dtype=np.float64)
    mujoco_module.mju_quat2Mat(matrix, quat)
    return matrix.reshape(3, 3)


def _mesh_vertices_in_body_frame(model, geom_id: int) -> np.ndarray:
    import mujoco

    mesh_id = int(model.geom_dataid[geom_id])
    start = int(model.mesh_vertadr[mesh_id])
    count = int(model.mesh_vertnum[mesh_id])
    vertices = np.array(model.mesh_vert[start:start + count], copy=True)
    rotation = _quat_to_matrix(mujoco, model.geom_quat[geom_id])
    return vertices @ rotation.T + model.geom_pos[geom_id]


def _select_tip_surface_point(vertices: np.ndarray, *, band_thickness: float = 0.0015) -> np.ndarray:
    centered = vertices - vertices.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    projection = vertices @ axis
    if abs(float(projection.max())) < abs(float(projection.min())):
        axis = -axis
        projection = -projection
    band_vertices = vertices[projection >= float(projection.max()) - band_thickness]
    centroid = band_vertices.mean(axis=0)
    closest = int(np.argmin(np.linalg.norm(band_vertices - centroid, axis=1)))
    return band_vertices[closest]


def _compute_fingertip_offsets(model, leaf_body_names: list[str]) -> dict[str, list[float]]:
    """Compute fingertip offsets in local body frame from mesh surface points."""
    import mujoco

    offsets = {}

    for bname in leaf_body_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
        if body_id < 0:
            offsets[bname] = [0.0, 0.0, 0.02]
            continue

        mesh_vertices: list[np.ndarray] = []
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid] != body_id:
                continue
            if int(model.geom_type[gid]) != int(mujoco.mjtGeom.mjGEOM_MESH):
                continue
            mesh_vertices.append(_mesh_vertices_in_body_frame(model, gid))

        if mesh_vertices:
            offsets[bname] = _select_tip_surface_point(np.vstack(mesh_vertices)).tolist()
        else:
            offsets[bname] = [0.0, 0.0, 0.02]

    return offsets


def _select_fingertip_bodies(model, leaf_body_names: list[str], tip_offsets: dict[str, list[float]]) -> list[str]:
    """Collapse duplicate terminal bodies that belong to the same finger.

    Some URDFs export several sibling leaf bodies near a fingertip (sensor shell,
    connector, distal cap). We only want the farthest candidate for each lateral
    finger slot.
    """
    import mujoco

    if len(leaf_body_names) <= 5:
        return leaf_body_names

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    tip_points: list[np.ndarray] = []
    valid_bodies: list[str] = []
    tip_distances: list[float] = []
    for body_name in leaf_body_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            continue
        world_tip = data.xpos[body_id] + _body_world_rotation(data, body_id) @ np.asarray(
            tip_offsets.get(body_name, [0.0, 0.0, 0.02]),
            dtype=np.float64,
        )
        tip_points.append(world_tip)
        valid_bodies.append(body_name)
        tip_distances.append(float(np.linalg.norm(world_tip)))

    if len(valid_bodies) <= 5:
        return valid_bodies

    def _group_key(body_name: str) -> str:
        tokens = [token for token in re.split(r"[_\\-]+", body_name.lower()) if token]
        filtered = [
            token for token in tokens
            if token not in {"left", "right", "l", "r", "link", "body"}
            and not token.isdigit()
        ]
        return filtered[0] if filtered else body_name.lower()

    grouped: dict[str, list[int]] = {}
    for index, body_name in enumerate(valid_bodies):
        grouped.setdefault(_group_key(body_name), []).append(index)

    duplicate_groups = [indices for indices in grouped.values() if len(indices) > 1]
    if duplicate_groups and 4 <= len(grouped) <= 6:
        chosen_indices = [
            max(indices, key=lambda index: tip_distances[index])
            for indices in grouped.values()
        ]
        return [valid_bodies[index] for index in chosen_indices]

    tip_points_array = np.asarray(tip_points, dtype=np.float64)
    centered = tip_points_array - tip_points_array.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    major_axis = vh[0]
    lateral_basis = vh[1:3]
    longitudinal = centered @ major_axis
    lateral = centered @ lateral_basis.T

    lateral_span = float(np.max(np.linalg.norm(lateral, axis=1))) if len(lateral) else 0.0
    merge_threshold = max(0.008, 0.18 * lateral_span)

    kept_indices: list[int] = []
    for candidate_index in np.argsort(longitudinal)[::-1]:
        if any(
            np.linalg.norm(lateral[candidate_index] - lateral[kept_index]) < merge_threshold
            for kept_index in kept_indices
        ):
            continue
        kept_indices.append(int(candidate_index))

    if len(kept_indices) < 2:
        return valid_bodies
    return [valid_bodies[index] for index in kept_indices]


def _find_all_joints(root: ET.Element) -> list[dict]:
    """Extract all joint elements with their attributes from MJCF XML."""
    joints = []
    for joint in root.iter("joint"):
        name = joint.get("name", "")
        jrange = joint.get("range", "")
        if name and jrange:
            joints.append({"name": name, "range": jrange})
    return joints


def _extract_mimic_joints(root: ET.Element) -> dict[str, dict[str, float | str]]:
    mimic_joints: dict[str, dict[str, float | str]] = {}
    for joint_elem in root.findall("joint"):
        name = joint_elem.get("name")
        mimic_elem = joint_elem.find("mimic")
        if not name or mimic_elem is None:
            continue
        parent_joint = mimic_elem.get("joint")
        if not parent_joint:
            raise ValueError(f"mimic joint {name} is missing the source joint")
        mimic_joints[name] = {
            "joint": parent_joint,
            "multiplier": float(mimic_elem.get("multiplier", "1")),
            "offset": float(mimic_elem.get("offset", "0")),
        }
    return mimic_joints


def _load_rohand_math_module(urdf_path: Path):
    module_path = None
    for parent in (urdf_path.parent, *urdf_path.parents):
        candidate = parent / "scripts" / "FingerMathURDF.py"
        if candidate.exists():
            module_path = candidate
            break
    if module_path is None:
        return None
    spec = importlib.util.spec_from_file_location("somehand_rohand_math", module_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fit_joint_polynomial(samples_x: np.ndarray, samples_y: np.ndarray) -> list[float]:
    coefficients = np.polyfit(samples_x, samples_y, deg=4)
    return [float(value) for value in coefficients[::-1]]


def _build_rohand_couplings(urdf_path: Path) -> dict[str, dict[str, object]]:
    rohand_math = _load_rohand_math_module(urdf_path)
    if rohand_math is None:
        return {}

    couplings: dict[str, dict[str, object]] = {}

    finger_joint_sets = {
        1: ("if_slider_link", ["if_slider_abpart_link", "if_proximal_link", "if_distal_link", "if_connecting_link"]),
        2: ("mf_slider_link", ["mf_slider_abpart_link", "mf_proximal_link", "mf_distal_link", "mf_connecting_link"]),
        3: ("rf_slider_link", ["rf_slider_abpart_link", "rf_proximal_link", "rf_distal_link", "rf_connecting_link"]),
        4: ("lf_slider_link", ["lf_slider_abpart_link", "lf_proximal_link", "lf_distal_link", "lf_connecting_link"]),
    }

    slider_range = np.linspace(0.0, 0.019, num=32, dtype=np.float64)
    for finger_id, (source_joint, passive_joints) in finger_joint_sets.items():
        sampled_angles = np.asarray(
            [rohand_math.HAND_FingerPosToAngle(finger_id, float(position)) for position in slider_range],
            dtype=np.float64,
        )
        for joint_index, joint_name in enumerate(passive_joints):
            couplings[joint_name] = {
                "joint": source_joint,
                "polycoef": _fit_joint_polynomial(slider_range, sampled_angles[:, joint_index]),
            }

    thumb_slider_range = np.linspace(0.0, 0.010, num=32, dtype=np.float64)
    thumb_angles = np.asarray(
        [rohand_math.HAND_FingerPosToAngle(0, float(position)) for position in thumb_slider_range],
        dtype=np.float64,
    )
    for joint_index, joint_name in enumerate(("th_proximal_link", "th_connecting_link", "th_distal_link")):
        couplings[joint_name] = {
            "joint": "th_slider_link",
            "polycoef": _fit_joint_polynomial(thumb_slider_range, thumb_angles[:, joint_index]),
        }

    return couplings


def _infer_hand_side(urdf_path: Path, hand_name: str) -> str | None:
    tokens = f"{urdf_path.stem}-{hand_name}".lower()
    if "right" in tokens or re.search(r"(?:^|[_-])r(?:$|[_-])", tokens):
        return "right"
    if "left" in tokens or re.search(r"(?:^|[_-])l(?:$|[_-])", tokens):
        return "left"
    return None


def _build_hand_frame(
    origin: np.ndarray,
    primary: np.ndarray,
    lateral_a: np.ndarray,
    lateral_b: np.ndarray,
) -> np.ndarray:
    y_axis = primary - origin
    y_axis = y_axis / np.linalg.norm(y_axis)
    x_axis = lateral_a - lateral_b
    x_axis = x_axis - y_axis * np.dot(x_axis, y_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    return np.stack([x_axis, y_axis, z_axis], axis=1)


def _canonical_hand_frame(hand_side: str) -> np.ndarray:
    from somehand.acceptance import mirror_pose_to_left, synthetic_hand_pose
    from somehand.domain.preprocessing import preprocess_landmarks

    pose = synthetic_hand_pose("open")
    if hand_side == "left":
        pose = mirror_pose_to_left(pose)
    landmarks = preprocess_landmarks(pose, hand_side=hand_side)
    return _build_hand_frame(
        landmarks[9],
        landmarks[12],
        landmarks[5],
        landmarks[13],
    )


def _semantic_model_point(model, data, *, hand_side: str, name: str, obj_type) -> np.ndarray:
    from somehand.infrastructure.model_name_resolver import ModelNameResolver
    import mujoco

    resolver = ModelNameResolver(model, hand_side=hand_side)
    resolved_name = resolver.resolve(name, obj_type=obj_type, role="Hand-frame alignment")
    point_id = mujoco.mj_name2id(model, obj_type, resolved_name)
    if obj_type == mujoco.mjtObj.mjOBJ_SITE:
        return data.site_xpos[point_id].copy()
    return data.xpos[point_id].copy()


def _compute_hand_root_quat(model_path: Path, *, hand_side: str) -> str | None:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    try:
        middle_base = _semantic_model_point(
            model,
            data,
            hand_side=hand_side,
            name="middle_base",
            obj_type=mujoco.mjtObj.mjOBJ_BODY,
        )
        middle_tip = _semantic_model_point(
            model,
            data,
            hand_side=hand_side,
            name="middle_tip",
            obj_type=mujoco.mjtObj.mjOBJ_SITE,
        )
        index_base = _semantic_model_point(
            model,
            data,
            hand_side=hand_side,
            name="index_base",
            obj_type=mujoco.mjtObj.mjOBJ_BODY,
        )
        ring_base = _semantic_model_point(
            model,
            data,
            hand_side=hand_side,
            name="ring_base",
            obj_type=mujoco.mjtObj.mjOBJ_BODY,
        )
        model_frame = _build_hand_frame(
            middle_base,
            middle_tip,
            index_base,
            ring_base,
        )
    except Exception:
        return None

    canonical_frame = _canonical_hand_frame(hand_side)
    rotation = canonical_frame @ model_frame.T
    quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, rotation.reshape(-1))
    return " ".join(f"{value:.6f}" for value in quat)


def _wrap_worldbody_with_hand_root(worldbody: ET.Element, *, quat: str) -> None:
    existing = worldbody.find("./body[@name='hand_root']")
    if existing is not None:
        existing.set("quat", quat)
        return

    hand_root = ET.Element("body", {"name": "hand_root", "quat": quat})
    existing_children = list(worldbody)
    for child in existing_children:
        worldbody.remove(child)
        hand_root.append(child)
    worldbody.append(hand_root)


def _find_package_root(urdf_path: Path, package_name: str) -> Path:
    for parent in (urdf_path.parent, *urdf_path.parents):
        if parent.name == package_name:
            return parent
        package_xml = parent / "package.xml"
        if package_xml.exists():
            tree = ET.parse(package_xml)
            package_root = tree.getroot()
            declared_name = package_root.findtext("name", default="").strip()
            if declared_name == package_name:
                return parent
    raise FileNotFoundError(
        f"Could not resolve package://{package_name}/... from URDF {urdf_path}"
    )


def _find_relative_path_below(root: Path, relative_path: Path) -> Path | None:
    direct_path = (root / relative_path).resolve()
    if direct_path.exists():
        return direct_path

    target_parts = relative_path.parts
    for candidate in root.rglob(relative_path.name):
        if candidate.parts[-len(target_parts):] == target_parts:
            return candidate.resolve()
    return None


def _resolve_mesh_path(mesh_ref: str, urdf_path: Path, *, meshdir: str = "") -> Path:
    if mesh_ref.startswith("package://"):
        package_ref = mesh_ref[len("package://") :]
        package_name, _, package_relative = package_ref.partition("/")
        if not package_name or not package_relative:
            raise ValueError(f"Invalid package mesh reference: {mesh_ref}")
        package_root = _find_package_root(urdf_path, package_name)
        relative_path = Path(package_relative)
        resolved = _find_relative_path_below(package_root, relative_path)
        if resolved is not None:
            return resolved
        return (package_root / relative_path).resolve()

    mesh_path = Path(mesh_ref)
    if mesh_path.is_absolute():
        return mesh_path

    candidate_rel_paths: list[Path] = [mesh_path]
    meshdir_path = Path(meshdir) if meshdir else None
    if meshdir_path is not None and mesh_path.parts[: len(meshdir_path.parts)] != meshdir_path.parts:
        candidate_rel_paths.append(meshdir_path / mesh_path)

    for search_root in (urdf_path.parent, *urdf_path.parents):
        for candidate_rel_path in candidate_rel_paths:
            candidate_path = (search_root / candidate_rel_path).resolve()
            if candidate_path.exists():
                return candidate_path
    return (urdf_path.parent / mesh_path).resolve()


_FLOAT_PATTERN = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def _sanitize_limit_attributes(root: ET.Element) -> None:
    for limit_elem in root.iter("limit"):
        for attr in ("lower", "upper", "effort", "velocity"):
            value = limit_elem.get(attr)
            if value is None:
                continue
            try:
                float(value)
            except ValueError:
                match = _FLOAT_PATTERN.search(value)
                if match is None:
                    raise ValueError(f"Invalid numeric value for limit attribute {attr!r}: {value!r}")
                limit_elem.set(attr, match.group(0))


def _prepare_urdf_for_mujoco(
    urdf_path: Path,
    *,
    import_meshdir: Path,
) -> tuple[
    str,
    Path,
    dict[str, Path],
    dict[str, str],
    dict[str, dict[str, float | str]],
    dict[str, dict[str, object]],
]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    _sanitize_limit_attributes(root)
    mimic_joints = _extract_mimic_joints(root)
    rohand_couplings = _build_rohand_couplings(urdf_path)
    compiler = root.find("./mujoco/compiler")
    meshdir_attr = ""
    if compiler is not None:
        meshdir_attr = compiler.get("meshdir", "")

    resolved_meshes: list[tuple[ET.Element, Path]] = []
    for mesh_elem in root.iter("mesh"):
        filename = mesh_elem.get("filename")
        if not filename:
            continue
        resolved_path = _resolve_mesh_path(filename, urdf_path, meshdir=meshdir_attr)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Referenced mesh not found: {resolved_path}")
        resolved_meshes.append((mesh_elem, resolved_path))

    meshdir = Path(
        os.path.commonpath([str(path.parent) for _, path in resolved_meshes])
    ) if resolved_meshes else urdf_path.parent
    meshdir = meshdir.resolve()

    if compiler is None:
        mujoco_elem = root.find("./mujoco")
        if mujoco_elem is None:
            mujoco_elem = ET.SubElement(root, "mujoco")
        compiler = ET.SubElement(mujoco_elem, "compiler")
    compiler.set("meshdir", f"{import_meshdir.resolve()}/")
    compiler.set("balanceinertia", "true")

    staged_mesh_names: dict[str, Path] = {}
    import_name_to_relative_path: dict[str, str] = {}
    mesh_assets: dict[str, Path] = {}
    for mesh_elem, resolved_path in resolved_meshes:
        relative_path = resolved_path.relative_to(meshdir).as_posix()
        existing = mesh_assets.get(relative_path)
        if existing is not None and existing != resolved_path:
            raise ValueError(
                f"Mesh relative-path collision for {relative_path}: {existing} vs {resolved_path}"
            )
        mesh_assets[relative_path] = resolved_path
        staged_name = resolved_path.name
        duplicate_index = 1
        while staged_name in staged_mesh_names and staged_mesh_names[staged_name] != resolved_path:
            staged_name = f"{resolved_path.stem}_{duplicate_index}{resolved_path.suffix}"
            duplicate_index += 1
        staged_mesh_names[staged_name] = resolved_path
        import_name_to_relative_path[staged_name] = relative_path
        shutil.copy2(resolved_path, import_meshdir / staged_name)
        mesh_elem.set("filename", staged_name)

    return (
        ET.tostring(root, encoding="unicode"),
        meshdir,
        mesh_assets,
        import_name_to_relative_path,
        mimic_joints,
        rohand_couplings,
    )


def convert_urdf_to_mjcf(
    urdf_path: str,
    output_dir: str,
    hand_name: str | None = None,
) -> str:
    """Convert a URDF file to MJCF with actuators and fingertip sites.

    Args:
        urdf_path: Path to the source URDF file.
        output_dir: Directory to write the output MJCF and meshes.
        hand_name: Optional name for the model. Defaults to URDF filename stem.

    Returns:
        Path to the generated MJCF model.xml file.
    """
    import mujoco

    urdf_path = Path(urdf_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if hand_name is None:
        hand_name = urdf_path.stem

    mesh_dst = output_dir / "meshes"
    if mesh_dst.exists():
        shutil.rmtree(mesh_dst)
    mesh_dst.mkdir(parents=True, exist_ok=True)

    # Load URDF via MuJoCo. We resolve vendor-specific mesh references up
    # front, then stage them into a flat temporary import directory because
    # MuJoCo's URDF importer drops nested mesh subdirectories like
    # collision/foo.stl when resolving assets.
    with tempfile.TemporaryDirectory(prefix="somehand-mujoco-import-") as import_tmp:
        (
            urdf_text,
            _,
            resolved_meshes,
            import_name_to_relative_path,
            mimic_joints,
            rohand_couplings,
        ) = _prepare_urdf_for_mujoco(
            urdf_path,
            import_meshdir=Path(import_tmp),
        )
        model = mujoco.MjModel.from_xml_string(urdf_text)

    for relative_path, source_path in resolved_meshes.items():
        destination = mesh_dst / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
    tmp_xml = output_dir / "_converted.xml"
    mujoco.mj_saveLastXML(str(tmp_xml), model)

    # Post-process: add actuators and fingertip sites
    tree = ET.parse(tmp_xml)
    root = tree.getroot()

    # Fix mesh directory
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("meshdir", "meshes/")
    compiler.set("balanceinertia", "true")

    asset = root.find("asset")
    if asset is not None:
        for mesh_elem in asset.findall("mesh"):
            filename = mesh_elem.get("file")
            if filename:
                mesh_elem.set("file", import_name_to_relative_path.get(filename, Path(filename).as_posix()))

    # Collect joints
    joints = _find_all_joints(root)

    # Add actuators if not present
    actuator_elem = root.find("actuator")
    if actuator_elem is None:
        actuator_elem = ET.SubElement(root, "actuator")
    existing_actuators = {a.get("joint") for a in actuator_elem}
    for joint in joints:
        if joint["name"] in mimic_joints:
            continue
        if joint["name"] not in existing_actuators:
            ET.SubElement(
                actuator_elem,
                "position",
                name=f"act_{joint['name']}",
                joint=joint["name"],
                ctrlrange=joint["range"],
                kp="10",
            )

    if mimic_joints:
        equality_elem = root.find("equality")
        if equality_elem is None:
            equality_elem = ET.SubElement(root, "equality")
        existing_equalities = {
            (item.get("joint1"), item.get("joint2"))
            for item in equality_elem.findall("joint")
        }
        for joint_name, mimic_data in mimic_joints.items():
            source_joint = str(mimic_data["joint"])
            relation = (joint_name, source_joint)
            if relation in existing_equalities:
                continue
            offset = float(mimic_data["offset"])
            multiplier = float(mimic_data["multiplier"])
            ET.SubElement(
                equality_elem,
                "joint",
                joint1=joint_name,
                joint2=source_joint,
                polycoef=f"{offset:.8g} {multiplier:.8g} 0 0 0",
            )

    if rohand_couplings:
        equality_elem = root.find("equality")
        if equality_elem is None:
            equality_elem = ET.SubElement(root, "equality")
        existing_equalities = {
            (item.get("joint1"), item.get("joint2"))
            for item in equality_elem.findall("joint")
        }
        for joint_name, coupling in rohand_couplings.items():
            source_joint = str(coupling["joint"])
            relation = (joint_name, source_joint)
            if relation in existing_equalities:
                continue
            polycoef = " ".join(f"{float(value):.8g}" for value in coupling["polycoef"])
            ET.SubElement(
                equality_elem,
                "joint",
                joint1=joint_name,
                joint2=source_joint,
                polycoef=polycoef,
            )

    # Add fingertip sites on leaf bodies at actual fingertip positions.
    # We compute positions by loading the model and finding geom extents.
    leaf_bodies = _find_leaf_bodies(root.find(".//worldbody"))
    tip_offsets = _compute_fingertip_offsets(model, leaf_bodies)
    fingertip_bodies = _select_fingertip_bodies(model, leaf_bodies, tip_offsets)

    if fingertip_bodies:
        for body_elem in root.iter("body"):
            bname = body_elem.get("name", "")
            if bname in fingertip_bodies:
                existing_sites = [s.get("name") for s in body_elem.findall("site")]
                site_name = f"{bname}_tip"
                if site_name not in existing_sites:
                    offset = tip_offsets.get(bname, [0, 0, 0.02])
                    ET.SubElement(
                        body_elem,
                        "site",
                        name=site_name,
                        pos=f"{offset[0]:.5f} {offset[1]:.5f} {offset[2]:.5f}",
                        size="0.004",
                        rgba="1 0 0 1",
                    )

    provisional_model_path = output_dir / "_with_sites.xml"
    tree.write(str(provisional_model_path), xml_declaration=True, encoding="unicode")

    hand_side = _infer_hand_side(urdf_path, hand_name)
    if hand_side is not None:
        root_quat = _compute_hand_root_quat(provisional_model_path, hand_side=hand_side)
        if root_quat is not None:
            worldbody = root.find(".//worldbody")
            if worldbody is not None:
                _wrap_worldbody_with_hand_root(worldbody, quat=root_quat)

    # Write final model
    model_path = output_dir / "model.xml"
    tree.write(str(model_path), xml_declaration=True, encoding="unicode")
    tmp_xml.unlink()
    provisional_model_path.unlink()

    # Validate by loading
    mujoco.MjModel.from_xml_path(str(model_path))

    # Print summary
    print(f"Converted: {urdf_path.name} -> {model_path}")
    print(f"  Joints ({len(joints)}):")
    for j in joints:
        print(f"    {j['name']}  range=[{j['range']}]")
    print(f"  Fingertip sites ({len(fingertip_bodies)}):")
    for b in fingertip_bodies:
        print(f"    {b}_tip (on body '{b}')")

    return str(model_path)
