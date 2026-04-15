"""Convert URDF hand models to MJCF for use with MuJoCo/Mink."""

import os
import re
import shutil
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


def _find_package_root(urdf_path: Path, package_name: str) -> Path:
    for parent in (urdf_path.parent, *urdf_path.parents):
        if parent.name == package_name:
            return parent
    raise FileNotFoundError(
        f"Could not resolve package://{package_name}/... from URDF {urdf_path}"
    )


def _resolve_mesh_path(mesh_ref: str, urdf_path: Path) -> Path:
    if mesh_ref.startswith("package://"):
        package_ref = mesh_ref[len("package://") :]
        package_name, _, package_relative = package_ref.partition("/")
        if not package_name or not package_relative:
            raise ValueError(f"Invalid package mesh reference: {mesh_ref}")
        package_root = _find_package_root(urdf_path, package_name)
        return (package_root / package_relative).resolve()

    mesh_path = Path(mesh_ref)
    if mesh_path.is_absolute():
        return mesh_path
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
) -> tuple[str, Path, dict[str, Path], dict[str, dict[str, float | str]]]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    _sanitize_limit_attributes(root)
    mimic_joints = _extract_mimic_joints(root)

    resolved_meshes: list[tuple[ET.Element, Path]] = []
    for mesh_elem in root.iter("mesh"):
        filename = mesh_elem.get("filename")
        if not filename:
            continue
        resolved_path = _resolve_mesh_path(filename, urdf_path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Referenced mesh not found: {resolved_path}")
        resolved_meshes.append((mesh_elem, resolved_path))

    meshdir = Path(
        os.path.commonpath([str(path.parent) for _, path in resolved_meshes])
    ) if resolved_meshes else urdf_path.parent
    meshdir = meshdir.resolve()

    compiler = root.find("./mujoco/compiler")
    if compiler is None:
        mujoco_elem = root.find("./mujoco")
        if mujoco_elem is None:
            mujoco_elem = ET.SubElement(root, "mujoco")
        compiler = ET.SubElement(mujoco_elem, "compiler")
    compiler.set("meshdir", f"{meshdir}/")
    compiler.set("balanceinertia", "true")

    mesh_assets: dict[str, Path] = {}
    for mesh_elem, resolved_path in resolved_meshes:
        relative_path = resolved_path.relative_to(meshdir).as_posix()
        existing = mesh_assets.get(relative_path)
        if existing is not None and existing != resolved_path:
            raise ValueError(
                f"Mesh relative-path collision for {relative_path}: {existing} vs {resolved_path}"
            )
        mesh_assets[relative_path] = resolved_path
        mesh_elem.set("filename", relative_path)

    return ET.tostring(root, encoding="unicode"), meshdir, mesh_assets, mimic_joints


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

    # Load URDF via MuJoCo. We first rewrite every mesh filename to an
    # absolute path so vendor-specific forms like ../meshes/... and
    # package://revo2_description/... both work reliably.
    urdf_text, _, resolved_meshes, mimic_joints = _prepare_urdf_for_mujoco(urdf_path)
    for relative_path, source_path in resolved_meshes.items():
        destination = mesh_dst / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)

    model = mujoco.MjModel.from_xml_string(urdf_text)
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
                mesh_elem.set("file", Path(filename).as_posix())

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

    # Add fingertip sites on leaf bodies at actual fingertip positions.
    # We compute positions by loading the model and finding geom extents.
    leaf_bodies = _find_leaf_bodies(root.find(".//worldbody"))
    tip_offsets = _compute_fingertip_offsets(model, leaf_bodies)

    if leaf_bodies:
        for body_elem in root.iter("body"):
            bname = body_elem.get("name", "")
            if bname in leaf_bodies:
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

    # Write final model
    model_path = output_dir / "model.xml"
    tree.write(str(model_path), xml_declaration=True, encoding="unicode")
    tmp_xml.unlink()

    # Validate by loading
    mujoco.MjModel.from_xml_path(str(model_path))

    # Print summary
    print(f"Converted: {urdf_path.name} -> {model_path}")
    print(f"  Joints ({len(joints)}):")
    for j in joints:
        print(f"    {j['name']}  range=[{j['range']}]")
    print(f"  Fingertip sites ({len(leaf_bodies)}):")
    for b in leaf_bodies:
        print(f"    {b}_tip (on body '{b}')")

    return str(model_path)
