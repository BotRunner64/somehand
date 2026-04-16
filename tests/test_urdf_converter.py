import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from somehand.urdf_converter import _find_package_root, _prepare_urdf_for_mujoco, _resolve_mesh_path


def test_find_package_root_supports_package_xml_name_mismatch(tmp_path):
    package_root = tmp_path / "omnihand_description-omnihandT2_1"
    package_root.mkdir()
    (package_root / "package.xml").write_text(
        "<package><name>omnihand_description</name></package>",
        encoding="utf-8",
    )
    urdf_path = package_root / "assets" / "urdf" / "omnihand_right.urdf"
    urdf_path.parent.mkdir(parents=True)
    urdf_path.write_text("<robot />", encoding="utf-8")

    assert _find_package_root(urdf_path, "omnihand_description") == package_root


def test_resolve_package_mesh_path_searches_inside_package_root(tmp_path):
    package_root = tmp_path / "omnihand_description-omnihandT2_1"
    package_root.mkdir()
    (package_root / "package.xml").write_text(
        "<package><name>omnihand_description</name></package>",
        encoding="utf-8",
    )
    mesh_path = package_root / "assets" / "meshes" / "l_palm.STL"
    mesh_path.parent.mkdir(parents=True)
    mesh_path.write_text("solid", encoding="utf-8")
    urdf_path = package_root / "assets" / "urdf" / "omnihand_left.urdf"
    urdf_path.parent.mkdir(parents=True)
    urdf_path.write_text("<robot />", encoding="utf-8")

    resolved = _resolve_mesh_path("package://omnihand_description/meshes/l_palm.STL", urdf_path)

    assert resolved == mesh_path.resolve()


def test_resolve_mesh_path_searches_ancestor_roots(tmp_path):
    robot_root = tmp_path / "g1_description"
    mesh_path = robot_root / "meshes" / "left_base_link.STL"
    mesh_path.parent.mkdir(parents=True)
    mesh_path.write_text("solid", encoding="utf-8")
    urdf_path = robot_root / "inspire_hand" / "FTP_left_hand.urdf"
    urdf_path.parent.mkdir(parents=True)
    urdf_path.write_text("<robot />", encoding="utf-8")

    resolved = _resolve_mesh_path("meshes/left_base_link.STL", urdf_path, meshdir="meshes")

    assert resolved == mesh_path.resolve()


def test_resolve_mesh_path_uses_compiler_meshdir_for_bare_filenames(tmp_path):
    robot_root = tmp_path / "robot"
    mesh_path = robot_root / "meshes" / "finger.stl"
    mesh_path.parent.mkdir(parents=True)
    mesh_path.write_text("solid", encoding="utf-8")
    urdf_path = robot_root / "urdf" / "hand.urdf"
    urdf_path.parent.mkdir(parents=True)
    urdf_path.write_text("<robot />", encoding="utf-8")

    resolved = _resolve_mesh_path("finger.stl", urdf_path, meshdir="meshes")

    assert resolved == mesh_path.resolve()


def test_prepare_urdf_for_mujoco_stages_nested_meshes_for_import(tmp_path):
    package_root = tmp_path / "omnihand_description-omnihandT2_1"
    package_root.mkdir()
    (package_root / "package.xml").write_text(
        "<package><name>omnihand_description</name></package>",
        encoding="utf-8",
    )
    visual_mesh = package_root / "assets" / "meshes" / "r_thumb_roll.STL"
    visual_mesh.parent.mkdir(parents=True, exist_ok=True)
    visual_mesh.write_text("solid visual", encoding="utf-8")
    collision_mesh = package_root / "assets" / "meshes" / "collision" / "r_thumb_roll_col.STL"
    collision_mesh.parent.mkdir(parents=True, exist_ok=True)
    collision_mesh.write_text("solid collision", encoding="utf-8")
    urdf_path = package_root / "assets" / "urdf_mesh_col" / "omnihand_right.urdf"
    urdf_path.parent.mkdir(parents=True, exist_ok=True)
    urdf_path.write_text(
        """
<robot name="omnihand">
  <mujoco>
    <compiler meshdir="meshes"/>
  </mujoco>
  <link name="base_link"/>
  <joint name="fixed" type="fixed">
    <parent link="base_link"/>
    <child link="thumb_link"/>
  </joint>
  <link name="thumb_link">
    <visual>
      <geometry>
        <mesh filename="package://omnihand_description/meshes/r_thumb_roll.STL"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <mesh filename="package://omnihand_description/meshes/collision/r_thumb_roll_col.STL"/>
      </geometry>
    </collision>
  </link>
</robot>
""".strip(),
        encoding="utf-8",
    )

    import_meshdir = tmp_path / "import_meshes"
    import_meshdir.mkdir()

    urdf_text, meshdir, mesh_assets, import_name_to_relative_path, _, _ = _prepare_urdf_for_mujoco(
        urdf_path,
        import_meshdir=import_meshdir,
    )

    assert meshdir == (package_root / "assets" / "meshes").resolve()
    assert f'meshdir="{import_meshdir.resolve()}/"' in urdf_text
    assert 'filename="r_thumb_roll.STL"' in urdf_text
    assert 'filename="r_thumb_roll_col.STL"' in urdf_text
    assert (import_meshdir / "r_thumb_roll.STL").read_text(encoding="utf-8") == "solid visual"
    assert (import_meshdir / "r_thumb_roll_col.STL").read_text(encoding="utf-8") == "solid collision"
    assert mesh_assets == {
        "r_thumb_roll.STL": visual_mesh.resolve(),
        "collision/r_thumb_roll_col.STL": collision_mesh.resolve(),
    }
    assert import_name_to_relative_path == {
        "r_thumb_roll.STL": "r_thumb_roll.STL",
        "r_thumb_roll_col.STL": "collision/r_thumb_roll_col.STL",
    }
