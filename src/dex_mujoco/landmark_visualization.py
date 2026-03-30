"""3D MediaPipe landmark visualization via viser."""

from __future__ import annotations

import numpy as np

_HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)

_LANDMARK_COLORS = np.array(
    [
        [255, 255, 255],
        [255, 170, 90], [255, 170, 90], [255, 170, 90], [255, 170, 90],
        [90, 220, 255], [90, 220, 255], [90, 220, 255], [90, 220, 255],
        [120, 255, 160], [120, 255, 160], [120, 255, 160], [120, 255, 160],
        [255, 230, 110], [255, 230, 110], [255, 230, 110], [255, 230, 110],
        [255, 130, 210], [255, 130, 210], [255, 130, 210], [255, 130, 210],
    ],
    dtype=np.uint8,
)
_BONE_COLORS = np.array(
    [_LANDMARK_COLORS[end] for _, end in _HAND_CONNECTIONS],
    dtype=np.uint8,
)


def _raw_to_viser(landmarks_3d: np.ndarray) -> np.ndarray:
    """Remap raw MediaPipe world landmarks into a Z-up viewer frame."""
    out = np.empty_like(landmarks_3d, dtype=np.float32)
    out[:, 0] = landmarks_3d[:, 0]
    out[:, 1] = -landmarks_3d[:, 2]
    out[:, 2] = -landmarks_3d[:, 1]
    return out


def _local_to_viser(landmarks_3d: np.ndarray) -> np.ndarray:
    """Local retargeting coordinates are already in a robot-aligned frame."""
    return np.asarray(landmarks_3d, dtype=np.float32)


class MediaPipe3DVisualizer:
    """Browser-based 3D visualization for MediaPipe hand landmarks."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8080,
        point_size: float = 0.015,
        space: str = "local",
    ):
        try:
            import viser
        except ImportError as exc:
            raise RuntimeError(
                "viser is not installed. Reinstall project dependencies or run `pip install viser`."
            ) from exc

        if space not in {"raw", "local"}:
            raise ValueError(f"Unsupported visualization space: {space}")

        self.space = space
        label = "MediaPipe 3D (local)" if space == "local" else "MediaPipe 3D (raw)"
        self._server = viser.ViserServer(host=host, port=port, label=label)
        self._server.scene.world_axes.visible = False
        self._server.scene.set_up_direction("+z")
        self._server.initial_camera.position = (0.22, -0.22, 0.16)
        self._server.initial_camera.look_at = (0.0, 0.0, 0.0)

        points = np.zeros((21, 3), dtype=np.float32)
        line_points = np.zeros((len(_HAND_CONNECTIONS), 2, 3), dtype=np.float32)
        line_colors = np.repeat(_BONE_COLORS[:, None, :], 2, axis=1)

        self._point_cloud = self._server.scene.add_point_cloud(
            "/mediapipe/landmarks",
            points=points,
            colors=_LANDMARK_COLORS,
            point_size=point_size,
            point_shape="circle",
            precision="float32",
        )
        self._bones = self._server.scene.add_line_segments(
            "/mediapipe/bones",
            points=line_points,
            colors=line_colors,
            line_width=3.0,
        )

    @property
    def url(self) -> str:
        return f"http://{self._server.get_host()}:{self._server.get_port()}"

    def update(self, landmarks_3d: np.ndarray):
        if self.space == "local":
            points = _local_to_viser(landmarks_3d)
        else:
            points = _raw_to_viser(landmarks_3d)
        self._point_cloud.points = points
        self._bones.points = points[np.asarray(_HAND_CONNECTIONS, dtype=np.int32)]

    def close(self):
        self._server.stop()
