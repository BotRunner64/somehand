"""Single-hand and bi-hand landmark viewer implementations."""

from __future__ import annotations

import mujoco
import numpy as np

from .viewer_camera import (
    DEFAULT_BIHAND_LANDMARK_CAMERA,
    DEFAULT_LANDMARK_CAMERA,
    HAND_CONNECTIONS,
    LANDMARK_COLORS,
    LANDMARK_VIEWER_XML,
    append_bihand_landmark_geoms,
    append_single_landmark_geoms,
    configure_free_camera,
    try_frame_camera_to_points,
)
from .viewer_passive import ManagedPassiveViewer, set_viewer_window_title
from .vector_visualization import (
    DISTANCE_RGBA,
    VectorPair,
    append_landmark_angle_geoms,
    append_landmark_frame_geoms,
    append_landmark_vector_geoms,
)

DIAGNOSTIC_BASE_POINT_ALPHA = 0.30
DIAGNOSTIC_BASE_BONE_ALPHA = 0.16


class LandmarkVisualizer:
    """Real-time MuJoCo visualization of the input hand landmarks."""

    def __init__(
        self,
        *,
        window_title: str | None = None,
        vector_pairs: list[VectorPair] | None = None,
        distance_pairs: list[VectorPair] | None = None,
        frame_triples: list[tuple[int, int, int]] | None = None,
        angle_triples: list[tuple[int, int, int]] | None = None,
    ):
        self.model = mujoco.MjModel.from_xml_string(LANDMARK_VIEWER_XML)
        self.data = mujoco.MjData(self.model)
        self._vector_pairs = [] if vector_pairs is None else [tuple(pair) for pair in vector_pairs]
        self._distance_pairs = [] if distance_pairs is None else [tuple(pair) for pair in distance_pairs]
        self._frame_triples = [] if frame_triples is None else [tuple(triple) for triple in frame_triples]
        self._angle_triples = [] if angle_triples is None else [tuple(triple) for triple in angle_triples]
        self._diagnostic = bool(
            self._vector_pairs
            or self._distance_pairs
            or self._frame_triples
            or self._angle_triples
        )
        self.viewer = ManagedPassiveViewer(
            model=self.model,
            data=self.data,
            show_left_ui=False,
            show_right_ui=False,
            window_title=window_title,
        )
        set_viewer_window_title(self.viewer, window_title)
        self._max_overlay_geoms = (
            len(LANDMARK_COLORS)
            + len(HAND_CONNECTIONS)
            + 2 * len(self._vector_pairs)
            + 2 * len(self._distance_pairs)
            + 6 * len(self._frame_triples)
            + 4 * len(self._angle_triples)
        )
        if self.viewer.user_scn is None:
            raise RuntimeError("MuJoCo passive viewer does not expose a user scene")
        if self.viewer.user_scn.maxgeom < self._max_overlay_geoms:
            raise RuntimeError(
                f"MuJoCo viewer user scene only supports {self.viewer.user_scn.maxgeom} geoms, "
                f"but landmark overlay needs {self._max_overlay_geoms}"
            )
        self._configure_camera(**DEFAULT_LANDMARK_CAMERA)
        self._camera_initialized = False

    def _configure_camera(
        self,
        *,
        distance: float,
        azimuth: float,
        elevation: float,
        lookat: tuple[float, float, float],
    ) -> None:
        with self.viewer.lock():
            configure_free_camera(
                self.viewer.cam,
                distance=distance,
                azimuth=azimuth,
                elevation=elevation,
                lookat=lookat,
            )
        self.viewer.sync(state_only=True)

    def update(self, landmarks: np.ndarray):
        with self.viewer.lock():
            mujoco.mj_forward(self.model, self.data)
            if not self._camera_initialized and try_frame_camera_to_points(
                self.viewer.cam,
                model=self.model,
                points=landmarks,
                azimuth=DEFAULT_LANDMARK_CAMERA["azimuth"],
                elevation=DEFAULT_LANDMARK_CAMERA["elevation"],
            ):
                self._camera_initialized = True
            self._update_landmark_overlay(landmarks)
        self.viewer.sync()

    def _update_landmark_overlay(self, landmarks: np.ndarray) -> None:
        scene = self.viewer.user_scn
        scene.ngeom = 0
        append_single_landmark_geoms(
            scene,
            landmarks,
            point_alpha=DIAGNOSTIC_BASE_POINT_ALPHA if self._diagnostic else None,
            bone_alpha=DIAGNOSTIC_BASE_BONE_ALPHA if self._diagnostic else None,
        )
        append_landmark_vector_geoms(scene, landmarks, self._vector_pairs)
        append_landmark_vector_geoms(scene, landmarks, self._distance_pairs, rgba=DISTANCE_RGBA)
        append_landmark_frame_geoms(scene, landmarks, self._frame_triples)
        append_landmark_angle_geoms(scene, landmarks, self._angle_triples)

    @property
    def is_running(self) -> bool:
        return self.viewer.is_running()

    def close(self):
        if self.viewer.is_running():
            self.viewer.close()


class BiHandLandmarkVisualizer:
    """Real-time MuJoCo visualization of both input-hand landmark sets."""

    def __init__(
        self,
        *,
        window_title: str | None = None,
        left_vector_pairs: list[VectorPair] | None = None,
        right_vector_pairs: list[VectorPair] | None = None,
        left_distance_pairs: list[VectorPair] | None = None,
        right_distance_pairs: list[VectorPair] | None = None,
        left_frame_triples: list[tuple[int, int, int]] | None = None,
        right_frame_triples: list[tuple[int, int, int]] | None = None,
        left_angle_triples: list[tuple[int, int, int]] | None = None,
        right_angle_triples: list[tuple[int, int, int]] | None = None,
    ):
        self.model = mujoco.MjModel.from_xml_string(LANDMARK_VIEWER_XML)
        self.data = mujoco.MjData(self.model)
        self._left_vector_pairs = [] if left_vector_pairs is None else [tuple(pair) for pair in left_vector_pairs]
        self._right_vector_pairs = [] if right_vector_pairs is None else [tuple(pair) for pair in right_vector_pairs]
        self._left_distance_pairs = [] if left_distance_pairs is None else [tuple(pair) for pair in left_distance_pairs]
        self._right_distance_pairs = [] if right_distance_pairs is None else [tuple(pair) for pair in right_distance_pairs]
        self._left_frame_triples = [] if left_frame_triples is None else [tuple(triple) for triple in left_frame_triples]
        self._right_frame_triples = [] if right_frame_triples is None else [tuple(triple) for triple in right_frame_triples]
        self._left_angle_triples = [] if left_angle_triples is None else [tuple(triple) for triple in left_angle_triples]
        self._right_angle_triples = [] if right_angle_triples is None else [tuple(triple) for triple in right_angle_triples]
        self._diagnostic = bool(
            self._left_vector_pairs
            or self._right_vector_pairs
            or self._left_distance_pairs
            or self._right_distance_pairs
            or self._left_frame_triples
            or self._right_frame_triples
            or self._left_angle_triples
            or self._right_angle_triples
        )
        self.viewer = ManagedPassiveViewer(
            model=self.model,
            data=self.data,
            show_left_ui=False,
            show_right_ui=False,
            window_title=window_title,
        )
        set_viewer_window_title(self.viewer, window_title)
        self._max_overlay_geoms = (
            2 * (len(LANDMARK_COLORS) + len(HAND_CONNECTIONS))
            + 2 * len(self._left_vector_pairs)
            + 2 * len(self._right_vector_pairs)
            + 2 * len(self._left_distance_pairs)
            + 2 * len(self._right_distance_pairs)
            + 6 * len(self._left_frame_triples)
            + 6 * len(self._right_frame_triples)
            + 4 * len(self._left_angle_triples)
            + 4 * len(self._right_angle_triples)
        )
        if self.viewer.user_scn is None:
            raise RuntimeError("MuJoCo passive viewer does not expose a user scene")
        if self.viewer.user_scn.maxgeom < self._max_overlay_geoms:
            raise RuntimeError(
                f"MuJoCo viewer user scene only supports {self.viewer.user_scn.maxgeom} geoms, "
                f"but bi-hand landmark overlay needs {self._max_overlay_geoms}"
            )
        self._configure_camera(**DEFAULT_BIHAND_LANDMARK_CAMERA)
        self._camera_initialized = False

    def _configure_camera(
        self,
        *,
        distance: float,
        azimuth: float,
        elevation: float,
        lookat: tuple[float, float, float],
    ) -> None:
        with self.viewer.lock():
            configure_free_camera(
                self.viewer.cam,
                distance=distance,
                azimuth=azimuth,
                elevation=elevation,
                lookat=lookat,
            )
        self.viewer.sync(state_only=True)

    def update(
        self,
        hands: np.ndarray,
    ):
        with self.viewer.lock():
            mujoco.mj_forward(self.model, self.data)
            finite_mask = np.isfinite(hands).all(axis=2)
            has_both_hands = bool(np.any(finite_mask[0]) and np.any(finite_mask[1]))
            finite_points = hands[finite_mask]
            if has_both_hands and not self._camera_initialized and try_frame_camera_to_points(
                self.viewer.cam,
                model=self.model,
                points=finite_points.reshape(-1, 3),
                azimuth=DEFAULT_BIHAND_LANDMARK_CAMERA["azimuth"],
                elevation=DEFAULT_BIHAND_LANDMARK_CAMERA["elevation"],
            ):
                self._camera_initialized = True
            self._update_landmark_overlay(hands)
        self.viewer.sync()

    def _update_landmark_overlay(self, hands: np.ndarray) -> None:
        scene = self.viewer.user_scn
        scene.ngeom = 0
        append_bihand_landmark_geoms(
            scene,
            hands,
            point_alpha=DIAGNOSTIC_BASE_POINT_ALPHA if self._diagnostic else None,
            bone_alpha=DIAGNOSTIC_BASE_BONE_ALPHA if self._diagnostic else None,
        )
        append_landmark_vector_geoms(scene, hands[0], self._left_vector_pairs)
        append_landmark_vector_geoms(scene, hands[1], self._right_vector_pairs)
        append_landmark_vector_geoms(scene, hands[0], self._left_distance_pairs, rgba=DISTANCE_RGBA)
        append_landmark_vector_geoms(scene, hands[1], self._right_distance_pairs, rgba=DISTANCE_RGBA)
        append_landmark_frame_geoms(scene, hands[0], self._left_frame_triples)
        append_landmark_frame_geoms(scene, hands[1], self._right_frame_triples)
        append_landmark_angle_geoms(scene, hands[0], self._left_angle_triples)
        append_landmark_angle_geoms(scene, hands[1], self._right_angle_triples)

    @property
    def is_running(self) -> bool:
        return self.viewer.is_running()

    def close(self):
        if self.viewer.is_running():
            self.viewer.close()


__all__ = ["LandmarkVisualizer", "BiHandLandmarkVisualizer"]
