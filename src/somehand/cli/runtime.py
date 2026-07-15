"""Runtime factories for CLI command handlers."""

from __future__ import annotations

import argparse

from somehand.app import (
    BiHandRetargetingEngine,
    BiHandRetargetingSession,
    ControlledRetargetingSession,
    RetargetingEngine,
    RetargetingSession,
)
from somehand.domain import RetargetingConfig
from somehand.runtime import (
    AsyncBiHandLandmarkOutputSink,
    AsyncLandmarkOutputSink,
    BiHandOutputWindowSink,
    BiHandVideoOutputSink,
    LinkerHandModelAdapter,
    LinkerHandSdkController,
    MujocoSimController,
    OpenCvPreviewWindow,
    RobotHandOutputSink,
    RobotHandTargetOutputSink,
    RobotHandVideoOutputSink,
    infer_linkerhand_model_family,
)

ViewerMode = str


def close_resource(resource: object) -> None:
    close_fn = getattr(resource, "close", None)
    if callable(close_fn):
        close_fn()


def _close_sinks(frame_sinks: list[object], sinks: list[object]) -> None:
    for sink in reversed(frame_sinks):
        close_resource(sink)
    for sink in reversed(sinks):
        close_resource(sink)


def _robot_vector_specs(config: RetargetingConfig) -> list[tuple[int, str, str, str, str]]:
    return [
        (
            index,
            constraint.robot[0],
            constraint.robot_types[0],
            constraint.robot[1],
            constraint.robot_types[1],
        )
        for index, constraint in enumerate(config.vector_constraints)
        if constraint.robot[0] != "world"
    ]


def _robot_distance_specs(config: RetargetingConfig) -> list[tuple[int, str, str, str, str]]:
    return [
        (
            index,
            constraint.robot[0],
            constraint.robot_types[0],
            constraint.robot[1],
            constraint.robot_types[1],
        )
        for index, constraint in enumerate(getattr(config, "distance_constraints", ()))
    ]


def _robot_frame_specs(config: RetargetingConfig) -> list[tuple[int, str, str, str, str, str, str]]:
    return [
        (
            index,
            constraint.robot_origin,
            constraint.robot_types[0],
            constraint.robot_primary,
            constraint.robot_types[1],
            constraint.robot_secondary,
            constraint.robot_types[2],
        )
        for index, constraint in enumerate(getattr(config, "frame_constraints", ()))
    ]


def _robot_angle_specs(config: RetargetingConfig) -> list[tuple[int, str]]:
    return [(index, constraint.joint) for index, constraint in enumerate(getattr(config, "angle_constraints", ()))]


def _human_distance_pairs(config: RetargetingConfig) -> list[tuple[int, int]]:
    return [tuple(constraint.human) for constraint in getattr(config, "distance_constraints", ())]


def _human_frame_triples(config: RetargetingConfig) -> list[tuple[int, int, int]]:
    return [
        (constraint.human_origin, constraint.human_primary, constraint.human_secondary)
        for constraint in getattr(config, "frame_constraints", ())
    ]


def _human_angle_triples(config: RetargetingConfig) -> list[tuple[int, int, int]]:
    return [tuple(constraint.landmarks) for constraint in getattr(config, "angle_constraints", ())]


def _build_visual_sinks(
    engine: RetargetingEngine,
    *,
    backend: str,
    viewer_mode: ViewerMode = "normal",
    key_callback=None,
    include_landmark_viewer: bool = True,
    include_sim_state_viewer: bool = True,
) -> tuple[list[object], list[object]]:
    sinks: list[object] = []
    frame_sinks: list[object] = []
    diagnostic = viewer_mode == "diagnostic"
    robot_vector_specs = _robot_vector_specs(engine.config) if diagnostic else None
    robot_distance_specs = _robot_distance_specs(engine.config) if diagnostic else None
    robot_frame_specs = _robot_frame_specs(engine.config) if diagnostic else None
    robot_angle_specs = _robot_angle_specs(engine.config) if diagnostic else None
    human_vector_pairs = [tuple(pair) for pair in engine.config.human_vector_pairs] if diagnostic else None
    human_distance_pairs = _human_distance_pairs(engine.config) if diagnostic else None
    human_frame_triples = _human_frame_triples(engine.config) if diagnostic else None
    human_angle_triples = _human_angle_triples(engine.config) if diagnostic else None
    if include_landmark_viewer:
        landmark_sink = AsyncLandmarkOutputSink(
            window_title="Input Landmarks",
            vector_pairs=human_vector_pairs,
            distance_pairs=human_distance_pairs,
            frame_triples=human_frame_triples,
            angle_triples=human_angle_triples,
        )
        frame_sinks.append(landmark_sink)
    if backend == "sim":
        sinks.append(
            RobotHandTargetOutputSink(
                engine.hand_model,
                key_callback=key_callback,
                window_title="Retargeting",
                viewer_mode=viewer_mode,
                hand_side=engine.config.hand.side if diagnostic else None,
                robot_vector_specs=robot_vector_specs,
                robot_distance_specs=robot_distance_specs,
                robot_frame_specs=robot_frame_specs,
                robot_angle_specs=robot_angle_specs,
            )
        )
        if include_sim_state_viewer:
            sinks.append(
                RobotHandOutputSink(
                    engine.hand_model,
                    key_callback=key_callback,
                    window_title="Sim State",
                    viewer_mode=viewer_mode,
                    hand_side=engine.config.hand.side if diagnostic else None,
                    robot_vector_specs=robot_vector_specs,
                    robot_distance_specs=robot_distance_specs,
                    robot_frame_specs=robot_frame_specs,
                    robot_angle_specs=robot_angle_specs,
                )
            )
    else:
        sinks.append(
            RobotHandOutputSink(
                engine.hand_model,
                key_callback=key_callback,
                window_title="Retargeting",
                viewer_mode=viewer_mode,
                hand_side=engine.config.hand.side if diagnostic else None,
                robot_vector_specs=robot_vector_specs,
                robot_distance_specs=robot_distance_specs,
                robot_frame_specs=robot_frame_specs,
                robot_angle_specs=robot_angle_specs,
            )
        )
    return sinks, frame_sinks


def _build_control_visual_sinks(
    engine: RetargetingEngine,
    *,
    backend: str,
    viewer_mode: ViewerMode = "normal",
    key_callback=None,
    include_landmark_viewer: bool = True,
    include_sim_state_viewer: bool = True,
) -> tuple[list[object], list[object]]:
    sinks: list[object] = []
    frame_sinks: list[object] = []
    diagnostic = viewer_mode == "diagnostic"
    robot_vector_specs = _robot_vector_specs(engine.config) if diagnostic else None
    robot_distance_specs = _robot_distance_specs(engine.config) if diagnostic else None
    robot_frame_specs = _robot_frame_specs(engine.config) if diagnostic else None
    robot_angle_specs = _robot_angle_specs(engine.config) if diagnostic else None
    human_vector_pairs = [tuple(pair) for pair in engine.config.human_vector_pairs] if diagnostic else None
    human_distance_pairs = _human_distance_pairs(engine.config) if diagnostic else None
    human_frame_triples = _human_frame_triples(engine.config) if diagnostic else None
    human_angle_triples = _human_angle_triples(engine.config) if diagnostic else None
    if include_landmark_viewer:
        landmark_sink = AsyncLandmarkOutputSink(
            window_title="Input Landmarks",
            vector_pairs=human_vector_pairs,
            distance_pairs=human_distance_pairs,
            frame_triples=human_frame_triples,
            angle_triples=human_angle_triples,
        )
        frame_sinks.append(landmark_sink)
    if backend == "sim":
        sinks.append(
            RobotHandTargetOutputSink(
                engine.hand_model,
                key_callback=key_callback,
                window_title="Retargeting",
                viewer_mode=viewer_mode,
                hand_side=engine.config.hand.side if diagnostic else None,
                robot_vector_specs=robot_vector_specs,
                robot_distance_specs=robot_distance_specs,
                robot_frame_specs=robot_frame_specs,
                robot_angle_specs=robot_angle_specs,
            )
        )
        if include_sim_state_viewer:
            sinks.append(
                RobotHandOutputSink(
                    engine.hand_model,
                    key_callback=key_callback,
                    window_title="Sim State",
                    viewer_mode=viewer_mode,
                    hand_side=engine.config.hand.side if diagnostic else None,
                    robot_vector_specs=robot_vector_specs,
                    robot_distance_specs=robot_distance_specs,
                    robot_frame_specs=robot_frame_specs,
                    robot_angle_specs=robot_angle_specs,
                )
            )
    elif backend == "real":
        sinks.append(
            RobotHandTargetOutputSink(
                engine.hand_model,
                key_callback=key_callback,
                window_title="Retargeting",
                viewer_mode=viewer_mode,
                hand_side=engine.config.hand.side if diagnostic else None,
                robot_vector_specs=robot_vector_specs,
                robot_distance_specs=robot_distance_specs,
                robot_frame_specs=robot_frame_specs,
                robot_angle_specs=robot_angle_specs,
            )
        )
    else:
        sinks.append(
            RobotHandOutputSink(
                engine.hand_model,
                key_callback=key_callback,
                window_title="Retargeting",
                viewer_mode=viewer_mode,
                hand_side=engine.config.hand.side if diagnostic else None,
                robot_vector_specs=robot_vector_specs,
                robot_distance_specs=robot_distance_specs,
                robot_frame_specs=robot_frame_specs,
                robot_angle_specs=robot_angle_specs,
            )
        )
    return sinks, frame_sinks


def _append_video_sink(
    sinks: list[object],
    *,
    hand_model,
    video_output_path: str | None,
    video_output_fps: int | None,
) -> None:
    if video_output_path is None:
        return
    if video_output_fps is None:
        raise ValueError("video_output_fps is required when video_output_path is provided")
    sinks.append(
        RobotHandVideoOutputSink(
            hand_model,
            output_path=video_output_path,
            fps=video_output_fps,
        )
    )


def _build_bihand_visual_sinks(
    engine: BiHandRetargetingEngine,
    *,
    viewer_mode: ViewerMode = "normal",
    key_callback=None,
) -> tuple[list[object], list[object]]:
    diagnostic = viewer_mode == "diagnostic"
    landmark_sink = AsyncBiHandLandmarkOutputSink(
        left_pos=engine.config.viewer.left_pos,
        right_pos=engine.config.viewer.right_pos,
        left_quat=engine.config.viewer.left_quat,
        right_quat=engine.config.viewer.right_quat,
        left_vector_pairs=[tuple(pair) for pair in engine.left_engine.config.human_vector_pairs] if diagnostic else None,
        right_vector_pairs=[tuple(pair) for pair in engine.right_engine.config.human_vector_pairs] if diagnostic else None,
        left_distance_pairs=_human_distance_pairs(engine.left_engine.config) if diagnostic else None,
        right_distance_pairs=_human_distance_pairs(engine.right_engine.config) if diagnostic else None,
        left_frame_triples=_human_frame_triples(engine.left_engine.config) if diagnostic else None,
        right_frame_triples=_human_frame_triples(engine.right_engine.config) if diagnostic else None,
        left_angle_triples=_human_angle_triples(engine.left_engine.config) if diagnostic else None,
        right_angle_triples=_human_angle_triples(engine.right_engine.config) if diagnostic else None,
    )
    frame_sinks = [landmark_sink]
    sinks = [
        BiHandOutputWindowSink(
            engine.left_engine.hand_model,
            engine.right_engine.hand_model,
            key_callback=key_callback,
            panel_width=engine.config.viewer.panel_width,
            panel_height=engine.config.viewer.panel_height,
            window_name=engine.config.viewer.window_name,
            left_pos=engine.config.viewer.left_pos,
            right_pos=engine.config.viewer.right_pos,
            camera_lookat=engine.config.viewer.camera_lookat,
            left_quat=engine.config.viewer.left_quat,
            right_quat=engine.config.viewer.right_quat,
            viewer_mode=viewer_mode,
            left_hand_side=engine.left_engine.config.hand.side if diagnostic else None,
            right_hand_side=engine.right_engine.config.hand.side if diagnostic else None,
            left_robot_vector_specs=_robot_vector_specs(engine.left_engine.config) if diagnostic else None,
            right_robot_vector_specs=_robot_vector_specs(engine.right_engine.config) if diagnostic else None,
            left_robot_distance_specs=_robot_distance_specs(engine.left_engine.config) if diagnostic else None,
            right_robot_distance_specs=_robot_distance_specs(engine.right_engine.config) if diagnostic else None,
            left_robot_frame_specs=_robot_frame_specs(engine.left_engine.config) if diagnostic else None,
            right_robot_frame_specs=_robot_frame_specs(engine.right_engine.config) if diagnostic else None,
            left_robot_angle_specs=_robot_angle_specs(engine.left_engine.config) if diagnostic else None,
            right_robot_angle_specs=_robot_angle_specs(engine.right_engine.config) if diagnostic else None,
        )
    ]
    return sinks, frame_sinks


def build_engine(args: argparse.Namespace, *, input_type: str) -> RetargetingEngine:
    return RetargetingEngine.from_config_path(args.config, input_type=input_type)


def build_bihand_engine(args: argparse.Namespace, *, input_type: str) -> BiHandRetargetingEngine:
    return BiHandRetargetingEngine.from_config_path(args.config, input_type=input_type)


def build_session(
    engine: RetargetingEngine,
    *,
    backend: str = "viewer",
    viewer_mode: ViewerMode = "normal",
    visualize: bool,
    show_preview: bool,
    key_callback=None,
    video_output_path: str | None = None,
    video_output_fps: int | None = None,
) -> RetargetingSession:
    sinks: list[object] = []
    frame_sinks: list[object] = []
    if visualize:
        try:
            sinks, frame_sinks = _build_visual_sinks(
                engine,
                backend=backend,
                viewer_mode=viewer_mode,
                key_callback=key_callback,
            )
        except BaseException:
            _close_sinks(frame_sinks, sinks)
            raise
    _append_video_sink(
        sinks,
        hand_model=engine.hand_model,
        video_output_path=video_output_path,
        video_output_fps=video_output_fps,
    )
    preview_window = OpenCvPreviewWindow() if show_preview else None
    return RetargetingSession(engine, sinks=sinks, frame_sinks=frame_sinks, preview_window=preview_window)


def build_control_backend(args: argparse.Namespace, engine: RetargetingEngine):
    if args.backend == "sim":
        return MujocoSimController(
            engine.config.hand.mjcf_path,
            control_rate_hz=args.control_rate,
            sim_rate_hz=args.sim_rate,
        )
    if args.backend == "real":
        family = args.model_family or engine.config.controller.model_family or infer_linkerhand_model_family(
            engine.config.hand.name
        )
        adapter = LinkerHandModelAdapter(
            engine.hand_model,
            family=family,
            hand_side=engine.config.hand.side,
            sdk_root="" if args.sdk_root is None else args.sdk_root,
        )
        return LinkerHandSdkController(
            adapter,
            transport=args.transport,
            can_interface=args.can_interface,
            modbus_port=args.modbus_port,
            default_speed=engine.config.controller.default_speed or adapter.default_speed,
            default_torque=engine.config.controller.default_torque or adapter.default_torque,
            sdk_root="" if args.sdk_root is None else args.sdk_root,
        )
    raise ValueError(f"Unsupported backend: {args.backend}")


def build_runtime_session(
    engine: RetargetingEngine,
    args: argparse.Namespace,
    *,
    visualize: bool,
    show_preview: bool,
    key_callback=None,
    video_output_path: str | None = None,
    video_output_fps: int | None = None,
    include_landmark_viewer: bool = True,
    include_sim_state_viewer: bool = True,
):
    if args.backend == "viewer":
        return build_session(
            engine,
            backend=args.backend,
            viewer_mode=getattr(args, "viewer_mode", "normal"),
            visualize=visualize,
            show_preview=show_preview,
            key_callback=key_callback,
            video_output_path=video_output_path,
            video_output_fps=video_output_fps,
        )

    sinks: list[object] = []
    frame_sinks: list[object] = []
    if visualize:
        try:
            sinks, frame_sinks = _build_control_visual_sinks(
                engine,
                backend=args.backend,
                viewer_mode=getattr(args, "viewer_mode", "normal"),
                key_callback=key_callback,
                include_landmark_viewer=include_landmark_viewer,
                include_sim_state_viewer=include_sim_state_viewer,
            )
        except BaseException:
            _close_sinks(frame_sinks, sinks)
            raise
    _append_video_sink(
        sinks,
        hand_model=engine.hand_model,
        video_output_path=video_output_path,
        video_output_fps=video_output_fps,
    )
    preview_window = OpenCvPreviewWindow() if show_preview else None
    controller = build_control_backend(args, engine)
    return ControlledRetargetingSession(
        engine,
        controller,
        sinks=sinks,
        frame_sinks=frame_sinks,
        preview_window=preview_window,
    )


def build_bihand_session(
    engine: BiHandRetargetingEngine,
    *,
    viewer_mode: ViewerMode = "normal",
    visualize: bool,
    show_preview: bool,
    key_callback=None,
    video_output_path: str | None = None,
    video_output_fps: int | None = None,
) -> BiHandRetargetingSession:
    sinks: list[object] = []
    frame_sinks: list[object] = []
    if visualize:
        try:
            sinks, frame_sinks = _build_bihand_visual_sinks(
                engine,
                viewer_mode=viewer_mode,
                key_callback=key_callback,
            )
        except BaseException:
            _close_sinks(frame_sinks, sinks)
            raise
    if video_output_path is not None:
        if video_output_fps is None:
            raise ValueError("video_output_fps is required when video_output_path is provided")
        sinks.append(
            BiHandVideoOutputSink(
                engine.left_engine.hand_model,
                engine.right_engine.hand_model,
                output_path=video_output_path,
                fps=video_output_fps,
                panel_width=engine.config.viewer.panel_width,
                panel_height=engine.config.viewer.panel_height,
                left_pos=engine.config.viewer.left_pos,
                right_pos=engine.config.viewer.right_pos,
                camera_lookat=engine.config.viewer.camera_lookat,
                left_quat=engine.config.viewer.left_quat,
                right_quat=engine.config.viewer.right_quat,
            )
        )
    preview_window = OpenCvPreviewWindow("Bi-Hand Detection") if show_preview else None
    return BiHandRetargetingSession(engine, sinks=sinks, frame_sinks=frame_sinks, preview_window=preview_window)


__all__ = [
    "build_bihand_engine",
    "build_bihand_session",
    "build_control_backend",
    "build_engine",
    "build_runtime_session",
    "build_session",
    "close_resource",
]
