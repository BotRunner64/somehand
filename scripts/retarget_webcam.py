"""Real-time hand retargeting from webcam."""

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dex_mujoco.hand_detector import HandDetector
from dex_mujoco.hand_model import HandModel
from dex_mujoco.landmark_visualization import MediaPipe3DVisualizer
from dex_mujoco.retargeting_config import RetargetingConfig
from dex_mujoco.vector_retargeting import preprocess_landmarks
from dex_mujoco.vector_retargeting import VectorRetargeter
from dex_mujoco.visualization import HandVisualizer


def main():
    parser = argparse.ArgumentParser(description="Real-time hand retargeting from webcam")
    parser.add_argument(
        "--config",
        default="configs/retargeting/linkerhand_l20.yaml",
        help="Path to retargeting config YAML",
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index")
    parser.add_argument(
        "--hand",
        choices=["Left", "Right"],
        default="Right",
        help="Actual operator hand to retarget",
    )
    parser.add_argument(
        "--swap-hands",
        action="store_true",
        help="Swap MediaPipe Left/Right labels if your capture pipeline reports the opposite hand",
    )
    parser.add_argument(
        "--viser",
        action="store_true",
        help="Show MediaPipe 3D landmarks in a browser via viser",
    )
    parser.add_argument(
        "--viser-host",
        default="127.0.0.1",
        help="Host address for the viser server",
    )
    parser.add_argument(
        "--viser-port",
        type=int,
        default=8080,
        help="Port for the viser server",
    )
    parser.add_argument(
        "--viser-space",
        choices=["local", "raw"],
        default="local",
        help="Which landmark coordinates to render in viser",
    )
    args = parser.parse_args()

    config = RetargetingConfig.load(args.config)
    hand_model = HandModel(config.hand.mjcf_path)
    retargeter = VectorRetargeter(hand_model, config)
    detector = HandDetector(target_hand=args.hand, swap_handedness=args.swap_hands)
    visualizer = HandVisualizer(hand_model)
    mp_visualizer = None
    if args.viser:
        mp_visualizer = MediaPipe3DVisualizer(
            host=args.viser_host,
            port=args.viser_port,
            space=args.viser_space,
        )

    print(f"Model: {config.hand.name} ({hand_model.nq} DOF)")
    print(f"Retargeting: {len(config.human_vector_pairs)} vector pairs")
    print(f"Tracking operator hand: {args.hand} | Swap hands: {args.swap_hands}")
    if mp_visualizer is not None:
        print(f"MediaPipe 3D viewer ({args.viser_space}): {mp_visualizer.url}")
    print("Press 'q' to quit.")

    for frame in HandDetector.create_source(args.camera):
        detection = detector.detect(frame)

        if detection is not None:
            retargeter.update_targets(detection.landmarks_3d, detection.handedness)
            qpos = retargeter.solve()
            visualizer.update(qpos)
            if mp_visualizer is not None:
                if args.viser_space == "local":
                    landmarks_for_vis = preprocess_landmarks(
                        detection.landmarks_3d,
                        handedness=detection.handedness,
                        frame=config.preprocess.frame,
                    )
                else:
                    landmarks_for_vis = detection.landmarks_3d
                mp_visualizer.update(landmarks_for_vis)
            frame = detector.draw_landmarks(frame, detection)

        cv2.imshow("Hand Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if not visualizer.is_running:
            break

    detector.close()
    if mp_visualizer is not None:
        mp_visualizer.close()
    visualizer.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
