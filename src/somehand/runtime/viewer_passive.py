"""Passive MuJoCo viewer wrappers and process-safe helpers."""

from __future__ import annotations

import atexit
import queue
import sys
import threading

import mujoco
import mujoco.viewer


def mujoco_key_callback(handler):
    if handler is None:
        return None

    def _callback(keycode: int) -> None:
        if keycode < 0 or keycode > 255:
            return
        handler(chr(keycode))

    return _callback


def set_viewer_overlay_label(viewer, label: str | None) -> None:
    set_texts = getattr(viewer, "set_texts", None)
    if not callable(set_texts):
        return
    label = "" if label is None else label
    set_texts(
        (
            mujoco.mjtFontScale.mjFONTSCALE_150,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            label,
            "",
        )
    )


def set_viewer_window_title(viewer, title: str | None) -> None:
    if not title:
        return
    get_sim = getattr(viewer, "_get_sim", None)
    if not callable(get_sim):
        return
    sim = get_sim()
    if sim is None:
        return
    try:
        sim.filename = title
    except Exception:
        return


def launch_passive_internal_with_window_title(
    model,
    data,
    *,
    handle_return,
    key_callback=None,
    show_left_ui=False,
    show_right_ui=False,
    window_title: str | None = None,
) -> None:
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    pert = mujoco.MjvPerturb()
    user_scn = mujoco.MjvScene(model, mujoco.viewer._Simulate.MAX_GEOM)
    simulate = mujoco.viewer._Simulate(cam, opt, pert, user_scn, False, key_callback)

    simulate.ui0_enable = show_left_ui
    simulate.ui1_enable = show_right_ui

    if mujoco.viewer._MJPYTHON is None:
        if not mujoco.viewer.glfw.init():
            raise mujoco.FatalError("could not initialize GLFW")
        atexit.register(mujoco.viewer.glfw.terminate)

    def _loader():
        return model, data, window_title or ""

    def notify_loaded():
        handle_return.put_nowait(mujoco.viewer.Handle(simulate, cam, opt, pert, user_scn))

    side_thread = threading.Thread(target=mujoco.viewer._reload, args=(simulate, _loader, notify_loaded))

    def _exit_simulate():
        simulate.exit()

    atexit.register(_exit_simulate)
    side_thread.start()
    simulate.render_loop()
    atexit.unregister(_exit_simulate)
    side_thread.join()
    simulate.destroy()


def compile_model_with_name(mjcf_path: str, model_name: str) -> tuple[mujoco.MjModel, mujoco.MjData]:
    spec = mujoco.MjSpec.from_file(mjcf_path)
    spec.modelname = model_name
    model = spec.compile()
    data = mujoco.MjData(model)
    return model, data


class ManagedPassiveViewer:
    """Wrap MuJoCo's passive viewer and wait for its render thread to exit."""

    def __init__(
        self,
        model,
        data,
        *,
        key_callback=None,
        show_left_ui=False,
        show_right_ui=False,
        window_title: str | None = None,
    ):
        if sys.platform == "darwin":
            self._handle = mujoco.viewer.launch_passive(
                model=model,
                data=data,
                key_callback=key_callback,
                show_left_ui=show_left_ui,
                show_right_ui=show_right_ui,
            )
            self._thread = None
            return

        handle_return = queue.Queue(1)
        if not window_title:
            target = mujoco.viewer._launch_internal
            args = (model, data)
            kwargs = dict(
                run_physics_thread=False,
                handle_return=handle_return,
                key_callback=key_callback,
                show_left_ui=show_left_ui,
                show_right_ui=show_right_ui,
            )
        else:
            target = launch_passive_internal_with_window_title
            args = (model, data)
            kwargs = dict(
                handle_return=handle_return,
                key_callback=key_callback,
                show_left_ui=show_left_ui,
                show_right_ui=show_right_ui,
                window_title=window_title,
            )
        self._thread = threading.Thread(
            target=target,
            args=args,
            kwargs=kwargs,
            name="somehand-passive-viewer",
            daemon=True,
        )
        self._thread.start()
        self._handle = handle_return.get()

    def __getattr__(self, name: str):
        return getattr(self._handle, name)

    def close(self, *, timeout: float = 2.0) -> None:
        self._handle.close()
        if self._thread is not None and self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=timeout)


__all__ = [
    "ManagedPassiveViewer",
    "compile_model_with_name",
    "launch_passive_internal_with_window_title",
    "mujoco_key_callback",
    "set_viewer_overlay_label",
    "set_viewer_window_title",
]
