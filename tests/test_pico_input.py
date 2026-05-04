import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import somehand.pico_input as pico_input
from somehand.runtime.source_adapters import BiHandPicoInputSource


def _hand(active=True, value=0.0):
    joints = np.zeros((26, 7), dtype=np.float64)
    for index in range(26):
        joints[index, :3] = [value + index, value + index + 0.25, value + index + 0.5]
    return SimpleNamespace(active=active, joints=joints)


def _frame(seq=1, *, left=True, right=True):
    return SimpleNamespace(
        seq=seq,
        left_hand=_hand(active=left, value=100.0),
        right_hand=_hand(active=right, value=200.0),
    )


def test_pico_hand_to_landmarks_uses_pico_bridge_joint_order():
    state = np.zeros((26, 7), dtype=np.float64)
    for index in range(26):
        state[index, :3] = [index, index + 10.0, index + 20.0]

    landmarks = pico_input.pico_hand_to_landmarks(state)

    assert landmarks.shape == (21, 3)
    np.testing.assert_allclose(landmarks[0], [1.0, -21.0, 11.0])
    np.testing.assert_allclose(landmarks[4], [5.0, -25.0, 15.0])
    np.testing.assert_allclose(landmarks[8], [10.0, -30.0, 20.0])


def test_pico_provider_reads_active_hand_from_pico_bridge(monkeypatch):
    class _FakeReceiver:
        fps = 72

        def __init__(self, **kwargs):
            self.frames = [_frame(seq=1, left=False), _frame(seq=2, left=True)]
            self.closed = False
            self.kwargs = kwargs

        def is_available(self):
            return not self.closed

        def wait_frame(self, *, timeout=None, after_seq=None):
            for frame in self.frames:
                if after_seq is None or frame.seq > after_seq:
                    return frame
            raise TimeoutError("no frame")

        def latest_frame(self):
            return self.frames[-1]

        def close(self):
            self.closed = True

        def stats_snapshot(self):
            return {"fps": 72}

    monkeypatch.setattr(pico_input, "PicoBridgeReceiver", _FakeReceiver)

    provider = pico_input.PicoHandProvider(
        "left",
        timeout=1.0,
        host="127.0.0.1",
        port=64000,
        discovery=False,
        advertise_ip="192.168.1.2",
    )

    detection = provider.get_detection()
    snapshot = provider.latest_detection_snapshot()

    assert provider.fps == 72
    assert provider.is_available() is True
    assert detection.hand_side == "left"
    assert snapshot is not None
    assert snapshot[0] == 2
    np.testing.assert_allclose(detection.landmarks_3d[0], [101.0, -101.5, 101.25])
    provider.close()
    assert provider.is_available() is False


def test_pico_provider_times_out_when_hand_stays_inactive(monkeypatch):
    class _FakeReceiver:
        fps = 80

        def __init__(self, **kwargs):
            self._seq = 0

        def is_available(self):
            return True

        def wait_frame(self, *, timeout=None, after_seq=None):
            self._seq += 1
            return _frame(seq=self._seq, left=False)

        def latest_frame(self):
            return None

        def close(self):
            return None

        def stats_snapshot(self):
            return {}

    monkeypatch.setattr(pico_input, "PicoBridgeReceiver", _FakeReceiver)

    provider = pico_input.PicoHandProvider("left", timeout=0.001)

    with pytest.raises(TimeoutError, match="No active PICO Bridge left hand frame"):
        provider.get_detection()


def test_bihand_pico_source_uses_one_receiver_for_both_hands(monkeypatch):
    created = []

    class _FakeReceiver:
        fps = 90

        def __init__(self, **kwargs):
            created.append(kwargs)
            self.frame = _frame(seq=9, left=True, right=True)

        def is_available(self):
            return True

        def wait_frame(self, *, timeout=None, after_seq=None):
            return self.frame

        def latest_frame(self):
            return self.frame

        def close(self):
            return None

        def stats_snapshot(self):
            return {"fps": 90, "connected": True}

    monkeypatch.setattr("somehand.runtime.source_adapters.PicoBridgeReceiver", _FakeReceiver)

    source = BiHandPicoInputSource(
        timeout=2.0,
        host="127.0.0.1",
        port=64001,
        discovery=False,
        advertise_ip="192.168.1.3",
    )
    frame = source.get_frame().detection
    snapshot = source.latest_bihand_frame_snapshot()

    assert len(created) == 1
    assert created[0]["host"] == "127.0.0.1"
    assert created[0]["port"] == 64001
    assert created[0]["discovery"] is False
    assert created[0]["advertise_ip"] == "192.168.1.3"
    assert source.fps == 90
    assert frame is not None
    assert frame.left is not None
    assert frame.right is not None
    assert snapshot is not None
    assert snapshot[0] == 9
