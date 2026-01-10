from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from kinect_pen.algo.body_tracker import MediaPipePoseBodyTracker, MediaPipePoseConfig
from kinect_pen.core.types import Frame


def test_rgb_tracking_uses_camera_3(monkeypatch):
    calls = []
    dummy_bgr = np.zeros((480, 640, 3), dtype=np.uint8)

    class DummyCap:
        def __init__(self, idx: int):
            calls.append(idx)

        def isOpened(self):
            return True

        def read(self):
            return True, dummy_bgr.copy()

        def release(self):
            return None

    monkeypatch.setattr(cv2, "VideoCapture", lambda idx: DummyCap(idx))

    class DummyPose:
        def process(self, rgb):
            lm0 = SimpleNamespace(x=0.25, y=0.5, visibility=0.9)
            return SimpleNamespace(pose_landmarks=SimpleNamespace(landmark=[lm0]))

    import kinect_pen.algo.body_tracker as bt

    monkeypatch.setattr(bt.mp.solutions.pose, "Pose", lambda **kwargs: DummyPose())

    tracker = MediaPipePoseBodyTracker(MediaPipePoseConfig())
    cap = cv2.VideoCapture(3)
    ok, frame = cap.read()
    cap.release()
    assert ok is True
    out = tracker.track_rgb(frame)
    assert calls == [3]
    assert out is not None
    name, u, v, conf = out[0]
    assert name == "nose"
    assert u == pytest.approx(0.25 * 640)
    assert v == pytest.approx(0.5 * 480)
    assert conf == pytest.approx(0.9)


def test_rgbd_lifting_backprojects_with_median_depth(monkeypatch):
    import kinect_pen.algo.body_tracker as bt

    monkeypatch.setattr(bt.mp.solutions.pose, "Pose", lambda **kwargs: object())
    tracker = MediaPipePoseBodyTracker(MediaPipePoseConfig(depth_window=5))
    u = 60.5
    v = 40.5
    conf = 0.8
    monkeypatch.setattr(tracker, "track_rgb", lambda color_bgr: [("left_wrist", u, v, conf)])

    depth = np.zeros((100, 100), dtype=np.uint16)
    ui = int(round(u))
    vi = int(round(v))
    depth[vi - 2 : vi + 3, ui - 2 : ui + 3] = 1000
    depth[vi, ui] = 2000
    depth[vi - 1, ui - 1] = 0

    intrinsics = np.array([500.0, 500.0, 50.0, 50.0], dtype=np.float32)
    frame = Frame(
        timestamp=0.0,
        frame_id=0,
        color=np.zeros((100, 100, 3), dtype=np.uint8),
        depth=depth,
        ir=None,
        intrinsics=intrinsics,
    )

    sk = tracker.track(frame)
    assert sk is not None
    name_to_joint = {j.name: j for j in sk.joints}
    assert "left_wrist" in name_to_joint
    assert "left_hand" in name_to_joint
    wrist = name_to_joint["left_wrist"]
    hand = name_to_joint["left_hand"]
    assert wrist.confidence == pytest.approx(conf)
    assert hand.confidence == pytest.approx(conf)

    expected_z = 1.0
    expected_x = (u - 50.0) * expected_z / 500.0
    expected_y = (v - 50.0) * expected_z / 500.0
    assert wrist.position[0] == float(np.float32(expected_x))
    assert wrist.position[1] == float(np.float32(expected_y))
    assert wrist.position[2] == float(np.float32(expected_z))
    assert np.allclose(hand.position, wrist.position, atol=0.0)
    assert sk.confidence == float(np.float32(conf))
