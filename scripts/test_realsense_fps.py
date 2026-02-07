import sys
import os
import time
import pyrealsense2 as rs

cur = os.path.dirname(os.path.abspath(__file__))
src = os.path.abspath(os.path.join(cur, "..", "src"))
if src not in sys.path:
    sys.path.append(src)

from ir_pen_tracker.io.realsense_camera import RealSenseCamera

def test_wrapper(seconds=5):
    cam = RealSenseCamera(depth_width=1280, depth_height=720, color_width=1920, color_height=1080, fps=30, enable_ir=True)
    cam.open()
    t0 = time.time()
    cnt = 0
    while time.time() - t0 < seconds:
        f = cam.read_frame()
        if f is not None:
            cnt += 1
    cam.close()
    return cnt / seconds

def test_raw(seconds=5):
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 30)
    cfg.enable_stream(rs.stream.infrared, 2, 1280, 720, rs.format.y8, 30)
    pipe.start(cfg)
    t0 = time.time()
    cnt = 0
    while time.time() - t0 < seconds:
        frames = pipe.wait_for_frames()
        if frames:
            cnt += 1
    pipe.stop()
    return cnt / seconds

def main():
    w = test_wrapper(5)
    r = test_raw(5)
    print("wrapper_fps", w)
    print("raw_fps", r)

if __name__ == "__main__":
    main()
