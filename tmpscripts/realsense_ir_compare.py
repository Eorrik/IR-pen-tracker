import sys
import os
import time
import cv2
import numpy as np
import pyrealsense2 as rs

USE_MANUAL = False
EXPOSURE_US = 4000.0
DEPTH_W, DEPTH_H = 1280, 720
FPS = 30

current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "..", "src")
if src_path not in sys.path:
    sys.path.append(src_path)

def manual_ir_loop():
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, DEPTH_W, DEPTH_H, rs.format.z16, FPS)
    cfg.enable_stream(rs.stream.infrared, 1, DEPTH_W, DEPTH_H, rs.format.y8, FPS)
    profile = pipe.start(cfg)
    dev = profile.get_device()
    depth_sensor = dev.first_depth_sensor()
    depth_sensor.set_option(rs.option.enable_auto_exposure, 0)
    depth_sensor.set_option(rs.option.exposure, float(EXPOSURE_US))
    cv2.namedWindow("IR Compare", cv2.WINDOW_NORMAL)
    while True:
        frames = pipe.wait_for_frames()
        ir_frame = frames.get_infrared_frame(1)
        if ir_frame is None:
            continue
        arr = np.asanyarray(ir_frame.get_data())
        if arr.dtype == np.uint8:
            vis = arr
        elif arr.dtype == np.uint16:
            vis = (arr / 256).astype(np.uint8)
        else:
            vis = arr.astype(np.uint8)
        cv2.putText(vis, "Source: manual", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
        cv2.imshow("IR Compare", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    pipe.stop()
    cv2.destroyAllWindows()

def camera_ir_loop():
    from ir_pen_tracker.io.realsense_camera import RealSenseCamera
    cam = RealSenseCamera(depth_width=DEPTH_W, depth_height=DEPTH_H, color_width=1920, color_height=1080, fps=FPS, enable_ir=True, preset="high_accuracy", exposure=float(EXPOSURE_US))
    ok = cam.open()
    if not ok:
        return
    cv2.namedWindow("IR Compare", cv2.WINDOW_NORMAL)
    while True:
        f = cam.read_frame()
        if f is None:
            continue
        ir = f.ir_main if f.ir_main is not None else f.ir
        if ir is None:
            continue
        vis = (np.clip(ir / 256.0, 0, 255)).astype(np.uint8)
        cv2.putText(vis, "Source: class", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
        cv2.imshow("IR Compare", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cam.close()
    cv2.destroyAllWindows()

def main():
    if USE_MANUAL:
        manual_ir_loop()
    else:
        camera_ir_loop()

if __name__ == "__main__":
    main()
