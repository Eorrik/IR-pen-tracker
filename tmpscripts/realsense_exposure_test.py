import pyrealsense2 as rs
import numpy as np
import cv2
import time

def main():
    pipe = rs.pipeline()
    cfg = rs.config()#
    cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 30)
    cfg.enable_stream(rs.stream.infrared, 2, 1280, 720, rs.format.y8, 30)

    profile = pipe.start(cfg)
    dev = profile.get_device()
    depth_sensor = dev.first_depth_sensor()
    depth_sensor.set_option(rs.option.enable_auto_exposure, 0)
    depth_sensor.set_option(rs.option.exposure, float(4000))
    print(depth_sensor.get_option(rs.option.enable_auto_exposure))
    cv2.namedWindow("IR", cv2.WINDOW_NORMAL)
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
        cv2.imshow("IR", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    pipe.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
