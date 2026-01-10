import pyrealsense2 as rs
import numpy as np

def main():
    pipe = rs.pipeline()
    cfg = rs.config()
    try:
        cfg.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 30)
    except Exception:
        cfg.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
    profile = pipe.start(cfg)
    try:
        frames = pipe.wait_for_frames()
        try:
            ir_frame = frames.get_infrared_frame(1)
        except Exception:
            ir_frame = frames.get_infrared_frame()
        if ir_frame is None:
            print("IR frame: None")
            return
        arr = np.asanyarray(ir_frame.get_data())
        print("IR dtype:", arr.dtype)
        print("IR shape:", arr.shape)
        print("IR strides:", arr.strides)
        print("IR itemsize:", arr.itemsize)
        sp = ir_frame.get_profile().as_video_stream_profile()
        print("IR width:", sp.width())
        print("IR height:", sp.height())
    finally:
        try:
            pipe.stop()
        except Exception:
            pass

if __name__ == "__main__":
    main()
