import pyrealsense2 as rs
import numpy as np

def main():
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 30)
    profile = pipe.start(cfg)
    frames = pipe.wait_for_frames()
    ir_frame = frames.get_infrared_frame(1)
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
    pipe.stop()

if __name__ == "__main__":
    main()
