import os
import sys
import numpy as np
import pyrealsense2 as rs

def main():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    pipeline.start(config)
    ret, frames = pipeline.try_wait_for_frames()
    color_frame = frames.get_color_frame()
    color_profile = color_frame.get_profile()
    color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
    coeffs = np.array(list(color_intrinsics.coeffs), dtype=np.float32)
    print("RealSense Color 畸变系数:", coeffs.tolist())
    pipeline.stop()

if __name__ == "__main__":
    main()
