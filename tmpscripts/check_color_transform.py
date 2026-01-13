import pyk4a
from pyk4a import PyK4A, Config, ColorResolution, ImageFormat, FPS, DepthMode
import cv2
import numpy as np

def main():
    print("Checking Color -> Depth transformation...")
    cfg = Config(
        color_resolution=ColorResolution.RES_720P,
        color_format=ImageFormat.COLOR_NV12,
        camera_fps=FPS.FPS_5,
        depth_mode=DepthMode.NFOV_UNBINNED,
        synchronized_images_only=True,
    )
    
    cam = PyK4A(cfg)
    cam.start()
    
    capture = cam.get_capture()
    if capture is not None:
        print("Capture received.")
        if hasattr(capture, "transformed_color"):
            print("capture.transformed_color exists!")
            trans_color = capture.transformed_color
            if trans_color is not None:
                print(f"Transformed Color Shape: {trans_color.shape}")
            else:
                print("Transformed Color is None.")
        else:
            print("capture.transformed_color DOES NOT exist.")
    cam.stop()

if __name__ == "__main__":
    main()
