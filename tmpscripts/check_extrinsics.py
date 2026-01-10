import cv2
import numpy as np
import sys
import os
import pyk4a
from pyk4a import PyK4A, Config, ColorResolution, ImageFormat, FPS, DepthMode

def main():
    print("Initializing Kinect...")
    cfg = Config(
        color_resolution=ColorResolution.RES_1080P,
        color_format=ImageFormat.COLOR_BGRA32,
        camera_fps=FPS.FPS_30,
        depth_mode=DepthMode.NFOV_UNBINNED,
        synchronized_images_only=True,
    )
    k4a = PyK4A(cfg)
    k4a.start()

    print("Kinect started. Press 'q' to quit.")
    print("Press '+' or '-' to adjust blending alpha.")

    alpha = 0.5

    try:
        while True:
            capture = k4a.get_capture()
            if capture.depth is None or capture.color is None:
                continue

            # 1. Get Color Image (BGRA -> BGR)
            color_frame = capture.color
            if color_frame.shape[2] == 4:
                color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGRA2BGR)
            
            # 2. Get Transformed Depth (Depth aligned to Color Camera)
            # This is the "Ground Truth" alignment provided by SDK
            transformed_depth = capture.transformed_depth
            
            # Normalize Depth for Visualization
            # Use same range for both to verify alignment
            depth_vis = np.zeros_like(color_frame, dtype=np.uint8)
            if transformed_depth is not None:
                # Clip to useful range (e.g. 0.2m to 2.0m)
                d_min, d_max = 200, 2000
                d_norm = np.clip(transformed_depth, d_min, d_max)
                d_norm = (d_norm - d_min) / (d_max - d_min) * 255.0
                d_uint8 = d_norm.astype(np.uint8)
                depth_vis = cv2.applyColorMap(d_uint8, cv2.COLORMAP_JET)

            # 3. Blend
            # Resize depth to match color if needed (SDK transform should match color res)
            if depth_vis.shape != color_frame.shape:
                depth_vis = cv2.resize(depth_vis, (color_frame.shape[1], color_frame.shape[0]))

            blended = cv2.addWeighted(color_frame, alpha, depth_vis, 1.0 - alpha, 0)

            # Resize for display (1080p is big)
            display_img = cv2.resize(blended, (960, 540))
            
            cv2.putText(display_img, f"Alpha: {alpha:.1f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("Check Extrinsics (SDK Transformed Depth)", display_img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('+') or key == ord('='):
                alpha = min(1.0, alpha + 0.1)
            elif key == ord('-') or key == ord('_'):
                alpha = max(0.0, alpha - 0.1)

    except KeyboardInterrupt:
        pass
    finally:
        k4a.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
