import sys
import os
import time
import cv2
import numpy as np

# Add src to path so we can import kinect_pen
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "..", "src")
sys.path.append(src_path)

from kinect_pen.io.kinect_camera import KinectCamera

# Global variables for mouse interaction
mouse_x, mouse_y = -1, -1

def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y

def main():
    print("Initializing Kinect Camera...")
    cam = KinectCamera()
    if not cam.open():
        print("Failed to open Kinect Camera.")
        return

    print("Camera opened. Hover mouse over image to see raw IR value.")
    print("Press 'q' to quit.")

    window_name = "Raw IR Debug"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    try:
        while True:
            frame = cam.read_frame()
            if frame is None:
                continue

            if frame.ir is not None:
                # Prepare display image
                # Normalize for visibility: 0-10000 -> 0-255
                vis_scale = 5000.0 # Adjust this to see darker areas if needed
                ir_disp = (np.clip(frame.ir / vis_scale, 0, 1.0) * 255).astype(np.uint8)
                vis_img = cv2.cvtColor(ir_disp, cv2.COLOR_GRAY2BGR)

                # Get raw value at mouse position
                raw_val = 0
                val_str = "N/A"
                dtype_str = str(frame.ir.dtype)
                
                if 0 <= mouse_x < frame.ir.shape[1] and 0 <= mouse_y < frame.ir.shape[0]:
                    raw_val = frame.ir[mouse_y, mouse_x]
                    val_str = f"{raw_val}"

                # Overlay info
                # Top-left: Global info
                cv2.putText(vis_img, f"Dtype: {dtype_str}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # At mouse cursor: Value
                if mouse_x >= 0 and mouse_y >= 0:
                    text = f"Val: {val_str}"
                    # Draw text near cursor but ensure it stays in bounds
                    tx = min(mouse_x + 15, frame.ir.shape[1] - 150)
                    ty = max(mouse_y - 15, 30)
                    
                    # Background box for readability
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(vis_img, (tx-5, ty-th-5), (tx+tw+5, ty+5), (0,0,0), -1)
                    cv2.putText(vis_img, text, (tx, ty), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                cv2.imshow(window_name, vis_img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cam.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
