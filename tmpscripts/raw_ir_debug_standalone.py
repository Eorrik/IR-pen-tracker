import cv2
import numpy as np
import pyk4a
from pyk4a import PyK4A, Config, ColorResolution, ImageFormat, FPS, DepthMode

# Global variables for mouse interaction
mouse_x, mouse_y = -1, -1

def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y

def main():
    print("Initializing PyK4A directly...")
    
    # Same config as the project
    cfg = Config(
        color_resolution=ColorResolution.RES_720P,
        color_format=ImageFormat.COLOR_NV12,
        camera_fps=FPS.FPS_5,
        depth_mode=DepthMode.NFOV_UNBINNED,
        synchronized_images_only=True,
    )

    try:
        cam = PyK4A(cfg)
        cam.start()
        print("Camera started successfully.")
        print("Displaying RAW IR (capture.ir) - NOT transformed/aligned.")
        print("Hover mouse to see pixel value. Press 'q' to quit.")

        window_name = "Standalone Raw IR"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_callback)

        while True:
            capture = cam.get_capture()
            if capture is None:
                continue

            # Access raw IR directly from sensor (not aligned to color)
            # This is usually 16-bit
            ir_raw = capture.ir
            ir_trans = capture.transformed_ir

            if ir_raw is not None:
                # Force uint16 interpretation just in case
                ir_data = ir_raw.astype(np.uint16)
                
                # Also process transformed IR if available
                ir_trans_data = None
                if ir_trans is not None:
                    ir_trans_data = ir_trans.astype(np.uint16)

                # Visualization scaling
                # Raw IR values can be up to 65535, but usually < 5000 for scene, > 10000 for retroreflectors
                vis_scale = 5000.0 
                
                # Prepare Raw Display
                ir_disp = (np.clip(ir_data / vis_scale, 0, 1.0) * 255).astype(np.uint8)
                vis_img = cv2.cvtColor(ir_disp, cv2.COLOR_GRAY2BGR)
                
                # Prepare Transformed Display (resize to match raw height for side-by-side)
                if ir_trans_data is not None:
                    ir_trans_disp = (np.clip(ir_trans_data / vis_scale, 0, 1.0) * 255).astype(np.uint8)
                    vis_trans_img = cv2.cvtColor(ir_trans_disp, cv2.COLOR_GRAY2BGR)
                    
                    # Resize trans to match raw height
                    h, w = vis_img.shape[:2]
                    th, tw = vis_trans_img.shape[:2]
                    scale = h / th
                    new_w = int(tw * scale)
                    vis_trans_resized = cv2.resize(vis_trans_img, (new_w, h))
                    
                    # Concatenate
                    final_vis = np.hstack((vis_img, vis_trans_resized))
                    
                    # Update mouse coordinate check for split view
                    # Mouse is on global window
                    pass
                else:
                    final_vis = vis_img

                # Info at mouse cursor
                raw_val = 0
                trans_val = 0
                val_str = "N/A"
                trans_val_str = "N/A"
                dtype_str = str(ir_data.dtype)
                
                h, w = ir_data.shape
                
                # Check if mouse is on Raw (Left) or Transformed (Right)
                # Note: This simple mouse logic assumes Raw is on the left. 
                # For precise debugging, let's just show values from Raw if on left, Transformed if on right.
                
                if ir_trans_data is not None:
                    # Raw is 0..w
                    # Trans is w..w+new_w
                    raw_w = w
                    trans_w = vis_trans_resized.shape[1]
                    
                    if 0 <= mouse_x < raw_w and 0 <= mouse_y < h:
                        raw_val = ir_data[mouse_y, mouse_x]
                        val_str = f"Raw: {raw_val}"
                    elif raw_w <= mouse_x < raw_w + trans_w and 0 <= mouse_y < h:
                        # Map back to transformed coordinates
                        rel_x = mouse_x - raw_w
                        orig_x = int(rel_x / scale)
                        orig_y = int(mouse_y / scale)
                        
                        if 0 <= orig_y < ir_trans_data.shape[0] and 0 <= orig_x < ir_trans_data.shape[1]:
                            trans_val = ir_trans_data[orig_y, orig_x]
                            val_str = f"Trans: {trans_val}"
                else:
                    if 0 <= mouse_x < w and 0 <= mouse_y < h:
                        raw_val = ir_data[mouse_y, mouse_x]
                        val_str = f"Raw: {raw_val}"

                # Overlay info
                cv2.putText(final_vis, f"Left: Raw IR | Right: Transformed IR", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if mouse_x >= 0 and mouse_y >= 0:
                    text = f"{val_str}"
                    # Draw text near cursor
                    tx = min(mouse_x + 15, final_vis.shape[1] - 200)
                    ty = max(mouse_y - 15, 30)
                    
                    # Background box
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(final_vis, (tx-5, ty-th-5), (tx+tw+5, ty+5), (0,0,0), -1)
                    cv2.putText(final_vis, text, (tx, ty), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                cv2.imshow(window_name, final_vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except Exception as e:
        print(f"Error: {e}")
    finally:
        try:
            cam.stop()
        except:
            pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
