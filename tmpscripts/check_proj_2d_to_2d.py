import cv2
import numpy as np
import pyk4a
from pyk4a import PyK4A, Config, ColorResolution, ImageFormat, FPS, DepthMode

# Global variables for mouse state
mouse_depth = {'x': -1, 'y': -1, 'active': False}
mouse_color = {'x': -1, 'y': -1, 'active': False}

def mouse_callback_depth(event, x, y, flags, param):
    global mouse_depth, mouse_color
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_depth['x'] = x
        mouse_depth['y'] = y
        mouse_depth['active'] = True
        mouse_color['active'] = False

def mouse_callback_color(event, x, y, flags, param):
    global mouse_depth, mouse_color
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_color['x'] = x
        mouse_color['y'] = y
        mouse_color['active'] = True
        mouse_depth['active'] = False

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
    
    # Open windows
    cv2.namedWindow("Raw Depth View")
    cv2.setMouseCallback("Raw Depth View", mouse_callback_depth)
    
    cv2.namedWindow("Raw Color View")
    cv2.setMouseCallback("Raw Color View", mouse_callback_color)

    print("Kinect started.")
    print("Hover over Depth View to project to Color View.")
    print("Hover over Color View to project to Depth View.")
    print("Press 'q' to quit.")

    try:
        while True:
            capture = k4a.get_capture()
            if capture.depth is None or capture.color is None:
                continue

            # 1. Get Raw Images
            raw_depth = capture.depth # (H_d, W_d) uint16 mm
            raw_color = capture.color # (H_c, W_c, 4) BGRA
            
            # Helper for Color->Depth lookup
            # We use transformed_depth ONLY to look up Z for a given color pixel
            # because getting Z for a color pixel otherwise requires raycasting.
            transformed_depth = capture.transformed_depth 

            if raw_color.shape[2] == 4:
                raw_color_bgr = cv2.cvtColor(raw_color, cv2.COLOR_BGRA2BGR)
            else:
                raw_color_bgr = raw_color.copy()

            # Visualization for Depth
            # Normalize for display
            d_min, d_max = 200, 2000
            d_vis = np.clip(raw_depth, d_min, d_max)
            d_vis = (d_vis - d_min) / (d_max - d_min) * 255.0
            d_vis_bgr = cv2.applyColorMap(d_vis.astype(np.uint8), cv2.COLORMAP_JET)

            # --- Projection Logic ---
            
            # Case 1: Mouse on Depth -> Project to Color
            if mouse_depth['active']:
                u_d, v_d = mouse_depth['x'], mouse_depth['y']
                
                # Draw crosshair on Depth
                cv2.drawMarker(d_vis_bgr, (u_d, v_d), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
                
                # Check bounds
                if 0 <= v_d < raw_depth.shape[0] and 0 <= u_d < raw_depth.shape[1]:
                    z_mm = float(raw_depth[v_d, u_d])
                    
                    if z_mm > 0:
                        try:
                            # 1. Unproject Depth(u,v,z) -> 3D Color Frame (Directly)
                            # calib.convert_2d_to_3d can transform to target camera
                            calib = k4a.calibration
                            
                            point3d_color = calib.convert_2d_to_3d(
                                (u_d, v_d),
                                z_mm,
                                pyk4a.CalibrationType.DEPTH,
                                pyk4a.CalibrationType.COLOR
                            )
                            
                            # 2. Project 3D Color Frame -> Color(u,v)
                            uv_color = calib.convert_3d_to_2d(
                                point3d_color,
                                pyk4a.CalibrationType.COLOR,
                                pyk4a.CalibrationType.COLOR
                            )
                            
                            u_c, v_c = int(uv_color[0]), int(uv_color[1])
                            
                            # Draw projected point on Color
                            cv2.drawMarker(raw_color_bgr, (u_c, v_c), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
                            cv2.putText(raw_color_bgr, f"Proj: {u_c},{v_c}", (u_c+10, v_c), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            
                        except Exception as e:
                            # print(f"Projection error: {e}")
                            pass

            # Case 2: Mouse on Color -> Project to Depth
            if mouse_color['active']:
                u_c, v_c = mouse_color['x'], mouse_color['y']
                
                # Draw crosshair on Color
                cv2.drawMarker(raw_color_bgr, (u_c, v_c), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
                
                # Check bounds
                if transformed_depth is not None and 0 <= v_c < transformed_depth.shape[0] and 0 <= u_c < transformed_depth.shape[1]:
                    # Lookup Z from transformed depth (this is the Z value of the surface at this color pixel)
                    z_mm = float(transformed_depth[v_c, u_c])
                    
                    if z_mm > 0:
                        try:
                            calib = k4a.calibration
                            
                            # 1. Unproject Color(u,v,z) -> 3D Depth Frame (Directly)
                            point3d_depth = calib.convert_2d_to_3d(
                                (u_c, v_c),
                                z_mm,
                                pyk4a.CalibrationType.COLOR,
                                pyk4a.CalibrationType.DEPTH
                            )
                            
                            # 2. Project 3D Depth Frame -> Depth(u,v)
                            uv_depth = calib.convert_3d_to_2d(
                                point3d_depth,
                                pyk4a.CalibrationType.DEPTH,
                                pyk4a.CalibrationType.DEPTH
                            )
                            
                            u_d, v_d = int(uv_depth[0]), int(uv_depth[1])
                            
                            # Draw projected point on Depth
                            cv2.drawMarker(d_vis_bgr, (u_d, v_d), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
                            cv2.putText(d_vis_bgr, f"Proj: {u_d},{v_d}", (u_d+10, v_d), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                            
                        except Exception as e:
                            pass

            # Display
            cv2.imshow("Raw Depth View", d_vis_bgr)
            cv2.imshow("Raw Color View", raw_color_bgr)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        k4a.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
