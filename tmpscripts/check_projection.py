import sys
import os
import cv2
import numpy as np
import json
import time

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from kinect_pen.io.kinect_camera import KinectCamera
from pyk4a import ColorResolution, FPS, DepthMode

# --- Constants ---
DEPTH_H_TARGET = 540
COLOR_H_TARGET = 540

# --- Globals ---
mouse_x, mouse_y = -1, -1
mouse_in_side = None # 'left' (depth) or 'right' (color)

def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y, mouse_in_side
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y

def project_point(p3d, intrinsics, dist_coeffs=None, scale=1.0):
    fx, fy, cx, cy = [float(x) for x in intrinsics]
    
    if dist_coeffs is None:
        # Simple projection
        x, y, z = p3d
        if z <= 0: return None
        u = (x * fx) / z + cx
        v = (y * fy) / z + cy
        return (int(u * scale), int(v * scale))
    else:
        # Distortion
        pts = np.array([p3d], dtype=np.float32)
        # Construct Camera Matrix
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        D = np.array(dist_coeffs, dtype=np.float32)
        
        img_pts, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), K, D)
        u, v = img_pts[0][0]
        return (int(u * scale), int(v * scale))

def unproject_point(u, v, z, intrinsics):
    fx, fy, cx, cy = [float(x) for x in intrinsics]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.array([x, y, z])

def transform_point(p, R, t):
    # p: (3,)
    # R: (3,3)
    # t: (3,)
    # out = R @ p + t
    # Ensure t is in same units as p (meters)
    return R @ p + t

def main():
    print("Initializing Camera...")
    cam = KinectCamera(
        color_resolution=ColorResolution.RES_1080P,
        camera_fps=FPS.FPS_30,
        depth_mode=DepthMode.NFOV_UNBINNED
    )
    if not cam.open():
        print("Failed to open camera.")
        return

    # Load Calibration (Desk Plane)
    desk_plane = None
    calib_file = os.path.join(project_root, "desk_calibration.json")
    if os.path.exists(calib_file):
        try:
            with open(calib_file, 'r') as f:
                data = json.load(f)
                if "plane_equation" in data:
                    desk_plane = np.array(data["plane_equation"])
                    print(f"Loaded Desk Plane: {desk_plane}")
        except Exception as e:
            print(f"Failed to load desk calibration: {e}")

    cv2.namedWindow("Check Projection", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Check Projection", mouse_callback)

    print("Running... Press 'q' to quit.")
    
    try:
        while True:
            frame = cam.read_frame()
            if frame is None:
                continue

            # --- 1. Prepare Images ---
            
            # Depth Image (Left)
            # Normalize for visualization
            if frame.ir is not None:
                ir_float = frame.ir.astype(np.float32)
                ir_norm = np.clip(ir_float / 5000.0, 0, 1.0) # Adjust gain as needed
                depth_vis = (ir_norm * 255).astype(np.uint8)
                depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
                depth_vis = cv2.applyColorMap(depth_vis[:,:,0], cv2.COLORMAP_JET)
            else:
                depth_vis = np.zeros((576, 640, 3), dtype=np.uint8)

            # Color Image (Right)
            if frame.color is not None:
                color_vis = frame.color.copy()
                if color_vis.shape[2] == 4:
                    color_vis = cv2.cvtColor(color_vis, cv2.COLOR_BGRA2BGR)
            else:
                color_vis = np.zeros((1080, 1920, 3), dtype=np.uint8)

            # Resize to Target Heights
            h_d, w_d = depth_vis.shape[:2]
            scale_d = DEPTH_H_TARGET / h_d
            w_d_new = int(w_d * scale_d)
            depth_vis_resized = cv2.resize(depth_vis, (w_d_new, DEPTH_H_TARGET))

            h_c, w_c = color_vis.shape[:2]
            scale_c = COLOR_H_TARGET / h_c
            w_c_new = int(w_c * scale_c)
            color_vis_resized = cv2.resize(color_vis, (w_c_new, COLOR_H_TARGET))

            # --- 2. Handle Mouse & Projection ---
            
            # Calibration Data
            calib_data = cam.get_calibration_data()
            intrinsics_d = np.array(calib_data["depth_intrinsics"]) if calib_data["depth_intrinsics"] else None
            intrinsics_c = np.array(calib_data["color_intrinsics"]) if calib_data["color_intrinsics"] else None
            dist_c = np.array(calib_data["color_dist_coeffs"]) if calib_data["color_dist_coeffs"] else None
            
            ext_d2c = calib_data["extrinsics_depth_to_color"]
            ext_c2d = calib_data["extrinsics_color_to_depth"]

            # Determine where mouse is
            # Left image ends at w_d_new
            pt_to_draw_on_color = None
            pt_to_draw_on_depth = None
            
            # Access raw calibration object from PyK4A
            # We need to access the protected member _cam for this script
            k4a_calib = cam._cam.calibration
            
            if 0 <= mouse_x < w_d_new and 0 <= mouse_y < DEPTH_H_TARGET:
                # Mouse in Depth
                # Map to Original Depth Coordinates
                u_d = int(mouse_x / scale_d)
                v_d = int(mouse_y / scale_d)
                
                # Draw Crosshair on Depth
                cv2.drawMarker(depth_vis_resized, (mouse_x, mouse_y), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
                
                # Get Depth Value
                if frame.depth is not None and 0 <= u_d < w_d and 0 <= v_d < h_d:
                    z_mm = float(frame.depth[v_d, u_d])
                    
                    if z_mm > 0:
                        try:
                            # Use SDK convert_2d_to_2d (Depth -> Color)
                            # source_point2d: (x, y), source_depth: float (mm), source_camera, target_camera
                            uv_c_float = k4a_calib.convert_2d_to_2d(
                                (u_d, v_d),
                                z_mm,
                                pyk4a.CalibrationType.DEPTH,
                                pyk4a.CalibrationType.COLOR
                            )
                            
                            if uv_c_float is not None:
                                u_c, v_c = uv_c_float
                                # Apply Scale for Visualization
                                u_c_vis = int(u_c * scale_c)
                                v_c_vis = int(v_c * scale_c)
                                pt_to_draw_on_color = (u_c_vis, v_c_vis)
                        except Exception as e:
                            # Conversion might fail if point is out of bounds
                            pass
                            
            elif w_d_new <= mouse_x < (w_d_new + w_c_new) and 0 <= mouse_y < COLOR_H_TARGET:
                # Mouse in Color
                mx_rel = mouse_x - w_d_new
                my_rel = mouse_y
                
                # Map to Original Color Coordinates
                u_c = int(mx_rel / scale_c)
                v_c = int(my_rel / scale_c)
                
                # Draw Crosshair on Color
                cv2.drawMarker(color_vis_resized, (mx_rel, my_rel), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
                
                # For Color -> Depth, we need the depth of the pixel in Color Camera.
                # We calculate this by intersecting with the Desk Plane.
                
                if desk_plane is not None and intrinsics_c is not None and ext_c2d is not None:
                    # We still need 3D math to find the Z-depth on the desk plane
                    # But the final projection to Depth pixel can use convert_2d_to_2d
                    
                    # 1. Ray in Color Frame (Origin 0,0,0)
                    ray_dir_c = unproject_point(u_c, v_c, 1.0, intrinsics_c)
                    ray_dir_c /= np.linalg.norm(ray_dir_c)
                    ray_orig_c = np.array([0, 0, 0])
                    
                    # 2. Transform Ray to Depth Frame (to intersect with desk_plane which is in Depth Frame)
                    R_c2d = np.array(ext_c2d["R"])
                    t_c2d = np.array(ext_c2d["t"]).flatten() / 1000.0
                    
                    ray_orig_d = transform_point(ray_orig_c, R_c2d, t_c2d)
                    ray_dir_d = R_c2d @ ray_dir_c
                    
                    # 3. Intersect with Plane
                    n = desk_plane[:3]
                    d_plane = desk_plane[3]
                    
                    denom = np.dot(n, ray_dir_d)
                    if abs(denom) > 1e-6:
                        t_val = -(np.dot(n, ray_orig_d) + d_plane) / denom
                        if t_val > 0:
                            # Intersection Point in Depth Frame
                            p_int_d = ray_orig_d + t_val * ray_dir_d
                            
                            # We need Z in COLOR Frame for convert_2d_to_2d(COLOR -> DEPTH)
                            # P_c = R_d2c * P_d + t_d2c
                            # OR just project p_int_d back to Color Frame to get Z
                            
                            R_d2c = np.array(ext_d2c["R"])
                            t_d2c = np.array(ext_d2c["t"]).flatten() / 1000.0
                            
                            p_int_c = transform_point(p_int_d, R_d2c, t_d2c)
                            z_c_mm = p_int_c[2] * 1000.0
                            
                            if z_c_mm > 0:
                                try:
                                    # Use SDK convert_2d_to_2d (Color -> Depth)
                                    uv_d_float = k4a_calib.convert_2d_to_2d(
                                        (u_c, v_c),
                                        z_c_mm,
                                        pyk4a.CalibrationType.COLOR,
                                        pyk4a.CalibrationType.DEPTH
                                    )
                                    
                                    if uv_d_float is not None:
                                        u_d, v_d = uv_d_float
                                        u_d_vis = int(u_d * scale_d)
                                        v_d_vis = int(v_d * scale_d)
                                        pt_to_draw_on_depth = (u_d_vis, v_d_vis)
                                except:
                                    pass

            # Draw Points
            if pt_to_draw_on_color:
                cv2.circle(color_vis_resized, pt_to_draw_on_color, 5, (0, 0, 255), -1)
                cv2.putText(color_vis_resized, "Proj from Depth", (pt_to_draw_on_color[0]+10, pt_to_draw_on_color[1]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            if pt_to_draw_on_depth:
                cv2.circle(depth_vis_resized, pt_to_draw_on_depth, 5, (0, 255, 255), -1)
                cv2.putText(depth_vis_resized, "Raycast on Desk", (pt_to_draw_on_depth[0]+10, pt_to_draw_on_depth[1]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # --- 3. Combine and Show ---
            combined = np.hstack((depth_vis_resized, color_vis_resized))
            
            # Add Labels
            cv2.putText(combined, "Depth View (IR)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(combined, "Color View (Raw)", (w_d_new + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Check Projection", combined)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        cam.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
