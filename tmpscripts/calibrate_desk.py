import cv2
import numpy as np
import json
import os
import sys
import time
import pyk4a

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from kinect_pen.io.kinect_camera import KinectCamera
from kinect_pen.core.config_loader import load_config

# --- Configuration (User ChAruco) ---
config = load_config(os.path.join(project_root, "config.json"))
board_cfg = config.get("calibration_board", {})

CHARUCOBOARD_ROWCOUNT = board_cfg.get("rows", 8)
CHARUCOBOARD_COLCOUNT = board_cfg.get("cols", 12)
SQUARE_LENGTH = board_cfg.get("square_length", 0.024) # m
MARKER_LENGTH = board_cfg.get("marker_length", 0.018) # m
dict_val = board_cfg.get("dictionary_id", "DICT_4X4_100")

if isinstance(dict_val, str) and hasattr(cv2.aruco, dict_val):
    ARUCO_DICT_ID = getattr(cv2.aruco, dict_val)
else:
    ARUCO_DICT_ID = dict_val if isinstance(dict_val, int) else cv2.aruco.DICT_4X4_100

OUTPUT_FILE = "desk_calibration.json"

def main():
    print(f"Initializing Kinect Camera...")
    try:
        # Use Raw Color for calibration to ensure independent data streams
        cam = KinectCamera(use_raw_color=True)
        if not cam.open():
            print("Failed to open camera (cam.open() returned False)")
            return
    except Exception as e:
        print(f"Error opening camera: {e}")
        return

    print("Camera initialized.")
    
    # --- Setup ChAruco ---
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    # Check OpenCV version for API compatibility
    try:
        # OpenCV 4.7+ API
        board = cv2.aruco.CharucoBoard(
            (CHARUCOBOARD_COLCOUNT, CHARUCOBOARD_ROWCOUNT),
            SQUARE_LENGTH,
            MARKER_LENGTH,
            aruco_dict
        )
        board.setLegacyPattern(True)
    except AttributeError:
        # OpenCV < 4.7 API
        print("Using legacy OpenCV ArUco API")
        board = cv2.aruco.CharucoBoard_create(
            CHARUCOBOARD_COLCOUNT, CHARUCOBOARD_ROWCOUNT,
            SQUARE_LENGTH,
            MARKER_LENGTH,
            aruco_dict
        )
    
    # Create detector parameters
    detector_params = cv2.aruco.DetectorParameters()
    
    print(f"Looking for ChAruco Board: {CHARUCOBOARD_COLCOUNT}x{CHARUCOBOARD_ROWCOUNT}")
    print("Press 'SPACE' to capture and calibrate.")
    print("Press 'q' to quit.")

    try:
        while True:
            frame = cam.read_frame()
            if frame is None:
                continue

            # We can use either IR or Color for calibration.
            # User requested using Color Camera for calibration, but ensuring we have extrinsics/intrinsics handled.
            # Actually, "Need to initialize color camera for calibration instead of depth/ir camera... ensure color/depth and ir are raw images."
            # This implies we should calibrate using the Color Stream? Or just have it available?
            # Usually desk calibration is relative to the TRACKING camera (IR/Depth).
            # If we calibrate using Color, we get Plane in Color Frame.
            # Then we need Color->Depth Extrinsics to transform Plane to Depth Frame.
            
            # Let's assume the user wants to calibrate the DESK relative to the COLOR camera?
            # OR relative to IR but using Color for better detection?
            # "Use extra matrix to align coordinate system" suggests:
            # Detect in Color -> Get Pose in Color -> Transform to Depth.
            
            # Let's use Color Image for detection
            if frame.color is not None:
                vis_img = frame.color.copy()
                gray = cv2.cvtColor(vis_img, cv2.COLOR_BGR2GRAY)
            else:
                continue
                
            # 1. Detect Markers in Color Image
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=detector_params)
            
            charuco_retval = 0
            charuco_corners = None
            charuco_ids = None
            
            if len(corners) > 0:
                cv2.aruco.drawDetectedMarkers(vis_img, corners, ids)
                
                # 2. Interpolate ChAruco Corners using Color Intrinsics
                if frame.color_intrinsics is not None:
                    fx, fy, cx, cy = frame.color_intrinsics
                    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
                    dist_coeffs = np.zeros((5, 1)) # Assuming minimal distortion for now
                    
                    charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                        corners, ids, gray, board, cameraMatrix=camera_matrix, distCoeffs=dist_coeffs
                    )
                    
                    if charuco_retval > 0:
                        cv2.aruco.drawDetectedCornersCharuco(vis_img, charuco_corners, charuco_ids, (0, 255, 0))
                        cv2.putText(vis_img, f"Corners: {charuco_retval}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Resize for display if too large (720p is fine usually)
            cv2.imshow("Desk Calibration (Color)", vis_img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == 32: # SPACE
                if charuco_retval > 4 and frame.color_intrinsics is not None: 
                    print("Capturing...")
                    
                    # Solve PnP in Color Frame
                    fx, fy, cx, cy = frame.color_intrinsics
                    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
                    dist_coeffs = np.zeros((5, 1)) 
                    
                    valid_pose = False
                    rvec = None
                    tvec = None
                    
                    try:
                        all_obj_points = board.getChessboardCorners()
                        if charuco_ids is not None:
                            obj_points = []
                            for i in range(len(charuco_ids)):
                                # Fix deprecation warning by extracting scalar
                                cid_arr = charuco_ids[i]
                                cid = int(cid_arr[0]) if isinstance(cid_arr, (list, np.ndarray)) else int(cid_arr)
                                obj_points.append(all_obj_points[cid])
                            
                            obj_points = np.array(obj_points, dtype=np.float32)
                            
                            success, rvec, tvec = cv2.solvePnP(obj_points, charuco_corners, camera_matrix, dist_coeffs)
                            valid_pose = success
                    except Exception as e:
                        print(f"Error in PnP: {e}")
                        pass

                    if valid_pose:
                        print("Calibration Successful (In Color Frame)!")
                        
                        # We have Plane in Color Frame.
                        # We need to transform it to Depth Frame (World for Pen Tracker).
                        # Transformation T_depth_color (Color to Depth) ? Or T_color_depth?
                        # Usually SDK gives Extrinsics.
                        # We need to fetch extrinsics from PyK4A.
                        
                        # Fetch extrinsics on demand since we didn't fully implement in KinectCamera yet
                        # We can access the internal _cam object if needed, or update KinectCamera properly.
                        # For now, let's grab it here via private access or update KinectCamera properly.
                        # Actually I updated KinectCamera but left a pass.
                        # Let's get it directly here for now.
                        
                        try:
                            calib = cam._cam.calibration
                            # transform_from_color_to_depth: 
                            # R (3x3), T (3x1)
                            # P_depth = R * P_color + T
                            
                            # pyk4a calibration object exposes `get_extrinsic_parameters`
                            # returns (R, T)
                            # source_camera, target_camera
                            
                            # We want to transform Point from Color to Depth.
                            # So Source=Color, Target=Depth.
                            
                            extrinsics = calib.get_extrinsic_parameters(
                                pyk4a.CalibrationType.COLOR,
                                pyk4a.CalibrationType.DEPTH
                            )
                            
                            R_c2d = np.array(extrinsics[0], dtype=np.float32) # 3x3
                            T_c2d = np.array(extrinsics[1], dtype=np.float32).reshape(3, 1) # 3x1 (in mm? or m?)
                            # PyK4A usually returns meters if configured? No, SDK is usually mm.
                            # Wait, PyK4A wraps SDK. SDK uses mm.
                            # But our pen tracker uses meters.
                            # We need to check units.
                            # PyK4A get_extrinsic_parameters returns translation in MILLIMETERS usually.
                            # We should convert to meters.
                            
                            T_c2d_m = T_c2d / 1000.0
                            
                            print(f"Extrinsics Loaded. T={T_c2d_m.flatten()}")

                            # Transform Plane
                            # Plane in Color: n_c, d_c
                            # Point on plane p_c.
                            # p_d = R * p_c + T
                            # Normal n_d = R * n_c
                            
                            # 1. Get Plane in Color
                            R_board, _ = cv2.Rodrigues(rvec)
                            normal_c = R_board[:, 2] # Z axis of board
                            point_c = tvec.flatten() # Origin of board
                            
                            # 2. Transform to Depth
                            normal_d = R_c2d @ normal_c
                            point_d = (R_c2d @ point_c.reshape(3,1) + T_c2d_m).flatten()
                            
                            # 3. New Plane Equation
                            a, b, c = normal_d
                            d = -np.dot(normal_d, point_d)
                            
                            print(f"Plane Normal (Depth): [{a:.3f}, {b:.3f}, {c:.3f}]")
                            print(f"Plane D (Depth): {d:.3f}")
                            
                            # Save
                            calib_data = {
                                "desk_plane": [float(a), float(b), float(c), float(d)], # Adjusted key to match main.py expectation
                                "plane_equation": [float(a), float(b), float(c), float(d)], # Keep old key for compatibility
                                "square_size_m": SQUARE_LENGTH,
                                "marker_length_m": MARKER_LENGTH,
                                "type": "charuco_color_aligned",
                                "extrinsics_R": R_c2d.tolist(),
                                "extrinsics_T": T_c2d_m.tolist()
                            }
                            
                            with open(OUTPUT_FILE, 'w') as f:
                                json.dump(calib_data, f, indent=4)
                            
                            print(f"Saved to {os.path.abspath(OUTPUT_FILE)}")
                            
                            cv2.putText(vis_img, "SAVED!", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                            cv2.imshow("Desk Calibration (Color)", vis_img)
                            cv2.waitKey(1000)
                            break
                            
                        except Exception as e:
                            print(f"Failed to get/apply extrinsics: {e}")

                    else:
                        print("SolvePnP Failed.")
                else:
                    print("Not enough corners detected.")

    except KeyboardInterrupt:
        pass
    finally:
        cam.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
