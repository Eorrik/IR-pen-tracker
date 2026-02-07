import json
import os
import cv2
import numpy as np

class DeskCalibration:
    def __init__(self, calibration_file="desk_calibration.json"):
        self.calibration_file = calibration_file
        self.plane_equation = None # [a, b, c, d] in Depth Camera Coordinates
        self.board_origin = None   # [x, y, z] in Depth Camera Coordinates (meters)
        self.board_rotation = None # 3x3 rotation matrix: Board -> Depth Camera

    def load(self):
        if os.path.exists(self.calibration_file):
            with open(self.calibration_file, 'r') as f:
                data = json.load(f)
                self.plane_equation = np.array(data["desk_plane"])
                bo = data.get("board_origin", None)
                br = data.get("board_rotation", None)
                if bo is not None:
                    self.board_origin = np.array(bo, dtype=np.float32)
                if br is not None:
                    self.board_rotation = np.array(br, dtype=np.float32)
                print(f"Loaded desk calibration: {self.plane_equation}")
                return True
        return False

    def save(self):
        if self.plane_equation is not None:
            data = {"desk_plane": self.plane_equation.tolist()}
            if self.board_origin is not None:
                data["board_origin"] = self.board_origin.tolist()
            if self.board_rotation is not None:
                data["board_rotation"] = self.board_rotation.tolist()
            with open(self.calibration_file, 'w') as f:
                json.dump(data, f, indent=4)
            print(f"Saved desk calibration to {self.calibration_file}")

    def detect_board(self, image, camera_matrix, dist_coeffs=None, board_config=None):
        """
        Detects ChArUco board in the image and estimates the board pose (Z-up typically).
        Returns the plane equation [a, b, c, d] in Camera Coordinates.
        board_config: dict with keys: rows, cols, square_length, marker_length, dictionary_id
        """
        if board_config is None:
            # Default fallback
            rows = 5
            cols = 7
            square_len = 0.04
            marker_len = 0.03
            dict_id = cv2.aruco.DICT_4X4_50
        else:
            rows = board_config.get("rows", 5)
            cols = board_config.get("cols", 7)
            square_len = board_config.get("square_length", 0.04)
            marker_len = board_config.get("marker_length", 0.03)
            # Handle dictionary_id (int or string)
            dict_val = board_config.get("dictionary_id", cv2.aruco.DICT_4X4_50)
            if isinstance(dict_val, str):
                # We need to resolve string to int here or assume caller resolved it.
                # Let's import the helper if possible or just use a local map/getattr
                if hasattr(cv2.aruco, dict_val):
                    dict_id = getattr(cv2.aruco, dict_val)
                else:
                    dict_id = cv2.aruco.DICT_4X4_50 # Fallback
            else:
                dict_id = dict_val

        dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
        
        # Create Board with Version Compatibility
        board = cv2.aruco.CharucoBoard((cols, rows), square_len, marker_len, dictionary)
        board.setLegacyPattern(True)

        # Use grayscale for detection and refinement
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary)
        
        if ids is not None and len(ids) > 0:
            ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
            
            if charuco_corners is not None and len(charuco_corners) > 4:
                valid, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, board, camera_matrix, dist_coeffs, None, None)
                
                if valid:
                    R, _ = cv2.Rodrigues(rvec)
                    cam_in_board = -R.T @ tvec.flatten()
                    flip_x = cam_in_board[0] <= 0
                    flip_z = cam_in_board[2] <= 0
                    if flip_x:
                        R[:, 0] = -R[:, 0]
                    if flip_z:
                        R[:, 2] = -R[:, 2]
                    if (flip_x ^ flip_z):
                        R[:, 1] = -R[:, 1]
                    rvec_corr, _ = cv2.Rodrigues(R)
                    normal_cam = R @ np.array([0, 0, 1])
                    point_cam = tvec.flatten()
                    d = -np.dot(normal_cam, point_cam)
                    plane = np.array([normal_cam[0], normal_cam[1], normal_cam[2], d])
                    return plane, (rvec_corr, tvec), len(charuco_corners)
                    
        return None, None, 0

    def transform_plane_color_to_depth(self, plane_color, extrinsics_c2d):
        """
        Transforms plane equation from Color Camera Frame to Depth Camera Frame.
        extrinsics_c2d: {'R': [[...]], 't': [...]} (Rotation and Translation from Color to Depth)
        P_depth = R * P_color + t
        """
        R_c2d = np.array(extrinsics_c2d['R'])
        # Extrinsics are usually in mm, but our system uses meters.
        # Flatten to ensure 1D array.
        t_c2d = np.array(extrinsics_c2d['t']).flatten() / 1000.0
        
        n_color = plane_color[:3]
        d_color = plane_color[3]
        
        # New Normal: n_depth = R * n_color
        # Note: Normal vector is a direction, so just rotate it? 
        # Wait, P_d = R P_c + t.
        # Plane: n_c . P_c + d_c = 0
        # P_c = R^T (P_d - t)
        # n_c . (R^T (P_d - t)) + d_c = 0
        # (R n_c) . (P_d - t) + d_c = 0
        # (R n_c) . P_d - (R n_c) . t + d_c = 0
        # n_d = R n_c
        # d_d = d_c - n_d . t
        
        n_depth = R_c2d @ n_color
        d_depth = d_color - np.dot(n_depth, t_c2d)
        
        plane_depth = np.array([n_depth[0], n_depth[1], n_depth[2], d_depth])
        return plane_depth

    @staticmethod
    def transform_pose_color_to_depth(rvec_color, tvec_color, extrinsics_c2d):
        R_c2d = np.array(extrinsics_c2d['R'])
        t_c2d = np.array(extrinsics_c2d['t']).flatten() / 1000.0
        R_color, _ = cv2.Rodrigues(rvec_color)
        origin_depth = R_c2d @ tvec_color.flatten() + t_c2d
        R_depth = R_c2d @ R_color
        return R_depth, origin_depth

    @staticmethod
    def get_world_transform(desk_plane, board_rotation=None, board_origin=None):
        if board_rotation is not None and board_origin is not None:
            R_w = board_rotation.T
            origin_w = board_origin
            return R_w, origin_w
        if desk_plane is None:
            return None, None
            
        # Plane: ax + by + cz + d = 0
        n = desk_plane[:3]
        d = desk_plane[3]
        
        # 1. Compute Origin (Intersection of Camera Z axis: P=t*(0,0,1))
        # n.(0,0,t) + d = 0 => n_z * t + d = 0 => t = -d / n_z
        if abs(n[2]) < 1e-6:
            return None, None
            
        t = -d / n[2]
        origin = np.array([0, 0, t])
        
        # 2. Compute Basis Vectors
        z_w = n / np.linalg.norm(n)
        
        x_c = np.array([1, 0, 0])
        x_w = x_c - np.dot(x_c, z_w) * z_w
        if np.linalg.norm(x_w) < 1e-6:
            x_w = np.cross(np.array([0,1,0]), z_w)
        x_w = x_w / np.linalg.norm(x_w)
        
        y_w = np.cross(z_w, x_w)
        y_w = y_w / np.linalg.norm(y_w)
        
        R = np.vstack((x_w, y_w, z_w))
        
        return R, origin
