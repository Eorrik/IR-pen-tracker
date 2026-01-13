import sys
import os
import time
import json
import cv2
import numpy as np
from collections import deque

# Add src to path so we can import kinect_pen
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "..", "src")
sys.path.append(src_path)
from pyk4a import PyK4A, Config, ColorResolution, ImageFormat, FPS, DepthMode

from kinect_pen.io.kinect_camera import KinectCamera
from kinect_pen.algo.pen_tracker import IRPenTracker, IRPenConfig
from kinect_pen.core.types import Frame

# Global mouse state
mouse_x, mouse_y = -1, -1

def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y

def project_point(p3d, intrinsics, dist_coeffs=None, scale=1.0):
    fx, fy, cx, cy = [float(x) for x in intrinsics] # Ensure it's a list/array of 4
    
    if dist_coeffs is None:
        x, y, z = p3d
        if z <= 0:
            return None
        u = (x * fx) / z + cx
        v = (y * fy) / z + cy
        return (int(u * scale), int(v * scale))
    else:
        # Use cv2.projectPoints for distortion
        pts = np.array([p3d], dtype=np.float32)
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        D = np.array(dist_coeffs, dtype=np.float32)
        
        # Points are already in Camera Frame, so rvec=0, tvec=0
        img_pts, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), K, D)
        u, v = img_pts[0][0]
        return (int(u * scale), int(v * scale))

def transform_point_to_color(p3d, extrinsics):
    """
    p3d: [x, y, z] in Depth Frame (Meters)
    extrinsics: {'R': 3x3, 't': 3x1 (mm)}
    Returns: [x, y, z] in Color Frame (Meters)
    """
    R = np.array(extrinsics['R'])
    t = np.array(extrinsics['t']).flatten() / 1000.0 # mm to m
    return R @ p3d + t

def draw_axes(img, intrinsics, length=0.1): # length in meters
    origin_3d = np.array([0.0, 0.0, 0.5]) # Draw axes floating at 0.5m front for visibility? 
    # Or just draw World Origin? Camera is World Origin. 
    # Drawing camera axes on image is weird (always center).
    # Let's draw a small coordinate system at bottom left of screen
    
    h, w = img.shape[:2]
    cx, cy = 50, h - 50
    cv2.arrowedLine(img, (cx, cy), (cx+40, cy), (0, 0, 255), 2) # X - Red
    cv2.putText(img, "X", (cx+45, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    cv2.arrowedLine(img, (cx, cy), (cx, cy-40), (0, 255, 0), 2) # Y - Green (Inverted for image coords usually, but camera Y is down)
    # Actually Camera Y is Down. So screen Y is Camera Y.
    # Camera X is Right. Screen X is Camera X.
    # Camera Z is Forward.
    
    cv2.putText(img, "Y", (cx, cy-45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

def get_world_transform(desk_plane):
    """
    Compute transformation from Camera to World (Desk) Frame.
    World Origin: Intersection of Camera Z-axis with Desk Plane.
    World Y: Plane Normal (pointing UP, opposite to Camera Y).
    World X: Camera X projected on Plane.
    World Z: Forward on Plane (X cross Y).
    """
    # Plane: ax + by + cz + d = 0
    n = desk_plane[:3]
    d = desk_plane[3]
    
    # 1. Compute Origin (Intersection of Camera Z axis: P=t*(0,0,1))
    # n.(0,0,t) + d = 0 => n_z * t + d = 0 => t = -d / n_z
    if abs(n[2]) < 1e-6:
        # Plane parallel to Z axis? Unlikely for desk.
        return None, None
        
    t = -d / n[2]
    origin = np.array([0, 0, t])
    
    # 2. Compute Basis Vectors
    # Y_W: Plane Normal.
    # Check direction. Camera Y is Down (0,1,0). Normal usually points Down too (positive Y component).
    # We want World Y to be UP (Height). So flip normal if it points Down.
    # Camera Y dot n > 0 => n points Down.
    if np.dot(n, np.array([0, 1, 0])) > 0:
        y_w = -n
    else:
        y_w = n
    y_w = y_w / np.linalg.norm(y_w)
    
    # X_W: Project Camera X (1,0,0) onto Plane
    x_c = np.array([1, 0, 0])
    # x_proj = x_c - (x_c . y_w) * y_w (using y_w as normal)
    # Note: y_w is normal to plane.
    x_w = x_c - np.dot(x_c, y_w) * y_w
    if np.linalg.norm(x_w) < 1e-6:
        # Camera X is normal to plane? Unlikely. Fallback to Camera Y cross Normal
        x_w = np.cross(np.array([0,1,0]), y_w)
    x_w = x_w / np.linalg.norm(x_w)
    
    # Z_W: x_w cross y_w
    z_w = np.cross(x_w, y_w)
    z_w = z_w / np.linalg.norm(z_w)
    
    # Rotation Matrix (Rows are X_W, Y_W, Z_W)
    # P_w = R * (P_c - Origin)
    R = np.vstack((x_w, y_w, z_w))
    
    return R, origin

class OrthoVisualizer:
    def __init__(self, width=400, height=400, x_range=(-0.2, 0.2), y_range=(-0.05, 0.15), z_range=(-0.2, 0.2)):
        self.w = width
        self.h = height
        # World Frame Ranges
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        
        self.canvas_top = np.zeros((height//2, width, 3), dtype=np.uint8)
        self.canvas_side = np.zeros((height//2, width, 3), dtype=np.uint8)
        
        self.history = deque(maxlen=100) # Store (x, y, z) in World Frame

    def update(self, x, y, z):
        self.history.append((x, y, z))

    def _draw_grid_and_points(self, canvas, h_val_getter, v_val_getter, h_range, v_range, label):
        """
        Generic drawer that maintains aspect ratio.
        h_range: (min, max) for horizontal axis
        v_range: (min, max) for vertical axis
        """
        cw, ch = canvas.shape[1], canvas.shape[0]
        canvas.fill(30)
        
        # Calculate isotropic scale
        h_len = h_range[1] - h_range[0]
        v_len = v_range[1] - v_range[0]
        
        if h_len <= 0 or v_len <= 0:
            return

        # Determine scale (pixels per meter) to fit both dimensions
        scale_h = cw / h_len
        scale_v = ch / v_len
        scale = min(scale_h, scale_v) * 0.9 # 90% fill
        
        # Center offsets
        cx = cw / 2
        cy = ch / 2
        
        h_mid = (h_range[0] + h_range[1]) / 2
        v_mid = (v_range[0] + v_range[1]) / 2
        
        def to_uv(h_val, v_val):
            u = int(cx + (h_val - h_mid) * scale)
            v = int(cy - (v_val - v_mid) * scale) # Y-up (v-down)
            return u, v
            
        # Draw Grid (0.1m steps)
        # Find start/end for grid lines based on view range
        
        # Vertical lines (Horizontal axis)
        start_h = np.floor(h_range[0] * 10) / 10.0
        for x in np.arange(start_h, h_range[1], 0.1):
            u, _ = to_uv(x, v_mid)
            if 0 <= u < cw:
                cv2.line(canvas, (u, 0), (u, ch), (50, 50, 50), 1)
                # Label axis if it's 0 or 0.5 or 1.0
                if abs(x % 0.5) < 1e-5 or abs(x) < 1e-5:
                     cv2.putText(canvas, f"{x:.1f}", (u+2, ch-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

        # Horizontal lines (Vertical axis)
        start_v = np.floor(v_range[0] * 10) / 10.0
        for y in np.arange(start_v, v_range[1], 0.1):
            _, v = to_uv(h_mid, y)
            if 0 <= v < ch:
                cv2.line(canvas, (0, v), (cw, v), (50, 50, 50), 1)
                if abs(y % 0.5) < 1e-5 or abs(y) < 1e-5:
                     cv2.putText(canvas, f"{y:.1f}", (5, v-2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

        # Draw Points
        for i, p3d in enumerate(self.history):
            h_val = h_val_getter(p3d)
            v_val = v_val_getter(p3d)
            
            u, v = to_uv(h_val, v_val)
            
            if 0 <= u < cw and 0 <= v < ch:
                # Color by Z (always use Z for color to show depth)
                z_val = p3d[2]
                norm = np.clip((z_val - self.z_range[0]) / (self.z_range[1] - self.z_range[0]), 0, 1)
                hue = int((1.0 - norm) * 120) 
                color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
                c = (int(color[0]), int(color[1]), int(color[2]))
                
                radius = 2 if i < len(self.history)-1 else 4
                cv2.circle(canvas, (u, v), radius, c, -1)

        # Label
        cv2.putText(canvas, f"{label} (Scale: {scale:.1f} px/m)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def draw(self, desk_y=None):
        # 1. Top View (XZ Plane)
        # H: X, V: Z
        self._draw_grid_and_points(
            self.canvas_top, 
            lambda p: p[0], # X
            lambda p: p[2], # Z
            self.x_range, 
            self.z_range, 
            "Top View (XZ)"
        )
        
        # 2. Side View (ZY Plane)
        # H: Z, V: Y
        # Camera Y is Down. To show "Up" as Up on screen, we invert Y.
        
        self._draw_grid_and_points(
            self.canvas_side,
            lambda p: p[2], # Z (Horizontal)
            lambda p: -p[1], # -Y (Vertical, so Up is Up)
            self.z_range,
            (-self.y_range[1], -self.y_range[0]), # Invert Y range
            "Side View (ZY)"
        )

        # Draw Desk Plane in Side View
        if desk_y is not None:
            # desk_y is in Camera Coords (positive down)
            # We map -Y to vertical. So Desk is at -desk_y.
            # We need to map -desk_y to pixels.
            
            # _draw_grid_and_points clears the canvas, so we need to modify it or draw after.
            # But we can't easily access the internal scale/transform of _draw_grid_and_points.
            # Let's just modify _draw_grid_and_points to accept extra drawables or just copy-paste logic?
            # Or better: make _draw_grid_and_points return the transform function.
            pass # Too complex to refactor now.
            
            # Quick hack: manually redraw desk line
            # We know the logic:
            cw, ch = self.canvas_side.shape[1], self.canvas_side.shape[0]
            v_range = (-self.y_range[1], -self.y_range[0])
            v_len = v_range[1] - v_range[0]
            # ... scale calculation ...
            # Let's just accept we can't easily draw the line inside _draw_grid_and_points without refactor.
            # Refactoring OrthoVisualizer slightly to support external drawing.

        return np.vstack((self.canvas_top, self.canvas_side))

    def get_transform(self, width, height, h_range, v_range):
        cw, ch = width, height
        h_len = h_range[1] - h_range[0]
        v_len = v_range[1] - v_range[0]
        if h_len <= 0 or v_len <= 0: return None
        
        scale_h = cw / h_len
        scale_v = ch / v_len
        scale = min(scale_h, scale_v) * 0.9
        
        cx = cw / 2
        cy = ch / 2
        h_mid = (h_range[0] + h_range[1]) / 2
        v_mid = (v_range[0] + v_range[1]) / 2
        
        def to_uv(h_val, v_val):
            u = int(cx + (h_val - h_mid) * scale)
            v = int(cy - (v_val - v_mid) * scale)
            return u, v
        return to_uv

    def draw_world(self):
        # 1. Top View (XZ in World)
        # H: X, V: Z
        # World X is Right, World Z is Forward (Up on screen)
        # Let's map World Z to Screen -Y (Up)
        
        self._draw_grid_and_points(
            self.canvas_top, 
            lambda p: p[0], # X
            lambda p: p[2], # Z
            self.x_range, 
            self.z_range, 
            "Top View (World XZ)"
        )
        
        # 2. Side View (ZY in World)
        # H: Z (Forward), V: Y (Up)
        # We want Z to go Right. Y to go Up.
        # Screen Y is Down. So we map World Y to Screen -Y.
        
        # Ranges
        h_range = self.z_range
        v_range = self.y_range # World Y range (-0.05 to 0.15)
        
        self._draw_grid_and_points(
            self.canvas_side,
            lambda p: p[2], # Z (Horizontal)
            lambda p: p[1], # Y (Vertical, Up)
            h_range,
            v_range,
            "Side View (World ZY)"
        )
        
        # Draw Desk Line at Y=0
        # Get transform
        to_uv = self.get_transform(self.w, self.h//2, h_range, v_range)
        if to_uv:
            # Line from (min_z, 0) to (max_z, 0)
            u1, v1 = to_uv(h_range[0], 0.0)
            u2, v2 = to_uv(h_range[1], 0.0)
            
            cv2.line(self.canvas_side, (u1, v1), (u2, v2), (0, 255, 255), 2)
            cv2.putText(self.canvas_side, "Desk Plane (Y=0)", (10, v1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        return np.vstack((self.canvas_top, self.canvas_side))


def main():
    print("Initializing Kinect Camera (Depth View)...")
    # Use 1080P to match the scale=0.5 assumption (1920x1080 -> 960x540)
    # Or better, we will calculate scale dynamically.
    cam = KinectCamera(
        color_resolution=ColorResolution.RES_1080P,
        camera_fps=FPS.FPS_30
    )
    if not cam.open():
        print("Failed to open Kinect Camera.")
        return

    print("Initializing IR Pen Tracker...")
    config = IRPenConfig()
    config.ir_threshold = 60000 
    tracker = IRPenTracker(config)

    ortho_view = OrthoVisualizer()

    cv2.namedWindow("Pen Tracker Debug", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Pen Tracker Debug", mouse_callback)

    print("Starting loop. Press 'q' to quit.")
    
    # Error metrics
    jitter_history = deque(maxlen=30)
    last_pos = None

    # Constants for Real Tip Estimation
    LOWEST_MARKER_TO_TIP = 0.13 # 13cm
    
    # Desk Plane State
    desk_plane = None # [a, b, c, d]
    desk_y_manual = 0.2 # Fallback
    
    # Load calibration
    calib_file = "desk_calibration.json"
    if os.path.exists(calib_file):
        with open(calib_file, 'r') as f:
            calib = json.load(f)
            if "plane_equation" in calib:
                desk_plane = np.array(calib["plane_equation"])
                print(f"Loaded Desk Calibration: {desk_plane}")

    while True:
        frame = cam.read_frame()
        if frame is None:
            continue

            # Track
            result, debug_info = tracker.track_debug(frame)

            # Get Calibration Data
            calib_data = cam.get_calibration_data()
            color_intrinsics = None
            color_dist_coeffs = None
            extrinsics_d2c = None
            if calib_data.get("color_intrinsics"):
                color_intrinsics = np.array(calib_data["color_intrinsics"])
            if calib_data.get("color_dist_coeffs"):
                color_dist_coeffs = np.array(calib_data["color_dist_coeffs"])
            if calib_data.get("extrinsics_depth_to_color"):
                extrinsics_d2c = calib_data["extrinsics_depth_to_color"]

            # --- Color Visualization ---
            color_vis = None
            vis_scale_factor = 1.0 # Scale from Camera Res to Display Res

            if frame.color is not None:
                color_vis = frame.color.copy()
                if color_vis.shape[2] == 4:
                    color_vis = cv2.cvtColor(color_vis, cv2.COLOR_BGRA2BGR)
                
                # Resize to 960x540
                target_w, target_h = 960, 540
                orig_h, orig_w = color_vis.shape[:2]
                
                vis_scale_factor = target_w / orig_w # Assuming aspect ratio is preserved
                
                color_vis = cv2.resize(color_vis, (target_w, target_h))
                
                # Draw on Color Image
                if result.has_lock and color_intrinsics is not None and extrinsics_d2c is not None:
                    # Get points (Tip, Tail, Real Tip, Corrected Tip)
                    # We need to recalculate them or extract from existing logic
                    # Let's extract them later in the loop or duplicate logic here?
                    # Duplicate logic is safer to ensure standalone correctness.
                    
                    if "final_tip" in debug_info and "final_tail" in debug_info:
                        tip = debug_info["final_tip"]
                        tail = debug_info["final_tail"]
                    else:
                        tip = result.tip_pos_cam
                        direction = result.direction
                        tail = tip + direction * 0.1

                    direction = result.direction
                    real_tip_rigid = tip - direction * LOWEST_MARKER_TO_TIP
                    
                    corrected_tip = real_tip_rigid.copy()
                    is_touching = False
                    
                    if desk_plane is not None:
                         n = desk_plane[:3]
                         d_param = desk_plane[3]
                         dist = np.dot(n, real_tip_rigid) + d_param
                         if dist > 0:
                             is_touching = True
                             v = real_tip_rigid - tip
                             denom = np.dot(n, v)
                             if abs(denom) > 1e-6:
                                 dist_tip = np.dot(n, tip) + d_param
                                 t = -dist_tip / denom
                                 corrected_tip = tip + t * v
                    
                    # Transform and Project
                    def to_color_uv(p_d):
                        p_c = transform_point_to_color(p_d, extrinsics_d2c)
                        return project_point(p_c, color_intrinsics, dist_coeffs=color_dist_coeffs, scale=0.5)
                    
                    uv_tip = to_color_uv(tip)
                    uv_tail = to_color_uv(tail)
                    uv_real = to_color_uv(real_tip_rigid)
                    uv_corr = to_color_uv(corrected_tip)
                    
                    if uv_tip and uv_tail:
                        cv2.line(color_vis, uv_tip, uv_tail, (0, 255, 255), 2)
                        cv2.circle(color_vis, uv_tip, 5, (0, 0, 255), -1)
                        cv2.circle(color_vis, uv_tail, 5, (255, 0, 0), -1)
                        
                        if uv_real:
                             cv2.line(color_vis, uv_tip, uv_real, (200, 200, 200), 1)
                             cv2.circle(color_vis, uv_real, 4, (255, 255, 255), -1)
                        
                        if is_touching and uv_corr:
                             cv2.circle(color_vis, uv_corr, 6, (255, 0, 255), -1)
                        elif uv_corr:
                             cv2.circle(color_vis, uv_corr, 4, (0, 255, 255), -1)

            # --- Visualization ---
            
            # Base: IR Image (Jet Colormap)
            if frame.ir is not None:
                # Log scale for better visibility of dark areas? Or just linear clip.
                # User wants to see valid depth pixels too.
                vis_scale = 5000.0
                ir_norm = np.clip(frame.ir / vis_scale, 0, 1.0)
                ir_disp = (ir_norm * 255).astype(np.uint8)
                vis_img = cv2.applyColorMap(ir_disp, cv2.COLORMAP_JET)
            else:
                h, w = frame.depth.shape
                vis_img = np.zeros((h, w, 3), dtype=np.uint8)

            # Draw ROI Rectangle (Blue)
            if "roi_corners" in debug_info:
                corners = debug_info["roi_corners"]
                pts = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(vis_img, [pts], True, (255, 0, 0), 1)
                
                if "roi_width" in debug_info:
                    cv2.putText(vis_img, f"ROI W: {debug_info['roi_width']:.1f}", 
                                (int(corners[1][0]), int(corners[1][1])), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 1. Debug: Draw Sample Points (Green) & Fitted Line (Yellow)
            if "sample_points_3d" in debug_info and len(debug_info["sample_points_3d"]) > 0:
                pts = debug_info["sample_points_3d"]
                # Downsample for speed
                for i in range(0, len(pts), 5):
                    uv = project_point(pts[i], frame.intrinsics)
                    if uv:
                        cv2.circle(vis_img, uv, 1, (0, 255, 0), -1)

            if "fitted_line" in debug_info and debug_info["fitted_line"] is not None:
                l_pt, l_dir = debug_info["fitted_line"]
                # Draw a segment around the center
                p_start = l_pt - l_dir * 0.2
                p_end = l_pt + l_dir * 0.2
                uv_s = project_point(p_start, frame.intrinsics)
                uv_e = project_point(p_end, frame.intrinsics)
                if uv_s and uv_e:
                    cv2.line(vis_img, uv_s, uv_e, (0, 255, 255), 1)

            # 2. 3D Pen Reprojection (Tip & Tail & Axes)
            if result.has_lock:
                # Use actual fitted marker positions on the line, not fixed length
                if "final_tip" in debug_info and "final_tail" in debug_info:
                    tip = debug_info["final_tip"]
                    tail = debug_info["final_tail"]
                else:
                    tip = result.tip_pos_cam
                    direction = result.direction
                    tail = tip + direction * 0.1 # Fallback
                
                direction = result.direction # Tip -> Tail

                # --- Calculate Real Tip (Rigid) ---
                # Direction is Tip->Tail. Real tip is "before" tip.
                # real_tip = tip - direction * 0.13
                # Wait, direction = tail - tip. 
                # So tip - direction * val -> moves away from tail?
                # Yes. tip is at 0. tail is at L. direction is (L-0).
                # tip - (tail-tip)*k -> 0 - L*k. Moves in opposite direction. Correct.
                real_tip_rigid = tip - direction * LOWEST_MARKER_TO_TIP
                
                # --- Desk Correction ---
                corrected_tip = real_tip_rigid.copy()
                is_touching = False
                compression_mm = 0.0
                
                if desk_plane is not None:
                    # General Plane: ax + by + cz + d = 0
                    n = desk_plane[:3]
                    d_param = desk_plane[3]
                    
                    # Distance of rigid tip to plane (signed)
                    # dist = n.p + d
                    dist = np.dot(n, real_tip_rigid) + d_param
                    
                    # If dist < 0, we are below the plane (assuming normal points up/to-camera)
                    # Check sign convention from calibration. 
                    # Calibration said: Normal [-0.055, 0.871, 0.489], D -0.264
                    # Camera (0,0,0) -> dist = -0.264.
                    # So Camera is on NEGATIVE side.
                    # Points further down (larger Y) -> dot product increases (since ny > 0).
                    # So dist increases as we go down.
                    # So "Below" desk means dist > 0 ??
                    # Wait. Normal Y is positive (0.87). Camera Y is positive Down.
                    # So Normal points Down (Same as Camera Y).
                    # So points with larger Y (deeper) have larger n.y * y.
                    # So dist increases as we go deeper.
                    # Camera is at Y=0. Dist = -0.264.
                    # Desk is at Y ~ 0.3. 0.87*0.3 - 0.264 ~ 0.26 - 0.264 ~ 0.
                    # So Desk is at dist=0.
                    # Points "Below" desk (larger Y) will have dist > 0.
                    # Points "Above" desk (smaller Y, closer to camera) will have dist < 0.
                    # Camera (Y=0) has dist < 0. Correct.
                    
                    # So if dist > 0, we are penetrating the desk.
                    if dist > 0:
                        is_touching = True
                        
                        # Intersect Line (tip, real_tip_rigid) with Plane
                        # Line: L(t) = tip + t * (real_tip_rigid - tip) for t in [0, 1]
                        # Solve n.(tip + t*v) + d = 0
                        # n.tip + t*(n.v) + d = 0
                        # t = -(n.tip + d) / (n.v)
                        
                        v = real_tip_rigid - tip
                        denom = np.dot(n, v)
                        
                        if abs(denom) > 1e-6:
                            dist_tip = np.dot(n, tip) + d_param
                            t = -dist_tip / denom
                            # If t is roughly in [0, 1.something], it's valid.
                            corrected_tip = tip + t * v
                            
                            # Calculate compression
                            # Distance from corrected_tip to real_tip_rigid
                            compression_mm = np.linalg.norm(corrected_tip - real_tip_rigid) * 1000.0
                        else:
                            # Parallel to plane? Just project along normal?
                            corrected_tip = real_tip_rigid - dist * n
                
                else:
                    # Fallback Manual Y Plane
                    # Plane Y = desk_y_manual. Camera Y is Down.
                    # If y > desk_y_manual, we are below.
                    if real_tip_rigid[1] > desk_y_manual:
                        is_touching = True
                        corrected_tip[1] = desk_y_manual
                        compression_mm = (real_tip_rigid[1] - desk_y_manual) * 1000.0
                
                # Update Ortho Visualizer
                ortho_view.update(corrected_tip[0], corrected_tip[1], corrected_tip[2])
                
                # --- Visualization: Tip, Tail, Real Tip, Axes ---
                uv_tip = project_point(tip, frame.intrinsics)
                uv_tail = project_point(tail, frame.intrinsics)
                uv_real = project_point(real_tip_rigid, frame.intrinsics)
                uv_corr = project_point(corrected_tip, frame.intrinsics)
                
                if uv_tip and uv_tail:
                    cv2.line(vis_img, uv_tip, uv_tail, (0, 255, 255), 2) # Yellow line for fitted pose
                    cv2.circle(vis_img, uv_tip, 5, (0, 0, 255), -1) # Red Tip Marker
                    cv2.circle(vis_img, uv_tail, 5, (255, 0, 0), -1) # Blue Tail Marker
                    
                    # Draw extension to real tip (White dashed)
                    if uv_real:
                         cv2.line(vis_img, uv_tip, uv_real, (200, 200, 200), 1)
                         cv2.circle(vis_img, uv_real, 4, (255, 255, 255), -1) # White Real Tip (Rigid)

                    # Draw Corrected Tip (Purple if touching)
                    if is_touching and uv_corr:
                         cv2.circle(vis_img, uv_corr, 6, (255, 0, 255), -1)
                         cv2.putText(vis_img, f"Press: {compression_mm:.1f}mm", (uv_corr[0]+10, uv_corr[1]), 
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                    elif uv_corr:
                         cv2.circle(vis_img, uv_corr, 4, (0, 255, 255), -1) # Cyan if hovering

                    # --- Draw Axes at Tip ---
                    axis_len = 0.05 # 5cm
                    
                    # Z Axis (Pen Body) - Blue
                    p_z = tip + direction * axis_len
                    uv_z = project_point(p_z, frame.intrinsics)
                    if uv_z:
                        cv2.line(vis_img, uv_tip, uv_z, (255, 0, 0), 2)
                        
                    # Construct arbitrary X/Y axes
                    # World Up (approximate, Camera Y is down)
                    world_up = np.array([0, -1, 0])
                    x_axis = np.cross(world_up, direction)
                    if np.linalg.norm(x_axis) < 1e-3:
                        x_axis = np.array([1, 0, 0])
                    x_axis /= np.linalg.norm(x_axis)
                    
                    y_axis = np.cross(direction, x_axis)
                    y_axis /= np.linalg.norm(y_axis)
                    
                    # X Axis - Red
                    p_x = tip + x_axis * axis_len
                    uv_x = project_point(p_x, frame.intrinsics)
                    if uv_x:
                        cv2.line(vis_img, uv_tip, uv_x, (0, 0, 255), 2)

                    # Y Axis - Green
                    p_y = tip + y_axis * axis_len
                    uv_y = project_point(p_y, frame.intrinsics)
                    if uv_y:
                        cv2.line(vis_img, uv_tip, uv_y, (0, 255, 0), 2)
            
            # Draw Raw Detections (Green dots)
            blobs_uv = debug_info.get("blobs_uv", [])
            for u, v in blobs_uv:
                cv2.circle(vis_img, (int(u), int(v)), 3, (0, 255, 0), 1)

            # 3. Coordinate Axes (Screen Space)
            draw_axes(vis_img, frame.intrinsics)

            # Draw Ortho Views
            # Update Ortho View with World Coordinates if available
            if desk_plane is not None and result.has_lock and "final_tip" in debug_info:
                # Calculate World Transform
                R_w, origin_w = get_world_transform(desk_plane)
                if R_w is not None:
                    # Transform corrected tip to World
                    # P_w = R * (P_c - Origin)
                    p_c = corrected_tip
                    p_w = R_w @ (p_c - origin_w)
                    
                    # Update Visualizer
                    ortho_view.update(p_w[0], p_w[1], p_w[2])
            elif result.has_lock and "final_tip" in debug_info:
                # Fallback to Camera Frame (Y-flipped to look somewhat upright?)
                # Or just raw camera coordinates. Camera Y is Down.
                # Let's just update with raw and hope for best or skip?
                # The user asked for World Frame. If no desk, we can't do World Frame properly.
                # Let's update with 0,0,0 or just skip update?
                # Maybe fallback to "Manual Desk Y" logic if needed, but let's stick to skip for now to avoid confusion.
                pass
            
            ortho_img = ortho_view.draw_world()
            
            # Combine Views (Side-by-Side)
            # vis_img: Depth View (Variable size, usually 640x576)
            # color_vis: Color View (960x540)
            
            combined_img = None
            
            if color_vis is not None:
                # Resize Depth to match Color Height (540)
                target_h = color_vis.shape[0] # 540
                h, w = vis_img.shape[:2]
                scale_d = target_h / h
                new_w = int(w * scale_d)
                vis_resized = cv2.resize(vis_img, (new_w, target_h))
                
                # Stack
                combined_img = np.hstack((vis_resized, color_vis))
            else:
                combined_img = vis_img

            # Show
            cv2.imshow("Pen Tracker Debug", combined_img)
            cv2.imshow("Ortho View", ortho_img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('w'):
                if desk_plane is None:
                    desk_y_manual -= 0.005
                else:
                    desk_plane[3] -= 0.001
            elif key == ord('s'):
                if desk_plane is None:
                    desk_y_manual += 0.005
                else:
                    desk_plane[3] += 0.001
            elif key == ord('c'):
                desk_plane = None
                print("Calibration cleared. Using manual Y.")
    cam.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
