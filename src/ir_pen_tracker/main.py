import sys
import os
import cv2
import numpy as np
import time
import json
from datetime import datetime
import pyk4a

# Add src to path to ensure imports work if run from project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from ir_pen_tracker.io.kinect_camera import KinectCamera
from ir_pen_tracker.algo.pen_tracker import IRPenTracker, IRPenConfig
from ir_pen_tracker.algo.calibration import DeskCalibration
from ir_pen_tracker.vis.ortho_vis import OrthoVisualizer
from ir_pen_tracker.core.config_loader import load_config

class KinectPenApp:
    def __init__(self):
        self.config = load_config()
        print(f"Loaded config: {self.config.keys()}")
        
        self.state = "INIT" # INIT, CALIBRATE, MAIN
        self.is_recording = False
        
        # Components
        self.cam = None
        self.tracker = None
        self.calib = DeskCalibration()
        self.ortho_vis = OrthoVisualizer()
        self.camera_type = str(self.config.get("camera", {}).get("type", "kinect")).lower()
        self.debug_skip_calibration = bool(self.config.get("debug", {}).get("skip_calibration", False))
        
        # Calibration State
        self.desk_plane_color = None # Temporary during calibration
        
        # Recording State
        self.rec_dir = None
        self.rec_out_color = None
        self.rec_file_pen = None
        self.rec_frame_idx = 0
        self.rec_start_time = 0
        
        # UI State
        self.mouse_pos = (-1, -1)
        self.btn_rec_rect = (850, 20, 100, 40) # x, y, w, h (Relative to 960x540 view)
        
        # Runtime
        self.desk_plane_depth = None # Final loaded/calibrated plane
        self.R_w = None # World Rotation
        self.origin_w = None # World Origin
        self.lowest_marker_to_tip_m = float(self.config.get("pen", {}).get("lowest_marker_to_tip_m", 0.13))
        
    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_pos = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            # Check Recording Button
            bx, by, bw, bh = self.btn_rec_rect
            if bx <= x <= bx + bw and by <= y <= by + bh:
                if self.state == "MAIN":
                    if self.is_recording:
                        self.stop_recording()
                    else:
                        self.start_recording()

    @staticmethod
    def project_point(p3d, intrinsics, dist_coeffs=None, scale=1.0):
        # p3d: (N, 3) or (3,)
            
        fx, fy, cx, cy = [float(x) for x in intrinsics.tolist()]
        
        # Construct Camera Matrix
        # Note: If we use dist_coeffs, we should project on the original resolution 
        # and THEN scale the coordinates, because distortion is relative to the optical center 
        # and image size of the calibration.
        # Scaling intrinsics BEFORE projection with distortion is tricky/incorrect 
        # if distortion is non-trivial.
        # So: Project with original intrinsics -> Scale output UV.
        
        x, y, z = float(p3d[0]), float(p3d[1]), float(p3d[2])
        if z <= 0:
            return None
        u = int(x * fx / z + cx)
        v = int(y * fy / z + cy)
        
        # Apply scale to the resulting coordinates
        if scale != 1.0:
            u *= scale
            v *= scale
            
        return (int(u), int(v))

    @staticmethod
    def transform_point(p_source, extrinsics):
        """
        Generic rigid body transformation.
        P_target = R * P_source + t
        extrinsics: {'R': [...], 't': [...]}
        """
        R = np.array(extrinsics['R'])
        t = np.array(extrinsics['t']).flatten() / 1000.0 # Convert mm to meters
        
        # Ensure p_source is column vector or handle numpy broadcasting
        # p_source shape: (3,) or (N, 3)
        # R @ p_source + t
        
        return R @ p_source + t
    
    @staticmethod
    def _depth_to_vis(depth_mm, d_min=200, d_max=2000):
        if depth_mm is None:
            return None
        arr = depth_mm.astype(np.uint16)
        mask = arr == 0
        arr = np.clip(arr, d_min, d_max)
        arr = ((arr - d_min) * 255.0 / max(1, (d_max - d_min))).astype(np.uint8)
        vis = cv2.applyColorMap(arr, cv2.COLORMAP_JET)
        vis[mask] = (0, 0, 0)
        return vis
    
    @staticmethod
    def _ir_to_vis(ir_u16):
        if ir_u16 is None:
            return None
        x = ir_u16.astype(np.float32)
        valid = x[x > 0]
        if valid.size > 0:
            p95 = float(np.percentile(valid, 95))
            p95 = max(p95, 1.0)
            x = np.clip(x, 0.0, p95)
            x = (x / p95) * 255.0
        else:
            x = np.clip(x, 0.0, 1500.0) / 1500.0 * 255.0
        return x.astype(np.uint8)

    def initialize_camera(self):
        print("Initializing Camera...")
        if self.camera_type == "realsense":
            from ir_pen_tracker.io.realsense_camera import RealSenseCamera
            rs_cfg = self.config.get("camera", {}).get("realsense", {})
            fps = int(rs_cfg.get("fps", 30))
            enable_ir = bool(rs_cfg.get("enable_ir", True))
            preset = str(rs_cfg.get("preset", "high_accuracy")).lower()
            laser_power = rs_cfg.get("laser_power", None)
            exposure = rs_cfg.get("exposure", None)
            self.cam = RealSenseCamera(depth_width=1280, depth_height=720,
                                           color_width=1920, color_height=1080,
                                           fps=fps, enable_ir=enable_ir, preset=preset,
                                           laser_power=laser_power, exposure=exposure)
        else:
            self.cam = KinectCamera(
                use_raw_color=True,
                color_resolution=pyk4a.ColorResolution.RES_1080P,
                camera_fps=pyk4a.FPS.FPS_30,
                depth_mode=pyk4a.DepthMode.NFOV_UNBINNED
            )
        self.cam.open()

    def start_recording(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.rec_dir = os.path.join(project_root, "recordings", f"rec_{timestamp}")
        
        os.makedirs(self.rec_dir, exist_ok=True)
        os.makedirs(os.path.join(self.rec_dir, "depth"), exist_ok=True)
        os.makedirs(os.path.join(self.rec_dir, "ir"), exist_ok=True)
        
        # Save Metadata
        meta = self.cam.get_calibration_data()
        meta["desk_plane"] = self.desk_plane_depth.tolist() if self.desk_plane_depth is not None else None
        with open(os.path.join(self.rec_dir, "meta.json"), 'w') as f:
            json.dump(meta, f, indent=2)
            
        # Video Writer (MJPG 1080p)
        self.rec_out_color = cv2.VideoWriter(
            os.path.join(self.rec_dir, "color.avi"),
            cv2.VideoWriter_fourcc(*'MJPG'),
            30.0,
            (1920, 1080)
        )
        
        self.rec_file_pen = open(os.path.join(self.rec_dir, "pen_data.jsonl"), 'w')
        self.rec_frame_idx = 0
        self.rec_start_time = time.time()
        self.is_recording = True
        print(f"Recording started: {self.rec_dir}")

    def stop_recording(self):
        if not self.is_recording:
            return
            
        self.is_recording = False
        if self.rec_out_color:
            self.rec_out_color.release()
            self.rec_out_color = None
        if self.rec_file_pen:
            self.rec_file_pen.close()
            self.rec_file_pen = None
            
        print(f"Recording saved to {self.rec_dir}")
        self.rec_dir = None

    def record_frame(self, frame, tracker_result):
        if not self.is_recording:
            return
            
        # 1. Color Video
        if frame.color is not None:
            c_img = frame.color
            if len(c_img.shape) == 3 and c_img.shape[2] == 4:
                c_img = cv2.cvtColor(c_img, cv2.COLOR_BGRA2BGR)
            self.rec_out_color.write(c_img)
            
        # 2. Raw Depth/IR
        if frame.depth is not None:
            cv2.imwrite(os.path.join(self.rec_dir, "depth", f"{self.rec_frame_idx:06d}.png"), 
                        frame.depth, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        if frame.ir is not None:
            cv2.imwrite(os.path.join(self.rec_dir, "ir", f"{self.rec_frame_idx:06d}.png"), 
                        frame.ir, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            
        # 3. Pen Data
        pen_entry = {
            "frame_id": self.rec_frame_idx,
            "timestamp": frame.timestamp,
            "has_lock": tracker_result.has_lock,
            "tip": tracker_result.tip_pos_cam.tolist() if tracker_result.tip_pos_cam is not None else None,
            "direction": tracker_result.direction.tolist() if tracker_result.direction is not None else None,
        }
        self.rec_file_pen.write(json.dumps(pen_entry) + "\n")
        
        self.rec_frame_idx += 1

    def run(self):
        self.initialize_camera()
        pen_cfg = IRPenConfig.from_dict(self.config.get("pen", {}))
        self.tracker = IRPenTracker(pen_cfg)
        
        # 1. Check Calibration or skip if debug
        if self.debug_skip_calibration:
            if self.calib.load():
                self.desk_plane_depth = self.calib.plane_equation
            self.state = "MAIN"
        else:
            if self.calib.load():
                print("Calibration found.")
                print("Press 'c' to Recalibrate, or any other key to continue...")
                choice = input("Recalibrate? [y/N]: ").strip().lower()
                if choice == 'y':
                    self.state = "CALIBRATE"
                else:
                    self.desk_plane_depth = self.calib.plane_equation
                    self.state = "MAIN"
            else:
                print("No calibration found.")
                print("Entering Calibration Mode.")
                self.state = "CALIBRATE"

        # Update World Transform if we have plane
        if self.desk_plane_depth is not None:
            self.R_w, self.origin_w = DeskCalibration.get_world_transform(self.desk_plane_depth)

        print(f"Starting App in {self.state} mode.")
        
        cv2.namedWindow("Kinect Pen App", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Kinect Pen App", self.on_mouse)
        
        while True:
            frame = self.cam.read_frame()
            if frame is None:
                continue
            
            vis_img = None
            
            if self.state == "CALIBRATE":
                vis_img = self.process_calibration(frame)
            elif self.state == "MAIN":
                vis_img = self.process_main(frame)
            
            if vis_img is not None:
                cv2.imshow("Kinect Pen App", vis_img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == 32: # Space
                if self.state == "CALIBRATE":
                    self.confirm_calibration()
                elif self.state == "MAIN":
                    if self.is_recording:
                        self.stop_recording()
                    else:
                        self.start_recording()
        self.stop_recording()
        self.cam.close()
        cv2.destroyAllWindows()

    def process_calibration(self, frame):
        if frame.color is None:
            return np.zeros((540, 960, 3), dtype=np.uint8)
            
        # Resize for display (1080p is too big)
        display_img = cv2.resize(frame.color, (960, 540))
        if display_img.shape[2] == 4:
            display_img = cv2.cvtColor(display_img, cv2.COLOR_BGRA2BGR)
            
        # Detect Board
        # Note: Detection needs full res image for best accuracy, but let's try.
        # Intrinsics: frame.color_intrinsics
        # We need to construct camera matrix from intrinsics [fx, fy, cx, cy]
        fx, fy, cx, cy = frame.color_intrinsics
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        dist_coeffs = np.zeros(5) # Assuming rectified or minimal distortion for now
        
        # Use full res color for detection
        color_bgr = frame.color
        if color_bgr.shape[2] == 4:
            color_bgr = cv2.cvtColor(color_bgr, cv2.COLOR_BGRA2BGR)
        
        # Ensure contiguous layout for OpenCV 4.11+ strict checks
        color_bgr = np.ascontiguousarray(color_bgr)
            
        board_cfg = self.config.get("calibration_board", {})
        
        # Ensure camera matrix is float32
        camera_matrix = camera_matrix.astype(np.float32)
        dist_coeffs = dist_coeffs.astype(np.float32)
        
        plane_cam, pose, num_markers = self.calib.detect_board(color_bgr, camera_matrix, dist_coeffs, board_config=board_cfg)
        
        status_text = f"Markers: {num_markers}"
        color = (0, 0, 255)
        
        if plane_cam is not None:
            rvec, tvec = pose
            self.desk_plane_color = plane_cam
            status_text += " | Board Detected! Press SPACE to Confirm."
            color = (0, 255, 0)
            
            # Draw Axis on display image (approximate)
            # Project 0,0,0 and axes
            # Needs OpenCV's drawFrameAxes which takes rvec/tvec
            # But we resized display_img. Need to scale intrinsics or project on full and resize.
            # Easier to project on full and resize.
            cv2.drawFrameAxes(color_bgr, camera_matrix, dist_coeffs, rvec, tvec, 0.1)
            display_img = cv2.resize(color_bgr, (960, 540))
        else:
            status_text += " | Looking for board..."
            
        cv2.putText(display_img, "CALIBRATION MODE", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(display_img, status_text, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return display_img

    def confirm_calibration(self):
        if self.desk_plane_color is not None:
            print("Calibration Confirmed.")
            # Transform to Depth Frame
            calib_data = self.cam.get_calibration_data()
            extrinsics = calib_data.get("extrinsics_color_to_depth")
            
            if extrinsics:
                self.desk_plane_depth = self.calib.transform_plane_color_to_depth(self.desk_plane_color, extrinsics)
                self.calib.plane_equation = self.desk_plane_depth
                self.calib.save()
                
                # Update World Transform
                self.R_w, self.origin_w = DeskCalibration.get_world_transform(self.desk_plane_depth)
                
                print("Transitioning to Main Mode.")
                self.state = "MAIN"
            else:
                print("Error: Missing extrinsics for transformation.")
        else:
            print("Cannot confirm: No board detected.")

    def process_main(self, frame):
        # Tracking
        result = self.tracker.track(frame)
        
        # 1. Camera View (Color) Base
        c_h, c_w = 540, 960
        if frame.color is not None:
            cam_view = cv2.resize(frame.color, (c_w, c_h))
            if cam_view.shape[2] == 4:
                cam_view = cv2.cvtColor(cam_view, cv2.COLOR_BGRA2BGR)
        else:
            cam_view = np.zeros((c_h, c_w, 3), dtype=np.uint8)

        d_h, d_w = 540, 600
        if self.camera_type == "kinect":
            if frame.ir is not None:
                ir_float = frame.ir.astype(np.float32)
                ir_norm = np.clip(ir_float, 0, 1500) / 1500.0 * 255.0
                ir_uint8 = ir_norm.astype(np.uint8)
                ir_view = cv2.cvtColor(ir_uint8, cv2.COLOR_GRAY2BGR)
                ir_view = cv2.resize(ir_view, (d_w, d_h))
            else:
                ir_view = np.zeros((d_h, d_w, 3), dtype=np.uint8)
        else:
            iv = self._ir_to_vis(frame.ir) if frame.ir is not None else None
            if iv is None:
                ir_view = np.zeros((d_h, d_w, 3), dtype=np.uint8)
            else:
                ir_view = cv2.resize(cv2.cvtColor(iv, cv2.COLOR_GRAY2BGR), (d_w, d_h))
            
        # Prepare Calibration Data for Projection
        calib_data = self.cam.get_calibration_data()
        color_intrinsics = None
        color_dist_coeffs = None
        extrinsics_d2c = None
        depth_intrinsics = None
        
        if calib_data.get("color_intrinsics"):
            color_intrinsics = np.array(calib_data["color_intrinsics"])
            
        if calib_data.get("color_dist_coeffs"):
            color_dist_coeffs = np.array(calib_data["color_dist_coeffs"])
        
        if calib_data.get("extrinsics_depth_to_color"):
            extrinsics_d2c = calib_data["extrinsics_depth_to_color"]

        if calib_data.get("depth_intrinsics"):
            depth_intrinsics = np.array(calib_data["depth_intrinsics"])
        
        # Process Pen Data
        if result.has_lock:
            # Depth Frame Coordinates
            tip_d = result.tip_pos_cam
            direction_d = result.direction
            
            # Calculate derived points (in Depth Frame)
            # Tail
            tail_d = result.tail_pos_cam
            real_tip_d = tip_d - direction_d * float(self.lowest_marker_to_tip_m)
            
            # Corrected Tip
            corrected_tip_d = real_tip_d.copy()
            is_touching = False
            compression_mm = 0.0
            
            if self.desk_plane_depth is not None:
                n = self.desk_plane_depth[:3]
                d_param = self.desk_plane_depth[3]
                dist = np.dot(n, real_tip_d) + d_param
                
                # Check intersection if penetrating (dist > 0 usually means below plane if normal points to camera)
                if dist > 0:
                    is_touching = True
                    v = real_tip_d - tip_d
                    denom = np.dot(n, v)
                    if abs(denom) > 1e-6:
                        dist_tip = np.dot(n, tip_d) + d_param
                        t = -dist_tip / denom
                        corrected_tip_d = tip_d + t * v
                        compression_mm = np.linalg.norm(corrected_tip_d - real_tip_d) * 1000.0
            
            # Update Visualizer (World Frame)
            if self.R_w is not None and self.origin_w is not None:
                p_w = self.R_w @ (corrected_tip_d - self.origin_w)
                hover_height = p_w[1]
                self.ortho_vis.update(p_w[0], p_w[1], p_w[2], hover_height=hover_height)
                
            # Draw on Camera View (Color Frame)
            def to_cam_uv(p_d_m):
                if p_d_m is None:
                    return None
                if self.camera_type == "kinect" and hasattr(self.cam, "_cam") and hasattr(self.cam._cam, "calibration"):
                    calib = self.cam._cam.calibration
                    p_d_mm = p_d_m * 1000.0
                    uv_depth = calib.convert_3d_to_2d(
                        p_d_mm,
                        pyk4a.CalibrationType.DEPTH,
                        pyk4a.CalibrationType.DEPTH
                    )
                    p_c_mm = calib.convert_2d_to_3d(
                        uv_depth,
                        p_d_mm[2],
                        pyk4a.CalibrationType.DEPTH,
                        pyk4a.CalibrationType.COLOR
                    )
                    uv_color = calib.convert_3d_to_2d(
                        p_c_mm,
                        pyk4a.CalibrationType.COLOR,
                        pyk4a.CalibrationType.COLOR
                    )
                    return (int(uv_color[0] * 0.5), int(uv_color[1] * 0.5))
                else:
                    calib_data = self.cam.get_calibration_data()
                    color_intr = calib_data.get("color_intrinsics")
                    extr_d2c = calib_data.get("extrinsics_depth_to_color")
                    if extr_d2c is None:
                        ec2d = calib_data.get("extrinsics_color_to_depth")
                        if ec2d is None:
                            return None
                        R = np.array(ec2d["R"])
                        t_mm = np.array(ec2d["t"]).flatten()
                        R_inv = R.T
                        t_inv_mm = -R_inv @ t_mm
                        extr_d2c = {"R": R_inv.tolist(), "t": t_inv_mm.tolist()}
                    Rdc = np.array(extr_d2c["R"])
                    tdc_m = np.array(extr_d2c["t"]).flatten() / 1000.0
                    pc_m = Rdc @ p_d_m + tdc_m
                    fx, fy, cx, cy = [float(x) for x in np.array(color_intr).tolist()]
                    z = float(pc_m[2])
                    if abs(z) < 1e-6:
                        return None
                    u = fx * pc_m[0] / z + cx
                    v = fy * pc_m[1] / z + cy
                    h0, w0 = frame.color.shape[:2] if frame.color is not None else (1080, 1920)
                    ud = int(u * 960 / w0)
                    vd = int(v * 540 / h0)
                    return (ud, vd)

            uv_tip = to_cam_uv(tip_d)
            if uv_tip is None:
                print(f"DEBUG: Tip projection failed. Tip_D: {tip_d}")
             
            uv_tail = to_cam_uv(tail_d)
            uv_real = to_cam_uv(real_tip_d)
            uv_corr = to_cam_uv(corrected_tip_d)
            
            if uv_tip and uv_tail:
                cv2.line(cam_view, uv_tip, uv_tail, (0, 255, 255), 1) # Yellow Axis
                cv2.circle(cam_view, uv_tip, 1, (0, 0, 255), -1) # Red Tip Marker
                cv2.circle(cam_view, uv_tail, 1, (255, 0, 0), -1) # Blue Tail Marker
                
                if uv_real:
                    cv2.line(cam_view, uv_tip, uv_real, (200, 200, 200), 1)
                    cv2.circle(cam_view, uv_real, 1, (255, 255, 255), -1) # White Real Tip
                
                if is_touching and uv_corr:
                    cv2.circle(cam_view, uv_corr, 1, (255, 0, 255), -1)
                    cv2.putText(cam_view, f"{compression_mm:.1f}mm", (uv_corr[0]+10, uv_corr[1]), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                elif uv_corr:
                    cv2.circle(cam_view, uv_corr, 1, (0, 255, 255), -1) # Cyan Hover
            if frame.ir is not None and frame.ir_main_intrinsics is not None:
                ir_h, ir_w = frame.ir.shape[:2]
                sx = d_w / float(ir_w)
                sy = d_h / float(ir_h)
                fx_i, fy_i, cx_i, cy_i = [float(x) for x in frame.ir_main_intrinsics.tolist()]
                def to_ir_uv(p_m):
                    z = float(p_m[2])
                    if z <= 0:
                        return None
                    u = fx_i * float(p_m[0]) / z + cx_i
                    v = fy_i * float(p_m[1]) / z + cy_i
                    return (int(round(u * sx)), int(round(v * sy)))
                uv_tip_i = to_ir_uv(tip_d)
                uv_tail_i = to_ir_uv(tail_d)
                uv_real_i = to_ir_uv(real_tip_d)
                uv_corr_i = to_ir_uv(corrected_tip_d)
                if uv_tip_i and uv_tail_i:
                    cv2.line(ir_view, uv_tip_i, uv_tail_i, (0, 255, 255), 1)
                    cv2.circle(ir_view, uv_tip_i, 1, (0, 0, 255), -1)
                    cv2.circle(ir_view, uv_tail_i, 1, (255, 0, 0), -1)
                    if uv_real_i:
                        cv2.line(ir_view, uv_tip_i, uv_real_i, (200, 200, 200), 1)
                        cv2.circle(ir_view, uv_real_i, 1, (255, 255, 255), -1)
                    if uv_corr_i:
                        color_corr = (255, 0, 255) if is_touching else (0, 255, 255)
                        cv2.circle(ir_view, uv_corr_i, 1, color_corr, -1)

        # Record
        self.record_frame(frame, result)
        
        # UI Overlays
        # Tracking Status
        if result.has_lock:
            cv2.putText(cam_view, "TRACKING", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(ir_view, "TRACKING", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(cam_view, "LOST", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(ir_view, "LOST", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        # Recording Button
        bx, by, bw, bh = self.btn_rec_rect
        btn_color = (0, 0, 255) if self.is_recording else (100, 100, 100)
        cv2.rectangle(cam_view, (bx, by), (bx+bw, by+bh), btn_color, -1)
        cv2.putText(cam_view, "REC" if self.is_recording else "START", (bx+10, by+28), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        if self.is_recording:
            elapsed = time.time() - self.rec_start_time
            cv2.putText(cam_view, f"{elapsed:.1f}s", (bx, by+60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(cam_view, f"{self.rec_dir}", (30, c_h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
        # 2. Ortho Views
        ortho_img = self.ortho_vis.draw() # Returns stacked top/side views (400x400)
        
        # Combine Layout
        # Canvas: 960 (Color) + 600 (Depth) + 400 (Ortho) = 1960 x 540
        final_h = 540
        final_w = c_w + d_w + 400
        canvas = np.zeros((final_h, final_w, 3), dtype=np.uint8)
        
        canvas[0:540, 0:960] = cam_view
        canvas[0:540, 960:960+d_w] = ir_view
        
        # Resize ortho to fit height if needed (it is 400 high, cam is 540 high)
        # Ortho is 400x400 (stacked 200+200)
        # Let's center it vertically
        y_offset = (540 - 400) // 2
        canvas[y_offset:y_offset+400, 960+d_w:960+d_w+400] = ortho_img
        
        return canvas

if __name__ == "__main__":
    app = KinectPenApp()
    app.run()
