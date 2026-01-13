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
from kinect_pen.algo.pen_tracker import IRPenTracker, IRPenConfig
from kinect_pen.core.types import Frame

def main():
    print("Initializing Kinect Camera...")
    cam = KinectCamera()
    if not cam.open():
        print("Failed to open Kinect Camera.")
        return

    print("Initializing IR Pen Tracker...")
    # You can adjust config here if needed
    config = IRPenConfig()
    # config.ir_threshold = 3000
    tracker = IRPenTracker(config)

    print("Starting loop. Press 'q' to quit.")
    
    while True:
        frame = cam.read_frame()
        if frame is None:
            continue

        result, debug_info = tracker.track_debug(frame)

        if frame.ir is not None:
            vis_scale = 10000.0
            ir_norm = np.clip(frame.ir / vis_scale, 0, 1.0)
            ir_disp = (ir_norm * 255).astype(np.uint8)
            
            vis_img = cv2.applyColorMap(ir_disp, cv2.COLORMAP_JET)
            
            blobs_uv = debug_info.get("blobs_uv", [])
            for i, (u, v) in enumerate(blobs_uv):
                cv2.circle(vis_img, (int(u), int(v)), 5, (0, 255, 255), -1)
                cv2.putText(vis_img, str(i), (int(u)+10, int(v)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            best_pair = debug_info.get("best_pair_indices")
            tip_idx = debug_info.get("tip_index")
            
            if best_pair is not None:
                idx1, idx2 = best_pair
                u1, v1 = blobs_uv[idx1]
                u2, v2 = blobs_uv[idx2]
                
                cv2.line(vis_img, (int(u1), int(v1)), (int(u2), int(v2)), (0, 255, 0), 2)
                
                if tip_idx is not None:
                    tip_u, tip_v = blobs_uv[tip_idx]
                    tail_idx = idx2 if idx1 == tip_idx else idx1
                    tail_u, tail_v = blobs_uv[tail_idx]
                    cv2.circle(vis_img, (int(tip_u), int(tip_v)), 8, (0, 0, 255), 2)
                    cv2.putText(vis_img, "TIP", (int(tip_u)+10, int(tip_v)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    cv2.circle(vis_img, (int(tail_u), int(tail_v)), 8, (255, 0, 0), 2)
                    cv2.putText(vis_img, "TAIL", (int(tail_u)+10, int(tail_v)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            lines = [
                f"FPS: {1.0 / (time.time() - frame.timestamp + 1e-6):.1f}" if frame.timestamp > 0 else "FPS: N/A",
                f"Blobs: {len(blobs_uv)}",
                f"Locked: {result.has_lock}",
            ]
            
            if result.has_lock:
                lines.append(f"Tip Pos: [{result.tip_pos_cam[0]:.3f}, {result.tip_pos_cam[1]:.3f}, {result.tip_pos_cam[2]:.3f}]")
                lines.append(f"Dir: [{result.direction[0]:.2f}, {result.direction[1]:.2f}, {result.direction[2]:.2f}]")

            y = 30
            for line in lines:
                cv2.putText(vis_img, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y += 30

            cv2.imshow("IR Debug Vis", vis_img)
        
        mask_ir = debug_info.get("mask_ir")
        if mask_ir is not None:
            cv2.imshow("IR Mask", mask_ir)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
                
    print("Closing camera...")
    cam.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
