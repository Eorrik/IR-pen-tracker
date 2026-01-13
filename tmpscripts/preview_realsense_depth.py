import sys
import os
import time
import numpy as np
import cv2

proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(proj_root, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from kinect_pen.io.realsense_camera import RealSenseCamera

def depth_to_vis(depth_mm, d_min=200, d_max=2000):
    if depth_mm is None:
        return None
    arr = depth_mm.astype(np.uint16)
    mask = arr == 0
    arr = np.clip(arr, d_min, d_max)
    arr = ((arr - d_min) * 255.0 / max(1, (d_max - d_min))).astype(np.uint8)
    vis = cv2.applyColorMap(arr, cv2.COLORMAP_JET)
    vis[mask] = (0, 0, 0)
    return vis

def ir_to_vis(ir_u16):
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

def main():
    cam = RealSenseCamera(depth_width=1280, depth_height=720, color_width=1920, color_height=1080, fps=30, enable_ir=True, preset="high_accuracy")
    ok = cam.open()
    if not ok:
        return
    d_min, d_max = 200, 2000
    cv2.namedWindow("RealSense Preview", cv2.WINDOW_NORMAL)
    t0 = time.time()
    n = 0
    while True:
        f = cam.read_frame()
        if f is None:
            continue
        n += 1
        color = f.color
        depth_vis = depth_to_vis(f.depth, d_min, d_max)
        ir_vis = ir_to_vis(f.ir)
        if color is None:
            color = np.zeros((1080, 1920, 3), dtype=np.uint8)
        if depth_vis is None:
            depth_vis = np.zeros((f.depth.shape[0], f.depth.shape[1], 3), dtype=np.uint8)
        if ir_vis is None:
            ir_vis = np.zeros_like(f.depth, dtype=np.uint8)
        h_c, w_c = color.shape[:2]
        h_d, w_d = depth_vis.shape[:2]
        color_small = cv2.resize(color, (960, 540))
        depth_small = cv2.resize(depth_vis, (600, 540))
        ir_small = cv2.resize(cv2.cvtColor(ir_vis, cv2.COLOR_GRAY2BGR), (600, 540))
        left = depth_small
        right = color_small
        bottom = ir_small
        canvas_h = 540
        canvas_w = 960 + 600
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[:, :600] = left
        canvas[:, 600:] = right
        fps = n / max(1e-6, (time.time() - t0))
        cv2.putText(canvas, f"d[{d_min}-{d_max}]mm  fps:{fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("RealSense Preview", canvas)
        cv2.imshow("IR", bottom)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('+') or key == ord('='):
            d_max = min(6000, d_max + 100)
        elif key == ord('-') or key == ord('_'):
            d_max = max(d_min + 100, d_max - 100)
        elif key == ord('['):
            d_min = max(0, d_min - 50)
        elif key == ord(']'):
            d_min = min(d_max - 100, d_min + 50)
    cam.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
