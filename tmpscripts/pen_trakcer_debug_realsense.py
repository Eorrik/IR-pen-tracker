import sys
import os
import time
import cv2
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "..", "src")
sys.path.append(src_path)

from ir_pen_tracker.io.realsense_camera import RealSenseCamera
from ir_pen_tracker.algo.stereo_pen_tracker import IRPenTrackerStereo, IRPenStereoConfig
from ir_pen_tracker.core.config_loader import load_config


def main():
    cfg_all = load_config(os.path.join(current_dir, "..", "config.json"))
    rs_cfg = cfg_all.get("camera", {}).get("realsense", {})
    preset = str(rs_cfg.get("preset", "high_accuracy")).lower()
    enable_ir = bool(rs_cfg.get("enable_ir", True))
    fps = int(rs_cfg.get("fps", 30))
    use_max = bool(rs_cfg.get("use_max_resolution", True))
    laser_power = rs_cfg.get("laser_power", None)
    exposure = rs_cfg.get("exposure", None)
    if use_max:
        cam = RealSenseCamera(depth_width=1280, depth_height=720,
                              color_width=1920, color_height=1080,
                              fps=fps, enable_ir=enable_ir, preset=preset,
                              laser_power=laser_power, exposure=exposure)
    else:
        cam = RealSenseCamera(fps=fps, enable_ir=enable_ir, preset=preset,
                              laser_power=laser_power, exposure=exposure)
    ok = cam.open()
    if not ok:
        return

    cfg = IRPenStereoConfig()
    tracker = IRPenTrackerStereo(cfg)

    cv2.namedWindow("Stereo IR Pen Debug", cv2.WINDOW_NORMAL)

    n = 0
    while True:
        frame = cam.read_frame()
        if frame is None:
            continue
        result, dbg = tracker.track_debug(frame)

        irL = frame.ir_main if frame.ir_main is not None else frame.ir
        irR = frame.ir_aux

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

        d_h, d_w = 480, 640
        ivL = ir_to_vis(irL) if irL is not None else None
        ivR = ir_to_vis(irR) if irR is not None else None
        vis_left = np.zeros((d_h, d_w, 3), dtype=np.uint8) if ivL is None else cv2.resize(cv2.cvtColor(ivL, cv2.COLOR_GRAY2BGR), (d_w, d_h))
        vis_right = np.zeros((d_h, d_w, 3), dtype=np.uint8) if ivR is None else cv2.resize(cv2.cvtColor(ivR, cv2.COLOR_GRAY2BGR), (d_w, d_h))

        orig_wL = irL.shape[1] if irL is not None else d_w
        orig_hL = irL.shape[0] if irL is not None else d_h
        orig_wR = irR.shape[1] if irR is not None else d_w
        orig_hR = irR.shape[0] if irR is not None else d_h
        sxL = d_w / float(orig_wL)
        syL = d_h / float(orig_hL)
        sxR = d_w / float(orig_wR)
        syR = d_h / float(orig_hR)

        if "left_uv" in dbg and "right_uv" in dbg:
            for (u, v) in dbg["left_uv"]:
                uu = int(round(u * sxL))
                vv = int(round(v * syL))
                cv2.circle(vis_left, (uu, vv), 1, (255, 255, 255), -1)
            for (u, v) in dbg["right_uv"]:
                uu = int(round(u * sxR))
                vv = int(round(v * syR))
                cv2.circle(vis_right, (uu, vv), 1, (255, 255, 255), -1)

        if "matched_uv" in dbg:
            for (l, r) in dbg["matched_uv"]:
                uL = int(round(l[0] * sxL))
                vL = int(round(l[1] * syL))
                uR = int(round(r[0] * sxR))
                vR = int(round(r[1] * syR))
                cv2.circle(vis_left, (uL, vL), 1, (0, 255, 0), -1)
                cv2.circle(vis_right, (uR, vR), 1, (0, 200, 200), -1)

        maskL = dbg.get("mask_left")
        maskR = dbg.get("mask_right")
        if maskL is None and irL is not None:
            _, maskL = cv2.threshold(irL, int(cfg.ir_threshold), 65535, cv2.THRESH_BINARY)
        if maskR is None and irR is not None:
            _, maskR = cv2.threshold(irR, int(cfg.ir_threshold), 65535, cv2.THRESH_BINARY)
        if maskL is not None:
            maskL_u8 = maskL.astype(np.uint8)
            maskL_u8 = cv2.resize(maskL_u8, (d_w, d_h))
            red = (maskL_u8 > 0)
            vis_left[red] = (0, 0, 255)
        if maskR is not None:
            maskR_u8 = maskR.astype(np.uint8)
            maskR_u8 = cv2.resize(maskR_u8, (d_w, d_h))
            red = (maskR_u8 > 0)
            vis_right[red] = (0, 0, 255)

        color_img = frame.color if frame.color is not None else np.zeros((1080, 1920, 3), dtype=np.uint8)
        ch, cw = color_img.shape[:2]
        color_disp = cv2.resize(color_img, (960, 540))
        tip = dbg.get("final_tip")
        tail = dbg.get("final_tail")
        if tip is not None and tail is not None:
            fx, fy, cx, cy = [float(x) for x in (frame.ir_main_intrinsics if frame.ir_main_intrinsics is not None else frame.intrinsics).tolist()]
            def to_uv(p):
                x, y, z = float(p[0]), float(p[1]), float(p[2])
                if z <= 0:
                    return None
                u = int(x * fx / z + cx)
                v = int(y * fy / z + cy)
                return (u, v)
            uv_tip = to_uv(tip)
            uv_tail = to_uv(tail)
            if uv_tip and uv_tail:
                a = (int(round(uv_tip[0] * sxL)), int(round(uv_tip[1] * syL)))
                b = (int(round(uv_tail[0] * sxL)), int(round(uv_tail[1] * syL)))
                cv2.line(vis_left, a, b, (0, 255, 255), 1)
                cv2.circle(vis_left, a, 1, (0, 0, 255), -1)
                cv2.circle(vis_left, b, 1, (255, 0, 0), -1)
            if frame.color_intrinsics is not None and frame.extrinsics_color_to_depth is not None:
                R = np.array(frame.extrinsics_color_to_depth[0], dtype=np.float32)
                t_mm = np.array(frame.extrinsics_color_to_depth[1], dtype=np.float32).reshape(3)
                Rdc = R.T
                tdc_m = -Rdc @ (t_mm / 1000.0)
                fx_c, fy_c, cx_c, cy_c = [float(x) for x in frame.color_intrinsics.tolist()]
                def to_uv_color(p_m):
                    z = float(p_m[2])
                    if z <= 0:
                        return None
                    pc = Rdc @ p_m + tdc_m
                    zc = float(pc[2])
                    if zc <= 0:
                        return None
                    u = fx_c * pc[0] / zc + cx_c
                    v = fy_c * pc[1] / zc + cy_c
                    uu = int(u * 960 / float(cw))
                    vv = int(v * 540 / float(ch))
                    return (uu, vv)
                uv_tip_c = to_uv_color(tip) if tip is not None else None
                uv_tail_c = to_uv_color(tail) if tail is not None else None
                if uv_tip_c and uv_tail_c:
                    cv2.line(color_disp, uv_tip_c, uv_tail_c, (0, 255, 255), 2)
                    cv2.circle(color_disp, uv_tip_c, 4, (0, 0, 255), -1)
                    cv2.circle(color_disp, uv_tail_c, 4, (255, 0, 0), -1)

        vis_img = np.hstack([vis_left, vis_right])
        cv2.putText(vis_img, f"Lock: {result.has_lock} Qual: {result.quality:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_img, f"L {orig_wL}x{orig_hL}->{d_w}x{d_h} sx {sxL:.3f} sy {syL:.3f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)
        cv2.putText(vis_img, f"R {orig_wR}x{orig_hR}->{d_w}x{d_h} sx {sxR:.3f} sy {syR:.3f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)
        if n == 0:
            print(f"IRL {orig_wL}x{orig_hL} IRR {orig_wR}x{orig_hR} display {d_w}x{d_h} sxL {sxL:.3f} syL {syL:.3f} sxR {sxR:.3f} syR {syR:.3f}")
            n = 1

        cv2.imshow("Stereo IR Pen Debug", vis_img)
        cv2.imshow("Color Pen Debug", color_disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cam.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
