import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
from ..core.interfaces import IBrushTracker
from ..core.types import Frame, BrushPoseVis


@dataclass
class IRPenStereoConfig:
    ir_threshold: int = 55000
    min_area: int = 2
    max_area: int = 2000
    min_disparity_px: float = 0.5
    min_depth_m: float = 0.1
    max_depth_m: float = 3.0
    baseline_m: Optional[float] = None
    kf_enabled: bool = True
    kf_process_var: float = 1e-2
    kf_measurement_var: float = 1e-4
    smoothing_alpha: float = 0.0

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "IRPenStereoConfig":
        cfg = IRPenStereoConfig()
        for k, v in d.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg


def _weighted_centroids(mask: np.ndarray, img: np.ndarray, min_area: int, max_area: int) -> List[Tuple[float, float, float, float]]:
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out: List[Tuple[float, float, float, float]] = []
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        ys, xs = np.where(labels == i)
        if xs.size == 0:
            continue
        weights = img[ys, xs].astype(np.float32)
        wsum = float(np.sum(weights))
        if wsum <= 1e-6:
            continue
        cx = float(np.sum(xs * weights) / wsum)
        cy = float(np.sum(ys * weights) / wsum)
        r = float(np.sqrt(area / np.pi))
        out.append((cx, cy, r, wsum))
    return out


def _depth_from_disparity(u_l: float, v_l: float, u_r: float, fx: float, cx: float, baseline_m: float) -> Tuple[float, float, float]:
    d = float(u_l - u_r)
    if abs(d) < 1e-6:
        return np.nan, np.nan, np.nan
    z = fx * baseline_m / d
    x = (u_l - cx) * z / fx
    return x, float(v_l), z


def _reproject(u: float, v: float, z: float, intr: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = [float(x) for x in intr.tolist()]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.array([x, y, z], dtype=np.float32)


class IRPenTrackerStereo(IBrushTracker):
    def __init__(self, config: IRPenStereoConfig = IRPenStereoConfig()):
        self._cfg = config
        self._last_tip: Optional[np.ndarray] = None
        self._last_ts: Optional[float] = None
        self._kf_left_states: List[Optional[np.ndarray]] = [None, None]
        self._kf_left_Ps: List[Optional[np.ndarray]] = [None, None]
        self._kf_right_states: List[Optional[np.ndarray]] = [None, None]
        self._kf_right_Ps: List[Optional[np.ndarray]] = [None, None]
        self._ema_left: List[Optional[np.ndarray]] = [None, None]
        self._ema_right: List[Optional[np.ndarray]] = [None, None]

    def track_debug(self, frame: Frame) -> Tuple[BrushPoseVis, Dict[str, Any]]:
        dbg: Dict[str, Any] = {
            "mask_left": None,
            "mask_right": None,
            "left_uv": [],
            "right_uv": [],
            "matched_uv": [],
            "points_3d": [],
            "final_tip": None,
            "final_tail": None
        }

        irL = frame.ir_main
        irR = frame.ir_aux
        if irL is None or irR is None:
            return BrushPoseVis(frame.timestamp, np.zeros(3), np.zeros(3), 0.0, False), dbg

        thr = int(self._cfg.ir_threshold)
        _, maskL = cv2.threshold(irL, thr, 65535, cv2.THRESH_BINARY)
        _, maskR = cv2.threshold(irR, thr, 65535, cv2.THRESH_BINARY)
        maskL = maskL.astype(np.uint8)
        maskR = maskR.astype(np.uint8)
        dbg["mask_left"] = maskL
        dbg["mask_right"] = maskR

        blobsL = _weighted_centroids(maskL, irL, self._cfg.min_area, self._cfg.max_area)
        blobsR = _weighted_centroids(maskR, irR, self._cfg.min_area, self._cfg.max_area)
        if len(blobsL) < 2 or len(blobsR) < 2:
            return BrushPoseVis(frame.timestamp, np.zeros(3), np.zeros(3), 0.0, False), dbg
        #print("length:1"+str(len(blobsL))+" "+str(len(blobsR)))
        blobsL.sort(key=lambda t: t[1])
        blobsR.sort(key=lambda t: t[1])
        (uL1, vL1, _, _), (uL2, vL2, _, _) = blobsL[:2]
        (uR1, vR1, _, _), (uR2, vR2, _, _) = blobsR[:2]

        left_uv = [(uL1, vL1), (uL2, vL2)]
        right_uv = [(uR1, vR1), (uR2, vR2)]

        dt = 1.0 / 30.0
        if self._last_ts is not None:
            dtt = float(frame.timestamp - self._last_ts)
            if dtt > 1e-6:
                dt = dtt

        q = float(self._cfg.kf_process_var)
        r = float(self._cfg.kf_measurement_var)
        a = float(self._cfg.smoothing_alpha)

        if self._cfg.kf_enabled:
            F = np.array([
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], dtype=np.float32)
            H = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0]
            ], dtype=np.float32)
            Q = np.diag([q, q, q, q]).astype(np.float32) * dt
            R = np.diag([r, r]).astype(np.float32)

            def kf_update(state_list, P_list, uv, idx):
                u, v = float(uv[idx][0]), float(uv[idx][1])
                x = state_list[idx]
                P = P_list[idx]
                if x is None or P is None:
                    x = np.array([u, v, 0.0, 0.0], dtype=np.float32)
                    P = np.eye(4, dtype=np.float32)
                x = F @ x
                P = F @ P @ F.T + Q
                z = np.array([u, v], dtype=np.float32)
                y = z - (H @ x)
                S = H @ P @ H.T + R
                K = P @ H.T @ np.linalg.inv(S)
                x = x + (K @ y)
                I4 = np.eye(4, dtype=np.float32)
                P = (I4 - K @ H) @ P
                state_list[idx] = x
                P_list[idx] = P
                return float(x[0]), float(x[1])

            # Left camera KFs
            uL1, vL1 = kf_update(self._kf_left_states, self._kf_left_Ps, left_uv, 0)
            uL2, vL2 = kf_update(self._kf_left_states, self._kf_left_Ps, left_uv, 1)
            # Right camera KFs
            uR1, vR1 = kf_update(self._kf_right_states, self._kf_right_Ps, right_uv, 0)
            uR2, vR2 = kf_update(self._kf_right_states, self._kf_right_Ps, right_uv, 1)

        # EMA on 2D points
        def ema_update(ema_list, uv, idx):
            u, v = float(uv[idx][0]), float(uv[idx][1])
            if a > 0.0:
                e = ema_list[idx]
                if e is None:
                    e = np.array([u, v], dtype=np.float32)
                else:
                    e = (1.0 - a) * e + a * np.array([u, v], dtype=np.float32)
                ema_list[idx] = e
                return float(e[0]), float(e[1])
            return u, v

        uL1, vL1 = ema_update(self._ema_left, [(uL1, vL1), (uL2, vL2)], 0)
        uL2, vL2 = ema_update(self._ema_left, [(uL1, vL1), (uL2, vL2)], 1)
        uR1, vR1 = ema_update(self._ema_right, [(uR1, vR1), (uR2, vR2)], 0)
        uR2, vR2 = ema_update(self._ema_right, [(uR1, vR1), (uR2, vR2)], 1)

        # Update debug with smoothed UVs
        dbg["left_uv"] = [(uL1, vL1), (uL2, vL2)]
        dbg["right_uv"] = [(uR1, vR1), (uR2, vR2)]

        # Pair after smoothing
        if abs(vL1 - vR1) + abs(vL2 - vR2) <= abs(vL1 - vR2) + abs(vL2 - vR1):
            pairs = [((uL1, vL1), (uR1, vR1)), ((uL2, vL2), (uR2, vR2))]
        else:
            pairs = [((uL1, vL1), (uR2, vR2)), ((uL2, vL2), (uR1, vR1))]
        dbg["matched_uv"] = pairs

        intrL = frame.ir_main_intrinsics if frame.ir_main_intrinsics is not None else frame.intrinsics
        baseline_m = frame.stereo_baseline_m
        if intrL is None or baseline_m is None or baseline_m <= 0:
            return BrushPoseVis(frame.timestamp, np.zeros(3), np.zeros(3), 0.0, False), dbg

        fx, fy, cx, cy = [float(x) for x in intrL.tolist()]

        pts3: List[np.ndarray] = []
        for (uL, vL), (uR, vR) in pairs:
            d = float(uL - uR)
            if abs(d) < self._cfg.min_disparity_px:
                continue
            z = fx * baseline_m / d
            if not (self._cfg.min_depth_m <= z <= self._cfg.max_depth_m):
                continue
            x = (uL - cx) * z / fx
            y = (vL - cy) * z / fy
            pts3.append(np.array([x, y, z], dtype=np.float32))

        dbg["points_3d"] = pts3
        if len(pts3) < 2:
            return BrushPoseVis(frame.timestamp, np.zeros(3), np.zeros(3), 0.0, False), dbg

        p1, p2 = pts3[:2]
        tip = p1
        tail = p2
        if self._last_tip is not None:
            d1 = float(np.linalg.norm(p1 - self._last_tip))
            d2 = float(np.linalg.norm(p2 - self._last_tip))
            if d2 < d1:
                tip, tail = p2, p1
        else:
            if p2[1] > p1[1]:
                tip, tail = p2, p1

        self._last_ts = float(frame.timestamp)
        self._last_tip = tip
        dbg["final_tip"] = tip
        dbg["final_tail"] = tail

        dir_vec = tail - tip
        nrm = float(np.linalg.norm(dir_vec))
        if nrm > 1e-6:
            dir_vec = dir_vec / nrm

        res = BrushPoseVis(
            timestamp=frame.timestamp,
            tip_pos_cam=tip,
            direction=dir_vec,
            quality=1.0,
            has_lock=True,
            tail_pos_cam=tail,
            mask_left = maskL
        )
        return res, dbg

    def track(self, frame: Frame) -> BrushPoseVis:
        res, _ = self.track_debug(frame)
        return res
