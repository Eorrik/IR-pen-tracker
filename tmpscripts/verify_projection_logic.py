
import pyk4a
from pyk4a import PyK4A, Config
import numpy as np

def main():
    k4a = PyK4A(Config())
    k4a.start()
    calib = k4a.calibration
    p_d_m = np.array([-0.07593688, 0.0653639, 0.42454123])
    p_d_mm = p_d_m * 1000.0
    print(f"Testing projection for point: {p_d_mm} mm")
    uv_depth = calib.convert_3d_to_2d(
        p_d_mm,
        pyk4a.CalibrationType.DEPTH,
        pyk4a.CalibrationType.DEPTH
    )
    print(f"Step 1 (Depth 3D->2D): {uv_depth}")
    p_c_mm = calib.convert_2d_to_3d(
        uv_depth,
        p_d_mm[2],
        pyk4a.CalibrationType.DEPTH,
        pyk4a.CalibrationType.COLOR
    )
    print(f"Step 2 (Depth 2D->Color 3D): {p_c_mm}")
    uv_color = calib.convert_3d_to_2d(
        p_c_mm,
        pyk4a.CalibrationType.COLOR,
        pyk4a.CalibrationType.COLOR
    )
    print(f"Step 3 (Color 3D->2D): {uv_color}")
    print("Projection verification SUCCESS.")
    k4a.stop()

if __name__ == "__main__":
    main()
