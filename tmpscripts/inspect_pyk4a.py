import pyk4a
from pyk4a import PyK4A, Config

print(f"pyk4a version: {pyk4a.__version__}")

import inspect
print("pyk4a.calibration attributes:")
print(dir(pyk4a.calibration))

if hasattr(pyk4a.calibration, 'Calibration'):
    print("\nCalibration class members:")
    print(dir(pyk4a.calibration.Calibration))
