import pyk4a
print("pyk4a attributes:", dir(pyk4a))
if hasattr(pyk4a, 'calibration'):
    print("\npyk4a.calibration attributes:", dir(pyk4a.calibration))
    if hasattr(pyk4a.calibration, 'Calibration'):
        print("\nCalibration class members:", dir(pyk4a.calibration.Calibration))
elif hasattr(pyk4a, 'Calibration'):
    print("\nCalibration class members (directly in pyk4a):", dir(pyk4a.Calibration))
