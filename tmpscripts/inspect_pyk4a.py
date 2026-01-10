import pyk4a
from pyk4a import PyK4A, Config

print(f"pyk4a version: {pyk4a.__version__}")

try:
    # We might not have a device connected in this environment (the user has it), 
    # but we can try to instantiate PyK4A or inspect the class.
    # However, PyK4A usually requires a device to open.
    # Let's just print dir(pyk4a.calibration.Calibration) if possible without a device.
    
    import inspect
    print("pyk4a.calibration attributes:")
    print(dir(pyk4a.calibration))
    
    if hasattr(pyk4a.calibration, 'Calibration'):
        print("\nCalibration class members:")
        print(dir(pyk4a.calibration.Calibration))
        
except Exception as e:
    print(f"Error: {e}")
