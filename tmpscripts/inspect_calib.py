
import pyk4a
from pyk4a import PyK4A, Config

def main():
    k4a = PyK4A(Config())
    k4a.start()
    calib = k4a.calibration
    print("Calibration attributes:")
    print(dir(calib))
    k4a.stop()

if __name__ == "__main__":
    main()
