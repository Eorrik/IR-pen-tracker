import mediapipe
import os

print(f"MediaPipe version: {mediapipe.__version__}")
try:
    import mediapipe.python.solutions.pose as mp_pose
    print("Successfully imported mediapipe.python.solutions.pose")
except ImportError as e:
    print(f"Failed to import mediapipe.python.solutions.pose: {e}")

try:
    import mediapipe.python.solutions as solutions
    print("Successfully imported mediapipe.python.solutions")
except ImportError as e:
    print(f"Failed to import mediapipe.python.solutions: {e}")
