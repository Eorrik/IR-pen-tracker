import mediapipe
import os

print(f"MediaPipe path: {mediapipe.__path__}")
mp_dir = mediapipe.__path__[0]
print(f"Contents of {mp_dir}:")
for item in os.listdir(mp_dir):
    print(f"  {item}")

try:
    import mediapipe.python
    print("Successfully imported mediapipe.python")
except ImportError as e:
    print(f"Failed to import mediapipe.python: {e}")

try:
    import mediapipe.solutions
    print("Successfully imported mediapipe.solutions")
except ImportError as e:
    print(f"Failed to import mediapipe.solutions: {e}")
