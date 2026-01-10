import mediapipe as mp
try:
    print(f"mp.solutions: {mp.solutions}")
    print(f"mp.solutions.pose: {mp.solutions.pose}")
except AttributeError as e:
    print(f"Error accessing mp.solutions: {e}")
