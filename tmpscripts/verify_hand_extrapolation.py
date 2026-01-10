import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Hands Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
)

def main():
    # Try using default webcam
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Could not open webcam. Please ensure a camera is connected.")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        vis = frame.copy()

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 1. Get Key Keypoints
                # WRIST = Index 0
                # MIDDLE_FINGER_MCP = Index 9
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                
                # Convert to pixel coordinates
                px_wrist = np.array([wrist.x * w, wrist.y * h])
                px_middle = np.array([middle_mcp.x * w, middle_mcp.y * h])

                # 2. Visualize Hand (Green)
                cv2.circle(vis, tuple(px_wrist.astype(int)), 5, (0, 255, 0), -1)
                cv2.circle(vis, tuple(px_middle.astype(int)), 5, (0, 255, 0), -1)
                cv2.line(vis, tuple(px_wrist.astype(int)), tuple(px_middle.astype(int)), (0, 255, 0), 2)

                # 3. Extrapolate Forearm (Blue)
                # Direction from Middle MCP to Wrist
                direction = px_wrist - px_middle
                
                # Normalize? No, keep scale relative to hand size
                # Let's say forearm is roughly 1.5x to 2.0x the length from MCP to Wrist? 
                # Actually MCP to Wrist is palm length. Forearm is usually longer.
                # Let's estimate elbow at 2.5x palm vector length from wrist.
                
                estimated_elbow = px_wrist + direction * 2.5
                
                # Draw extrapolated forearm
                cv2.circle(vis, tuple(estimated_elbow.astype(int)), 5, (255, 0, 0), -1) # Blue Elbow
                cv2.line(vis, tuple(px_wrist.astype(int)), tuple(estimated_elbow.astype(int)), (255, 0, 0), 3)
                
                cv2.putText(vis, "Extrapolated Forearm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow("Hand + Extrapolated Arm", vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
