import cv2
import mediapipe as mp
import numpy as np
import csv
import os

# ================== Gesture List ==================
GESTURES = ["1","2","3","4","5","6","7","8","9","10","okay","no", "stop",
            "less","go","down","back"]

# ================== Utility Functions ==================
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))

    return np.degrees(angle)

def calculate_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)

# ================== MediaPipe Setup ==================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# ================== Dataset Setup ==================
dataset_path = "dataset/data.csv"
os.makedirs("dataset", exist_ok=True)

current_label_index = 0
sample_count = 0
samples_per_gesture = 30

if not os.path.exists(dataset_path):
    with open(dataset_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "thumb_angle", "index_angle", "middle_angle",
            "ring_angle", "pinky_angle",
            "thumb_index_dist", "index_middle_dist",
            "middle_ring_dist", "palm_angle", "label"
        ])

# ================== Main Loop ==================
while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    features = None  # IMPORTANT safety

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Extract landmarks
            lm = []
            for p in hand_landmarks.landmark:
                lm.append([p.x, p.y, p.z])

            # Feature engineering
            features = [
                calculate_angle(lm[2], lm[3], lm[4]),
                calculate_angle(lm[5], lm[6], lm[8]),
                calculate_angle(lm[9], lm[10], lm[12]),
                calculate_angle(lm[13], lm[14], lm[16]),
                calculate_angle(lm[17], lm[18], lm[20]),
                calculate_distance(lm[4], lm[8]),
                calculate_distance(lm[8], lm[12]),
                calculate_distance(lm[12], lm[16]),
                calculate_angle(lm[0], lm[5], lm[17])
            ]

    # Display info
    cv2.putText(
        frame,
        f"Gesture: {GESTURES[current_label_index]} | Sample: {sample_count}/30",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow("STEP 5: Dataset Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    # Save sample
    if key == ord('s') and features is not None:
        with open(dataset_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(features + [GESTURES[current_label_index]])

        sample_count += 1
        print(f"Saved {sample_count}/30 for {GESTURES[current_label_index]}")

        if sample_count == samples_per_gesture:
            sample_count = 0
            current_label_index += 1
            print("➡ Move to next gesture")

            if current_label_index == len(GESTURES):
                print("✅ Dataset collection complete")
                break

    # Exit
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()
