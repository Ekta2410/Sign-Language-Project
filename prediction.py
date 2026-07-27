import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque, Counter

# NLP + Speech
from textblob import TextBlob
from gtts import gTTS
import pygame
import os

# ================== Load Model ==================
model = joblib.load("sign_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")


# ================== Suggestion System ==================
def get_suggestions(text):
    try:
        blob = TextBlob(text.lower())
        suggestion = str(blob.correct())

        suggestions = [text.lower()]

        if suggestion != text.lower():
            suggestions.append(suggestion)

        return suggestions[:3]

    except:
        return [text]

# ================== Text to Speech ==================
def speak_text(text):
    try:
        tts = gTTS(text=text, lang='en')
        tts.save("output.mp3")

        pygame.init()
        pygame.mixer.init()
        pygame.mixer.music.load("output.mp3")
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            continue

        os.remove("output.mp3")

    except Exception as e:
        print("Speech Error:", e)

# ================== Variables ==================
current_word = ""
last_appended = ""
predicted_label = ""
suggestions = []

# ================== Prediction Smoothing ==================
prediction_buffer = deque(maxlen=10)
CONFIDENCE_THRESHOLD = 0.3

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

# ================== MediaPipe ==================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# ================== Main Loop ==================
while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            lm = [[p.x, p.y, p.z] for p in hand_landmarks.landmark]

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

            probs = model.predict_proba([features])[0]
            max_prob = np.max(probs)
            pred_index = np.argmax(probs)
            raw_pred = label_encoder.inverse_transform([pred_index])[0]

            # Debug
            cv2.putText(frame, f"Raw: {raw_pred} ({max_prob:.2f})",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            # Stable prediction
            if max_prob >= CONFIDENCE_THRESHOLD:
                prediction_buffer.append(raw_pred)

                if len(prediction_buffer) == prediction_buffer.maxlen:
                    predicted_label = Counter(prediction_buffer).most_common(1)[0][0]

            # Display detected letter
            cv2.putText(frame, f"Detected: {predicted_label}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # ================== Suggestions ==================
    if current_word != "":
        suggestions = get_suggestions(current_word)
    else:
        suggestions = []

    # ================== Display ==================
    cv2.putText(frame, f"Sentence: {current_word}",
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    # Show suggestions
    y = 160
    for i, s in enumerate(suggestions):
        cv2.putText(frame, f"{i+1}: {s}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        y += 30

    # Show accuracy
    cv2.putText(frame, accuracy_text,
                (10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    cv2.imshow("Sign Language → Speech System", frame)

    # ================== Controls ==================
    key = cv2.waitKey(1) & 0xFF

    # ENTER → add letter
    if key == 13 and predicted_label != "" and predicted_label != last_appended:
        current_word += predicted_label
        last_appended = predicted_label

    # SPACE
    elif key == 32:
        if len(current_word) > 0 and current_word[-1] != " ":
            current_word += " "
        last_appended = ""

    # BACKSPACE
    elif key == ord('b'):
        current_word = current_word[:-1]
        last_appended = ""

    # CLEAR
    elif key == ord('c'):
        current_word = ""
        last_appended = ""

    # SELECT SUGGESTION
    elif key == ord('1') and len(suggestions) >= 1:
        current_word = suggestions[0]

    elif key == ord('2') and len(suggestions) >= 2:
        current_word = suggestions[1]

    elif key == ord('3') and len(suggestions) >= 3:
        current_word = suggestions[2]

    # SPEAK
    elif key == ord('s'):
        if current_word.strip() != "":
            speak_text(current_word)

    # EXIT
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()

