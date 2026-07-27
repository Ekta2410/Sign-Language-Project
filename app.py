import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque, Counter
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from textblob import TextBlob
from gtts import gTTS
import os

# ================== CONFIG ==================
st.set_page_config(layout="wide")
st.title("🤟 Sign Language to Speech System")

# ================== LOAD MODEL ==================
model = joblib.load("sign_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ================== SESSION ==================
if "sentence" not in st.session_state:
    st.session_state.sentence = ""

if "predicted" not in st.session_state:
    st.session_state.predicted = ""

# ================== NLP ==================
def get_suggestions(text):
    blob = TextBlob(text.lower())
    suggestion = str(blob.correct())
    suggestions = [text.lower()]
    if suggestion != text.lower():
        suggestions.append(suggestion)
    return suggestions[:3]

# ================== SPEECH ==================
def speak_text(text):
    try:
        tts = gTTS(text=text, lang='en')
        tts.save("output.mp3")

        audio_file = open("output.mp3", "rb")
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/mp3")

        os.remove("output.mp3")

    except Exception as e:
        st.error(f"Speech Error: {e}")

# ================== MEDIAPIPE ==================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

CONFIDENCE_THRESHOLD = 0.3

# ================== FEATURE FUNCTIONS ==================
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def calculate_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

# ================== VIDEO PROCESSOR ==================
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.prediction_buffer = deque(maxlen=10)
        self.current_prediction = ""

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

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

                # 🔥 RAW + CONFIDENCE
                max_prob = np.max(probs)
                raw_pred = label_encoder.inverse_transform([np.argmax(probs)])[0]

                cv2.putText(
                    img,
                    f"Raw: {raw_pred} ({max_prob:.2f})",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

                # ✅ STABLE PREDICTION
                if max_prob >= CONFIDENCE_THRESHOLD:
                    self.prediction_buffer.append(raw_pred)

                    if len(self.prediction_buffer) == self.prediction_buffer.maxlen:
                        self.current_prediction = Counter(self.prediction_buffer).most_common(1)[0][0]

                # 🟢 DETECTED OUTPUT
                cv2.putText(
                    img,
                    f"Detected: {self.current_prediction}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

        return img

# ================== LAYOUT ==================
col_cam, col_controls = st.columns([2, 1])

# ================== CAMERA ==================
with col_cam:
    st.subheader("Camera Feed")

    webrtc_ctx = webrtc_streamer(
        key="gesture",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False}
    )

    if webrtc_ctx.video_processor:
        st.session_state.predicted = webrtc_ctx.video_processor.current_prediction

# ================== CONTROLS ==================
with col_controls:
    st.subheader("Controls")

    st.write("Detected Letter:", st.session_state.predicted)

    if st.button("Add Letter"):
        st.session_state.sentence += st.session_state.predicted

    if st.button("Space"):
        st.session_state.sentence += " "

    if st.button("Delete"):
        st.session_state.sentence = st.session_state.sentence[:-1]

    if st.button("Clear"):
        st.session_state.sentence = ""

    st.markdown("---")

    st.subheader("Sentence")
    st.write(st.session_state.sentence)

    if st.session_state.sentence:
        st.subheader("Suggestions")
        suggestions = get_suggestions(st.session_state.sentence)

        for i, s in enumerate(suggestions):
            if st.button(f"Use: {s}", key=i):
                st.session_state.sentence = s

    st.markdown("---")

    if st.button("🔊 Speak"):
        speak_text(st.session_state.sentence)

    st.download_button(
        "⬇ Download",
        st.session_state.sentence,
        file_name="output.txt"
    )

st.markdown("### 📚 Gesture Reference")
st.image("/Users/apple/project/images/symbol.png")
st.markdown("---")
st.image("/Users/apple/project/images/symbol 1-10 copy.tiff")
st.markdown("---")
st.image("/Users/apple/project/images/words symbol.png")

