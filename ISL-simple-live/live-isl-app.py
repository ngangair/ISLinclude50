import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import json, os, threading
import mediapipe as mp
import pyttsx3

# confirm your path
MODELS_PATH = r"C:\path\to\data\models"     # <---
SPLITS_PATH = r"C:\path\to\data\splits"     # <---

class TemporalAttention(tf.keras.layers.Layer):
    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.W = tf.keras.layers.Dense(units, activation='tanh')
        self.V = tf.keras.layers.Dense(1)
    def call(self, inputs):
        score   = self.V(self.W(inputs))
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(inputs * weights, axis=1)
        return context, weights

@st.cache_resource
def load_assets():
    m = tf.keras.models.load_model(
        os.path.join(MODELS_PATH, 'best_model_v2.keras'),
        custom_objects={'TemporalAttention': TemporalAttention}
    )
    with open(os.path.join(SPLITS_PATH, 'splits.json'), 'r') as f:
        splits = json.load(f)
    idx_to_word = {int(v): k for k, v in splits['label_map'].items()}
    return m, idx_to_word

model, idx_to_word = load_assets()

def normalize_hand(h):
    if h is None: return np.zeros(63, dtype=np.float32)
    c = np.array([[lm.x, lm.y, lm.z] for lm in h.landmark], dtype=np.float32)
    c -= c[0]
    s = np.linalg.norm(c[9])
    if s > 1e-6: c /= s
    return c.flatten()

def compute_angles(h):
    if h is None: return np.zeros(10, dtype=np.float32)
    lm = h.landmark
    def ang(a, b, c):
        va=np.array([lm[a].x,lm[a].y,lm[a].z])
        vb=np.array([lm[b].x,lm[b].y,lm[b].z])
        vc=np.array([lm[c].x,lm[c].y,lm[c].z])
        ba=va-vb; bc=vc-vb
        return np.arccos(np.clip(np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6),-1,1))
    return np.array([ang(1,2,3),ang(2,3,4),ang(5,6,7),ang(6,7,8),ang(9,10,11),
                     ang(10,11,12),ang(13,14,15),ang(14,15,16),ang(17,18,19),ang(18,19,20)],dtype=np.float32)

def compute_distances(h):
    if h is None: return np.zeros(5, dtype=np.float32)
    lm = h.landmark
    def d(a,b): return np.linalg.norm(np.array([lm[a].x-lm[b].x,lm[a].y-lm[b].y,lm[a].z-lm[b].z]))
    ref = d(0,9)+1e-6
    return np.array([d(4,8)/ref,d(4,20)/ref,d(8,20)/ref,d(5,17)/ref,d(0,12)/ref],dtype=np.float32)

def compute_palm_normal(h):
    if h is None: return np.zeros(3, dtype=np.float32)
    lm = h.landmark
    w=np.array([lm[0].x,lm[0].y,lm[0].z])
    i=np.array([lm[5].x,lm[5].y,lm[5].z])
    p=np.array([lm[17].x,lm[17].y,lm[17].z])
    n = np.cross(i-w, p-w)
    mag = np.linalg.norm(n)
    return (n/mag if mag>1e-6 else n).astype(np.float32)

def extract_frame_features(results, prev=None):
    static = np.concatenate([
        normalize_hand(results.left_hand_landmarks),
        normalize_hand(results.right_hand_landmarks),
        compute_angles(results.left_hand_landmarks),
        compute_angles(results.right_hand_landmarks),
        compute_distances(results.left_hand_landmarks),
        compute_distances(results.right_hand_landmarks),
        compute_palm_normal(results.left_hand_landmarks),
        compute_palm_normal(results.right_hand_landmarks),
    ])
    velocity = static - prev if prev is not None else np.zeros(162, dtype=np.float32)
    return np.concatenate([static, velocity]).astype(np.float32)

def speak_async(text):
    def _go():
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    threading.Thread(target=_go, daemon=True).start()

if "sentence"  not in st.session_state: st.session_state.sentence  = []
if "last_word" not in st.session_state: st.session_state.last_word = ""
if "sequence"  not in st.session_state: st.session_state.sequence  = []

st.set_page_config(page_title="ISL Recognition", layout="wide")
st.title("ISL Recognition")
st.caption("Sign to your webcam")

c1, c2 = st.columns(2)
c1.metric("Dataset",   "INCLUDE50")
c2.metric("ISL Signs", "50 Classes")
st.divider()

ctrl1, ctrl2, ctrl3 = st.columns([2, 1, 1])
with ctrl1:
    threshold = st.slider("Confidence Threshold", 0.4, 0.95, 0.65, 0.05)
with ctrl2:
    speak_btn = st.button("🔊  Speak Sentence", use_container_width=True)
with ctrl3:
    clear_btn = st.button("🗑️  Clear", use_container_width=True)

if clear_btn:
    st.session_state.sentence  = []
    st.session_state.last_word = ""
    st.session_state.sequence  = []

if speak_btn and st.session_state.sentence:
    speak_async(" ".join(st.session_state.sentence))

st.divider()

cam_col, info_col = st.columns([3, 2])
with cam_col:
    st.subheader("Camera Feed")
    frame_placeholder = st.empty()
with info_col:
    st.subheader("Recognized Sign")
    sign_placeholder = st.empty()
    conf_placeholder = st.empty()
    st.markdown("---")
    st.subheader("Sentence")
    sent_placeholder = st.empty()

sign_placeholder.markdown("### —")
conf_placeholder.progress(0.0, text="Waiting...")
sent_placeholder.markdown("*Start signing to build a sentence...*")

run = st.checkbox("▶️  Start Camera")

if run:
    cap      = cv2.VideoCapture(0)
    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_draw       = mp.solutions.drawing_utils
    prev_features = None
    frame_count   = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Cannot read from webcam.")
                break

            frame   = cv2.flip(frame, 1)
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            if results.left_hand_landmarks:
                mp_draw.draw_landmarks(frame, results.left_hand_landmarks,
                                       mp.solutions.holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_draw.draw_landmarks(frame, results.right_hand_landmarks,
                                       mp.solutions.holistic.HAND_CONNECTIONS)

            frame_placeholder.image(frame, channels="BGR", use_container_width=True)

            features      = extract_frame_features(results, prev_features)
            prev_features = features[:162]
            st.session_state.sequence.append(features)
            if len(st.session_state.sequence) > 30:
                st.session_state.sequence.pop(0)

            frame_count += 1
            if len(st.session_state.sequence) == 30 and frame_count % 3 == 0:
                X     = np.array(st.session_state.sequence, dtype=np.float32)[np.newaxis, ...]
                probs = model(X, training=False).numpy()[0]
                idx   = int(np.argmax(probs))
                conf  = float(probs[idx])
                word  = idx_to_word.get(idx, "?")
                display_word = word.replace('_', ' ')

                if conf >= threshold:
                    sign_placeholder.markdown(f"### {display_word}")
                    conf_placeholder.progress(conf, text=f"{conf*100:.0f}% confident")
                    if word != st.session_state.last_word:
                        st.session_state.sentence.append(display_word)
                        st.session_state.last_word = word
                        speak_async(display_word)
                        if len(st.session_state.sentence) > 12:
                            st.session_state.sentence.pop(0)
                else:
                    sign_placeholder.markdown("### ...")
                    conf_placeholder.progress(0.0, text="Waiting for clear sign...")

            if st.session_state.sentence:
                sent_placeholder.markdown(
                    "**" + "  →  ".join(st.session_state.sentence) + "**"
                )

    finally:
        cap.release()
        holistic.close()
