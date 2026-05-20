import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import json
import mediapipe as mp
from datetime import datetime

# confirm your path
MODELS_PATH   = r"data/models"          # <---
SPLITS_PATH   = r"data/splits"          # <---
NEW_DATA_PATH = r"data/new_samples"     # <---
os.makedirs(NEW_DATA_PATH, exist_ok=True)

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
        os.path.join(MODELS_PATH, 'best_model_v2.h5'),
        custom_objects={'TemporalAttention': TemporalAttention},
        compile=False
    )
    with open(os.path.join(SPLITS_PATH, 'splits.json'), 'r') as f:
        splits = json.load(f)
    idx_to_word = {int(v): k for k, v in splits['label_map'].items()}
    all_classes = sorted(splits['label_map'].keys())
    return m, idx_to_word, all_classes

model, idx_to_word, ALL_CLASSES = load_assets()

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

def save_sample(word, sequence_30frames):
    word_dir  = os.path.join(NEW_DATA_PATH, word)
    os.makedirs(word_dir, exist_ok=True)
    existing  = len([f for f in os.listdir(word_dir) if f.endswith('.npy')])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath  = os.path.join(word_dir, f"{word}_{timestamp}_{existing:03d}.npy")
    np.save(filepath, np.array(sequence_30frames, dtype=np.float32))
    return filepath

def count_by_word():
    counts = {}
    if not os.path.exists(NEW_DATA_PATH): return counts
    for word in os.listdir(NEW_DATA_PATH):
        d = os.path.join(NEW_DATA_PATH, word)
        if os.path.isdir(d):
            n = len([f for f in os.listdir(d) if f.endswith('.npy')])
            if n > 0: counts[word] = n
    return counts

for k, v in {
    "sentence":              [],
    "last_word":             "",
    "save_log":              [],
    "pending_confirmations": [],
    "confirm_id_counter":    0,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.set_page_config(page_title="ISL Live Recognition", layout="wide")
st.markdown("<style>.block-container{padding-top:1.2rem;}</style>", unsafe_allow_html=True)

st.title("ISL Live Recognition")
st.caption("⏬ Scroll down to... sign to your webcam... and to see all 50 words")

m1, m2 = st.columns(2)
m1.metric("Dataset",   "INCLUDE50")
m2.metric("ISL Signs", "50 Classes")
st.divider()

cam_col, panel_col = st.columns([3, 2])

with cam_col:
    st.subheader("Camera Feed")
    frame_placeholder = st.empty()

with panel_col:

    st.subheader("Recognized Sign")
    sign_placeholder = st.empty()
    conf_placeholder = st.empty()
    sign_placeholder.markdown("### —")
    conf_placeholder.progress(0.0, text="Waiting...")

    st.markdown("---")

    st.subheader("Recognized Words")
    sent_placeholder = st.empty()
    sent_placeholder.markdown("*Start signing to build a sentence...*")

    st.markdown("---")

    st.subheader("Select Words")
    st.caption("Tap to confirm correctly recognized words")

    if st.session_state.pending_confirmations:
        for item in list(st.session_state.pending_confirmations):
            col_word, col_yes, col_no = st.columns([3, 1, 1])
            col_word.markdown(f"**{item['display']}**")

            if col_yes.button("✅ Yes", key=f"yes_{item['id']}"):
                save_sample(item['word'], item['sequence'])
                ts = datetime.now().strftime("%H:%M:%S")
                st.session_state.save_log.append(f"[{ts}]  '{item['display']}' saved")
                st.session_state.pending_confirmations = [
                    p for p in st.session_state.pending_confirmations
                    if p['id'] != item['id']
                ]
                st.rerun()

            if col_no.button("❌ No", key=f"no_{item['id']}"):
                display = item['display']
                if display in st.session_state.sentence:
                    st.session_state.sentence.remove(display)
                st.session_state.pending_confirmations = [
                    p for p in st.session_state.pending_confirmations
                    if p['id'] != item['id']
                ]
                if st.session_state.last_word == item['word']:
                    st.session_state.last_word = ""
                st.rerun()
    else:
        st.caption("*Words will appear here once they are recognized & camera feed is stopped...*")

    st.markdown("---")

    st.subheader("Saved Samples")
    st.caption("Confirmed words are saved immediately as numeric feature sequences")
    if st.session_state.save_log:
        for entry in st.session_state.save_log[-5:]:
            st.markdown(f"✅ `{entry}`")
    else:
        st.caption("*No samples saved yet in this session*")

st.divider()

run = st.checkbox("▶️  Start Camera")

ctrl1, ctrl2 = st.columns([3, 1])
with ctrl1:
    threshold = st.slider("Confidence Threshold", 0.3, 0.95, 0.65, 0.05)
with ctrl2:
    clear_btn = st.button("🗑️  Clear All", use_container_width=True)

if clear_btn:
    st.session_state.sentence              = []
    st.session_state.last_word             = ""
    st.session_state.pending_confirmations = []
    st.rerun()

st.divider()

with st.expander("📋  Browse All 50 ISL Signs"):
    saved_counts = count_by_word()
    cols = st.columns(5)
    for i, word in enumerate(ALL_CLASSES):
        display = word.replace('_', ' ')
        n       = saved_counts.get(word, 0)
        label   = f"{display}  ✦{n}" if n > 0 else display
        cols[i % 5].markdown(f"- {label}")
    if saved_counts:
        st.caption("✦ thank you for contributing your word samples")

DISPLAY_WIDTH = 480

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
    sequence_buffer = []
    last_sentence   = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Cannot read from webcam.")
                break

            frame = cv2.flip(frame, 1)

            h, w  = frame.shape[:2]
            scale = DISPLAY_WIDTH / w
            display_frame = cv2.resize(frame, (DISPLAY_WIDTH, int(h * scale)))

            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            if results.left_hand_landmarks:
                mp_draw.draw_landmarks(display_frame, results.left_hand_landmarks,
                                       mp.solutions.holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_draw.draw_landmarks(display_frame, results.right_hand_landmarks,
                                       mp.solutions.holistic.HAND_CONNECTIONS)

            frame_placeholder.image(display_frame, channels="BGR",
                                    use_container_width=True)

            features      = extract_frame_features(results, prev_features)
            prev_features = features[:162]
            sequence_buffer.append(features)
            if len(sequence_buffer) > 30:
                sequence_buffer.pop(0)

            frame_count += 1
            if len(sequence_buffer) == 30 and frame_count % 3 == 0:
                X     = np.array(sequence_buffer, dtype=np.float32)[np.newaxis, ...]
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

                        st.session_state.confirm_id_counter += 1
                        st.session_state.pending_confirmations.append({
                            "id":       st.session_state.confirm_id_counter,
                            "word":     word,
                            "display":  display_word,
                            "sequence": list(sequence_buffer)
                        })

                        if len(st.session_state.pending_confirmations) > 10:
                            st.session_state.pending_confirmations.pop(0)

                        if len(st.session_state.sentence) > 12:
                            st.session_state.sentence.pop(0)
                else:
                    sign_placeholder.markdown("### ...")
                    conf_placeholder.progress(0.0, text="Waiting for clear sign...")

            if st.session_state.sentence != last_sentence:
                sent_placeholder.markdown(
                    "**" + "  →  ".join(st.session_state.sentence) + "**"
                )
                last_sentence = list(st.session_state.sentence)

    finally:
        cap.release()
        holistic.close()
