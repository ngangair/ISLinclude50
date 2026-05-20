import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import json
import mediapipe as mp
import av
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

st.set_page_config(page_title="ISL Live Recognition", layout="wide")

MODELS_PATH   = "data/models"
SPLITS_PATH   = "data/splits"
NEW_DATA_PATH = "data/new_samples"
os.makedirs(NEW_DATA_PATH, exist_ok=True)

# ─── CUSTOM ATTENTION LAYER ───
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

# ─── LOAD MODEL ───
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

# ─── FEATURE EXTRACTION ───
def normalize_hand(h):
    if h is None: return np.zeros(63, dtype=np.float32)
    c = np.array([[lm.x, lm.y, lm.z] for lm in h.landmark], dtype=np.float32)
    c -= c[0]; s = np.linalg.norm(c[9])
    if s > 1e-6: c /= s
    return c.flatten()

def compute_angles(h):
    if h is None: return np.zeros(10, dtype=np.float32)
    lm = h.landmark
    def ang(a, b, c):
        va=np.array([lm[a].x,lm[a].y,lm[a].z]); vb=np.array([lm[b].x,lm[b].y,lm[b].z]); vc=np.array([lm[c].x,lm[c].y,lm[c].z])
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
    w=np.array([lm[0].x,lm[0].y,lm[0].z]); i=np.array([lm[5].x,lm[5].y,lm[5].z]); p=np.array([lm[17].x,lm[17].y,lm[17].z])
    n = np.cross(i-w, p-w); mag = np.linalg.norm(n)
    return (n/mag if mag>1e-6 else n).astype(np.float32)

def extract_frame_features(results, prev=None):
    static = np.concatenate([
        normalize_hand(results.left_hand_landmarks), normalize_hand(results.right_hand_landmarks),
        compute_angles(results.left_hand_landmarks), compute_angles(results.right_hand_landmarks),
        compute_distances(results.left_hand_landmarks), compute_distances(results.right_hand_landmarks),
        compute_palm_normal(results.left_hand_landmarks), compute_palm_normal(results.right_hand_landmarks),
    ])
    velocity = static - prev if prev is not None else np.zeros(162, dtype=np.float32)
    return np.concatenate([static, velocity]).astype(np.float32)

# ─── DATA SAVING ───
def save_sample(word, sequence_30frames):
    word_dir = os.path.join(NEW_DATA_PATH, word)
    os.makedirs(word_dir, exist_ok=True)
    existing = len([f for f in os.listdir(word_dir) if f.endswith('.npy')])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    np.save(os.path.join(word_dir, f"{word}_{timestamp}_{existing:03d}.npy"),
            np.array(sequence_30frames, dtype=np.float32))

def count_by_word():
    counts = {}
    if not os.path.exists(NEW_DATA_PATH): return counts
    for word in os.listdir(NEW_DATA_PATH):
        d = os.path.join(NEW_DATA_PATH, word)
        if os.path.isdir(d):
            n = len([f for f in os.listdir(d) if f.endswith('.npy')])
            if n > 0: counts[word] = n
    return counts

# ─── VIDEO PROCESSOR ───
# TOPIC: VideoProcessorBase
# WHY: streamlit-webrtc calls recv() for every frame from the browser webcam.
# We subclass it to inject MediaPipe + model inference into the stream.
# Results are stored as instance attributes — Streamlit reads them via ctx.video_processor.
class ISLProcessor(VideoProcessorBase):
    def __init__(self):
        self.holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False, model_complexity=1,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.mp_draw       = mp.solutions.drawing_utils
        self.sequence_buf  = []
        self.prev_features = None
        self.frame_count   = 0
        # These are read by Streamlit UI outside recv()
        self.current_word  = ""
        self.current_conf  = 0.0
        self.new_word      = None    # set when a NEW word is detected; UI reads & clears it
        self.snapshot      = None    # sequence snapshot when new word detected

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb)

        features           = extract_frame_features(results, self.prev_features)
        self.prev_features = features[:162]
        self.sequence_buf.append(features)
        if len(self.sequence_buf) > 30:
            self.sequence_buf.pop(0)

        self.frame_count += 1
        if len(self.sequence_buf) == 30 and self.frame_count % 5 == 0:
            X     = np.array(self.sequence_buf, dtype=np.float32)[np.newaxis, ...]
            probs = model(X, training=False).numpy()[0]
            idx   = int(np.argmax(probs))
            self.current_conf = float(probs[idx])
            self.current_word = idx_to_word.get(idx, "?")
            # Signal a new word to the UI (UI reads & clears this)
            if self.current_conf >= 0.55:
                self.new_word = self.current_word
                self.snapshot = list(self.sequence_buf)

        # Draw landmarks and overlay on frame
        if results.left_hand_landmarks:
            self.mp_draw.draw_landmarks(img, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            self.mp_draw.draw_landmarks(img, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)

        # Word overlay on video
        color = (0, 255, 0) if self.current_conf >= 0.55 else (0, 165, 255)
        cv2.rectangle(img, (0, 0), (320, 60), (0, 0, 0), -1)
        cv2.putText(img, f"{self.current_word}  {self.current_conf*100:.0f}%",
                    (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ─── SESSION STATE ───
for k, v in {"sentence":[], "last_word":"", "save_log":[],
             "pending_confirmations":[], "confirm_id_counter":0}.items():
    if k not in st.session_state: st.session_state[k] = v

# ─── UI ───
st.markdown("<style>.block-container{padding-top:1.2rem;}</style>", unsafe_allow_html=True)
st.title("ISL Live Recognition")
st.caption("Click START to activate your camera — sign — confirm or correct recognized words")

m1, m2 = st.columns(2)
m1.metric("Dataset", "INCLUDE50")
m2.metric("ISL Signs", "50 Classes")
st.divider()

cam_col, panel_col = st.columns([3, 2])

with cam_col:
    st.subheader("Camera Feed")
    # TOPIC: RTCConfiguration with STUN server
    # WHY: WebRTC needs a STUN server to negotiate the connection between the
    # user's browser and the Streamlit Cloud server. Google's free STUN server works.
    RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    ctx = webrtc_streamer(
        key="isl",
        video_processor_factory=ISLProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with panel_col:
    st.subheader("Recognized Sign")
    sign_ph = st.empty()
    conf_ph = st.empty()
    sign_ph.markdown("### —")
    conf_ph.progress(0.0, text="Waiting...")
    st.markdown("---")

    st.subheader("Recognized Words")
    sent_ph = st.empty()
    sent_ph.markdown("*Start signing to build a sentence...*")
    st.markdown("---")

    st.subheader("Select Words")
    st.caption("Tap to confirm correctly recognized words")

    if st.session_state.pending_confirmations:
        for item in list(st.session_state.pending_confirmations):
            c1, c2, c3 = st.columns([3, 1, 1])
            c1.markdown(f"**{item['display']}**")
            if c2.button("✅ Yes", key=f"yes_{item['id']}"):
                save_sample(item['word'], item['sequence'])
                ts = datetime.now().strftime("%H:%M:%S")
                st.session_state.save_log.append(f"[{ts}] '{item['display']}' saved")
                st.session_state.pending_confirmations = [p for p in st.session_state.pending_confirmations if p['id'] != item['id']]
                st.rerun()
            if c3.button("❌ No", key=f"no_{item['id']}"):
                if item['display'] in st.session_state.sentence:
                    st.session_state.sentence.remove(item['display'])
                st.session_state.pending_confirmations = [p for p in st.session_state.pending_confirmations if p['id'] != item['id']]
                if st.session_state.last_word == item['word']:
                    st.session_state.last_word = ""
                st.rerun()
    else:
        st.caption("*Words appear here as they are recognized...*")

    st.markdown("---")
    st.subheader("Saved Samples")
    st.caption("Confirmed words saved as numeric feature sequences")
    if st.session_state.save_log:
        for entry in st.session_state.save_log[-5:]:
            st.markdown(f"✅ `{entry}`")
    else:
        st.caption("*No samples saved yet*")

# ─── READ FROM VIDEO PROCESSOR ───
# TOPIC: Reading processor state from outside recv()
# WHY: recv() runs in a background thread. We read its output here
# in the main Streamlit thread to update the UI and session_state.
# This is the standard streamlit-webrtc pattern for passing data out.
if ctx.video_processor:
    vp = ctx.video_processor
    if vp.current_word:
        color = "green" if vp.current_conf >= 0.55 else "orange"
        sign_ph.markdown(f"### {vp.current_word}")
        conf_ph.progress(min(vp.current_conf, 1.0), text=f"{vp.current_conf*100:.0f}% confident")

    # Pick up new word detections
    if vp.new_word and vp.new_word != st.session_state.last_word:
        word    = vp.new_word
        display = word.replace('_', ' ')
        st.session_state.sentence.append(display)
        st.session_state.last_word = word
        st.session_state.confirm_id_counter += 1
        st.session_state.pending_confirmations.append({
            "id": st.session_state.confirm_id_counter,
            "word": word, "display": display,
            "sequence": vp.snapshot or []
        })
        if len(st.session_state.pending_confirmations) > 10:
            st.session_state.pending_confirmations.pop(0)
        if len(st.session_state.sentence) > 12:
            st.session_state.sentence.pop(0)
        vp.new_word = None   # clear so we don't pick it up again

    if st.session_state.sentence:
        sent_ph.markdown("**" + "  →  ".join(st.session_state.sentence) + "**")

st.divider()

threshold = st.slider("Confidence Threshold", 0.3, 0.95, 0.65, 0.05)
clear_btn = st.button("🗑️  Clear All")
if clear_btn:
    st.session_state.sentence = []; st.session_state.last_word = ""
    st.session_state.pending_confirmations = []; st.rerun()

st.divider()

with st.expander("📋  Browse All 50 ISL Signs"):
    saved_counts = count_by_word()
    cols = st.columns(5)
    for i, word in enumerate(ALL_CLASSES):
        display = word.replace('_', ' ')
        n = saved_counts.get(word, 0)
        cols[i % 5].markdown(f"- {display}{'  ✦'+str(n) if n>0 else ''}")
    if saved_counts:
        st.caption("✦ = your contributed samples for that sign")
