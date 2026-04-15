# """
# Real-Time Waste Classification — Streamlit App
# Two decoupled background threads:
#   Thread 1 (capture)   — reads camera frames at full speed, encodes JPEG for display
#   Thread 2 (inference) — runs ONNX on latest frame independently; never blocks capture
# Fragment reads pre-built JPEG + latest probs — zero blocking I/O in UI thread.
# """

# import time
# import threading
# import collections
# from pathlib import Path

# import cv2
# import numpy as np
# import streamlit as st
# import onnxruntime as ort


# # ─────────────────────────── Softmax ────────────────────────────────
# def softmax(x: np.ndarray) -> np.ndarray:
#     x = x - np.max(x, axis=1, keepdims=True)
#     e = np.exp(x)
#     return e / np.sum(e, axis=1, keepdims=True)


# # ─────────────────────────── Page Config ────────────────────────────
# st.set_page_config(
#     page_title="♻️ WasteVision AI",
#     page_icon="♻️",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
# * { font-family: 'Space Grotesk', sans-serif; }
# .main { background: #0d1117; }
# .pred-card {
#     background: linear-gradient(135deg, #1a1f2e 0%, #0f1419 100%);
#     border: 2px solid #30363d; border-radius: 16px;
#     padding: 20px; text-align: center; margin-bottom: 12px;
# }
# .pred-class { font-size: 2.2rem; font-weight: 700; letter-spacing: -0.5px; }
# .pred-conf  { font-size: 1.1rem; color: #8b949e; margin-top: 4px; }
# .badge {
#     display: inline-block; padding: 4px 12px; border-radius: 20px;
#     font-size: 0.75rem; font-weight: 600; text-transform: uppercase;
#     letter-spacing: 1px; margin-top: 8px;
# }
# .badge-high   { background: #1a4731; color: #3fb950; }
# .badge-medium { background: #3d2b00; color: #d29922; }
# .badge-low    { background: #3d0a0a; color: #f85149; }
# .metric-box {
#     background: #161b22; border: 1px solid #30363d;
#     border-radius: 10px; padding: 14px; text-align: center;
# }
# .metric-val { font-size: 1.6rem; font-weight: 700; }
# .metric-lbl { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }
# .class-row  { display: flex; align-items: center; gap: 10px; padding: 6px 0; font-size: 0.9rem; }
# </style>
# """, unsafe_allow_html=True)


# # ─────────────────────────── Constants ──────────────────────────────
# CLASS_CONFIG = {
#     "Glass":     {"emoji": "🫙", "color": "#58a6ff", "tip": "Rinse and place in glass recycling bin."},
#     "Hazardous": {"emoji": "☢️",  "color": "#f85149", "tip": "Take to designated hazardous waste facility."},
#     "Metal":     {"emoji": "🔩", "color": "#d2a679", "tip": "Clean and place in metal recycling bin."},
#     "Organic":   {"emoji": "🌿", "color": "#3fb950", "tip": "Compost or place in organic waste bin."},
#     "Paper":     {"emoji": "📄", "color": "#e3b341", "tip": "Flatten and place in paper recycling."},
#     "Plastic":   {"emoji": "🧴", "color": "#bc8cff", "tip": "Check recycling number; most go in plastic bin."},
# }

# CLASS_NAMES      = ["Glass", "Hazardous", "Metal", "Organic", "Paper", "Plastic"]
# IMG_SIZE         = 260
# MODEL_PATH       = Path("waste_classifier.onnx")
# ONNX_INPUT_NAME  = "input"
# ONNX_OUTPUT_NAME = "output"


# # ─────────────────────────── Model ──────────────────────────────────
# @st.cache_resource(show_spinner="Loading ONNX model…")
# def load_model() -> ort.InferenceSession:
#     if not MODEL_PATH.exists():
#         st.error(f"Model not found at '{MODEL_PATH}'.")
#         st.stop()
#     providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
#     session   = ort.InferenceSession(MODEL_PATH.as_posix(), providers=providers)
#     ins  = [i.name for i in session.get_inputs()]
#     outs = [o.name for o in session.get_outputs()]
#     if ONNX_INPUT_NAME not in ins or ONNX_OUTPUT_NAME not in outs:
#         st.error(f"I/O name mismatch. Inputs={ins}  Outputs={outs}")
#         st.stop()
#     return session


# # ─────────────────────────── Pre-process ────────────────────────────
# def preprocess(frame_bgr: np.ndarray) -> np.ndarray:
#     rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
#     rsz  = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
#     t    = rsz.astype(np.float32) / 255.0
#     t    = t.transpose(2, 0, 1)
#     mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
#     std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
#     return ((t - mean) / std)[np.newaxis].astype(np.float32)


# # ─────────────────────────── Inference ──────────────────────────────
# def run_inference(session, tensor):
#     logits = session.run([ONNX_OUTPUT_NAME], {ONNX_INPUT_NAME: tensor})[0]
#     probs  = softmax(logits)[0]
#     idx    = int(np.argmax(probs))
#     return CLASS_NAMES[idx], float(probs[idx]), probs


# # ─────────────────────────── Smoother ───────────────────────────────
# class TemporalSmoother:
#     def __init__(self, window=10, threshold=0.45):
#         self.threshold = threshold
#         self.buffer    = collections.deque(maxlen=window)

#     def update(self, probs):
#         self.buffer.append(probs)

#     def get_smoothed(self):
#         if not self.buffer:
#             return None, 0.0, np.zeros(len(CLASS_NAMES))
#         avg  = np.mean(self.buffer, axis=0)
#         idx  = int(np.argmax(avg))
#         conf = float(avg[idx])
#         return ("Uncertain", conf, avg) if conf < self.threshold else (CLASS_NAMES[idx], conf, avg)


# # ═══════════════════════════════════════════════════════════════════
# #  CameraThread — two fully decoupled loops
# #
# #  _capture_loop : reads frames as fast as the camera delivers them.
# #                  Never waits for inference. Encodes JPEG each frame.
# #
# #  _inference_loop : takes latest raw frame, runs ONNX, stores probs.
# #                    Runs independently; capture never waits for it.
# # ═══════════════════════════════════════════════════════════════════
# class CameraThread:
#     def __init__(self, src: int = 0):
#         self.src      = src
#         self.ok       = False
#         self.active   = False
#         self._flip    = False
#         self._session = None

#         # raw frame shared capture→inference (protected by _frame_lock)
#         self._raw_frame  = None
#         self._frame_lock = threading.Lock()

#         # outputs shared inference→fragment (protected by _out_lock)
#         self._jpeg     = None
#         self._probs    = None
#         self._latency  = 0.0
#         self._out_lock = threading.Lock()

#         self._cap_thread = None
#         self._inf_thread = None

#     # ── Public API ────────────────────────────────────────────────
#     def start(self, session, flip: bool):
#         if self._cap_thread and self._cap_thread.is_alive():
#             return
#         self._session = session
#         self._flip    = flip
#         self.active   = True
#         self._cap_thread = threading.Thread(target=self._capture_loop,   daemon=True)
#         self._inf_thread = threading.Thread(target=self._inference_loop, daemon=True)
#         self._cap_thread.start()
#         self._inf_thread.start()

#     def stop(self):
#         self.active = False

#     def set_flip(self, flip: bool):
#         self._flip = flip

#     def read(self):
#         """Returns (jpeg_bytes | None, probs | None, latency_ms)."""
#         with self._out_lock:
#             jpeg    = self._jpeg
#             probs   = self._probs.copy() if self._probs is not None else None
#             latency = self._latency
#         return jpeg, probs, latency

#     # ── Thread 1: capture at full camera speed ────────────────────
#     def _capture_loop(self):
#         cap = cv2.VideoCapture(self.src)
#         cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#         self.ok = cap.isOpened()

#         while self.active:
#             ret, frame = cap.read()
#             if not ret or frame is None:
#                 time.sleep(0.005)
#                 continue

#             if self._flip:
#                 frame = cv2.flip(frame, 1)

#             # Store for inference thread (non-blocking)
#             with self._frame_lock:
#                 self._raw_frame = frame

#             # Read latest probs without waiting for inference
#             with self._out_lock:
#                 probs = self._probs.copy() if self._probs is not None else None

#             # Draw overlay
#             display = frame.copy()
#             h, w    = display.shape[:2]
#             overlay = display.copy()
#             cv2.rectangle(overlay, (0, h - 90), (w, h), (13, 17, 23), -1)
#             cv2.addWeighted(overlay, 0.8, display, 0.2, 0, display)

#             if probs is not None:
#                 cls_idx  = int(np.argmax(probs))
#                 cls_name = CLASS_NAMES[cls_idx]
#                 cfg = CLASS_CONFIG.get(cls_name, {"color": "#ffffff"})
#                 hx  = cfg["color"].lstrip("#")
#                 bgr = (int(hx[4:6], 16), int(hx[2:4], 16), int(hx[0:2], 16))
#                 cv2.putText(display, cls_name,
#                             (16, h - 52), cv2.FONT_HERSHEY_DUPLEX, 1.1, bgr, 2, cv2.LINE_AA)
#                 cv2.putText(display, f"Conf: {float(probs[cls_idx]):.0%}",
#                             (16, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (139, 148, 158), 1, cv2.LINE_AA)
#             else:
#                 cv2.putText(display, "Scanning...",
#                             (16, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (139, 148, 158), 2, cv2.LINE_AA)

#             _, buf = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 82])
#             with self._out_lock:
#                 self._jpeg = buf.tobytes()

#         cap.release()
#         self.ok = False
#         with self._out_lock:
#             self._jpeg = None

#     # ── Thread 2: inference at its own pace ───────────────────────
#     def _inference_loop(self):
#         last_frame = None
#         while self.active:
#             with self._frame_lock:
#                 frame = self._raw_frame

#             # Skip if no new frame yet
#             if frame is None or frame is last_frame:
#                 time.sleep(0.005)
#                 continue

#             last_frame = frame
#             t0 = time.time()
#             try:
#                 _, _, raw_probs = run_inference(self._session, preprocess(frame))
#             except Exception:
#                 raw_probs = None
#             latency = (time.time() - t0) * 1000

#             with self._out_lock:
#                 self._probs   = raw_probs
#                 self._latency = latency


# @st.cache_resource
# def get_camera() -> CameraThread:
#     return CameraThread(src=0)


# # ─────────────────────────── Session State ──────────────────────────
# for k, v in {
#     "running":        False,
#     "frame_cnt":      0,
#     "t_start":        None,
#     "latencies":      collections.deque(maxlen=60),
#     "smoother":       None,
#     "cfg_flip":       True,
#     "cfg_show_all":   True,
#     "cfg_smooth_win": 10,
#     "cfg_conf_thr":   0.45,
# }.items():
#     if k not in st.session_state:
#         st.session_state[k] = v


# # ─────────────────────────── Sidebar ────────────────────────────────
# with st.sidebar:
#     st.markdown("## ♻️ WasteVision AI")
#     st.markdown("Real-time waste classification using ONNX EfficientNet-B3")
#     st.divider()
#     st.markdown("### ⚙️ Settings")
#     smooth_window    = st.slider("Smoothing window (frames)", 1, 30, 10)
#     conf_threshold   = st.slider("Confidence threshold", 0.1, 0.9, 0.45, 0.05)
#     show_all_classes = st.toggle("Show all class probabilities", value=True)
#     flip_camera      = st.toggle("Mirror camera feed", value=True)
#     st.divider()
#     st.markdown("### 📂 Waste Guide")
#     for cls, cfg in CLASS_CONFIG.items():
#         st.markdown(
#             f'<div style="padding:6px 0;color:{cfg["color"]}">'
#             f'{cfg["emoji"]} <b>{cls}</b><br>'
#             f'<small style="color:#8b949e">{cfg["tip"]}</small></div>',
#             unsafe_allow_html=True,
#         )

# # Keep flip in sync without full restart
# if st.session_state.running:
#     get_camera().set_flip(flip_camera)


# # ─────────────────────────── Header ─────────────────────────────────
# st.markdown("# ♻️ WasteVision AI — Real-Time Classification")
# st.markdown("Point your webcam at any waste item for instant AI-powered sorting guidance.")
# st.divider()

# col_start, col_stop, _ = st.columns([1, 1, 3])
# with col_start:
#     if st.button("▶ Start Camera", type="primary", use_container_width=True):
#         session = load_model()
#         cam     = get_camera()
#         cam.start(session, flip=flip_camera)
#         for _ in range(20):
#             if cam.ok:
#                 break
#             time.sleep(0.1)
#         if not cam.ok:
#             st.error("❌ Could not open webcam. Check camera permissions.")
#         else:
#             st.session_state.cfg_flip       = flip_camera
#             st.session_state.cfg_show_all   = show_all_classes
#             st.session_state.cfg_smooth_win = smooth_window
#             st.session_state.cfg_conf_thr   = conf_threshold
#             st.session_state.running        = True
#             st.session_state.frame_cnt      = 0
#             st.session_state.t_start        = time.time()
#             st.session_state.latencies      = collections.deque(maxlen=60)
#             st.session_state.smoother       = TemporalSmoother(smooth_window, conf_threshold)

# with col_stop:
#     if st.button("⏹ Stop", use_container_width=True):
#         get_camera().stop()
#         st.session_state.running = False

# st.divider()

# # ── Stable layout placeholders ───────────────────────────────────────
# col_cam, col_info = st.columns([3, 2], gap="large")
# with col_cam:
#     cam_ph = st.empty()
# with col_info:
#     pred_ph      = st.empty()
#     metrics_ph   = st.empty()
#     breakdown_ph = st.empty()


# # ═══════════════════════════════════════════════════════════════════
# #  FRAGMENT — fires every 66 ms
# #  Reads pre-built JPEG + latest probs. No I/O. No inference.
# # ═══════════════════════════════════════════════════════════════════
# @st.fragment(run_every=0.066)
# def camera_loop():
#     if not st.session_state.running:
#         cam_ph.markdown("""
#         <div style="background:#161b22;border:2px dashed #30363d;border-radius:16px;
#                     height:420px;display:flex;align-items:center;justify-content:center;
#                     flex-direction:column;gap:16px">
#             <div style="font-size:4rem">📷</div>
#             <div style="font-size:1.2rem;color:#8b949e">Click ▶ Start Camera to begin</div>
#             <div style="font-size:0.85rem;color:#484f58">ONNX EfficientNet-B3 · Real-Time · 6 Classes</div>
#         </div>""", unsafe_allow_html=True)
#         pred_ph.markdown("""
#         <div class="pred-card">
#             <div style="font-size:2.5rem">♻️</div>
#             <div class="pred-class" style="color:#8b949e">Ready</div>
#             <div class="pred-conf">Awaiting camera feed</div>
#         </div>""", unsafe_allow_html=True)
#         return

#     smoother = st.session_state.smoother
#     show_all = st.session_state.cfg_show_all
#     if smoother is None:
#         return

#     jpeg, raw_probs, latency = get_camera().read()

#     if jpeg is None:
#         cam_ph.warning("⏳ Waiting for first camera frame…")
#         return

#     if raw_probs is not None:
#         st.session_state.latencies.append(latency)
#         smoother.update(raw_probs)

#     smooth_class, smooth_conf, smooth_probs = smoother.get_smoothed()

#     st.session_state.frame_cnt += 1
#     elapsed = time.time() - (st.session_state.t_start or time.time())
#     fps     = st.session_state.frame_cnt / max(elapsed, 1e-6)

#     # Decode JPEG → RGB numpy for st.image
#     frame_np = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
#     cam_ph.image(cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB), use_container_width=True)

#     # ── Prediction card ───────────────────────────────────────────────
#     if smooth_class and smooth_class != "Uncertain":
#         cfg       = CLASS_CONFIG.get(smooth_class, {"emoji": "❓", "color": "#8b949e", "tip": ""})
#         badge_cls = "badge-high" if smooth_conf >= 0.75 else ("badge-medium" if smooth_conf >= 0.5 else "badge-low")
#         badge_txt = "High Confidence" if smooth_conf >= 0.75 else ("Medium Confidence" if smooth_conf >= 0.5 else "Low Confidence")
#         pred_ph.markdown(f"""
#         <div class="pred-card" style="border-color:{cfg['color']}40">
#             <div style="font-size:3rem">{cfg['emoji']}</div>
#             <div class="pred-class" style="color:{cfg['color']}">{smooth_class}</div>
#             <div class="pred-conf">{smooth_conf:.1%} confidence</div>
#             <div class="badge {badge_cls}">{badge_txt}</div>
#             <div style="margin-top:14px;font-size:0.88rem;color:#8b949e;
#                         border-top:1px solid #30363d;padding-top:12px">
#                 💡 {cfg['tip']}
#             </div>
#         </div>""", unsafe_allow_html=True)
#     else:
#         pred_ph.markdown("""
#         <div class="pred-card">
#             <div style="font-size:3rem">🔍</div>
#             <div class="pred-class" style="color:#8b949e">Scanning…</div>
#             <div class="pred-conf">Point camera at waste item</div>
#         </div>""", unsafe_allow_html=True)

#     # ── Metrics ───────────────────────────────────────────────────────
#     avg_lat = float(np.mean(st.session_state.latencies)) if st.session_state.latencies else 0.0
#     metrics_ph.markdown(f"""
#     <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:12px">
#         <div class="metric-box">
#             <div class="metric-val" style="color:#3fb950">{fps:.0f}</div>
#             <div class="metric-lbl">FPS</div>
#         </div>
#         <div class="metric-box">
#             <div class="metric-val" style="color:#58a6ff">{avg_lat:.0f}<span style="font-size:0.9rem">ms</span></div>
#             <div class="metric-lbl">Latency</div>
#         </div>
#         <div class="metric-box">
#             <div class="metric-val" style="color:#e3b341">{st.session_state.frame_cnt}</div>
#             <div class="metric-lbl">Frames</div>
#         </div>
#     </div>""", unsafe_allow_html=True)

#     # ── Class breakdown ───────────────────────────────────────────────
#     if show_all:
#         rows = ""
#         for cls, p in sorted(zip(CLASS_NAMES, smooth_probs), key=lambda x: -x[1]):
#             cfg = CLASS_CONFIG.get(cls, {"emoji": "❓", "color": "#8b949e"})
#             pct = int(p * 100)
#             rows += f"""
#             <div class="class-row">
#                 <span style="width:80px;color:{cfg['color']}">{cfg['emoji']} {cls}</span>
#                 <div style="flex:1;background:#21262d;border-radius:3px;height:6px">
#                     <div style="width:{pct}%;background:{cfg['color']};
#                                 height:6px;border-radius:3px;transition:width 0.3s"></div>
#                 </div>
#                 <span style="width:42px;text-align:right;color:#8b949e;font-size:0.8rem">{pct}%</span>
#             </div>"""
#         breakdown_ph.markdown(
#             f'<div style="background:#161b22;border:1px solid #30363d;'
#             f'border-radius:10px;padding:14px">'
#             f'<div style="font-size:0.75rem;color:#8b949e;text-transform:uppercase;'
#             f'letter-spacing:1px;margin-bottom:10px">All Categories</div>'
#             f'{rows}</div>',
#             unsafe_allow_html=True,
#         )


# camera_loop()
"""
Real-Time Waste Classification — Streamlit App
Two decoupled background threads:
  Thread 1 (capture)   — reads camera frames at full speed, encodes JPEG for display
  Thread 2 (inference) — runs ONNX on latest frame independently; never blocks capture
Fragment reads pre-built JPEG + latest probs — zero blocking I/O in UI thread.

Camera sources supported:
  - Laptop/desktop webcam  → index 0, 1, 2 …
  - Android phone (DroidCam app) → http://<phone-ip>:4747/video
  - iPhone (EpocCam / Camo)     → http://<phone-ip>:8080/video
  - Any RTSP/HTTP stream         → paste URL directly
"""

import time
import threading
import collections
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import onnxruntime as ort


# ─────────────────────────── Softmax ────────────────────────────────
def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)


# ─────────────────────────── Page Config ────────────────────────────
st.set_page_config(
    page_title="♻️ WasteVision AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
* { font-family: 'Space Grotesk', sans-serif; }
.main { background: #0d1117; }
.pred-card {
    background: linear-gradient(135deg, #1a1f2e 0%, #0f1419 100%);
    border: 2px solid #30363d; border-radius: 16px;
    padding: 20px; text-align: center; margin-bottom: 12px;
}
.pred-class { font-size: 2.2rem; font-weight: 700; letter-spacing: -0.5px; }
.pred-conf  { font-size: 1.1rem; color: #8b949e; margin-top: 4px; }
.badge {
    display: inline-block; padding: 4px 12px; border-radius: 20px;
    font-size: 0.75rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 1px; margin-top: 8px;
}
.badge-high   { background: #1a4731; color: #3fb950; }
.badge-medium { background: #3d2b00; color: #d29922; }
.badge-low    { background: #3d0a0a; color: #f85149; }
.metric-box {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 10px; padding: 14px; text-align: center;
}
.metric-val { font-size: 1.6rem; font-weight: 700; }
.metric-lbl { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }
.class-row  { display: flex; align-items: center; gap: 10px; padding: 6px 0; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────── Constants ──────────────────────────────
CLASS_CONFIG = {
    "Glass":     {"emoji": "🫙", "color": "#58a6ff", "tip": "Rinse and place in glass recycling bin."},
    "Hazardous": {"emoji": "☢️",  "color": "#f85149", "tip": "Take to designated hazardous waste facility."},
    "Metal":     {"emoji": "🔩", "color": "#d2a679", "tip": "Clean and place in metal recycling bin."},
    "Organic":   {"emoji": "🌿", "color": "#3fb950", "tip": "Compost or place in organic waste bin."},
    "Paper":     {"emoji": "📄", "color": "#e3b341", "tip": "Flatten and place in paper recycling."},
    "Plastic":   {"emoji": "🧴", "color": "#bc8cff", "tip": "Check recycling number; most go in plastic bin."},
}

CLASS_NAMES      = ["Glass", "Hazardous", "Metal", "Organic", "Paper", "Plastic"]
IMG_SIZE         = 260
MODEL_PATH       = Path("waste_classifier.onnx")
ONNX_INPUT_NAME  = "input"
ONNX_OUTPUT_NAME = "output"


# ─────────────────────────── Model ──────────────────────────────────
@st.cache_resource(show_spinner="Loading ONNX model…")
def load_model() -> ort.InferenceSession:
    if not MODEL_PATH.exists():
        st.error(f"Model not found at '{MODEL_PATH}'.")
        st.stop()
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session   = ort.InferenceSession(MODEL_PATH.as_posix(), providers=providers)
    ins  = [i.name for i in session.get_inputs()]
    outs = [o.name for o in session.get_outputs()]
    if ONNX_INPUT_NAME not in ins or ONNX_OUTPUT_NAME not in outs:
        st.error(f"I/O name mismatch. Inputs={ins}  Outputs={outs}")
        st.stop()
    return session


# ─────────────────────────── Pre-process ────────────────────────────
def preprocess(frame_bgr: np.ndarray) -> np.ndarray:
    rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rsz  = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    t    = rsz.astype(np.float32) / 255.0
    t    = t.transpose(2, 0, 1)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    return ((t - mean) / std)[np.newaxis].astype(np.float32)


# ─────────────────────────── Inference ──────────────────────────────
def run_inference(session, tensor):
    logits = session.run([ONNX_OUTPUT_NAME], {ONNX_INPUT_NAME: tensor})[0]
    probs  = softmax(logits)[0]
    idx    = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]), probs


# ─────────────────────────── Smoother ───────────────────────────────
class TemporalSmoother:
    def __init__(self, window=10, threshold=0.45):
        self.threshold = threshold
        self.buffer    = collections.deque(maxlen=window)

    def update(self, probs):
        self.buffer.append(probs)

    def get_smoothed(self):
        if not self.buffer:
            return None, 0.0, np.zeros(len(CLASS_NAMES))
        avg  = np.mean(self.buffer, axis=0)
        idx  = int(np.argmax(avg))
        conf = float(avg[idx])
        return ("Uncertain", conf, avg) if conf < self.threshold else (CLASS_NAMES[idx], conf, avg)


# ═══════════════════════════════════════════════════════════════════
#  CameraThread — identical logic to original.
#  src accepts int (webcam index) OR str (HTTP/RTSP URL for phone).
# ═══════════════════════════════════════════════════════════════════
class CameraThread:
    def __init__(self, src=0):
        self.src      = src
        self.ok       = False
        self.active   = False
        self._flip    = False
        self._session = None

        self._raw_frame  = None
        self._frame_lock = threading.Lock()

        self._jpeg     = None
        self._probs    = None
        self._latency  = 0.0
        self._out_lock = threading.Lock()

        self._cap_thread = None
        self._inf_thread = None

    def start(self, session, flip: bool):
        if self._cap_thread and self._cap_thread.is_alive():
            return
        self._session = session
        self._flip    = flip
        self.active   = True
        self._cap_thread = threading.Thread(target=self._capture_loop,   daemon=True)
        self._inf_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._cap_thread.start()
        self._inf_thread.start()

    def stop(self):
        self.active = False

    def set_flip(self, flip: bool):
        self._flip = flip

    def read(self):
        with self._out_lock:
            jpeg    = self._jpeg
            probs   = self._probs.copy() if self._probs is not None else None
            latency = self._latency
        return jpeg, probs, latency

    def _capture_loop(self):
        src = self.src if isinstance(self.src, str) else int(self.src)
        cap = cv2.VideoCapture(src)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ok = cap.isOpened()

        while self.active:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            if self._flip:
                frame = cv2.flip(frame, 1)

            with self._frame_lock:
                self._raw_frame = frame

            with self._out_lock:
                probs = self._probs.copy() if self._probs is not None else None

            display = frame.copy()
            h, w    = display.shape[:2]
            overlay = display.copy()
            cv2.rectangle(overlay, (0, h - 90), (w, h), (13, 17, 23), -1)
            cv2.addWeighted(overlay, 0.8, display, 0.2, 0, display)

            if probs is not None:
                cls_idx  = int(np.argmax(probs))
                cls_name = CLASS_NAMES[cls_idx]
                cfg = CLASS_CONFIG.get(cls_name, {"color": "#ffffff"})
                hx  = cfg["color"].lstrip("#")
                bgr = (int(hx[4:6], 16), int(hx[2:4], 16), int(hx[0:2], 16))
                cv2.putText(display, cls_name,
                            (16, h - 52), cv2.FONT_HERSHEY_DUPLEX, 1.1, bgr, 2, cv2.LINE_AA)
                cv2.putText(display, f"Conf: {float(probs[cls_idx]):.0%}",
                            (16, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (139, 148, 158), 1, cv2.LINE_AA)
            else:
                cv2.putText(display, "Scanning...",
                            (16, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (139, 148, 158), 2, cv2.LINE_AA)

            _, buf = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 82])
            with self._out_lock:
                self._jpeg = buf.tobytes()

        cap.release()
        self.ok = False
        with self._out_lock:
            self._jpeg = None

    def _inference_loop(self):
        last_frame = None
        while self.active:
            with self._frame_lock:
                frame = self._raw_frame

            if frame is None or frame is last_frame:
                time.sleep(0.005)
                continue

            last_frame = frame
            t0 = time.time()
            try:
                _, _, raw_probs = run_inference(self._session, preprocess(frame))
            except Exception:
                raw_probs = None
            latency = (time.time() - t0) * 1000

            with self._out_lock:
                self._probs   = raw_probs
                self._latency = latency


# ─────────────────────────── Session State ──────────────────────────
for k, v in {
    "running":        False,
    "frame_cnt":      0,
    "t_start":        None,
    "latencies":      collections.deque(maxlen=60),
    "smoother":       None,
    "cfg_flip":       True,
    "cfg_show_all":   True,
    "cfg_smooth_win": 10,
    "cfg_conf_thr":   0.45,
    "cam_instance":   None,
    "cam_src_last":   None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────── Sidebar ────────────────────────────────
with st.sidebar:
    st.markdown("## ♻️ WasteVision AI")
    st.markdown("Real-time waste classification using ONNX EfficientNet-B3")
    st.divider()

    # ── Camera source ────────────────────────────────────────────────
    st.markdown("### 📷 Camera Source")
    cam_mode = st.radio(
        "cam_mode",
        ["💻 Laptop webcam", "📱 Phone (IP stream)", "🔢 Custom index"],
        label_visibility="collapsed",
    )

    if cam_mode == "💻 Laptop webcam":
        cam_src = st.selectbox("Webcam index (0 = default, try 1/2 if wrong)", [0, 1, 2, 3])
        st.caption("Index 0 is your default webcam.")

    elif cam_mode == "📱 Phone (IP stream)":
        st.markdown("""
**Steps:**
1. Install a streaming app on your phone:
   - **Android** → [DroidCam](https://play.google.com/store/apps/details?id=com.dev47apps.droidcam)
   - **iPhone** → [EpocCam](https://apps.apple.com/app/epoccam-webcam-for-mac-and-pc/id449133483)
2. Connect phone & laptop to **same Wi-Fi**
3. Open app → note the IP shown
4. Paste URL below ↓
""")
        cam_src = st.text_input(
            "Stream URL",
            value="http://192.168.1.100:4747/video",
            help="DroidCam: http://<ip>:4747/video | EpocCam: http://<ip>:8080/video",
        ).strip()

    else:
        cam_src = int(st.number_input("Device index", min_value=0, max_value=10, value=0, step=1))

    st.divider()
    st.markdown("### ⚙️ Settings")
    smooth_window    = st.slider("Smoothing window (frames)", 1, 30, 10)
    conf_threshold   = st.slider("Confidence threshold", 0.1, 0.9, 0.45, 0.05)
    show_all_classes = st.toggle("Show all class probabilities", value=True)
    flip_camera      = st.toggle("Mirror camera feed", value=True)
    st.divider()
    st.markdown("### 📂 Waste Guide")
    for cls, cfg in CLASS_CONFIG.items():
        st.markdown(
            f'<div style="padding:6px 0;color:{cfg["color"]}">'
            f'{cfg["emoji"]} <b>{cls}</b><br>'
            f'<small style="color:#8b949e">{cfg["tip"]}</small></div>',
            unsafe_allow_html=True,
        )

if st.session_state.running and st.session_state.cam_instance:
    st.session_state.cam_instance.set_flip(flip_camera)


# ─────────────────────────── Header ─────────────────────────────────
st.markdown("# ♻️ WasteVision AI — Real-Time Classification")
st.markdown("Point your camera at any waste item for instant AI-powered sorting guidance.")
st.divider()

col_start, col_stop, _ = st.columns([1, 1, 3])
with col_start:
    if st.button("▶ Start Camera", type="primary", use_container_width=True):
        session = load_model()

        # Create new CameraThread if source changed or first run
        if st.session_state.cam_src_last != cam_src or st.session_state.cam_instance is None:
            if st.session_state.cam_instance is not None:
                st.session_state.cam_instance.stop()
                time.sleep(0.2)
            cam = CameraThread(src=cam_src)
            st.session_state.cam_instance = cam
            st.session_state.cam_src_last = cam_src
        else:
            cam = st.session_state.cam_instance

        cam.start(session, flip=flip_camera)
        for _ in range(30):
            if cam.ok:
                break
            time.sleep(0.1)

        if not cam.ok:
            st.error(
                "❌ Could not open camera.\n\n"
                "**Laptop:** try index 1 or 2 in the sidebar.\n\n"
                "**Phone:** make sure the app is running, URL is correct, "
                "and both devices are on the same Wi-Fi network."
            )
        else:
            st.session_state.running        = True
            st.session_state.frame_cnt      = 0
            st.session_state.t_start        = time.time()
            st.session_state.latencies      = collections.deque(maxlen=60)
            st.session_state.smoother       = TemporalSmoother(smooth_window, conf_threshold)

with col_stop:
    if st.button("⏹ Stop", use_container_width=True):
        if st.session_state.cam_instance:
            st.session_state.cam_instance.stop()
        st.session_state.running = False

st.divider()

# ── Stable layout placeholders ───────────────────────────────────────
col_cam, col_info = st.columns([3, 2], gap="large")
with col_cam:
    cam_ph = st.empty()
with col_info:
    pred_ph      = st.empty()
    metrics_ph   = st.empty()
    breakdown_ph = st.empty()


# ═══════════════════════════════════════════════════════════════════
#  FRAGMENT — fires every 66 ms  (identical to original)
# ═══════════════════════════════════════════════════════════════════
@st.fragment(run_every=0.066)
def camera_loop():
    if not st.session_state.running:
        cam_ph.markdown("""
        <div style="background:#161b22;border:2px dashed #30363d;border-radius:16px;
                    height:420px;display:flex;align-items:center;justify-content:center;
                    flex-direction:column;gap:16px">
            <div style="font-size:4rem">📷</div>
            <div style="font-size:1.2rem;color:#8b949e">Click ▶ Start Camera to begin</div>
            <div style="font-size:0.85rem;color:#484f58">ONNX EfficientNet-B3 · Real-Time · 6 Classes</div>
        </div>""", unsafe_allow_html=True)
        pred_ph.markdown("""
        <div class="pred-card">
            <div style="font-size:2.5rem">♻️</div>
            <div class="pred-class" style="color:#8b949e">Ready</div>
            <div class="pred-conf">Awaiting camera feed</div>
        </div>""", unsafe_allow_html=True)
        return

    smoother = st.session_state.smoother
    show_all = st.session_state.cfg_show_all
    if smoother is None:
        return

    cam = st.session_state.cam_instance
    if cam is None:
        return

    jpeg, raw_probs, latency = cam.read()

    if jpeg is None:
        cam_ph.warning("⏳ Waiting for first camera frame…")
        return

    if raw_probs is not None:
        st.session_state.latencies.append(latency)
        smoother.update(raw_probs)

    smooth_class, smooth_conf, smooth_probs = smoother.get_smoothed()

    st.session_state.frame_cnt += 1
    elapsed = time.time() - (st.session_state.t_start or time.time())
    fps     = st.session_state.frame_cnt / max(elapsed, 1e-6)

    frame_np = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    cam_ph.image(cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB), use_container_width=True)

    # ── Prediction card ───────────────────────────────────────────────
    if smooth_class and smooth_class != "Uncertain":
        cfg       = CLASS_CONFIG.get(smooth_class, {"emoji": "❓", "color": "#8b949e", "tip": ""})
        badge_cls = "badge-high" if smooth_conf >= 0.75 else ("badge-medium" if smooth_conf >= 0.5 else "badge-low")
        badge_txt = "High Confidence" if smooth_conf >= 0.75 else ("Medium Confidence" if smooth_conf >= 0.5 else "Low Confidence")
        pred_ph.markdown(f"""
        <div class="pred-card" style="border-color:{cfg['color']}40">
            <div style="font-size:3rem">{cfg['emoji']}</div>
            <div class="pred-class" style="color:{cfg['color']}">{smooth_class}</div>
            <div class="pred-conf">{smooth_conf:.1%} confidence</div>
            <div class="badge {badge_cls}">{badge_txt}</div>
            <div style="margin-top:14px;font-size:0.88rem;color:#8b949e;
                        border-top:1px solid #30363d;padding-top:12px">
                💡 {cfg['tip']}
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        pred_ph.markdown("""
        <div class="pred-card">
            <div style="font-size:3rem">🔍</div>
            <div class="pred-class" style="color:#8b949e">Scanning…</div>
            <div class="pred-conf">Point camera at waste item</div>
        </div>""", unsafe_allow_html=True)

    # ── Metrics ───────────────────────────────────────────────────────
    avg_lat = float(np.mean(st.session_state.latencies)) if st.session_state.latencies else 0.0
    metrics_ph.markdown(f"""
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:12px">
        <div class="metric-box">
            <div class="metric-val" style="color:#3fb950">{fps:.0f}</div>
            <div class="metric-lbl">FPS</div>
        </div>
        <div class="metric-box">
            <div class="metric-val" style="color:#58a6ff">{avg_lat:.0f}<span style="font-size:0.9rem">ms</span></div>
            <div class="metric-lbl">Latency</div>
        </div>
        <div class="metric-box">
            <div class="metric-val" style="color:#e3b341">{st.session_state.frame_cnt}</div>
            <div class="metric-lbl">Frames</div>
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Class breakdown ───────────────────────────────────────────────
    if show_all:
        rows = ""
        for cls, p in sorted(zip(CLASS_NAMES, smooth_probs), key=lambda x: -x[1]):
            cfg = CLASS_CONFIG.get(cls, {"emoji": "❓", "color": "#8b949e"})
            pct = int(p * 100)
            rows += f"""
            <div class="class-row">
                <span style="width:80px;color:{cfg['color']}">{cfg['emoji']} {cls}</span>
                <div style="flex:1;background:#21262d;border-radius:3px;height:6px">
                    <div style="width:{pct}%;background:{cfg['color']};
                                height:6px;border-radius:3px;transition:width 0.3s"></div>
                </div>
                <span style="width:42px;text-align:right;color:#8b949e;font-size:0.8rem">{pct}%</span>
            </div>"""
        breakdown_ph.markdown(
            f'<div style="background:#161b22;border:1px solid #30363d;'
            f'border-radius:10px;padding:14px">'
            f'<div style="font-size:0.75rem;color:#8b949e;text-transform:uppercase;'
            f'letter-spacing:1px;margin-bottom:10px">All Categories</div>'
            f'{rows}</div>',
            unsafe_allow_html=True,
        )


camera_loop()
