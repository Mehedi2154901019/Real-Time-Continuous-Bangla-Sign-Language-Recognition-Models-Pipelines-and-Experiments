import streamlit as st
import numpy as np
import cv2
import pickle
import time
import threading
from collections import deque
import mediapipe as mp
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import psutil
import os
import logging
import av
import json

from streamlit_webrtc import webrtc_streamer


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "best_model.keras"
CLASSES_PATH = "classes.json"
ENCODER_PATH = "label_encoder.pkl"
SCALER_PATH = "scaler.pkl"

# ------------------------------------------------------------
# MODEL INPUT
# ------------------------------------------------------------

WINDOW_SIZE = 120

FEATURES_PER_FRAME = 130


# ------------------------------------------------------------
# PROCESSING RESOLUTION
#
# This stays 640x480.
# The displayed camera is made smaller using CSS/layout.
# ------------------------------------------------------------

FRAME_W = 640
FRAME_H = 480


# ------------------------------------------------------------
# DISPLAY SIZE
#
# This affects visual presentation only.
# ------------------------------------------------------------

DISPLAY_W = 480
DISPLAY_H = 360


# ------------------------------------------------------------
# MEDIAPIPE
# ------------------------------------------------------------

DETECTION_CONF = 0.5
TRACKING_CONF = 0.5


# ------------------------------------------------------------
# GESTURE SETTINGS
# ------------------------------------------------------------

MAX_DEQUE_CHUNKS = 10

MISSING_HAND_TOLERANCE = 5

MAX_GESTURE_FRAMES = 240


# ============================================================
# LOGGING
# ============================================================

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

logging.getLogger("mediapipe").setLevel(logging.ERROR)


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Realtime Gesture Recognition",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* -------------------------------------------------------
       Reduce overall page width usage
       ------------------------------------------------------- */

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
    }


    /* -------------------------------------------------------
       Keep camera visually compact
       ------------------------------------------------------- */

    div[data-testid="stVideo"] {
        max-width: 480px !important;
        margin-left: auto;
        margin-right: auto;
    }


    /* -------------------------------------------------------
       Metric cards / tables
       ------------------------------------------------------- */

    div[data-testid="stMetric"] {
        padding: 0.25rem;
    }


    /* -------------------------------------------------------
       Compact telemetry
       ------------------------------------------------------- */

    .telemetry-box {
        padding: 0.5rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


st.title("🖐️ Realtime Gesture Recognition")


# ============================================================
# LOAD MODEL + ASSETS
# ============================================================

@st.cache_resource
def load_assets():

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = load_model(MODEL_PATH)


    # --------------------------------------------------------
    # Label encoder
    # --------------------------------------------------------

    with open(ENCODER_PATH, "rb") as f:

        label_encoder = pickle.load(f)


    # --------------------------------------------------------
    # Scaler
    # --------------------------------------------------------

    with open(SCALER_PATH, "rb") as f:

        scaler: StandardScaler = pickle.load(f)


    # --------------------------------------------------------
    # Classes
    # --------------------------------------------------------

    classes = None

    if os.path.exists(CLASSES_PATH):

        try:

            with open(CLASSES_PATH, "r") as f:

                classes = json.load(f)

        except Exception as e:

            logging.warning(
                f"Could not load classes.json: {e}"
            )


    return (
        model,
        label_encoder,
        scaler,
        classes
    )


(
    model,
    label_encoder,
    scaler,
    classes
) = load_assets()


# ============================================================
# MODEL VALIDATION
# ============================================================

try:

    model_input_shape = model.input_shape

except Exception:

    model_input_shape = None


if (
    scaler.n_features_in_
    != FEATURES_PER_FRAME
):

    st.error(
        f"Scaler expects "
        f"{scaler.n_features_in_} features, "
        f"but this pipeline produces "
        f"{FEATURES_PER_FRAME} features per frame."
    )

    st.stop()


# ============================================================
# MEDIAPIPE
# ============================================================

mp_pose = mp.solutions.pose

mp_hands = mp.solutions.hands


# ============================================================
# SHARED STATE
# ============================================================

state_lock = threading.Lock()


runtime = {

    "label": "waiting...",

    "confidence": 0.0,

    "state": "WAITING",

    "hand_present": False,

    "fps": 0.0,

    "latency_ms": 0.0,

    "current_ram_mb": 0.0,

    "peak_ram_mb": 0.0,

    "total_frames": 0,

    "total_latency_ms": 0.0,

    "running": False,

}


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_landmark_features(
    results_pose,
    results_hands
):

    """
    130 features per frame.

    ----------------------------------------------------------
    Left hand
        21 landmarks × 3 = 63

    Right hand
        21 landmarks × 3 = 63

    Pose distances
        Left wrist  -> nose = 1
        Right wrist -> nose = 1
        Left elbow  -> nose = 1
        Right elbow -> nose = 1

    Total = 63 + 63 + 4 = 130
    ----------------------------------------------------------
    """

    features = []


    # ========================================================
    # DISTANCES
    # ========================================================

    dist_left_wrist = 0.0

    dist_right_wrist = 0.0

    dist_left_elbow = 0.0

    dist_right_elbow = 0.0


    # ========================================================
    # POSE
    # ========================================================

    if (
        results_pose
        and results_pose.pose_landmarks
    ):

        lm = (
            results_pose
            .pose_landmarks
            .landmark
        )


        try:

            # ------------------------------------------------
            # Nose
            # ------------------------------------------------

            nose = np.array([
                lm[0].x,
                lm[0].y,
                lm[0].z
            ])


            # ------------------------------------------------
            # Left wrist
            # ------------------------------------------------

            left_wrist = np.array([
                lm[15].x,
                lm[15].y,
                lm[15].z
            ])


            # ------------------------------------------------
            # Right wrist
            # ------------------------------------------------

            right_wrist = np.array([
                lm[16].x,
                lm[16].y,
                lm[16].z
            ])


            # ------------------------------------------------
            # Left elbow
            # ------------------------------------------------

            left_elbow = np.array([
                lm[13].x,
                lm[13].y,
                lm[13].z
            ])


            # ------------------------------------------------
            # Right elbow
            # ------------------------------------------------

            right_elbow = np.array([
                lm[14].x,
                lm[14].y,
                lm[14].z
            ])


            # ------------------------------------------------
            # Wrist → nose
            # ------------------------------------------------

            dist_left_wrist = np.linalg.norm(
                left_wrist - nose
            )


            dist_right_wrist = np.linalg.norm(
                right_wrist - nose
            )


            # ------------------------------------------------
            # Elbow → nose
            # ------------------------------------------------

            dist_left_elbow = np.linalg.norm(
                left_elbow - nose
            )


            dist_right_elbow = np.linalg.norm(
                right_elbow - nose
            )


        except Exception:

            pass


    # ========================================================
    # HANDS
    # ========================================================

    left_hand = [0.0] * 63

    right_hand = [0.0] * 63


    if (
        results_hands
        and results_hands.multi_hand_landmarks
        and results_hands.multi_handedness
    ):

        for (
            hand_landmarks,
            handedness
        ) in zip(
            results_hands.multi_hand_landmarks,
            results_hands.multi_handedness
        ):

            coords = []


            for lm in hand_landmarks.landmark:

                coords.extend([
                    lm.x,
                    lm.y,
                    lm.z
                ])


            label = (
                handedness
                .classification[0]
                .label
                .lower()
            )


            if label == "left":

                left_hand = coords

            else:

                right_hand = coords


    # ========================================================
    # COMBINE
    # ========================================================

    features.extend(left_hand)

    features.extend(right_hand)

    features.extend([
        dist_left_wrist,
        dist_right_wrist,
        dist_left_elbow,
        dist_right_elbow
    ])


    features = np.array(
        features,
        dtype=np.float32
    )


    # ========================================================
    # SAFETY CHECK
    # ========================================================

    if (
        features.shape[0]
        != FEATURES_PER_FRAME
    ):

        raise ValueError(
            f"Expected "
            f"{FEATURES_PER_FRAME} features, "
            f"got {features.shape[0]}"
        )


    return features


# ============================================================
# PREPARE SEQUENCE
# ============================================================

def prepare_sequence(frames):

    """
    Convert arbitrary gesture length into
    exactly 120 frames.

    ----------------------------------------------------------
    < 120:
        Repeat final frame.

    = 120:
        Use directly.

    > 120:
        Extract middle 120.
    ----------------------------------------------------------
    """

    if not frames:

        return None


    frames = np.asarray(
        frames,
        dtype=np.float32
    )


    # ========================================================
    # SHORT SEQUENCE
    # ========================================================

    if len(frames) < WINDOW_SIZE:

        padding_count = (
            WINDOW_SIZE - len(frames)
        )


        last_frame = frames[-1]


        padding = np.repeat(
            last_frame[np.newaxis, :],
            padding_count,
            axis=0
        )


        frames = np.vstack([
            frames,
            padding
        ])


    # ========================================================
    # EXACT WINDOW
    # ========================================================

    elif len(frames) == WINDOW_SIZE:

        pass


    # ========================================================
    # LONG SEQUENCE
    # ========================================================

    else:

        mid = len(frames) // 2


        start = (
            mid
            - WINDOW_SIZE // 2
        )


        start = max(
            0,
            min(
                start,
                len(frames)
                - WINDOW_SIZE
            )
        )


        frames = frames[
            start:start + WINDOW_SIZE
        ]


    # ========================================================
    # SAFETY CHECK
    # ========================================================

    if (
        frames.shape[0]
        != WINDOW_SIZE
    ):

        raise ValueError(
            f"Expected "
            f"{WINDOW_SIZE} frames, "
            f"got {frames.shape[0]}"
        )


    if (
        frames.shape[1]
        != FEATURES_PER_FRAME
    ):

        raise ValueError(
            f"Expected "
            f"{FEATURES_PER_FRAME} features, "
            f"got {frames.shape[1]}"
        )


    return frames


# ============================================================
# PREDICTION
# ============================================================

def predict_gesture_from_frames(frames):

    if not frames:

        return "waiting...", 0.0


    # ========================================================
    # PREPARE 120 × 130
    # ========================================================

    frames = prepare_sequence(
        frames
    )


    if frames is None:

        return "waiting...", 0.0


    # ========================================================
    # SCALE
    # ========================================================

    frames_scaled = scaler.transform(
        frames
    )


    # ========================================================
    # MODEL INPUT
    #
    # (batch, time, features)
    #
    # = (1, 120, 130)
    # ========================================================

    frames_scaled = frames_scaled.reshape(
        1,
        WINDOW_SIZE,
        FEATURES_PER_FRAME
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    preds = model.predict(
        frames_scaled,
        verbose=0
    )


    # ========================================================
    # CLASS
    # ========================================================

    pred_class = np.argmax(
        preds,
        axis=1
    )[0]


    # ========================================================
    # LABEL
    # ========================================================

    pred_label = (
        label_encoder
        .inverse_transform(
            [pred_class]
        )[0]
    )


    # ========================================================
    # CONFIDENCE
    # ========================================================

    confidence = float(
        np.max(preds)
    )


    return (
        pred_label,
        confidence
    )


# ============================================================
# VIDEO PROCESSOR
# ============================================================

class GestureProcessor:

    def __init__(self):

        # ----------------------------------------------------
        # Pose
        # ----------------------------------------------------

        self.pose = mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=DETECTION_CONF,
            min_tracking_confidence=TRACKING_CONF
        )


        # ----------------------------------------------------
        # Hands
        # ----------------------------------------------------

        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=DETECTION_CONF,
            min_tracking_confidence=TRACKING_CONF
        )


        # ----------------------------------------------------
        # Gesture buffer
        # ----------------------------------------------------

        self.frames_current_gesture = []


        # ----------------------------------------------------
        # Last prediction
        # ----------------------------------------------------

        self.last_label = "waiting..."

        self.last_conf = 0.0


        # ----------------------------------------------------
        # State
        # ----------------------------------------------------

        self.state = "WAITING"

        self.missing_counter = 0


        # ----------------------------------------------------
        # Previous predictions
        # ----------------------------------------------------

        self.previous_chunks = deque(
            maxlen=MAX_DEQUE_CHUNKS
        )


        # ----------------------------------------------------
        # Process
        # ----------------------------------------------------

        self.process = psutil.Process(
            os.getpid()
        )


    # ========================================================
    # PROCESS FRAME
    # ========================================================

    def process_frame(self, frame):

        start_time = time.perf_counter()


        # ====================================================
        # FRAME
        # ====================================================

        image = frame.to_ndarray(
            format="bgr24"
        )


        # ----------------------------------------------------
        # Processing resolution stays 640 × 480
        # ----------------------------------------------------

        image = cv2.resize(
            image,
            (
                FRAME_W,
                FRAME_H
            )
        )


        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )


        # ====================================================
        # MEDIAPIPE
        # ====================================================

        results_pose = (
            self.pose.process(rgb)
        )


        results_hands = (
            self.hands.process(rgb)
        )


        hand_present = (
            results_hands.multi_hand_landmarks
            is not None
        )


        # ====================================================
        # STATE MACHINE
        # ====================================================

        if self.state == "WAITING":

            if hand_present:

                self.state = "RECORDING"

                self.frames_current_gesture = []

                self.missing_counter = 0


        elif self.state == "RECORDING":

            # =================================================
            # HAND PRESENT
            # =================================================

            if hand_present:

                features = (
                    extract_landmark_features(
                        results_pose,
                        results_hands
                    )
                )


                self.frames_current_gesture.append(
                    features
                )


                self.missing_counter = 0


                # =============================================
                # MAXIMUM GESTURE LENGTH
                # =============================================

                if (
                    len(
                        self.frames_current_gesture
                    )
                    >= MAX_GESTURE_FRAMES
                ):

                    (
                        self.last_label,
                        self.last_conf
                    ) = (
                        predict_gesture_from_frames(
                            self.frames_current_gesture
                        )
                    )


                    self.previous_chunks.append(
                        (
                            list(
                                self.frames_current_gesture
                            ),
                            self.last_label,
                            self.last_conf
                        )
                    )


                    self.frames_current_gesture = []

                    self.missing_counter = 0

                    self.state = "WAITING"


            # =================================================
            # HAND MISSING
            # =================================================

            else:

                self.missing_counter += 1


                # ---------------------------------------------
                # Temporary disappearance
                # ---------------------------------------------

                if (
                    self.missing_counter
                    <= MISSING_HAND_TOLERANCE
                ):

                    if (
                        self.frames_current_gesture
                    ):

                        last_features = (
                            self.frames_current_gesture[-1]
                        )


                        self.frames_current_gesture.append(
                            last_features
                        )


                # ---------------------------------------------
                # Gesture ended
                # ---------------------------------------------

                else:

                    if (
                        self.frames_current_gesture
                    ):

                        (
                            self.last_label,
                            self.last_conf
                        ) = (
                            predict_gesture_from_frames(
                                self.frames_current_gesture
                            )
                        )


                        self.previous_chunks.append(
                            (
                                list(
                                    self.frames_current_gesture
                                ),
                                self.last_label,
                                self.last_conf
                            )
                        )


                    self.frames_current_gesture = []

                    self.missing_counter = 0

                    self.state = "WAITING"


        # ====================================================
        # PERFORMANCE
        # ====================================================

        end_time = time.perf_counter()


        elapsed = (
            end_time
            - start_time
        )


        latency_ms = (
            elapsed * 1000.0
        )


        fps = (
            1.0 / elapsed
            if elapsed > 0
            else 0.0
        )


        # ====================================================
        # RAM
        # ====================================================

        current_ram_mb = (
            self.process
            .memory_info()
            .rss
            / (1024 * 1024)
        )


        # ====================================================
        # SHARED STATE
        # ====================================================

        with state_lock:

            runtime["label"] = (
                self.last_label
            )

            runtime["confidence"] = (
                self.last_conf
            )

            runtime["state"] = (
                self.state
            )

            runtime["hand_present"] = (
                hand_present
            )

            runtime["fps"] = fps

            runtime["latency_ms"] = (
                latency_ms
            )

            runtime["current_ram_mb"] = (
                current_ram_mb
            )


            if (
                current_ram_mb
                > runtime["peak_ram_mb"]
            ):

                runtime["peak_ram_mb"] = (
                    current_ram_mb
                )


            runtime["total_frames"] += 1

            runtime["total_latency_ms"] += (
                latency_ms
            )

            runtime["running"] = True


        # ====================================================
        # OVERLAY
        # ====================================================

        overlay = image.copy()


        # ====================================================
        # TELEMETRY PANEL
        # ====================================================

        cv2.rectangle(
            overlay,
            (0, 0),
            (FRAME_W, 125),
            (0, 0, 0),
            -1
        )


        # ====================================================
        # GESTURE
        # ====================================================

        cv2.putText(
            overlay,
            (
                f"{self.last_label.upper()} "
                f"({self.last_conf:.2f})"
            ),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2
        )


        # ====================================================
        # HAND
        # ====================================================

        cv2.putText(
            overlay,
            (
                f"Hand: "
                f"{'Yes' if hand_present else 'No'}"
            ),
            (450, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (200, 200, 200),
            2
        )


        # ====================================================
        # FPS
        # ====================================================

        cv2.putText(
            overlay,
            f"FPS: {fps:.1f}",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2
        )


        # ====================================================
        # LATENCY
        # ====================================================

        cv2.putText(
            overlay,
            f"Latency: {latency_ms:.1f} ms",
            (115, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2
        )


        # ====================================================
        # RAM
        # ====================================================

        cv2.putText(
            overlay,
            f"RAM: {current_ram_mb:.1f} MB",
            (330, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2
        )


        # ====================================================
        # PEAK RAM
        # ====================================================

        with state_lock:

            peak_ram = (
                runtime["peak_ram_mb"]
            )


        cv2.putText(
            overlay,
            f"Peak RAM: {peak_ram:.1f} MB",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2
        )


        # ====================================================
        # STATE
        # ====================================================

        cv2.putText(
            overlay,
            f"State: {self.state}",
            (235, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (200, 200, 200),
            2
        )


        # ====================================================
        # FRAME COUNTER
        # ====================================================

        with state_lock:

            frame_count = (
                runtime["total_frames"]
            )


        cv2.putText(
            overlay,
            f"Frames: {frame_count}",
            (420, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            2
        )


        # ====================================================
        # RETURN
        # ====================================================

        return av.VideoFrame.from_ndarray(
            overlay,
            format="bgr24"
        )


    # ========================================================
    # CLEANUP
    # ========================================================

    def close(self):

        try:

            self.pose.close()

        except Exception:

            pass


        try:

            self.hands.close()

        except Exception:

            pass


# ============================================================
# PROCESSOR HOLDER
# ============================================================

processor_holder = {
    "processor": None
}


# ============================================================
# VIDEO CALLBACK
# ============================================================

def video_frame_callback(frame):

    if (
        processor_holder["processor"]
        is None
    ):

        processor_holder["processor"] = (
            GestureProcessor()
        )


    return (
        processor_holder["processor"]
        .process_frame(frame)
    )


# ============================================================
# VIDEO ENDED
# ============================================================

def on_video_ended():

    processor = (
        processor_holder.get(
            "processor"
        )
    )


    if processor is not None:

        processor.close()


    processor_holder["processor"] = None


    with state_lock:

        runtime["running"] = False


# ============================================================
# PAGE LAYOUT
# ============================================================

camera_col, telemetry_col = st.columns(
    [1.15, 1],
    gap="large"
)


# ============================================================
# CAMERA
# ============================================================

with camera_col:

    st.markdown(
        "### 📷 Camera"
    )

    st.caption(
        "Processing: 640 × 480 • "
        "Display: compact"
    )


    ctx = webrtc_streamer(

        key="gesture-recognition",

        video_frame_callback=(
            video_frame_callback
        ),

        on_video_ended=(
            on_video_ended
        ),

        media_stream_constraints={

            "video": {

                "width": FRAME_W,

                "height": FRAME_H

            },

            "audio": False

        },

        media_toggle_controls=False
    )


# ============================================================
# TELEMETRY
# ============================================================

with telemetry_col:

    st.markdown(
        "### 📊 Live Performance Telemetry"
    )


    metric_placeholder = st.empty()


    # --------------------------------------------------------
    # Initial state
    # --------------------------------------------------------

    metric_placeholder.markdown(
        """
| Metric | Value |
|---|---:|
| ⚡ FPS | **0.0** |
| ⏱️ Latency | **0.00 ms** |
| 💾 Current RAM | **0.00 MB** |
| 📈 Peak RAM | **0.00 MB** |
| 🎯 Gesture | **waiting...** |
| 🎲 Confidence | **0.00** |
| ✋ Hand | **No** |
| 🔄 State | **WAITING** |
| 🎞️ Frames | **0** |
| 🧩 Input Window | **120 frames** |
| 📐 Features / Frame | **130** |
"""
    )


# ============================================================
# FINAL PLACEHOLDER
# ============================================================

final_placeholder = st.empty()


# ============================================================
# POLLING LOOP
# ============================================================

while ctx.state.playing:

    with state_lock:

        fps = runtime["fps"]

        latency = runtime["latency_ms"]

        current_ram = (
            runtime["current_ram_mb"]
        )

        peak_ram = (
            runtime["peak_ram_mb"]
        )

        frames = (
            runtime["total_frames"]
        )

        label = runtime["label"]

        confidence = (
            runtime["confidence"]
        )

        gesture_state = (
            runtime["state"]
        )

        hand = (
            runtime["hand_present"]
        )


    # ========================================================
    # LIVE TELEMETRY
    # ========================================================

    metric_placeholder.markdown(
        f"""
| Metric | Value |
|---|---:|
| ⚡ FPS | **{fps:.1f}** |
| ⏱️ Latency | **{latency:.2f} ms** |
| 💾 Current RAM | **{current_ram:.2f} MB** |
| 📈 Peak RAM | **{peak_ram:.2f} MB** |
| 🎯 Gesture | **{label}** |
| 🎲 Confidence | **{confidence:.2f}** |
| ✋ Hand | **{'Yes' if hand else 'No'}** |
| 🔄 State | **{gesture_state}** |
| 🎞️ Frames | **{frames}** |
| 🧩 Input Window | **{WINDOW_SIZE} frames** |
| 📐 Features / Frame | **{FEATURES_PER_FRAME}** |
"""
    )


    time.sleep(0.2)


# ============================================================
# FINAL STATISTICS
# ============================================================

with state_lock:

    final_frames = (
        runtime["total_frames"]
    )

    final_total_latency = (
        runtime["total_latency_ms"]
    )

    final_peak_ram = (
        runtime["peak_ram_mb"]
    )


if final_frames > 0:

    final_avg_latency = (
        final_total_latency
        / final_frames
    )


    final_avg_fps = (
        1000.0 / final_avg_latency
        if final_avg_latency > 0
        else 0.0
    )


    st.markdown("---")


    st.markdown(
        "## 📈 TRUE PIPELINE PERFORMANCE SUMMARY"
    )


    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.metric(
            "Total Frames",
            f"{final_frames}"
        )


    with c2:

        st.metric(
            "Average FPS",
            f"{final_avg_fps:.2f}"
        )


    with c3:

        st.metric(
            "Average Latency",
            f"{final_avg_latency:.2f} ms"
        )


    with c4:

        st.metric(
            "Peak RAM",
            f"{final_peak_ram:.2f} MB"
        )


    st.success(
        "✅ Webcam session finished. "
        "The statistics above are the measurements "
        "collected during the actual processing pipeline."
    )