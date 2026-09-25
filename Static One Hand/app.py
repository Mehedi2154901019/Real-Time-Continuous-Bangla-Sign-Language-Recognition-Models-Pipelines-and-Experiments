# ============================================================
# BdSL47 - Signer-Independent Real-Time Inference Pipeline
# ============================================================
#
# Pipeline:
# Webcam frame
#     ↓
# MediaPipe Hands
#     ↓
# Number of hands detected?
#     ├── 0 → No hand detected
#     ├── 2 → Two hands detected (unsupported)
#     └── 1 → Extract 21 landmarks × XYZ = 63 features
#                ↓
#          Translation normalization
#                ↓
#          Scale normalization
#                ↓
#          Flatten to 63
#                ↓
#          StandardScaler
#                ↓
#          MLP (.keras)
#                ↓
#          BdSL47 class prediction
#
# Metrics:
# - FPS
# - End-to-end latency
# - Peak RAM
# - Prediction confidence
#
# Bangla display:
# - Pillow + Noto Sans Bengali
#
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

import os
import json
import time
import threading

import cv2
import numpy as np
import joblib
import psutil
import mediapipe as mp
import tensorflow as tf

from tensorflow.keras.models import load_model

from PIL import Image, ImageDraw, ImageFont


# ============================================================
# 2. PATHS
# ============================================================

DATA_DIR = (
    r"D:\aaa EAAI Major Revision\static one handed"
)

MODEL_PATH = os.path.join(
    DATA_DIR,
    "final_model.keras"
)

SCALER_PATH = os.path.join(
    DATA_DIR,
    "scaler.pkl"
)

CONFIG_PATH = os.path.join(
    DATA_DIR,
    "pipeline_config.json"
)

# Bangla Unicode font
BANGLA_FONT_PATH = os.path.join(
    DATA_DIR,
    "NotoSansBengali-Regular.ttf"
)


# ============================================================
# 3. CLASS MAPPING
# ============================================================

class_map = {
    0: "০",
    1: "১",
    2: "২",
    3: "৩",
    4: "৪",
    5: "৫",
    6: "৬",
    7: "৭",
    8: "৮",
    9: "৯",

    10: "অ/য়",
    11: "আ",
    12: "ই/ঈ",
    13: "উ/ঊ",
    14: "র/ঋ/ড়/ঢ়",
    15: "এ",
    16: "ঐ",
    17: "ও",
    18: "ঔ",
    19: "ক",

    20: "খ/ক্ষ",
    21: "গ",
    22: "ঘ",
    23: "ঙ",
    24: "চ",
    25: "ছ",
    26: "জ/য",
    27: "ঝ",
    28: "ঞ",
    29: "ট",

    30: "ঠ",
    31: "ড",
    32: "ঢ",
    33: "ণ/ন",
    34: "ত",
    35: "থ",
    36: "দ",
    37: "ধ",
    38: "প",
    39: "ফ",

    40: "ব/ভ",
    41: "ম",
    42: "ল",
    43: "শ/ষ/স",
    44: "হ",
    45: "ং",
    46: "ঁ"
}


# ============================================================
# 4. CHECK REQUIRED FILES
# ============================================================

print("=" * 75)
print("BdSL47 REAL-TIME INFERENCE PIPELINE")
print("=" * 75)

required_files = [
    MODEL_PATH,
    SCALER_PATH,
    CONFIG_PATH,
    BANGLA_FONT_PATH
]

for path in required_files:

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}"
        )

print("\nAll required files found.")

print(f"\nModel: {MODEL_PATH}")
print(f"Scaler: {SCALER_PATH}")
print(f"Config: {CONFIG_PATH}")
print(f"Font: {BANGLA_FONT_PATH}")


# ============================================================
# 5. LOAD BANGLA FONTS
# ============================================================

# Large prediction text
BANGLA_FONT_LARGE = ImageFont.truetype(
    BANGLA_FONT_PATH,
    40
)

# Smaller information text
BANGLA_FONT_MEDIUM = ImageFont.truetype(
    BANGLA_FONT_PATH,
    27
)

# Normal English text
ENGLISH_FONT = ImageFont.truetype(
    BANGLA_FONT_PATH,
    25
)


# ============================================================
# 6. UNICODE TEXT DRAWING FUNCTION
# ============================================================

def draw_unicode_text(
    frame,
    text,
    position,
    font,
    fill=(255, 255, 255)
):
    """
    Draw Unicode/Bangla text on an OpenCV frame
    using Pillow.

    OpenCV's cv2.putText() does not properly support
    Bangla Unicode characters.
    """

    # OpenCV BGR → RGB
    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Convert NumPy image to PIL
    pil_image = Image.fromarray(
        frame_rgb
    )

    draw = ImageDraw.Draw(
        pil_image
    )

    draw.text(
        position,
        text,
        font=font,
        fill=fill
    )

    # PIL RGB → OpenCV BGR
    frame[:] = cv2.cvtColor(
        np.array(pil_image),
        cv2.COLOR_RGB2BGR
    )

    return frame


# ============================================================
# 7. LOAD CONFIGURATION
# ============================================================

with open(
    CONFIG_PATH,
    "r",
    encoding="utf-8"
) as f:

    config = json.load(f)


APPLY_TRANSLATION_NORMALIZATION = config.get(
    "apply_translation_normalization",
    True
)

APPLY_SCALE_NORMALIZATION = config.get(
    "apply_scale_normalization",
    True
)

APPLY_ROTATION_NORMALIZATION = config.get(
    "apply_rotation_normalization",
    False
)

FEATURE_COLUMNS = config.get(
    "feature_columns",
    []
)

NUM_CLASSES = config.get(
    "num_classes",
    47
)


print("\n" + "=" * 75)
print("PIPELINE CONFIGURATION")
print("=" * 75)

print(
    f"Translation normalization: "
    f"{APPLY_TRANSLATION_NORMALIZATION}"
)

print(
    f"Scale normalization:       "
    f"{APPLY_SCALE_NORMALIZATION}"
)

print(
    f"Rotation normalization:    "
    f"{APPLY_ROTATION_NORMALIZATION}"
)

print(
    f"Number of classes:          "
    f"{NUM_CLASSES}"
)

print(
    f"Number of features:        "
    f"{len(FEATURE_COLUMNS)}"
)


# ============================================================
# 8. LOAD MODEL
# ============================================================

print("\nLoading Keras model...")

model = load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully.")


# ============================================================
# 9. LOAD STANDARD SCALER
# ============================================================

print("\nLoading StandardScaler...")

scaler = joblib.load(
    SCALER_PATH
)

print("Scaler loaded successfully.")


# ============================================================
# 10. VERIFY MODEL AND PIPELINE
# ============================================================

print("\n" + "=" * 75)
print("PIPELINE VERIFICATION")
print("=" * 75)


if len(class_map) != NUM_CLASSES:

    raise ValueError(
        f"Class mapping contains "
        f"{len(class_map)} classes, "
        f"but configuration specifies "
        f"{NUM_CLASSES}."
    )


if model.output_shape[-1] != NUM_CLASSES:

    raise ValueError(
        f"Model outputs "
        f"{model.output_shape[-1]} classes, "
        f"but expected "
        f"{NUM_CLASSES}."
    )


if len(FEATURE_COLUMNS) != 63:

    raise ValueError(
        f"Expected 63 feature columns, "
        f"but configuration contains "
        f"{len(FEATURE_COLUMNS)}."
    )


print("Class mapping:             OK")
print("Model output classes:      OK")
print("Feature count:             63")
print("Scaler:                    OK")
print("Pipeline configuration:    OK")

print("\nPipeline verification successful.")


# ============================================================
# 11. MEDIAPIPE INITIALIZATION
# ============================================================

mp_hands = mp.solutions.hands

mp_drawing = (
    mp.solutions.drawing_utils
)

mp_drawing_styles = (
    mp.solutions.drawing_styles
)


# ============================================================
# 12. LANDMARK NORMALIZATION
# ============================================================

def normalize_landmarks(landmarks):

    """
    Applies exactly the same geometric normalization
    used during model training.

    Input:
        landmarks: (21, 3)

    Output:
        normalized landmarks: (21, 3)
    """

    landmarks = landmarks.copy()

    # --------------------------------------------------------
    # Wrist
    # --------------------------------------------------------

    wrist = landmarks[0].copy()

    # --------------------------------------------------------
    # Translation normalization
    # --------------------------------------------------------

    if APPLY_TRANSLATION_NORMALIZATION:

        landmarks = (
            landmarks - wrist
        )

    # --------------------------------------------------------
    # Scale normalization
    # --------------------------------------------------------

    if APPLY_SCALE_NORMALIZATION:

        distances = np.linalg.norm(
            landmarks,
            axis=1
        )

        scale = np.max(
            distances
        )

        if scale > 1e-8:

            landmarks = (
                landmarks / scale
            )

    # --------------------------------------------------------
    # Rotation normalization
    #
    # Disabled for the current trained model.
    # --------------------------------------------------------

    if APPLY_ROTATION_NORMALIZATION:

        index_mcp = landmarks[5]

        middle_mcp = landmarks[9]

        pinky_mcp = landmarks[17]

        axis_u = (
            middle_mcp - wrist
        )

        norm_u = np.linalg.norm(
            axis_u
        )

        if norm_u > 1e-8:

            axis_u = (
                axis_u / norm_u
            )

            axis_v = (
                index_mcp - wrist
            )

            axis_v = (
                axis_v
                - np.dot(
                    axis_v,
                    axis_u
                ) * axis_u
            )

            norm_v = np.linalg.norm(
                axis_v
            )

            if norm_v > 1e-8:

                axis_v = (
                    axis_v / norm_v
                )

                axis_w = np.cross(
                    axis_u,
                    axis_v
                )

                norm_w = np.linalg.norm(
                    axis_w
                )

                if norm_w > 1e-8:

                    axis_w = (
                        axis_w / norm_w
                    )

                    axis_v = np.cross(
                        axis_w,
                        axis_u
                    )

                    axis_v = (
                        axis_v
                        / np.linalg.norm(
                            axis_v
                        )
                    )

                    rotation_matrix = (
                        np.column_stack(
                            [
                                axis_v,
                                axis_u,
                                axis_w
                            ]
                        )
                    )

                    landmarks = (
                        landmarks
                        @ rotation_matrix
                    )

    return landmarks


# ============================================================
# 13. PREPARE LANDMARKS FOR MODEL
# ============================================================

def prepare_landmarks_for_model(
    landmarks
):

    """
    Converts MediaPipe's 21×3 landmarks
    into the exact 63-dimensional representation
    used during model training.
    """

    landmarks = np.asarray(
        landmarks,
        dtype=np.float32
    )

    if landmarks.shape != (21, 3):

        raise ValueError(
            f"Expected landmark shape "
            f"(21, 3), got "
            f"{landmarks.shape}"
        )

    # Geometric normalization
    landmarks = normalize_landmarks(
        landmarks
    )

    # 21 × 3 → 63
    features = landmarks.reshape(
        1,
        63
    )

    # StandardScaler
    features = scaler.transform(
        features
    )

    return features.astype(
        np.float32
    )


# ============================================================
# 14. PREDICTION FUNCTION
# ============================================================

def predict_landmarks(
    landmarks
):

    """
    Returns:

        predicted_class
        predicted_label
        confidence
    """

    features = (
        prepare_landmarks_for_model(
            landmarks
        )
    )

    probabilities = (
        model.predict(
            features,
            verbose=0
        )[0]
    )

    predicted_class = int(
        np.argmax(
            probabilities
        )
    )

    confidence = float(
        probabilities[
            predicted_class
        ]
    )

    predicted_label = (
        class_map[
            predicted_class
        ]
    )

    return (
        predicted_class,
        predicted_label,
        confidence
    )


# ============================================================
# 15. RAM MONITOR
# ============================================================

process = psutil.Process(
    os.getpid()
)

peak_ram_mb = 0.0

monitoring = True


def monitor_ram():

    global peak_ram_mb

    while monitoring:

        try:

            ram_mb = (
                process
                .memory_info()
                .rss
                / (1024 ** 2)
            )

            if ram_mb > peak_ram_mb:

                peak_ram_mb = ram_mb

        except Exception:
            pass

        time.sleep(
            0.01
        )


ram_thread = threading.Thread(
    target=monitor_ram,
    daemon=True
)

ram_thread.start()


# ============================================================
# 16. PERFORMANCE STORAGE
# ============================================================

latencies_ms = []

fps_values = []

prediction_count = 0


# ============================================================
# 17. CAMERA
# ============================================================

CAMERA_INDEX = 0

cap = cv2.VideoCapture(
    CAMERA_INDEX
)

if not cap.isOpened():

    raise RuntimeError(
        "Could not open webcam."
    )


# ============================================================
# 18. CAMERA SETTINGS
# ============================================================

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)


# ============================================================
# 19. MEDIAPIPE HANDS
# ============================================================

with mp_hands.Hands(

    static_image_mode=False,

    max_num_hands=2,

    model_complexity=1,

    min_detection_confidence=0.5,

    min_tracking_confidence=0.5

) as hands:

    previous_frame_time = (
        time.perf_counter()
    )

    while True:

        # ====================================================
        # START END-TO-END TIMER
        # ====================================================

        start_time = (
            time.perf_counter()
        )

        # ----------------------------------------------------
        # Capture frame
        # ----------------------------------------------------

        success, frame = (
            cap.read()
        )

        if not success:

            print(
                "Failed to read frame."
            )

            break

        # ----------------------------------------------------
        # BGR → RGB
        # ----------------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------------------
        # MediaPipe processing
        # ----------------------------------------------------

        results = hands.process(
            rgb_frame
        )

        # ====================================================
        # DEFAULT VALUES
        # ====================================================

        display_text = (
            "No hand detected"
        )

        confidence_text = ""

        hand_count = 0

        predicted_class = None

        predicted_label = None

        confidence = None

        # ====================================================
        # HAND COUNT
        # ====================================================

        if results.multi_hand_landmarks:

            hand_count = len(
                results.multi_hand_landmarks
            )

        # ====================================================
        # TWO HANDS
        # ====================================================

        if hand_count >= 2:

            display_text = (
                "Two hands detected"
            )

            confidence_text = (
                "Unsupported - "
                "dataset contains one-hand gestures"
            )

            # Draw both hands

            for hand_landmarks in (
                results.multi_hand_landmarks
            ):

                mp_drawing.draw_landmarks(

                    frame,

                    hand_landmarks,

                    mp_hands.HAND_CONNECTIONS,

                    mp_drawing_styles
                    .get_default_hand_landmarks_style(),

                    mp_drawing_styles
                    .get_default_hand_connections_style()
                )

        # ====================================================
        # ONE HAND
        # ====================================================

        elif hand_count == 1:

            hand_landmarks = (
                results.multi_hand_landmarks[0]
            )

            # ------------------------------------------------
            # Draw landmarks
            # ------------------------------------------------

            mp_drawing.draw_landmarks(

                frame,

                hand_landmarks,

                mp_hands.HAND_CONNECTIONS,

                mp_drawing_styles
                .get_default_hand_landmarks_style(),

                mp_drawing_styles
                .get_default_hand_connections_style()
            )

            # ------------------------------------------------
            # Extract 21 × XYZ
            # ------------------------------------------------

            landmark_array = np.array(

                [
                    [
                        landmark.x,
                        landmark.y,
                        landmark.z
                    ]

                    for landmark
                    in hand_landmarks.landmark
                ],

                dtype=np.float32
            )

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            try:

                (
                    predicted_class,
                    predicted_label,
                    confidence
                ) = predict_landmarks(
                    landmark_array
                )

                display_text = (
                    f"Class "
                    f"{predicted_class}: "
                    f"{predicted_label}"
                )

                confidence_text = (
                    f"Confidence: "
                    f"{confidence * 100:.2f}%"
                )

                prediction_count += 1

            except Exception as e:

                display_text = (
                    "Prediction error"
                )

                confidence_text = str(e)

        # ====================================================
        # END-TO-END LATENCY
        # ====================================================

        end_time = (
            time.perf_counter()
        )

        latency_ms = (
            end_time - start_time
        ) * 1000.0

        latencies_ms.append(
            latency_ms
        )

        # ====================================================
        # FPS
        # ====================================================

        current_time = (
            time.perf_counter()
        )

        frame_interval = (
            current_time
            - previous_frame_time
        )

        previous_frame_time = (
            current_time
        )

        if frame_interval > 0:

            current_fps = (
                1.0 / frame_interval
            )

            fps_values.append(
                current_fps
            )

        else:

            current_fps = 0.0

        # ====================================================
        # DISPLAY - BANGLA PREDICTION
        # ====================================================

        frame = draw_unicode_text(

            frame,

            display_text,

            (30, 20),

            BANGLA_FONT_LARGE,

            fill=(0, 255, 0)
        )

        # ====================================================
        # DISPLAY - CONFIDENCE
        # ====================================================

        frame = draw_unicode_text(

            frame,

            confidence_text,

            (30, 70),

            BANGLA_FONT_MEDIUM,

            fill=(255, 255, 0)
        )

        # ====================================================
        # DISPLAY - HAND COUNT
        # ====================================================

        frame = draw_unicode_text(

            frame,

            f"Hands detected: {hand_count}",

            (30, 110),

            ENGLISH_FONT,

            fill=(255, 255, 255)
        )

        # ====================================================
        # DISPLAY - FPS
        # ====================================================

        frame = draw_unicode_text(

            frame,

            f"FPS: {current_fps:.2f}",

            (30, 150),

            ENGLISH_FONT,

            fill=(0, 255, 255)
        )

        # ====================================================
        # DISPLAY - LATENCY
        # ====================================================

        frame = draw_unicode_text(

            frame,

            f"Latency: {latency_ms:.2f} ms",

            (30, 190),

            ENGLISH_FONT,

            fill=(0, 255, 255)
        )

        # ====================================================
        # DISPLAY - PEAK RAM
        # ====================================================

        frame = draw_unicode_text(

            frame,

            f"Peak RAM: {peak_ram_mb:.2f} MB",

            (30, 230),

            ENGLISH_FONT,

            fill=(0, 255, 255)
        )

        # ====================================================
        # DISPLAY - EXIT
        # ====================================================

        frame = draw_unicode_text(

            frame,

            "Press Q to quit",

            (30, 270),

            ENGLISH_FONT,

            fill=(255, 255, 255)
        )

        # ====================================================
        # SHOW FRAME
        # ====================================================

        cv2.imshow(

            "BdSL47 Signer-Independent Recognition",

            frame
        )

        # ====================================================
        # QUIT
        # ====================================================

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        if key == ord("q"):

            break


# ============================================================
# 20. CLEANUP
# ============================================================

monitoring = False

cap.release()

cv2.destroyAllWindows()


# ============================================================
# 21. PERFORMANCE SUMMARY
# ============================================================

print("\n")

print("=" * 75)
print("PERFORMANCE SUMMARY")
print("=" * 75)


# ------------------------------------------------------------
# LATENCY
# ------------------------------------------------------------

if len(latencies_ms) > 0:

    latency_array = np.array(
        latencies_ms
    )

    print(
        f"Frames processed:        "
        f"{len(latency_array)}"
    )

    print(
        f"Mean latency:            "
        f"{np.mean(latency_array):.2f} ms"
    )

    print(
        f"Median latency:          "
        f"{np.median(latency_array):.2f} ms"
    )

    print(
        f"Minimum latency:         "
        f"{np.min(latency_array):.2f} ms"
    )

    print(
        f"Maximum latency:         "
        f"{np.max(latency_array):.2f} ms"
    )

    print(
        f"95th percentile latency: "
        f"{np.percentile(latency_array, 95):.2f} ms"
    )


# ------------------------------------------------------------
# FPS
# ------------------------------------------------------------

if len(fps_values) > 0:

    fps_array = np.array(
        fps_values
    )

    print(
        f"\nMean FPS:                "
        f"{np.mean(fps_array):.2f}"
    )

    print(
        f"Median FPS:              "
        f"{np.median(fps_array):.2f}"
    )

    print(
        f"Minimum FPS:             "
        f"{np.min(fps_array):.2f}"
    )

    print(
        f"Maximum FPS:             "
        f"{np.max(fps_array):.2f}"
    )


# ------------------------------------------------------------
# OTHER METRICS
# ------------------------------------------------------------

print(
    f"\nPrediction count:        "
    f"{prediction_count}"
)

print(
    f"Peak RAM usage:          "
    f"{peak_ram_mb:.2f} MB"
)


# ============================================================
# 22. FINAL
# ============================================================

print("=" * 75)
print("Inference completed successfully.")
print("=" * 75)