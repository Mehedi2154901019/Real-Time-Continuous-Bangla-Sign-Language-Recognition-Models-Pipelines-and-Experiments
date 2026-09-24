import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import json
import time
import psutil
import os

# Initialize process tracker for RAM usage
process = psutil.Process(os.getpid())

# -----------------------
# Load Classes & Config
# -----------------------
MODEL_PATH = "final_model.keras"
CLASSES_PATH = "classes.json"
INDICES_PATH = "class_indices.json"
CONFIG_PATH = "config.json"

with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

# Read class indices mapping if needed
if os.path.exists(INDICES_PATH):
    with open(INDICES_PATH, "r", encoding="utf-8") as f:
        class_indices = json.load(f)

# Rebuild MobileNetV2 Architecture to bypass deserialization/quantization bugs
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights=None
)

inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
outputs = tf.keras.layers.Dense(len(class_names), activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

# Load weights from the final trained model
model.load_weights(MODEL_PATH)

# -----------------------
# Mediapipe Setup
# -----------------------
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5
)

# -----------------------
# Helper Functions
# -----------------------
def detect_and_crop_palm(image, padding=0.25):
    """
    Detects hand(s) and returns a single combined bounding box crop.
    Adds padding to cover full palms (default 25%).
    """
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if not results.multi_hand_landmarks:
        return None, image

    h, w, _ = image.shape
    boxes = []

    for hand_landmarks in results.multi_hand_landmarks:
        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]

        x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
        y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)

        boxes.append((x_min, y_min, x_max, y_max))

    # Merge into one big bounding box
    x_min = min(b[0] for b in boxes)
    y_min = min(b[1] for b in boxes)
    x_max = max(b[2] for b in boxes)
    y_max = max(b[3] for b in boxes)

    # Add padding
    box_w, box_h = x_max - x_min, y_max - y_min

    x_min = max(0, x_min - int(padding * box_w))
    y_min = max(0, y_min - int(padding * box_h))
    x_max = min(w, x_max + int(padding * box_w))
    y_max = min(h, y_max + int(padding * box_h))

    # Draw bounding box
    cv2.rectangle(
        image,
        (x_min, y_min),
        (x_max, y_max),
        (0, 0, 255),
        2
    )

    cropped = image[y_min:y_max, x_min:x_max]

    return cropped, image


def preprocess_image(cropped):
    """
    Resize and preprocess the cropped hand image using
    the official MobileNetV2 preprocessing function.

    This matches the preprocessing used during model training.
    """
    img_resized = cv2.resize(cropped, (224, 224))

    # Convert to float32
    img_resized = img_resized.astype("float32")

    # Apply MobileNetV2 preprocessing.
    # This scales pixel values according to the preprocessing
    # expected by an ImageNet-pretrained MobileNetV2 model.
    img_resized = tf.keras.applications.mobilenet_v2.preprocess_input(
        img_resized
    )

    # Add batch dimension
    img_resized = np.expand_dims(img_resized, axis=0)

    return img_resized


def predict_class(img_tensor):
    """Run prediction and return class label & confidence."""
    preds = model.predict(img_tensor, verbose=0)

    class_idx = np.argmax(preds)
    confidence = float(np.max(preds))

    return class_names[class_idx], confidence


# -----------------------
# Streamlit App
# -----------------------
st.title("🤚 Real-Time Sign Language Recognition (MobileNetV2)")

run = st.checkbox("Start Webcam")

# Telemetry placeholders for live stats banner
metrics_placeholder = st.empty()
FRAME_WINDOW = st.image([])

if run:

    cap = cv2.VideoCapture(0)

    # Session Metrics Trackers
    total_latency = 0.0
    frame_count = 0
    peak_ram_mb = 0.0

    while run:

        # Start timer for end-to-end frame loop
        loop_start_time = time.perf_counter()

        ret, frame = cap.read()

        if not ret:
            st.error("Failed to access webcam.")
            break

        # Mirror effect for user-friendly webcam view
        frame = cv2.flip(frame, 1)

        # Detect hands and crop the detected region
        cropped, annotated = detect_and_crop_palm(
            frame,
            padding=0.25
        )

        if cropped is not None:

            # Preprocess using MobileNetV2 preprocessing
            img_tensor = preprocess_image(cropped)

            # Model prediction
            pred_class, confidence = predict_class(img_tensor)

            cv2.putText(
                annotated,
                f"{pred_class} ({confidence * 100:.1f}%)",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

        # End timer & calculate metrics
        loop_end_time = time.perf_counter()

        elapsed_time = loop_end_time - loop_start_time

        latency_ms = elapsed_time * 1000

        current_fps = (
            1.0 / elapsed_time
            if elapsed_time > 0
            else 0.0
        )

        # Accumulate metrics for session averages
        total_latency += latency_ms
        frame_count += 1

        # Track Peak RAM
        current_ram_mb = (
            process.memory_info().rss / (1024 * 1024)
        )

        if current_ram_mb > peak_ram_mb:
            peak_ram_mb = current_ram_mb

        # Update live telemetry metrics banner
        metrics_placeholder.markdown(
            f"⚡ **Live FPS:** `{current_fps:.1f}` | "
            f"⏱️ **Latency:** `{latency_ms:.1f} ms` | "
            f"💾 **Current RAM:** `{current_ram_mb:.1f} MB` | "
            f"📈 **Peak RAM:** `{peak_ram_mb:.1f} MB`"
        )

        # Display annotated webcam frame
        FRAME_WINDOW.image(
            cv2.cvtColor(
                annotated,
                cv2.COLOR_BGR2RGB
            )
        )

    cap.release()

    # Calculate true session averages when loop stops
    avg_latency = (
        total_latency / frame_count
        if frame_count > 0
        else 0.0
    )

    avg_fps = (
        1000.0 / avg_latency
        if avg_latency > 0
        else 0.0
    )

    # -----------------------
    # Streamlit Summary
    # -----------------------
    st.success(
        "✅ Webcam session ended. Performance Summary:"
    )

    st.markdown(
        f"""
        - **Total Frames Processed:** {frame_count}
        - **True Average Latency:** ~{avg_latency:.2f} ms
        - **True Average Operational FPS:** ~{avg_fps:.1f}
        - **Peak RAM Consumption:** {peak_ram_mb:.2f} MB
        """
    )

    # -----------------------
    # Terminal Summary
    # -----------------------
    print("\n" + "=" * 45)
    print(" TRUE PIPELINE PERFORMANCE SUMMARY ")
    print("=" * 45)
    print(
        f" Total Frames Processed      : {frame_count}"
    )
    print(
        f" Peak RAM Consumption        : {peak_ram_mb:.2f} MB"
    )
    print(
        f" True Average Latency        : ~{avg_latency:.2f} ms"
    )
    print(
        f" True Average Operational FPS: ~{avg_fps:.1f}"
    )
    print("=" * 45)

else:
    st.write("☝️ Check 'Start Webcam' to begin")