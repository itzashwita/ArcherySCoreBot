import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
from datetime import datetime
import os

st.set_page_config(page_title="Archery YOLOv8 Demo", layout="wide")
st.title("Ashwita Ramanavelan's Archery Detection using YOLOv8 🎯")

@st.cache_resource
def load_model():
    return YOLO("runs/archery_yolov88/weights/best.pt")

model = load_model()

# ---------------- USER INPUT ----------------
st.sidebar.header("🏹 Participant Info")

name = st.sidebar.text_input("Enter Archer Name")
category = st.sidebar.selectbox(
    "Choose Category",
    ["Practice", "Qualification", "Final Round"]
)

input_type = st.radio("Select Input Type", ["Image", "Video", "Webcam"])

SCORE_PER_ARROW = 10
EXCEL_FILE = "scores.xlsx"

# ---------------- SAVE TO EXCEL FUNCTION ----------------
def save_score(name, category, input_type, arrows, score):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    new_row = pd.DataFrame([{
        "Name": name,
        "Category": category,
        "Input Type": input_type,
        "Arrows Detected": arrows,
        "Total Score": score,
        "Timestamp": timestamp
    }])

    if os.path.exists(EXCEL_FILE):
        existing = pd.read_excel(EXCEL_FILE)
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row

    combined.to_excel(EXCEL_FILE, index=False)

# ---------------- IMAGE ----------------
if input_type == "Image":
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_image:
        bytes_data = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(bytes_data, cv2.IMREAD_COLOR)

        results = model(image, conf=0.85)
        annotated = results[0].plot()

        num_arrows = len(results[0].boxes)
        total_score = num_arrows * SCORE_PER_ARROW

        col1, col2 = st.columns([3, 1])

        with col1:
            st.image(annotated, channels="BGR")

        with col2:
            st.subheader("📊 Score Board")
            st.metric("Arrows Detected", num_arrows)
            st.metric("Total Score", total_score)

            if st.button("💾 Save Score"):
                if name.strip() == "":
                    st.warning("Please enter archer name.")
                else:
                    save_score(name, category, input_type, num_arrows, total_score)
                    st.success("Score saved successfully!")

# ---------------- VIDEO ----------------
elif input_type == "Video":
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if uploaded_video:
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_video.read())

        cap = cv2.VideoCapture("temp_video.mp4")
        frame_slot = st.empty()
        final_arrows = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=0.85)
            annotated = results[0].plot()
            final_arrows = len(results[0].boxes)

            col1, col2 = frame_slot.columns([3, 1])

            with col1:
                st.image(annotated, channels="BGR")

            with col2:
                st.subheader("📊 Score Board")
                st.metric("Arrows Detected", final_arrows)
                st.metric("Total Score", final_arrows * SCORE_PER_ARROW)

        cap.release()

        if st.button("💾 Save Final Score"):
            save_score(name, category, input_type, final_arrows, final_arrows * SCORE_PER_ARROW)
            st.success("Score saved successfully!")

# ---------------- WEBCAM ----------------
elif input_type == "Webcam":
    start = st.checkbox("Start Webcam")
    cap = cv2.VideoCapture(0)
    frame_slot = st.empty()
    last_arrows = 0

    while start:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.85)
        annotated = results[0].plot()

        last_arrows = count_valid_arrows(results[0])
        total_score = last_arrows * SCORE_PER_ARROW

        col1, col2 = frame_slot.columns([3, 1])

        with col1:
            st.image(annotated, channels="BGR")

        with col2:
            st.subheader("📊 Score Board")
            st.metric("Arrows Detected", last_arrows)
            st.metric("Total Score", total_score)

    cap.release()

    if st.button("💾 Save Final Score"):
        save_score(name, category, input_type, last_arrows, total_score)
        st.success("Score saved successfully!")
