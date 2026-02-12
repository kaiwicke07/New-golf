import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile

st.title("GitHub Codespaces Golf Analyzer 🏌️")

st.write("Upload a video to analyze the swing.")

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    cap = cv2.VideoCapture(tfile.name)
    st_frame = st.empty()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process
            results = pose.process(image)

            # Draw
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

            # Display
            st_frame.image(image, use_container_width=True)

    cap.release()
