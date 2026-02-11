import os
import cv2
import numpy as np
import mediapipe as mp
import base64
from flask import Flask, render_template, request, jsonify

app = Flask(__name__, static_folder='static', template_folder='templates')

# --- Configuration ---
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- Golf Analysis Logic ---
mp_pose = mp.solutions.pose
COLORS = {
    "join": (0, 20, 200),   # Red-ish
    "marks": (0, 255, 255), # Yellow
    "text": (0, 255, 0),    # Green
    "axis": (200, 200, 200) # Light Gray
}

def calculate_angle(a, b, c):
    """Calculate angle ABC (in degrees)"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

def process_swing_video(input_path, output_path):
    """
    Processes video: tracks landmarks, draws overlay, returns path to processed video
    and a base64 encoded 'key frame' for Gemini analysis.
    """
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    mid_frame_b64 = None
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            current_frame += 1
            
            # 1. Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            
            # 2. Draw Visuals
            if results.pose_landmarks:
                h, w, _ = frame.shape
                landmarks = results.pose_landmarks.landmark
                
                def get_coord(lm): return (int(lm.x * w), int(lm.y * h))
                
                # Key Points: Ear, Hip, Wrist (Right side assumed for DTL view)
                p_ear = get_coord(landmarks[mp_pose.PoseLandmark.RIGHT_EAR])
                p_hip = get_coord(landmarks[mp_pose.PoseLandmark.RIGHT_HIP])
                p_wrist = get_coord(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST])
                
                # Draw Skeleton
                cv2.line(frame, p_ear, p_hip, COLORS["join"], 3)
                cv2.line(frame, p_hip, p_wrist, COLORS["join"], 3)
                cv2.circle(frame, p_ear, 5, COLORS["marks"], -1)
                cv2.circle(frame, p_hip, 5, COLORS["marks"], -1)
                cv2.circle(frame, p_wrist, 5, COLORS["marks"], -1)
                
                # Calculate Angle
                angle = calculate_angle(p_ear, p_hip, p_wrist)
                cv2.putText(frame, f"{int(angle)} deg", (p_hip[0] + 20, p_hip[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS["text"], 2)
                
                # Draw Axis (Bottom Left)
                cv2.line(frame, (50, h-50), (50, h-100), COLORS["axis"], 2) # Y
                cv2.line(frame, (50, h-50), (100, h-50), COLORS["axis"], 2) # X
            
            # 3. Capture "Middle" frame for Gemini Analysis 
            if current_frame == int(frame_count / 2):
                _, buffer = cv2.imencode('.jpg', frame)
                mid_frame_b64 = base64.b64encode(buffer).decode('utf-8')

            out.write(frame)
            
    cap.release()
    out.release()
    return mid_frame_b64

# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze_video', methods=['POST'])
def analyze_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    video = request.files['video']
    if video.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    input_filename = "input_" + video.filename
    output_filename = "processed_" + video.filename
    input_path = os.path.join(UPLOAD_FOLDER, input_filename)
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
    
    video.save(input_path)
    
    try:
        best_frame_b64 = process_swing_video(input_path, output_path)
        return jsonify({
            'processed_video_url': f"/{output_path}",
            'analysis_frame': best_frame_b64
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)