from flask import Flask, render_template, Response, jsonify, send_from_directory
import cv2
import time
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from threading import Lock
from pathlib import Path

app = Flask(__name__)

# Define the path for the saved graph
GRAPH_PATH = "static/velocity_graph.png"
MAX_LOG_SAMPLES = 3000

# Initialize video capture but do not keep it running
cap = None
stream_lock = Lock()
is_streaming = False

# Setup Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Motion tracking variables
data_log = []
data_lock = Lock()

def calculate_angle(a, b, c):
    """Calculate the angle between three points (a, b, c) in degrees."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def generate_frames():
    """Capture video frames and process motion tracking."""
    global cap, is_streaming

    with stream_lock:
        is_streaming = True
        cap = cv2.VideoCapture(0)  # Open camera when streaming starts

    if not cap.isOpened():
        with stream_lock:
            is_streaming = False
            cap = None
        return

    try:
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                # Convert image to RGB for Mediapipe processing
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                current_time = time.time()
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark

                    # Extract joint positions
                    joints = {
                        "shoulder": [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y],
                        "elbow": [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y],
                        "wrist": [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y],
                        "hip": [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y],
                        "knee": [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y],
                        "ankle": [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]
                    }

                    # Calculate angles
                    angles = {
                        "elbow": calculate_angle(joints["shoulder"], joints["elbow"], joints["wrist"]),
                        "knee": calculate_angle(joints["hip"], joints["knee"], joints["ankle"]),
                        "shoulder": calculate_angle(joints["hip"], joints["shoulder"], joints["elbow"]),
                        "hip": calculate_angle(joints["shoulder"], joints["hip"], joints["knee"])
                    }

                    # Store data
                    with data_lock:
                        data_log.append([current_time, angles["elbow"], angles["shoulder"], angles["hip"], angles["knee"]])
                        if len(data_log) > MAX_LOG_SAMPLES:
                            data_log.pop(0)

                    # Draw Pose Landmarks
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Encode frame for video stream
                _, buffer = cv2.imencode('.jpg', image)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        with stream_lock:
            if cap and cap.isOpened():
                cap.release()
            cap = None
            is_streaming = False

@app.route('/')
def index():
    return render_template('Index.html')

@app.route('/video_feed')
def video_feed():
    """Start video stream."""
    if is_streaming:
        return jsonify({"error": "A stream is already running. Stop it before starting a new one."}), 409
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_video', methods=['POST'])
def stop_video():
    """Stop the camera feed."""
    global cap, is_streaming
    with stream_lock:
        if cap and cap.isOpened():
            cap.release()
        cap = None
        is_streaming = False
    return jsonify({"message": "Video stopped successfully"})

@app.route('/generate_velocity_graph', methods=['POST'])
def generate_velocity_graph():
    """Generate velocity graph and save it as a PNG file."""
    with data_lock:
        if not data_log:
            return jsonify({"error": "No motion data recorded yet!"}), 400
        data_array = np.array(data_log, dtype=np.float64)

    # Use relative time (seconds since first sample) for readable charts.
    data_array[:, 0] = data_array[:, 0] - data_array[0, 0]

    if data_array.shape[0] < 2:
        return jsonify({"error": "Need at least two samples to generate velocity data."}), 400

    # Create figure
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))

    # Plot angles over time
    axs[0].plot(data_array[:, 0], data_array[:, 1], label="Elbow Angle")
    axs[0].plot(data_array[:, 0], data_array[:, 2], label="Shoulder Angle")
    axs[0].plot(data_array[:, 0], data_array[:, 3], label="Hip Angle")
    axs[0].plot(data_array[:, 0], data_array[:, 4], label="Knee Angle")
    axs[0].set_title("Joint Angles Over Time")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Angle (Degrees)")
    axs[0].legend()
    axs[0].grid(True)

    # Plot velocities over time (simulated using gradient)
    velocities = np.gradient(data_array[:, 1:], data_array[:, 0], axis=0)

    axs[1].plot(data_array[:, 0], velocities[:, 0], label="Elbow Velocity")
    axs[1].plot(data_array[:, 0], velocities[:, 1], label="Shoulder Velocity")
    axs[1].plot(data_array[:, 0], velocities[:, 2], label="Hip Velocity")
    axs[1].plot(data_array[:, 0], velocities[:, 3], label="Knee Velocity")
    axs[1].set_title("Joint Velocities Over Time")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Velocity (units/s)")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    Path(GRAPH_PATH).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(GRAPH_PATH)
    plt.close()

    return jsonify({"graph_url": f"/static/velocity_graph.png"})

@app.route('/reset_session', methods=['POST'])
def reset_session():
    """Reset motion tracking data for a new capture session."""
    with data_lock:
        data_log.clear()
    return jsonify({"message": "Session reset successfully"})

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True)


