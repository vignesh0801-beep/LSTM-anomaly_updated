from flask import Flask, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
import threading
from twilio.rest import Client

app = Flask(__name__)

# ✅ Load LSTM Model
from tensorflow.keras.models import load_model
model_path = "lstm-model.h5"
model = None
try:
    model = load_model(model_path)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("⚠ Running without model prediction.")

# ✅ Twilio Configuration
ACCOUNT_SID = "AC055caf982adc2f9d44fed9fd56c862c0"
AUTH_TOKEN = "a6764c6bad1186298f31a7c2f20eb403"
TO_PHONE = "+918788030694"
FROM_PHONE = "+15674061129"

twilio_client = Client(ACCOUNT_SID, AUTH_TOKEN)

# ✅ Initialize Mediapipe Pose Detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# ✅ Open Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

# ✅ Parameters
SEQUENCE_LENGTH = 20
sequence = []
last_alert_time = 0
ALERT_INTERVAL = 30
stop_stream = False  # Flag to stop the stream

# ✅ Extract Pose Landmarks
def extract_landmarks(results):
    if results.pose_landmarks:
        return [coord for lm in results.pose_landmarks.landmark for coord in (lm.x, lm.y, lm.z, lm.visibility)]
    return None

# ✅ Draw Landmarks
def draw_landmarks(frame, results):
    mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    return frame

# ✅ Send Twilio Alert
def send_alert(action):
    global last_alert_time
    current_time = time.time()
    
    if current_time - last_alert_time > ALERT_INTERVAL:
        message = twilio_client.messages.create(
            body=f"⚠ Alert: Violent action detected - {action}!",
            from_=FROM_PHONE,
            to=TO_PHONE
        )
        print(f"📩 Alert sent! SID: {message.sid}")
        last_alert_time = current_time
    else:
        print("⏳ Alert skipped (Cooldown in effect)")

# ✅ Predict Action
def predict_action(model, sequences):
    if model and len(sequences) == SEQUENCE_LENGTH:
        sequences = np.expand_dims(np.array(sequences), axis=0)
        prediction = model.predict(sequences)[0]
        if prediction[1] > 0.5:
            return "Punching"
        elif prediction[2] > 0.5:
            return "Slapping"
        return "Neutral"
    return "Neutral"

# ✅ Generate Video Frames
def generate_frames():
    global sequence, stop_stream
    while not stop_stream:
        success, frame = cap.read()
        if not success or stop_stream:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        landmarks = extract_landmarks(results)
        if landmarks:
            sequence.append(landmarks)

            if len(sequence) > SEQUENCE_LENGTH:
                sequence.pop(0)  # Maintain rolling window of 20 frames

            if len(sequence) == SEQUENCE_LENGTH:
                action = predict_action(model, sequence)
                print(f"🔍 Predicted Action: {action}")
                if action in ["Punching", "Slapping"]:
                    send_alert(action)

            frame = draw_landmarks(frame, results)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    print("🛑 Video stream stopped.")

# ✅ Flask Endpoint to Stop Video
@app.route('/stop_video', methods=['POST'])
def stop_video():
    global stop_stream
    stop_stream = True
    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Video stream stopped via API.")
    return jsonify({"message": "Video stream stopped successfully!"})

# ✅ Flask Endpoint: Stream Webcam Feed
@app.route('/video_feed')
def video_feed():
    global stop_stream
    stop_stream = False  # Reset when starting stream
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ✅ Flask Endpoint: Get Current Prediction
@app.route('/get_prediction', methods=['GET'])
def get_prediction():
    return jsonify({"current_action": predict_action(model, sequence)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)