from flask import Flask, request, render_template, Response, jsonify
import torch
import cv2
import numpy as np
from PIL import Image
import os
import pygame
import threading

app = Flask(__name__)

# Folders for uploaded and processed files
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Alarm sound file
ALARM_SOUND = "alarm.mp3"

# Initialize pygame for playing sound
pygame.mixer.init()

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='final_best_4.pt', force_reload=True)
model.eval()

CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence threshold for detection
ALERT_ANIMALS = ['Bear', 'Elephant', 'Fox', 'Leopard', 'Panther', 'cheetah', 'hyena', 'lion', 'tiger', 'wolf']  # List of animals that should trigger the alarm
alarm_playing = False  # Flag to track if the alarm is playing
alarm_thread = None  # Thread for playing alarm sound


def play_alarm():
    """Plays the alarm continuously until manually stopped."""
    global alarm_playing
    if alarm_playing:
        return  # Do nothing if the alarm is already playing

    alarm_playing = True
    pygame.mixer.music.load(ALARM_SOUND)

    while alarm_playing:
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy() and alarm_playing:
            continue  # Wait for the sound to finish before looping


def stop_alarm():
    """Stops the alarm manually."""
    global alarm_playing
    alarm_playing = False
    pygame.mixer.music.stop()


def detect_objects(image_path, output_path):
    """Detects objects in an image and plays the alarm if needed."""
    global alarm_playing, alarm_thread

    image = Image.open(image_path)
    results = model(image)
    df = results.pandas().xyxy[0]

    img = cv2.imread(image_path)
    alert_triggered = False

    for _, row in df.iterrows():
        confidence = row['confidence']
        class_name = row['name']
        if confidence >= CONFIDENCE_THRESHOLD and class_name in ALERT_ANIMALS:
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f"{class_name} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            alert_triggered = True

    if alert_triggered and not alarm_playing:
        alarm_thread = threading.Thread(target=play_alarm, daemon=True)
        alarm_thread.start()

    cv2.imwrite(output_path, img)
    return output_path


def detect_video_stream(video_path):
    """Processes a video stream and plays the alarm only if the detected animal is in the alert list."""
    global alarm_playing, alarm_thread

    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = model(image)
        df = results.pandas().xyxy[0]

        alert_triggered = False

        for _, row in df.iterrows():
            confidence = row['confidence']
            class_name = row['name']
            if confidence >= CONFIDENCE_THRESHOLD and class_name in ALERT_ANIMALS:
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                alert_triggered = True

        if alert_triggered and not alarm_playing:
            alarm_thread = threading.Thread(target=play_alarm, daemon=True)
            alarm_thread.start()

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    stop_alarm()  # Stop alarm when the video ends


@app.route('/')
def index():
    """Renders the index.html page."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles file uploads and processes images or videos."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    if file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        return jsonify({"video_url": file_path})
    else:
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], 'output.jpg')
        detect_objects(file_path, output_path)
        return jsonify({"image_url": output_path})


@app.route('/detect_video')
def detect_video():
    """Streams video with real-time detection."""
    video_path = request.args.get('video_path')
    return Response(detect_video_stream(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stop_alarm', methods=['POST'])
def stop_alarm_api():
    """API endpoint to manually stop the alarm."""
    stop_alarm()
    return jsonify({"message": "Alarm stopped"})


if __name__ == '__main__':
    app.run(debug=True)