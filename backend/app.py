from flask import Flask, Response, jsonify, render_template
import cv2
import numpy as np
import threading
import time
from ai_model import analyze_class
import atexit
import os

# Get the directory of the current file
basedir = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, 
            template_folder=os.path.join(basedir, 'templates'),
            static_folder=os.path.join(basedir, 'static'))

# Deploy environments (like Heroku) typically don't have a webcam available.
# Fall back to a blank frame so the server can still run.

def _init_camera():
    try:
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cam.isOpened():
            cam.release()
            return None
        return cam
    except Exception:
        return None

camera = _init_camera()

confused = 0
attentive = 0
percent = 0
history = []

raw_frame = None
latest_faces = []
lock = threading.Lock()

stop_event = threading.Event()

# Camera reading thread (very fast)
def read_camera():
    global raw_frame
    while not stop_event.is_set():
        if camera is None:
            # No webcam available (e.g., deployed on Heroku). Use a blank frame.
            with lock:
                raw_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            time.sleep(0.1)
            continue

        success, frame = camera.read()
        if not success:
            time.sleep(0.1)
            continue
        
        with lock:
            raw_frame = frame
        
        time.sleep(0.01)

# Background processing thread (slower, decoupled)
def process_frames():
    global raw_frame, latest_faces, confused, attentive, percent

    while not stop_event.is_set():
        with lock:
            frame = raw_frame.copy() if raw_frame is not None else None
            
        if frame is None:
            time.sleep(0.1)
            continue

        faces_data, c, a, p = analyze_class(frame)

        with lock:
            latest_faces = faces_data
            confused = c
            attentive = a
            percent = p

# Streaming generator (fast, overlay cache)
def generate_frames():
    global raw_frame, latest_faces

    while not stop_event.is_set():
        with lock:
            has_frame = raw_frame is not None
            if has_frame:
                frame = raw_frame.copy()
                faces = list(latest_faces)
        
        if not has_frame:
            time.sleep(0.1)
            continue
            
        # Draw all cached faces onto the raw frame
        for face_data in faces:
            x, y, w, h = face_data['box']
            color = face_data['color']
            emotion = face_data['emotion']
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(
                frame,
                emotion,
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        time.sleep(0.03) # Cap to ~30 FPS


@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/video")
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/stats")
def stats():
    global confused, attentive, percent, history

    history.append(percent)
    if len(history) > 20:
        history.pop(0)

    alert = ""
    if percent > 60:
        alert = " High confusion detected!"

    return jsonify({
        "confused": confused,
        "attentive": attentive,
        "confusion": percent,
        "history": history,
        "alert": alert
    })


def shutdown():
    stop_event.set()
    time.sleep(0.2) # Give threads a moment to finish
    if camera is not None and camera.isOpened():
        camera.release()
        print("Camera properly released and closed.")
    else:
        print("No camera to release (running in headless/deployed mode).")


atexit.register(shutdown)

#  Start background threads
threading.Thread(target=read_camera, daemon=True).start()
threading.Thread(target=process_frames, daemon=True).start()

if __name__ == "__main__":
    import os
    debug = os.environ.get("FLASK_DEBUG", "False") == "True"
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=debug, use_reloader=False)