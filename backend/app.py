from flask import Flask, Response, jsonify, render_template
import cv2
import numpy as np
import threading
import time
# from ai_model import analyze_class  # TODO: restore once ai_model dependency is resolved
import atexit
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Detect if running on Vercel (serverless environment)
VERCEL_ENV = os.environ.get('VERCEL') == '1'
logger.info(f"Running on Vercel: {VERCEL_ENV}")

# Get the directory of the current file
basedir = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, 
            template_folder=os.path.join(basedir, 'templates'),
            static_folder=os.path.join(basedir, 'static'))

logger.info("Initializing Confusion Detection Application")

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
    try:
        while not stop_event.is_set():
            try:
                if camera is None:
                    # No webcam available (e.g., deployed). Use a blank frame.
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
            except Exception as e:
                logger.error(f"Error in read_camera loop: {e}")
                time.sleep(0.5)
    except Exception as e:
        logger.error(f"Critical error in read_camera: {e}")

# Background processing thread (slower, decoupled)
def process_frames():
    global raw_frame, latest_faces, confused, attentive, percent
    try:
        while not stop_event.is_set():
            try:
                with lock:
                    frame = raw_frame.copy() if raw_frame is not None else None
                    
                if frame is None:
                    time.sleep(0.1)
                    continue

                # faces_data, c, a, p = analyze_class(frame)  # TODO: restore once ai_model dependency is resolved
                faces_data, c, a, p = [], 0, 0, 0  # Stub: ai_model unavailable

                with lock:
                    latest_faces = faces_data
                    confused = c
                    attentive = a
                    percent = p
                    
                time.sleep(0.3)  # Slower processing
            except Exception as e:
                logger.error(f"Error in process_frames loop: {e}")
                time.sleep(0.5)
    except Exception as e:
        logger.error(f"Critical error in process_frames: {e}")

# Streaming generator (fast, overlay cache)
def generate_frames():
    global raw_frame, latest_faces
    
    try:
        while not stop_event.is_set():
            try:
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
                    try:
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
                    except Exception as e:
                        logger.debug(f"Error drawing face: {e}")
                        continue

                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    continue
                    
                frame = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                
                time.sleep(0.03)  # Cap to ~30 FPS
            except Exception as e:
                logger.error(f"Error in generate_frames loop: {e}")
                time.sleep(0.1)
    except Exception as e:
        logger.error(f"Critical error in generate_frames: {e}")


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


@app.route("/health")
def health():
    """Health check endpoint for deployment monitoring."""
    return jsonify({"status": "ok", "message": "Confusion Detection API is running"}), 200


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors gracefully."""
    logger.warning(f"404 error: {error}")
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors gracefully."""
    logger.error(f"500 error: {error}")
    return jsonify({"error": "Internal server error"}), 500


def shutdown():
    """Graceful shutdown - clean up resources."""
    try:
        logger.info("Shutting down application...")
        stop_event.set()
        time.sleep(0.5)  # Give threads time to finish
        
        if camera is not None:
            try:
                if camera.isOpened():
                    camera.release()
                    logger.info("Camera released successfully.")
            except Exception as e:
                logger.warning(f"Error releasing camera: {e}")
        else:
            logger.info("No camera to release (running in headless/deployed mode).")
        
        logger.info("Application shutdown complete.")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


atexit.register(shutdown)

# Start background threads (skip on Vercel serverless)
if not VERCEL_ENV:
    logger.info("Starting background threads...")
    threading.Thread(target=read_camera, daemon=True).start()
    threading.Thread(target=process_frames, daemon=True).start()
    logger.info("Background threads started successfully.")
else:
    logger.warning("Skipping background threads on Vercel (serverless environment)")

if __name__ == "__main__":
    import os
    debug = os.environ.get("FLASK_DEBUG", "False") == "True"
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"Starting Flask app on port {port} (debug={debug})")
    app.run(host="0.0.0.0", port=port, debug=debug, use_reloader=False)