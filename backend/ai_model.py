import cv2
import logging
from deepface import DeepFace

logger = logging.getLogger(__name__)

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

MAX_FACES = 10

def analyze_class(frame):
    """Analyze frame for faces and emotions. Returns gracefully on any error."""
    try:
        if frame is None:
            return [], 0, 0, 0
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.1, 3)
        faces = faces[:MAX_FACES]

        confused = 0
        attentive = 0
        faces_data = []

        for (x, y, w, h) in faces:
            try:
                face = frame[y:y+h, x:x+w]

                result = DeepFace.analyze(
                    face,
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend='opencv'
                )

                emotion = result[0]['dominant_emotion']

                if emotion in ["sad", "fear", "angry", "disgust"]:
                    confused += 1
                    color = (0, 0, 255)
                else:
                    attentive += 1
                    color = (0, 255, 0)
                    
                faces_data.append({
                    "box": (x, y, w, h),
                    "color": color,
                    "emotion": emotion
                })

            except Exception as e:
                logger.debug(f"Error analyzing face: {e}")
                continue

        total = confused + attentive
        percent = int((confused / total) * 100) if total > 0 else 0
        
        return faces_data, confused, attentive, percent
        
    except Exception as e:
        logger.error(f"Critical error in analyze_class: {e}")
        return [], 0, 0, 0