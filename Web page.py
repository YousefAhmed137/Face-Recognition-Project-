from flask import Flask, render_template, render_template_string, Response
import logging
import cv2
import numpy as np
import pickle
import os
from keras_facenet import FaceNet
from sklearn.svm import SVC

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "svm_model_160x160.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoder.pkl")
CASCADE_PATH = os.path.join(BASE_DIR, "models", "haarcascade_frontalface_default.xml")

# Load the trained model and encoder
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(ENCODER_PATH, 'rb') as f:
        encoder = pickle.load(f)
    logger.info("Model and encoder loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model files: {str(e)}")
    model = None
    encoder = None

# Initialize FaceNet
embedder = FaceNet()

# Load Haar Cascade
try:
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
except Exception as e:
    logger.error(f"Error loading Haar Cascade: {str(e)}")
    face_cascade = None

# Open the webcam
camera = cv2.VideoCapture(0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/face_recognition')
def face_recognition():
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Face Recognition</title>
        <style>
            body {
                background: linear-gradient(to right, #141e30, #243b55);
                color: white;
                text-align: center;
                font-family: sans-serif;
                padding: 20px;
            }
            h1 {
                color: #00ffd5;
            }
            #videoFeed {
                border: 4px solid #00aaff;
                border-radius: 10px;
                box-shadow: 0 0 20px rgba(0, 170, 255, 0.5);
                width: 90%;
                max-width: 800px;
            }
            .btn {
                display: inline-block;
                padding: 10px 20px;
                margin-top: 20px;
                background-color: #00aaff;
                color: #fff;
                text-decoration: none;
                border-radius: 5px;
            }
            .btn:hover {
                background-color: #0088cc;
            }
        </style>
    </head>
    <body>
        <h1>Real-time Face Recognition</h1>
        <img id="videoFeed" src="{{ url_for('video_feed') }}" />
        <br>
        <a href="/" class="btn">Back to Home</a>
    </body>
    </html>
    ''')

# Generator function for live video streaming
def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5) if face_cascade else []

        for (x, y, w, h) in faces:
            if w > 50 and h > 50:
                face = frame[y:y+h, x:x+w]
                try:
                    face_resized = cv2.resize(face, (160, 160))
                    embedding = embedder.embeddings([face_resized])[0]
                    probabilities = model.predict_proba([embedding])[0]
                    max_conf = max(probabilities)

                    if max_conf > 0.5:
                        predicted_label = encoder.inverse_transform([np.argmax(probabilities)])[0]
                    else:
                        predicted_label = "Unknown"

                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(frame, f"{predicted_label} ({max_conf:.2f})", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                except Exception as e:
                    logger.warning(f"Face processing error: {e}")

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        logger.error(f"Error running the Flask app: {str(e)}")
    finally:
        camera.release()
        cv2.destroyAllWindows()
