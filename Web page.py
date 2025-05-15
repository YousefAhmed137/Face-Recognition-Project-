from flask import Flask, render_template, render_template_string, Response
import logging
import cv2
import numpy as np
import pickle
from keras_facenet import FaceNet
from sklearn.svm import SVC

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load the trained model and encoder
try:
    with open(r"H:\AI_Projects\Facenetwithfalsk\svm_model_160x160.pkl", 'rb') as f:
        model = pickle.load(f)
    with open(r"H:\AI_Projects\Facenetwithfalsk\label_encoder.pkl", 'rb') as f:
        encoder = pickle.load(f)
    logger.info("Model and encoder loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model files: {str(e)}")
    model = None
    encoder = None

# Initialize FaceNet
embedder = FaceNet()

# Load Haar Cascade for face detection
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception as e:
    logger.error(f"Error loading Haar Cascade: {str(e)}")
    face_cascade = None

# Global camera
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
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Face Recognition</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(to right, #141e30, #243b55);
                color: #f0f0f0;
                margin: 0;
                padding: 20px;
                text-align: center;
            }
            h1 {
                color: #00ffd5;
                margin-bottom: 30px;
            }
            .container {
                max-width: 800px;
                margin: 0 auto;
            }
            #videoFeed {
                max-width: 100%;
                border: 3px solid #00aaff;
                border-radius: 8px;
                box-shadow: 0 0 20px rgba(0, 170, 255, 0.4);
            }
            .btn {
                display: inline-block;
                padding: 12px 30px;
                margin-top: 20px;
                font-size: 1.1em;
                background-color: #00aaff;
                color: #fff;
                border: none;
                border-radius: 8px;
                text-decoration: none;
                transition: all 0.3s ease;
            }
            .btn:hover {
                background-color: #0088cc;
                transform: translateY(-2px);
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Real-time Face Recognition</h1>
            <div>
                <img id="videoFeed" src="{{ url_for('video_feed') }}" alt="Video Feed">
            </div>
            <a href="/" class="btn">Back to Home</a>
        </div>
    </body>
    </html>
    ''')

# Function to capture frames and perform face recognition
def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Preprocess image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)  # Reduce noise
        gray = cv2.equalizeHist(gray)  # Enhance contrast

        # Detect faces with stricter parameters
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5) if face_cascade else []

        for (x, y, w, h) in faces:
            if w > 50 and h > 50:  # Filter small detections
                face = frame[y:y+h, x:x+w]
                try:
                    face_resized = cv2.resize(face, (160, 160))
                    embedding = embedder.embeddings([face_resized])[0]
                    probabilities = model.predict_proba([embedding])[0]
                    max_confidence = max(probabilities)
                    predicted_label = encoder.inverse_transform([np.argmax(probabilities)])[0] if max_confidence > 0.5 else "Unknown"
                    
                    # Draw rectangle and label
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(frame, f"{predicted_label} ({max_confidence:.2f})", 
                                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
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