import cv2
import numpy as np
import pickle
from keras_facenet import FaceNet
from sklearn.svm import SVC

# Load the trained model and label encoder
with open("H:\AI_Projects\Facenetwithfalsk\svm_model_160x160.pkl", 'rb') as f:
    model = pickle.load(f)


with open("H:\AI_Projects\Facenetwithfalsk\label_encoder.pkl", 'rb') as f:
    encoder = pickle.load(f)

# Initialize FaceNet for embeddings
embedder = FaceNet()

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier("D:/Downloads/haarcascade_frontalface_default.xml")

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        # Extract the face
        face = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (160, 160))
        
        # Extract face embedding using FaceNet
        face_embedding = embedder.embeddings([face_resized])[0]
        
        # Predict the class (person) using the trained model
        prediction = model.predict([face_embedding])
        predicted_label = encoder.inverse_transform(prediction)[0]
        
        # Draw rectangle around the face and display the predicted label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, predicted_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow("Face Recognition", frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
