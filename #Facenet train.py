#Facenet train 
import os
import cv2
import numpy as np
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle
from tqdm import tqdm

# Load FaceNet model
embedder = FaceNet()

# Paths
dataset_path = "H:/AI_Projects/DATA"
face_cascade = cv2.CascadeClassifier("D:/Downloads/haarcascade_frontalface_default.xml")

# Variables
X, y = [], []

# Loop over each person
for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_folder):
        continue

    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y_, w, h) in faces:
            face = img[y_:y_+h, x:x+w]
            face = cv2.resize(face, (160, 160))
            face_embedding = embedder.embeddings([face])[0]
            X.append(face_embedding)
            y.append(person_name)
            break  # only one face per image assumed

# Convert to numpy
X = np.asarray(X)
y = np.asarray(y)

# Encode labels
encoder = LabelEncoder()
y_enc = encoder.fit_transform(y)

# Train SVM
model = SVC(kernel='linear', probability=True)
model.fit(X, y_enc)

# Save model and labels
with open("svm_model_160x160.pkl", 'wb') as f:
    pickle.dump(model, f)

with open("label_encoder.pkl", 'wb') as f:
    pickle.dump(encoder, f)

# Optional: save embeddings
np.savez_compressed("faces_embeddings_done_4classes.npz", embeddings=X, labels=y_enc)

print("✅ Training complete!")
