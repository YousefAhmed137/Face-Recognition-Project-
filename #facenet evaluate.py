#facenet evaluate.py
import os
import cv2
import numpy as np
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
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

    for img_name in tqdm(os.listdir(person_folder), desc=f"Processing {person_name}"):
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

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)

# Train the SVM model on the training set
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Model Evaluation Report:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

# Save model and label encoder
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

with open(os.path.join(model_dir, "svm_model_160x160.pkl"), 'wb') as f:
    pickle.dump(model, f)

with open(os.path.join(model_dir, "label_encoder.pkl"), 'wb') as f:
    pickle.dump(encoder, f)

# Save the embeddings
np.savez_compressed(os.path.join(model_dir, "faces_embeddings_done_4classes.npz"), embeddings=X, labels=y_enc)

print("✅ Training complete and evaluation done!")
