import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from insightface.app import FaceAnalysis
import joblib

# ============ Step 1: Load ArcFace Model =============
# Load the InsightFace model (ArcFace) with CPU execution provider
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)  # ctx_id=0 means CPU, not GPU

# ============ Step 2: Prepare Dataset =============
def load_data(dataset_path):
    X = []  # Face embeddings
    y = []  # Corresponding labels (person names)
    
    for label in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, label)
        if not os.path.isdir(person_path):
            continue

        for img_file in os.listdir(person_path):
            img_path = os.path.join(person_path, img_file)
            img = cv2.imread(img_path)
            faces = app.get(img)
            if faces:
                embedding = faces[0].embedding  # Get 512-dimensional face embedding
                X.append(embedding)
                y.append(label)
    
    return np.array(X), np.array(y)

# ============ Step 3: Train Classifier =============
def train_model(X, y):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)  # Convert names to numbers

    # Splitting data into training and testing (validation) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.3, random_state=42)
    
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X_train, y_train)  # Train the SVM classifier

    # Predicting on the test set
    y_pred = clf.predict(X_test)
    
    # Calculate accuracy on the test set
    acc = accuracy_score(y_test, y_pred)
    print("Validation Accuracy:", acc)

    return clf, le

# ============ Step 4: Real-Time Recognition =============
def recognize_real_time(clf, le):
    cap = cv2.VideoCapture(0)  # Open webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = app.get(frame)
        for face in faces:
            emb = face.embedding.reshape(1, -1)  # Reshape for prediction
            pred = clf.predict(emb)[0]
            prob = clf.predict_proba(emb)[0][pred]
            name = le.inverse_transform([pred])[0]  # Convert back to label

            x, y, w, h = face.bbox.astype(int)  # Bounding box
            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({prob:.2f})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("ArcFace Real-Time Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()

# ============ Step 5: Run Everything =============
if __name__ == "__main__":
    dataset_dir = '/home/farah/farah/yolo_pics/dataset'  # Folder structure: dataset/person1/*.jpg, dataset/person2/*.jpg
    
    # Check if trained models already exist
    if os.path.exists("arcface_clf.pkl") and os.path.exists("label_encoder.pkl"):
        print("Loading pre-trained model and label encoder...")
        clf = joblib.load("arcface_clf.pkl")
        le = joblib.load("label_encoder.pkl")
    else:
        print("Loading data...")
        X, y = load_data(dataset_dir)

        print("Training model...")
        clf, le = train_model(X, y)

        # Save trained classifier and label encoder
        joblib.dump(clf, "arcface_clf.pkl")
        joblib.dump(le, "label_encoder.pkl")
    
    print("Starting real-time recognition...")
    recognize_real_time(clf, le)
