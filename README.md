# 🎓 AI Face Recognition System

[![Watch the video](https://img.youtube.com/vi/mFkWZtsga0Q/0.jpg)](https://www.youtube.com/watch?v=mFkWZtsga0Q)

> 🎥 Click the image above to watch our face recognition system in action!

This repository contains the graduation project developed for the *Digital Egypt Pioneers Initiative (DEPI)*.  
The system is an **AI-based face recognition application** that leverages deep learning and classical machine learning techniques to recognize and classify faces in real time.

---

## 📌 Project Overview

The goal of this project is to build a reliable and efficient face recognition system using modern AI technologies. The system consists of:

- **Face Detection** using Haar Cascade or YOLOv5  
- **Face Embedding** using FaceNet  
- **Face Classification** using Support Vector Machine (SVM)  
- **Web Interface** built with Flask and styled HTML/CSS  
- **Real-time Recognition** via webcam or uploaded images  

---

## 🧠 Technologies Used

| Component        | Technology                         |
|------------------|------------------------------------|
| Face Detection   | OpenCV Haar Cascade, YOLOv5        |
| Face Embeddings  | FaceNet (Keras/TensorFlow)         |
| Classifier       | Scikit-learn SVM, LabelEncoder     |
| Web Application  | Flask                              |
| Frontend         | HTML, CSS                          |
| Language         | Python                             |

---

## 🚀 Features

- Real-time face detection and recognition
- Train the model on new users easily
- Supports both YOLO and Haar Cascade for detection
- User-friendly web interface
- High accuracy using deep learning face embeddings

---

## 🛠 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/YousefAhmed137/Face-Recognition-Project-.git
cd Face-Recognition-Project-
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the application

```bash
python app.py
```

### 4. Access the app

Open your browser and go to:  
[http://localhost:5000](http://localhost:5000)

---

## 🧑‍💻 Training on New Users

To add and recognize new users:

1. **Add images** of the user in a new folder:

```
data/<username>/
```

2. **Generate face embeddings**:

```bash
python utils/generate_embeddings.py
```

3. **Train the classifier**:

```bash
python utils/train_classifier.py
```

Now you’re ready to recognize new faces through the app.

---

## 📷 Sample Output

> **Prediction:** Yousef Ahmed  
> **Confidence:** 97.5%

---

## 📂 Project Structure

```
face-recognition-depi/
├── app.py                         # Main Flask app
├── data/                          # Raw user face images
├── models/                        # YOLO, Haar Cascade, SVM, FaceNet embeddings, LabelEncoder
│   ├── yolov5s-face.pt
│   ├── haarcascade_frontalface_default.xml
│   ├── svm_model.pkl
│   ├── face_embeddings.npy
│   └── label_encoder.pkl
├── static/                        # CSS/JS/Images/output
│   └── sample_output.png
├── templates/                     # HTML templates
├── utils/                         # Preprocessing, embedding, classifier scripts
│   ├── generate_embeddings.py
│   └── train_classifier.py
├── requirements.txt
└── README.md
```

---

## ⚠️ Notes

- **YOLOv5** must be installed separately or imported via [Ultralytics](https://github.com/ultralytics/yolov5) if you're using the `.pt` model.
- This is an academic project. For real-world use, ensure you implement **secure data handling**, **encryption**, and **privacy policies**.

---

## 📝 Licens
This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for more details.

---

## 🙌 Contributors

This project was developed by the DEPI team:

- [@YousefAhmed137](https://github.com/YousefAhmed137)  
- [@farah-mahmoud](https://github.com/farah-mahmoud)
- [@MennaWaleed-eng](https://github.com/MennaWaleed-eng)
- [@nada-mohamed878](https://github.com/nada-mohamed878)
- 

