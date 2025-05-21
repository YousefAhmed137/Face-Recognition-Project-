# Facial Recognition System

A real-time facial recognition system using ArcFace for face embedding and an SVM classifier for identity recognition. This project demonstrates a full machine learning pipeline from data preprocessing to live video face verification.

## 🚀 Features
- Real-time facial recognition using live webcam input
- Face embedding extraction with the ArcFace deep learning model
- Identity classification using Support Vector Machine (SVM)
- Pre-trained on a dataset of well-known public figures

## 🧠 Technologies Used
- Python
- PyTorch
- OpenCV
- NumPy
- Scikit-learn

## 📁 Project Structure
```
facial-recognition-system/
│
├── arc_face.py               # Main script for real-time recognition
├── requirements.txt          # Required Python packages
├── dataset/                  # Face images organized per identity / Replace it with your custom dataset
```

## 📦 Installation

1. **Clone the repository**
```bash
git clone https://github.com/farah-mahmoud/facial-recognition-system.git
cd facial-recognition-system
```

2. **Install the dependencies**
```bash
pip install -r requirements.txt
```

## ▶️ Running the System

To launch the facial recognition system with live camera input, run:

```bash
python arc_face.py
```

The system will open your webcam and begin detecting and recognizing faces in real time.

## 📚 How It Works

1. **Data Preprocessing:**  
   Face images from the dataset are detected, aligned, and resized.

2. **Embedding Generation:**  
   The ArcFace model encodes each face into a 512-dimensional feature vector.

3. **Classifier Training:**  
   These embeddings are used to train an SVM classifier for identity recognition.

4. **Real-Time Inference:**  
   The live video feed is processed frame by frame. Detected faces are recognized and labeled using the trained SVM.

## 📝 Future Improvements
- Expand dataset with more identities
- Add support for on-the-fly training
- Improve performance under low-light or occluded conditions
- Deploy on edge devices (e.g., Raspberry Pi with Coral or Jetson Nano)

## 👩‍💻 Author
**Farah Mahmoud Kamal**  
[LinkedIn](https://www.linkedin.com/in/farahmahmoud) • [GitHub](https://github.com/farah-mahmoud)
