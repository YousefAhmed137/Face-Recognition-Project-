import os
import time
import cv2
from ultralytics import YOLO

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "models", "yolov8_best.pt")
    CROP_DIR = os.path.join(BASE_DIR, "data")

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please add your YOLOv8 model weights.")

    os.makedirs(CROP_DIR, exist_ok=True)

    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(0)
    crop_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        results = model.predict(source=frame, verbose=False)
        boxes = results[0].boxes.xyxy

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            crop = frame[y1:y2, x1:x2]
            crop_filename = os.path.join(CROP_DIR, f"crop_{crop_count}.jpg")
            cv2.imwrite(crop_filename, crop)
            crop_count += 1
            time.sleep(0.1)

        annotated_frame = results[0].plot()
        cv2.imshow("YOLOv8 Live Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
