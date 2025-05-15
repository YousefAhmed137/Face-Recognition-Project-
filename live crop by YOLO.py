from ultralytics import YOLO
import cv2
import os
import time

model = YOLO(r"H:\AI_Projects/yolov8_best.pt")  # Update the correct path to your model

cap = cv2.VideoCapture(0)
crop_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, verbose=False)
    boxes = results[0].boxes.xyxy
    
    #print(boxes)
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        #print(x1,y1,x2)
        crop = frame[y1:y2, x1:x2]
        crop_filename = os.path.join(r"H:\AI_Projects\DATA", f"crop6_{crop_count}.jpg")  # Use raw string here
        cv2.imwrite(crop_filename, crop)
        crop_count += 1
        time.sleep(0.1)
        
    annot = results[0].plot()
    cv2.imshow("YOLOv8 Live Detection", annot)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

cap.release()
cv2.destroyAllWindows()
