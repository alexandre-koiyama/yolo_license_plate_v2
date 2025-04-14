
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr

cap = cv2.VideoCapture('VIDEO_PLATE.mp4')

model = YOLO('best.pt')

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out_put = cv2.VideoWriter('output_dd.mp4', fourcc, fps, (width, height))
reader = easyocr.Reader(["en"])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            class_id = int(box.cls[0])
            class_name = model.model.names[class_id]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 92, 255), 3)
            
            #MASK
            plate_crop = frame[y1:y2, x1:x2]
            plate_crop = cv2.resize(plate_crop, (0, 0), fx=10, fy=10)
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(32, 32))
            gray = clahe.apply(gray)
            filtered = cv2.bilateralFilter(gray, d=12, sigmaColor=75, sigmaSpace=75)
            _, thresh = cv2.threshold(filtered, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            output = reader.readtext(morph)
            for out in output:
                text_bbox, text, text_score = out
                if text_score > 0.3:
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    out_put.write(frame)

cap.release()
out_put.release()
cv2.destroyAllWindows()

print('Process completed')
