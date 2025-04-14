import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR

def ocr_with_paddle(img):
    finaltext = ''
    ocr = PaddleOCR(lang='ch', use_angle_cls=True)
    img_path = 'exp.jpeg'
    result = ocr.ocr(img)

    for i in range(len(result[0])):
        text = result[0][i][1][0]
        finaltext += ' '+ text
    return finaltext


cap = cv2.VideoCapture('video_original.mp4')

model = YOLO('best.pt')

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out_put = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))



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

            plate_crop = frame[y1:y2, x1:x2]
            plate_crop = cv2.resize(plate_crop, (0, 0), fx=6, fy=6)

            try:
              text_out = ocr_with_paddle(plate_crop)
              cv2.putText(frame, text_out, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
              #print(text_out)
            except:
              pass
            
    out_put.write(frame)

cap.release()
out_put.release()
cv2.destroyAllWindows()

print('Process completed')
