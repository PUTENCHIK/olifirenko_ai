import cv2
import time
from pathlib import Path
from ultralytics import YOLO

data_path = Path(__file__).parent.parent / "data"
model_path = data_path / "best.pt"

model = YOLO(model_path)

window_name = "YOLO"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    
    results = model(frame)
    
    for result in results:
        if not len(result.boxes.xyxy):
            continue
        x1, y1, x2, y2 = result.boxes.xyxy[0].numpy().astype("int")
        index = result.boxes.cls[0].item()
        name = result.names[index]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
        cv2.putText(frame, f"{name}", (x1+20, y1+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow(window_name, frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
