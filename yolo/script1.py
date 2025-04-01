import cv2
import time
from pathlib import Path
from ultralytics import YOLO

data_path = Path(__file__).parent.parent / "data"
img_path = data_path / "corgi.jpg"
model_path = data_path / "yolo11n.pt"
image = cv2.imread(img_path)

model = YOLO(model_path)
t = time.perf_counter()
results = model(img_path)
print(f"Time: {time.perf_counter() - t}")

for result in results:
    x1, y1, x2, y2 = result.boxes.xyxy[0].numpy().astype("int")
    index = result.boxes.cls[0].item()
    name = result.names[index]
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 4)
    cv2.putText(image, f"{name}", (x1+20, y1+20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

window_name = "YOLO"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

cv2.imshow(window_name, image)
cv2.waitKey()
