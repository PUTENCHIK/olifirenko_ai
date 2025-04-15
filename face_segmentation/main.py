import cv2
import time
import numpy as np
from pathlib import Path
from skimage import draw
from ultralytics import YOLO


data_path = Path(__file__).parent.parent / "data"
model_path = data_path / "facial_best.pt"
oranges_orig = cv2.imread(data_path / "oranges.png")

camera = cv2.VideoCapture(0)
camera_window = "Camera"
cv2.namedWindow(camera_window, cv2.WINDOW_NORMAL)
result_window = "Orange"
cv2.namedWindow(result_window, cv2.WINDOW_NORMAL)

model = YOLO(model_path)

while camera.isOpened():
    ret, image = camera.read()
    
    cv2.imshow(camera_window, image)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('f'):
        print("processing...")
        original = image.copy()
        oranges = oranges_orig.copy()
        hsv_oranges = cv2.cvtColor(oranges, cv2.COLOR_BGR2HSV)

        lower = (10, 220, 220)
        upper = (25, 255, 255)

        r = 4
        rr, cc = draw.disk((r, r), r)
        struct = np.zeros((2*r+1, 2*r+1), np.uint8)
        struct[rr, cc] = 1
        mask = cv2.inRange(hsv_oranges, lower, upper)
        mask = cv2.dilate(mask, struct)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key=cv2.contourArea)
        m = cv2.moments(sorted_contours[-1])
        cy = int(m["m10"] / m["m00"])
        cx = int(m["m01"] / m["m00"])

        bbox = cv2.boundingRect(sorted_contours[-1])
        print("bbox gotten")
        
        result = model(image)[0]
        print("model predicted...")

        masks = result.masks
        annotated = result.plot()

        if masks is None or len(masks) == 0:
            print("No faces")
            continue

        global_mask = masks[0].data.numpy()[0, :, :]
        for mask in masks[1:]:
            global_mask += mask.data.numpy()[0, :, :]

        global_mask = cv2.resize(global_mask, (image.shape[1],
                                            image.shape[0])).astype("uint8")

        # r = 3
        # rr, cc = draw.disk((r, r,), r)
        # struct = np.zeros((2*r+1, 2*r+1), np.uint8)
        # struct[rr, cc] = 1
        # global_mask = cv2.dilate(global_mask, struct)

        global_mask = global_mask.reshape(image.shape[0], image.shape[1], 1)
        parts = (original * global_mask).astype("uint8")

        pos = np.where(global_mask > 0)
        min_y, max_y = int(np.min(pos[0]) * 0.75), int(np.max(pos[0]) * 1.3)
        min_x, max_x = int(np.min(pos[1]) * 0.75), int(np.max(pos[1]) * 1.3)
        global_mask = global_mask[min_y:max_y, min_x:max_x]
        parts = parts[min_y:max_y, min_x:max_x]

        x, y, w, h = bbox
        resized_parts = cv2.resize(parts, (w, h))
        resized_mask = cv2.resize(global_mask, (w, h)) * 255

        roi = oranges[y:y+h, x:x+w]
        bg = cv2.bitwise_and(roi,
                            roi,
                            mask=cv2.bitwise_not(resized_mask))
        combined = cv2.add(bg, resized_parts)

        oranges[y:y+h, x:x+w] = combined

        cv2.imshow(result_window, oranges)

camera.release()
cv2.destroyAllWindows()
