import cv2
import time
import numpy as np
from pathlib import Path
from skimage import draw
from ultralytics import YOLO


data_path = Path(__file__).parent.parent / "data"
model_path = data_path / "facial_best.pt"
image = cv2.imread(data_path / "gennadiy.jpg")
original = image.copy()

image_window = "Image"
cv2.namedWindow(image_window, cv2.WINDOW_NORMAL)
mask_window = "Mask"
cv2.namedWindow(image_window, cv2.WINDOW_NORMAL)

model = YOLO(model_path)
result = model(image)[0]

masks = result.masks
annotated = result.plot()

global_mask = masks[0].data.numpy()[0, :, :]
for mask in masks[1:]:
    global_mask += mask.data.numpy()[0, :, :]

global_mask = cv2.resize(global_mask, (image.shape[1],
                                       image.shape[0])).astype("uint8")

r = 5
rr, cc = draw.disk((r, r,), r)
struct = np.zeros((2*r+1, 2*r+1), np.uint8)
struct[rr, cc] = 1

global_mask = cv2.dilate(global_mask,
                         struct,
                         iterations=2)
global_mask = global_mask.reshape(image.shape[0], image.shape[1], 1)

face = (original * global_mask).astype("uint8")

cv2.imshow("Image", annotated)
cv2.imshow("Mask", face)
cv2.waitKey()
