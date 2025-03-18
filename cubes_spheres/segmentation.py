import cv2
import numpy as np
from pathlib import Path


def get_color(cm: int) -> tuple:
    if cm == 1:
        return (255, 0, 0)
    else:
        return (0, 255, 255)


def callback(event, x, y, flags, param):
    global updated
    if event == cv2.EVENT_LBUTTONDOWN:
        updated = True
        cv2.circle(markers, (x, y), 5, (current_marker), -1)
        cv2.circle(image_copy, (x, y), 5, get_color(current_marker), -1)


root = Path(__file__).parent
image = cv2.imread(str(root / "images" / "cube_20_1.png"))

main_name = "Main"
segments_name = "Segments"
cv2.namedWindow(main_name)
cv2.namedWindow(segments_name)
cv2.setMouseCallback(main_name, callback)

markers = np.zeros(image.shape[:-1], dtype="int32")
segments = np.zeros_like(image)
image_copy = image.copy()
updated = False
current_marker = 1

while True:
    cv2.imshow(main_name, image_copy)
    cv2.imshow(segments_name, segments)
    
    key = cv2.waitKey(1)
    if key in [ord('q'), 27]:
        break
    elif key == ord('c'):
        image_copy = image.copy()
        markers = np.zeros(image.shape[:-1], dtype="int32")
        segments = np.zeros_like(image)
    elif key > 0 and chr(key).isdigit():
        current_marker = 1 if int(chr(key)) == 1 else 2
    
    if updated:
        markers_copy = markers.copy()
        cv2.watershed(image, markers_copy)
        segments = np.zeros(image.shape, dtype=np.uint8)
        
        segments[markers_copy == 1] = get_color(1)
        segments[markers_copy == 2] = get_color(2)
        updated = False
    
cv2.destroyAllWindows()
