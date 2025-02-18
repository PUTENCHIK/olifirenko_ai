import cv2
import numpy as np

# from tensorflow.keras.models import load_model
# from pathlib import Path


# path = Path(__file__).parent
# model = load_model(str(path / 'model.keras'))
# model.summary()


def draw_callback(event, x, y, *args):
    global history, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    if event == cv2.EVENT_LBUTTONUP:
        drawing = False
        history += [canvas.copy()]
        print(f"History: {len(history)}")
    if event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(canvas, (x, y), brush_size, 255, -1)


window_name = "Paint"
cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL)
cv2.setMouseCallback(window_name, draw_callback)

canvas = np.zeros((280, 280), dtype="uint8")
history = [canvas.copy()]
brush_size = 3
drawing = False

while True:
    cv2.imshow(window_name, canvas)
    
    key = cv2.waitKey(1)
    if key in [ord('q'), 27]:
        break
    elif key == ord('u'):
        brush_size += 1
        print(f"Brush size: {brush_size}")
    elif key == ord('i'):
        brush_size -= 1
        print(f"Brush size: {brush_size}")
    elif key == ord('z'):
        if len(history) > 1:
            history.pop()
            canvas = history[-1].copy()
            print(f"History: {len(history)}")
    elif key == ord('c'):
        canvas[:] = 0

cv2.destroyAllWindows()
