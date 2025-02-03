import cv2
import pathlib
import matplotlib.pyplot as plt


def detect_object(img, classifier, scaleFactor = None, minNeighbors = None):
    result = img.copy()
    rects = classifier.detectMultiScale(result,
                                        scaleFactor=scaleFactor,
                                        minNeighbors=minNeighbors)
    for (x, y, w, h) in rects:
        print(x, y, w, h)
    #     cv2.rectangle(result, (x, y), (x+w, y+h), (255, 255, 255), 1)
    print()
    return rects


def add_glasses(frame, glasses, eyes_coords):
    result = frame.copy()
    x, y, w, h = eyes_coords
    
    glasses = cv2.resize(glasses, (w, h))
    roi = result[y:y+h, x:x+w]
    
    glasses_gray = cv2.cvtColor(glasses, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(glasses_gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    fg = cv2.bitwise_and(glasses, glasses, mask=mask)
    
    combined = cv2.add(bg, fg)
    result[y:y+h, x:x+w] = combined
    
    # result = mask
    
    return result


camera = cv2.VideoCapture(0)
window_name = "Main"
window = cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL)

path = pathlib.Path(__file__).parent
cascade1_path = "haarcascades/haarcascade_eye_tree_eyeglasses.xml"
cascade2_path = "haarcascades/haarcascade_righteye_2splits.xml"
cascade1 = cv2.CascadeClassifier(str(path / cascade1_path))
cascade2 = cv2.CascadeClassifier(str(path / cascade2_path))

glasses = cv2.imread(str(path / "dealwithit.png"))

while camera.isOpened():
    _, frame = camera.read()
    eyes = detect_object(frame, cascade1, 1.2, 5)
    
    if len(eyes) == 2:
        x1, y1, w1, h1 = eyes[0]
        x2, y2, w2, h2 = eyes[1]
        eyes_coords = [min(int(x1), int(x2)),
                       min(int(y1), int(y2)),
                       int(x2 + w2 - x1) if x2 > x1 else int(x1 + w1 - x2),
                       max(int(h1), int(h2))]
        
        frame = add_glasses(frame, glasses, eyes_coords)
    
    cv2.imshow(window_name, frame)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()