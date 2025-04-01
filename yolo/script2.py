import cv2
import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator


def dist(p1, p2) -> float:
    x1, y1 = p1
    x2, y2 = p2
    return ((x2-x1)**2 + (y2-y1)**2)**0.5


def angle(a, b, c) -> int:
    aa = dist(b, c)
    bb = dist(a, c)
    cc = dist(a, b)
    
    cosB = (aa**2 + cc**2 - bb**2) / (2 * aa * cc)
    B = np.arccos(cosB)
    return np.rad2deg(B)


def process(image, keypoints):
    nose_seen = keypoints[0][0] > 0 and keypoints[0][1] > 0
    left_ear_seen = keypoints[3][0] > 0 and keypoints[3][1] > 0
    right_ear_seen = keypoints[3][0] > 0 and keypoints[3][1] > 0
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_hip = keypoints[11]
    right_hip = keypoints[12]
    left_knee = keypoints[13]
    right_knee = keypoints[14]
    left_ankle = keypoints[15]
    right_ankle = keypoints[16]
    
    try:
        if left_ear_seen and not right_ear_seen:
            angle_knee = angle(left_hip, left_knee, left_ankle)
            x, y = int(left_knee[0]) + 10, int(left_knee[1]) + 10
        else:
            angle_knee = angle(right_hip, right_knee, right_ankle)
            x, y = int(right_knee[0]) + 10, int(right_knee[1]) + 10
        cv2.putText(image, f"{int(angle_knee)}", (x, y),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (25, 25, 255), 2)
        return angle_knee
    except ZeroDivisionError:
        pass
    

data_path = Path(__file__).parent.parent / "data"
img_path = data_path / "pose.jpg"
model_path = data_path / "yolo11n-pose.pt"

model = YOLO(model_path)

windows_name = "YOLO"
cap = cv2.VideoCapture(0)

writer = cv2.VideoWriter(data_path / "out.mp4",
                         cv2.VideoWriter_fourcc(*"MP4V"),
                         10, (640, 480))

last_time = time.perf_counter()
is_down = False
counter = 0
last_down = time.perf_counter()
while cap.isOpened():
    ret, frame = cap.read()
    writer.write(frame)
    copy = frame.copy()
    cur_time = time.perf_counter()
    cv2.putText(frame, f"fps: {(1 / (cur_time - last_time)):.1f}",
                (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.5,
                (0, 255, 0), 1)
    last_time = cur_time
    
    # results = model(frame)
    results = None
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    
    if results is not None:
        result = results[0]
        keypoints = result.keypoints.xy.tolist()
        if not keypoints:
            continue
        
        keypoints = keypoints[0]
        if not keypoints:
            continue
        
        annotator = Annotator(copy)
        annotator.kpts(result.keypoints.data[0],
                    result.orig_shape, 5, True)
        annotated = annotator.result()
        angle_ = process(annotated, keypoints)
        if angle_ is not None:
            old_value = is_down
            is_down = angle_ < 100
            if old_value and not is_down:
                counter += 1
                last_down = time.perf_counter()
        
        if time.perf_counter() - last_down > 10:
            counter = 0
        
        cv2.putText(frame, f"counter = {counter}", (10, 40),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 1)
    
        cv2.imshow("Pose", annotated)
    cv2.imshow(windows_name, frame)

writer.release()
cap.release()
cv2.destroyAllWindows()

