import cv2
import time
import numpy as np
from enum import Enum, auto
from typing import Optional
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator


class Position(Enum):
    not_in_frame = auto()
    not_in_rack = auto()
    up = auto()
    down = auto()
    middle = auto()


def now():
    return time.perf_counter()


def is_zero(point) -> bool:
    return point[0] == 0 and point[1] == 0


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
    angle_ = np.rad2deg(B)
    return int(angle_) if angle_ else None


def angle_to_abscissa(a, b) -> Optional[int]:
    if is_zero(a) or is_zero(b):
        return None
    c = (a[0], b[1])
    return angle(a, b, c)


def put_text(image,
             text: str,
             row: int = 1,
             color: tuple = (60, 60, 60),
             scale: float = 1):
    y_pos = 20 + 25*row + 40*(scale-1)
    cv2.putText(image, text,
                (10, y_pos), cv2.FONT_HERSHEY_PLAIN, 1.5*scale,
                color, scale+1)


def process(image, keypoints):
    left_ear_seen = keypoints[3][0] > 0 and keypoints[3][1] > 0
    right_ear_seen = keypoints[3][0] > 0 and keypoints[3][1] > 0
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    right_elbow = keypoints[7]
    left_elbow = keypoints[8]
    right_fist = keypoints[9]
    left_fist = keypoints[10]
    left_hip = keypoints[11]
    right_hip = keypoints[12]
    left_knee = keypoints[13]
    right_knee = keypoints[14]
    left_ankle = keypoints[15]
    right_ankle = keypoints[16]

    try:
        is_right = right_ear_seen and not left_ear_seen
        shoulder = right_shoulder if is_right else left_shoulder
        elbow = right_elbow if is_right else left_elbow
        fist = right_fist if is_right else left_fist
        hip = right_hip if is_right else left_hip
        knee = right_knee if is_right else left_knee
        ankle = right_ankle if is_right else left_ankle

        torso_abs_angle = angle_to_abscissa(shoulder, hip)
        leg_angle = angle(hip, knee, ankle)
        leg_abs_angle = angle_to_abscissa(hip, ankle)
        arm_angle = angle(shoulder, elbow, fist)
        if None in [torso_abs_angle, leg_angle, leg_abs_angle, arm_angle]:
            text = "not in frame"
            pos = Position.not_in_frame
            color = (0, 0, 255)
        elif torso_abs_angle < 30 and leg_angle > 120 and leg_abs_angle < 60:
            text = "in the rack, "
            color = (0, 255, 0)
            if arm_angle < 120:
                text += "down"
                pos = Position.down
            elif arm_angle > 150:
                text += "up"
                pos = Position.up
            else:
                text += "middle"
                pos = Position.middle
        else:
            text = "not in the rack"
            pos = Position.not_in_rack
            color = (0, 0, 255)
        put_text(image, text, color=color)
        put_text(image, f"arm: {arm_angle}", 2, scale=2)

        return pos

    except ZeroDivisionError:
        pass


login = "maxim"
password = "qwerty123"
ip = "192.168.1.10"
port = 8080
stream = "h264_ulaw"
url = f"rtsp://{login}:{password}@{ip}:{port}/{stream}.sdp"

home = Path(__file__).parent
data_path = Path(__file__).parent.parent / "data"
model_path = data_path / "yolo11n-pose.pt"

model = YOLO(model_path)
main_window = "Camera"
skeleton_window = "Annotated"
cv2.namedWindow(main_window, cv2.WINDOW_GUI_NORMAL)
cv2.namedWindow(skeleton_window, cv2.WINDOW_GUI_NORMAL)
camera = cv2.VideoCapture(url)

mp4s = [video for video in home.glob("*.mp4")]
writer = cv2.VideoWriter(home / f"out{len(mp4s)+1}.mp4",
                         cv2.VideoWriter_fourcc(*"MP4V"),
                         30, (640, 480))

last_time = now()
predict_time = last_time
counter = 0
position = Position.not_in_frame
is_down = False
last_pushup = now()
last_bad_pos = None
while camera.isOpened():
    ret, frame = camera.read()
    original = frame.copy()

    cur_time = now()
    put_text(frame, f"fps: {(1 / (cur_time - last_time)):.1f}")

    if cur_time - predict_time > 0.2:
        results = model(original, verbose=False)
        predict_time = now()
        if results is not None:
            result = results[0]
            keypoints = result.keypoints.xy.tolist()
            if not keypoints:
                continue
            
            keypoints = keypoints[0]
            if not keypoints:
                continue
            
            annotator = Annotator(original)
            annotator.kpts(result.keypoints.data[0],
                        result.orig_shape, 5, True)
            annotated = annotator.result()
            pos = process(annotated, keypoints)
            if pos is not None:
                position = pos
                if position == Position.down:
                    last_bad_pos = None
                    if not is_down and now() - last_pushup > 0.25:
                        is_down = True
                elif position == Position.up and is_down:
                    last_bad_pos = None
                    is_down = False
                    counter += 1
                    last_pushup = now()
                else:
                    if last_bad_pos is None:
                        last_bad_pos = now()
                    if now() - last_bad_pos > 3:
                        counter = 0
            else:
                position = Position.not_in_frame
                if last_bad_pos is None:
                    last_bad_pos = now()
                if now() - last_bad_pos > 3:
                    counter = 0
            cv2.imshow(skeleton_window, annotated)

    if now() - last_pushup > 10:
        counter = 0

    put_text(frame, f"{position.name}", 2)
    put_text(frame, f"PUSHUPS:{counter}", 3, scale=2)
    writer.write(frame)
    cv2.imshow(main_window, frame)
    last_time = cur_time
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

camera.release()
writer.release()
cv2.destroyAllWindows()
