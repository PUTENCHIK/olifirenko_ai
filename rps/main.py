import cv2
import time
from pathlib import Path
from ultralytics import YOLO

data_path = Path(__file__).parent.parent / "data"
model_path = data_path / "best.pt"

model = YOLO(model_path)

camera_window = "Camera"
yolo_window = "YOLO"
cv2.namedWindow(camera_window, cv2.WINDOW_NORMAL)
cv2.namedWindow(yolo_window, cv2.WINDOW_NORMAL)
cap = cv2.VideoCapture(0)

state = "idle"  # wait, result
prev_time = 0
cur_time = 0
player1_hand = ""
player2_hand = ""
timer = 0
DELAY_WAIT = 10
game_result = ""

while cap.isOpened():
    ret, frame = cap.read()
    
    cv2.putText(frame, f"{state} - {(DELAY_WAIT - timer):.1f}", (20, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow(camera_window, frame)
    
    if state == "wait":
        timer = time.time() - prev_time
    if timer >= DELAY_WAIT:
        timer = DELAY_WAIT
        if state == "wait":
            state = "result"
            if player1_hand == player2_hand:
                game_result = "tie"
            elif ((player1_hand == "scissors" and player2_hand == "rock") or
                (player1_hand == "rock" and player2_hand == "paper") or
                (player1_hand == "paper" and player2_hand == "scissors")):
                    game_result = "player 2 wins"
            else:
                game_result = "player 1 wins"
            print(f"{game_result}: {player1_hand} and {player2_hand}")
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('r'):
        timer = 0
        state = "idle"
        
    if state == "result":
        continue
    
    results = model(frame, verbose=False)
    result = results[0]
    if not result:
        continue
    
    if len(result.boxes.xyxy) == 2:
        labels = []
        for label, xyxy in zip(result.boxes.cls, result.boxes.xyxy):
            x1, y1, x2, y2 = xyxy.numpy().astype("int")
            label = result.names[label.item()]
            labels += [label.lower()]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
            cv2.putText(frame, f"{labels[-1]}", (x1+20, y1+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        player1_hand, player2_hand = labels
        if player1_hand == "rock" and player2_hand == "rock" and state == "idle":
            state = "wait"
            prev_time = time.time()

    cv2.imshow(yolo_window, frame)
    
cap.release()
cv2.destroyAllWindows()
