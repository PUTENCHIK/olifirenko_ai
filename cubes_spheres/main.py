import cv2
import pathlib
import matplotlib.pyplot as plt


root = pathlib.Path(__file__).parent
dataset = root / "dataset"

images_path = [path for path in dataset.glob("*.jpg")]
print(f"Pathes amount: {len(images_path)}")
images = [cv2.imread(str(path)) for path in images_path]

window_name = "Main"
window = cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL)

ID = 20
counter = 0
cubes = []
spheres = []
to_delete = []
waiting_enter = False
array_to_add = None

while counter < len(images_path):
    image = images[counter].copy()
    text_to_show = f"{counter})"
    # cv2.
    cv2.imshow(window_name, image)
    
    key = cv2.waitKey(1)
    # if key in [ord('s'), ord('c'), ord('d')]:
    #     if key == ord('s'):
    #         array_to_add = 's'
    #     if key == ord('c'):
    #         array_to_add = 'c'
    #     if key == ord('d'):
    #         array_to_add = 'd'
    #     waiting_enter = True
    
    # if waiting_enter and key == ord('p'):
    #     pass
    if key == ord('n'):
        counter = counter-1 if counter > 0 else len(images)-1
    if key == ord('m'):
        counter = counter+1 if counter < len(images)-1 else 0
    if key == ord('q'):
        break

cv2.destroyAllWindows()
