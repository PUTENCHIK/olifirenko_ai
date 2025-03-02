import cv2
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum, auto


class ObjectType(Enum):
    CUBE = auto()
    SPHERE = auto()


class Image:

    ID = 20
    NEED_SIZE = 480
    SPHERE_SYMBOL = 's'
    CUBE_SYMBOL = 'c'
    SPHERE_TEMPLATE = "sphere_{}_{}.png"
    CUBE_TEMPLATE = "cube_{}_{}.png"

    def __init__(self, number: int, original: cv2.typing.MatLike, type: str = 'c'):
        self.number = number
        self.original = original
        self.resized = self.get_resized()

        self.object_type = ObjectType.CUBE if type == 'c' else ObjectType.SPHERE
        self.to_delete = True
        self.center = (self.resized.shape[1]//2, self.resized.shape[0]//2)
    
    def is_vertical(self) -> bool:
        h, w, *other = self.original.shape
        return h > w

    def scale_size(self) -> tuple:
        h, w, *other = self.original.shape
        if w > h:
            return Image.NEED_SIZE, np.round(Image.NEED_SIZE * w/h).astype("uint32")
        else:
            return np.round(Image.NEED_SIZE * h/w).astype("uint32"), Image.NEED_SIZE
    
    def get_resized(self):
        copy = self.original.copy()
        size = self.scale_size()
        return cv2.resize(copy, size[::-1])
    
    def with_extra_info(self):
        copy = self.resized.copy()
        font_scale = 0.7
        font_thickness = 2
        text_color = (0, 0, 255)
        bound_color = (220, 0, 0)

        cv2.putText(copy, f"{counter}/{len(images)-1}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)
        cv2.putText(copy, f"{self.resized.shape}",
                    (10, copy.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)
        cv2.putText(copy, Image.CUBE_SYMBOL if self.object_type == ObjectType.CUBE else Image.SPHERE_SYMBOL,
                    (copy.shape[1]-50, 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale+0.3, text_color, font_thickness+1)
        cv2.putText(copy, "delete" if self.to_delete else "leave",
                    (copy.shape[1]-70, copy.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255) if self.to_delete else (0, 255, 0), font_thickness)
        cv2.putText(copy, f"s: {sphere_amount}, c: {cube_amount}",
                    (copy.shape[1]-150, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)
        
        cv2.circle(copy, self.center,
                   1, bound_color, 3)
        cv2.rectangle(copy, (self.center[0]-Image.NEED_SIZE//2, self.center[1]-Image.NEED_SIZE//2),
                            (self.center[0]+Image.NEED_SIZE//2, self.center[1]+Image.NEED_SIZE//2),
                            bound_color, 3)

        return copy


def on_mouse_callback(event, x, y, *params):
    if event == cv2.EVENT_LBUTTONDOWN:
        image = images[counter]
        cx, cy = image.center
        if image.is_vertical():
            if y < cy:
                image.center = (cx, y if y >= Image.NEED_SIZE//2 else Image.NEED_SIZE//2)
            else:
                image.center = (cx, y if y <= image.resized.shape[0]-Image.NEED_SIZE//2 else image.resized.shape[0]-Image.NEED_SIZE//2)
        else:
            if x < cx:
                image.center = (x if x >= Image.NEED_SIZE//2 else Image.NEED_SIZE//2, cy)
            else:
                image.center = (x if x <= image.resized.shape[1]-Image.NEED_SIZE//2 else image.resized.shape[1]-Image.NEED_SIZE//2, cy)


def count_objects() -> tuple:
    s_amount, c_amount = 0, 0
    for image in images:
        if not image.to_delete:
            if image.object_type == ObjectType.SPHERE:
                s_amount += 1
            if image.object_type == ObjectType.CUBE:
                c_amount += 1
    return s_amount, c_amount


root = pathlib.Path(__file__).parent
dataset = root / "dataset"

images_path = [path for path in dataset.glob("*.jpg")]
print(f"Pathes amount: {len(images_path)}")
images = [Image(i, cv2.imread(str(path)), 'c' if i < 75 else 's') for i, path in enumerate(images_path)]

window_name = "Main"
window = cv2.namedWindow(window_name, cv2.WINDOW_KEEPRATIO)
cv2.setMouseCallback(window_name, on_mouse_callback)

counter = 0
sphere_amount, cube_amount = count_objects()
to_process = False

while True:
    image = images[counter]
    cv2.imshow(window_name, image.with_extra_info())
    # plt.imshow(image.with_extra_info())
    # plt.show()
    
    key = cv2.waitKey(1)

    if key == ord('s') and image.object_type != ObjectType.SPHERE:
        image.object_type = ObjectType.SPHERE
        sphere_amount, cube_amount = count_objects()
    elif key == ord('c') and image.object_type != ObjectType.CUBE:
        image.object_type = ObjectType.CUBE
        sphere_amount, cube_amount = count_objects()
    if key == ord('d'):
        image.to_delete = not image.to_delete
        sphere_amount, cube_amount = count_objects()

    if key == ord('n'):
        counter = counter-1 if counter > 0 else len(images)-1
    if key == ord('m'):
        counter = counter+1 if counter < len(images)-1 else 0
    if key in [ord('q'), ord('p')]:
        if key == ord('p'):
            to_process = True
        break

cv2.destroyAllWindows()

if to_process:
    processed = root / "olifirenko_mv"
    processed.mkdir(exist_ok=True)
    train, test = processed / "train", processed / "test"
    train.mkdir(exist_ok=True)
    test.mkdir(exist_ok=True)

    limit = 10
    train_spheres, train_cubes = 0, 0
    test_spheres, test_cubes = 0, 0

    for i, image in enumerate(images):
        if image.to_delete:
            print(f"{i} deleted")
            continue
        
        is_sphere = image.object_type == ObjectType.SPHERE
        to_train = bool(np.random.randint(0, 2))
        result = image.resized[image.center[1]-Image.NEED_SIZE//2:image.center[1]+Image.NEED_SIZE//2,
                               image.center[0]-Image.NEED_SIZE//2:image.center[0]+Image.NEED_SIZE//2]
        if is_sphere:
            if to_train and train_spheres < limit:
                print(f"{i} {result.shape} sphere ({train_spheres}, {test_spheres}) -> train")
                train_spheres += 1
                cv2.imwrite(str(train / Image.SPHERE_TEMPLATE.format(Image.ID, train_spheres)),
                            result)
                continue
            if test_spheres < limit:
                print(f"{i} {result.shape} sphere ({train_spheres}, {test_spheres}) -> test")
                test_spheres += 1
                cv2.imwrite(str(test / Image.SPHERE_TEMPLATE.format(Image.ID, test_spheres)),
                            result)
            else:
                print(f"{i} sphere ({train_spheres}, {test_spheres}) -> skip")
        else:
            if to_train and train_cubes < limit:
                print(f"{i} {result.shape} cube ({train_cubes}, {test_cubes}) -> train")
                train_cubes += 1
                cv2.imwrite(str(train / Image.CUBE_TEMPLATE.format(Image.ID, train_cubes)),
                            result)
                continue
            if test_cubes < limit:
                print(f"{i} {result.shape} cube ({train_cubes}, {test_cubes}) -> test")
                test_cubes += 1
                cv2.imwrite(str(test / Image.CUBE_TEMPLATE.format(Image.ID, test_cubes)),
                            result)
            else:
                print(f"{i} cube ({train_cubes}, {test_cubes}) -> skip")

    print(train_spheres, test_spheres)
    print(train_cubes, test_cubes)
