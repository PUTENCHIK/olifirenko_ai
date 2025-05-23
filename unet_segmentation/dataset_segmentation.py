import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pprint import pprint


def read_dataset(path: Path) -> list:
    images_dir = path / "images"
    labels_dir = path / "labels"

    image_pathes = {str(p.stem)[:-4]: p for p in images_dir.glob("*.*")}
    label_pathes = {str(p.stem)[:-4]: p for p in labels_dir.glob("*.*")}

    dataset = []
    for id, img_path in image_pathes.items():
        if id not in label_pathes:
            print(f"[WARNING] no label for {id}")
            continue
        
        lbl_path = label_pathes[id]
        with open(lbl_path) as file:
            lines = file.readlines()
            if not len(lines):
                print(f"{lbl_path} is empty")
                continue
            line = lines[0]
            parts = line.split(" ")
            class_ = int(parts[0])
            points = [(float(parts[i-1]), float(parts[i])) for i in range(2, len(parts[1:]), 2)]
        
        dataset += [{
            'image_path': img_path,
            'labels_path': lbl_path,
            'class': class_,
            'points': points,
        }]
    return dataset


root = Path(__file__).parent
dataset_path = root / "dataset"
train_path = dataset_path / "train"
valid_path = dataset_path / "valid"
test_path = dataset_path / "test"

main_window = "Main"
mask_window = "Mask"
cv2.namedWindow(main_window, cv2.WINDOW_FREERATIO)
cv2.namedWindow(mask_window, cv2.WINDOW_FREERATIO)

train_ds = read_dataset(train_path)
valid_ds = read_dataset(valid_path)
test_ds = read_dataset(test_path)

print("train_ds:", len(train_ds))
print("valid_ds:", len(valid_ds))
print("test_ds:", len(test_ds))

shape = (64, 64)
ds = valid_ds
index = 0
image = None
mask = None

while True:
    if image is None or mask is None:
        image = cv2.imread(ds[index]["image_path"])
        image = cv2.resize(image, shape)
        cv2.putText(image, f"{index+1}/{len(ds)}",
                    (5, 10), 1, 0.6, (0, 255, 0), 1)
        
        w, h, _ = image.shape
        mask = np.zeros((h, w), dtype=np.uint8)

        points = ds[index]["points"]
        points = np.array([(int(x*w), int(y*h)) for x, y in points], dtype=np.int32)
        points = points.reshape((-1, 1, 2))

        cv2.fillPoly(mask, [points], color=(254, 254, 254))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask, contours, -1, (128, 128, 128), 2)

        mask[mask == 128] = 255
        mask[mask == 0] = 127
        mask[mask == 254] = 0

        # plt.figure(figsize=(15, 15))
        # plt.subplot(1, 2, 1)
        # plt.imshow(image)
        # plt.subplot(1, 2, 2)
        # plt.imshow(mask)
        # plt.show()

    cv2.imshow(main_window, image)
    cv2.imshow(mask_window, mask)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('n'):
        index = index-1 if index > 0 else len(ds)-1
        image = None
        mask = None
    elif key == ord('m'):
        index = index+1 if index+1 < len(ds) else 0
        image = None
        mask = None

cv2.destroyAllWindows()
