import cv2
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops

# import sys
# np.set_printoptions(threshold=sys.maxsize)


def read_texts(path: pathlib.Path) -> list:
    test_images = list()
    for test_image in path.glob("*.png"):
        test_images += [binary(plt.imread(str(test_image)))]
        # plt.imshow(img)
        # plt.show()

    return test_images


def image_to_size(image: np.array, size: list) -> np.array:
    h, w = size[0] - image.shape[0], size[1] - image.shape[1]
    return np.pad(image, ((h//2, int(h/2 + 0.5)), (w//2, int(w/2 + 0.5))))


def binary(image: np.array) -> np.array:
    image = image.mean(axis=2) if len(image.shape) == 3 else image
    image[image > 0] = 1
    return image


def read_dataset(path: pathlib.Path) -> tuple:
    data = list()                   # list of images
    labels = list()                 # list of numbers
    symbols2numbers = dict()        # dictionary of designations of symbols

    max_sizes = [0, 0]

    for directory in path.iterdir():
        if directory.is_dir():
            symbol = directory.stem
            label = len(symbols2numbers.items()) + 1
            symbols2numbers[symbol] = label

            for img_path in directory.glob("*.png"):
                image = binary(plt.imread(img_path))
                # print(f"{symbol}: {image.shape}")
                data += [image]
                labels += [label]

                for i in range(len(max_sizes)):
                    if image.shape[i] > max_sizes[i]:
                        max_sizes[i] = image.shape[i]
                # plt.imshow(image)
                # plt.show()
    print(f"Max sizes: {max_sizes}")

    for i, image in enumerate(data):
        image = image_to_size(image, max_sizes)
        data[i] = image.flatten()
        # print(f"{labels[i]}: {data[i].shape}")
        # plt.imshow(data[i])
        # plt.show()

    data = np.array(data)
    labels = np.array(labels)

    return data, labels, symbols2numbers, max_sizes


def extract_symbols(image: np.array, max_sizes: list) -> np.array:
    # print(image.shape)
    labeled = label(image)
    regions = regionprops(labeled)

    order = dict()
    for region in regions:
        order[int(region.centroid[1])] = region.label
    order = sorted(order.items())
    # print(order)
    for i, (x, lbl) in enumerate(order):
        if i > 0:
            diff = (x - order[i-1][0]) / image.shape[1]
            if diff < 0.02:
                labeled[labeled == lbl] = order[i-1][1]

    symbols = dict()
    regions = regionprops(labeled)
    for region in regions:
        symbols[int(region.centroid[1])] = binary(image_to_size(region.image, max_sizes))

    symbols = [image for x, image in sorted(symbols.items())]
    symbols = np.array(symbols)
    return symbols


root = pathlib.Path(__file__).parent
task_dir = root / "task"
train_dir = task_dir / "train"

texts = read_texts(task_dir)
dataset, labels, symbols2numbers, max_size_symbol = read_dataset(train_dir)

dataset = dataset.astype("f4")
labels = labels.reshape(-1, 1).astype("f4")

# for k, v in symbols2numbers.items():
#     print(f"{k}: {v}")

# print(symbols2numbers)
print(f"Texts amount: {len(texts)}")
print(f"Symbols amount: {len(symbols2numbers.items())}")
print(f"Dataset | labels shape: {dataset.shape} | {labels.shape}")

# knn = cv2.ml.KNearest_create()
# knn.train(dataset, cv2.ml.ROW_SAMPLE, labels)

for i, text_img in enumerate(texts):
    symbols_img = extract_symbols(text_img, max_size_symbol)
    print(f"{i}) {symbols_img.shape}")
    # for i, symbol in enumerate(symbols_img):
    #     plt.imshow(symbol)
    #     plt.title(str(i+1))
    #     plt.show()
