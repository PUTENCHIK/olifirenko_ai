import cv2
import numpy as np
from pathlib import Path
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils


def selective_search(image):
    #opencv-contrib-python
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    return rects


def non_max_supression(boxes, probs, overlap = 0.3):
    pick = []
    
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = boxes[:, 2], boxes[:, 3]
    
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(probs)
    
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick += [i]
        
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
        ratio = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs,
                         np.concatenate(([last],
                                         np.where(ratio > overlap)[0])))
    return boxes[pick].astype("int")


img_path = Path(__file__).parent.parent / "data" / "corgi.jpg"
print(img_path)
original = cv2.imread(img_path)
image = original.copy()

height, width = image.shape[:2]

rects = selective_search(image)

proposals = []
bboxes = []
for x, y, w, h in rects:
    if w/width < 0.1 or h/height < 0.1:
        continue
    if w/h > 1.2 or h/w > 1.2:
        continue
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    roi = original[y:y+h, x:x+w]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(roi, (224, 224))
    roi = img_to_array(roi)
    roi = preprocess_input(roi)
    proposals += [roi]
    bboxes += [(x, y, x+w, y+h)]

cv2.imshow("Original", image)
cv2.waitKey()

proposals = np.array(proposals)
model = ResNet50(weights="imagenet")
predictions = model.predict(proposals)
print(predictions)
predictions = imagenet_utils.decode_predictions(predictions, top=1)
print(predictions)

image = original.copy()

pboxes = []
pprob = []
for i, pred in enumerate(predictions):
    _, label, prob = pred[0]
    if prob > 0.6 and label == "Pembroke":
        x1, y1, x2, y2 = bboxes[i]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{label} - {prob:.2f}", (x1+10, y1+10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (128, 128, 0), 2)
        pboxes += [bboxes[i]]
        pprob += [prob]
        
cv2.imshow("Original", image)
cv2.waitKey()

boxes = non_max_supression(np.array(pboxes),
                           np.array(pprob))
print(len(boxes))

image = original.copy()
for b in boxes:
    x1, y1, x2, y2 = b
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.imshow("Original", image)
cv2.waitKey()
