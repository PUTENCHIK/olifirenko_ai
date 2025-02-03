import cv2
import pathlib
import matplotlib.pyplot as plt


def detect_face(img, classifier, scaleFactor = None, minNeighbors = None):
    result = img.copy()
    rects = classifier.detectMultiScale(result,
                                        scaleFactor=scaleFactor,
                                        minNeighbors=minNeighbors)
    for (x, y, w, h) in rects:
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return result


path = pathlib.Path(__file__).parent

lbp_cascade = "lbpcascades/lbpcascade_frontalface.xml"
haar_cascade = "haarcascades/haarcascade_frontalface_default.xml"
cooper = cv2.imread(str(path / "cooper.jpg"))
solvay = cv2.imread(str(path / "solvay-conference.jpg"))

face = cv2.CascadeClassifier(str(path / haar_cascade))      # лучше для одного лица
lbp = cv2.CascadeClassifier(str(path / lbp_cascade))        # для нескольких лиц

print(cooper.shape, solvay.shape)

plt.figure()
plt.imshow(detect_face(cooper, face, 1.2, 5))

plt.figure()
plt.imshow(detect_face(solvay, face, 1.2, 5))

plt.show()
