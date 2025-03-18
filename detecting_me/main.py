import cv2
import numpy as np
from pathlib import Path

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (Input, Dense, Dropout,
                                     GlobalAveragePooling2D)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.mobilenet import MobileNet


def make_model(classes: int = 2,
               width: int = 224,
               height: int = 224):
    base_model = MobileNet(include_top=False,
                           input_shape=(width, height, 3))
    for layer in base_model.layers[:]:
        layer.trainable = False
    
    input = Input(shape=(width, height, 3))
    custom_model = base_model(input)
    custom_model = GlobalAveragePooling2D()(custom_model)
    custom_model = Dense(64, activation="relu")(custom_model)
    custom_model = Dropout(0.5)(custom_model)
    predictions = Dense(classes, activation="softmax")(custom_model)
    
    model = Model(inputs=input, outputs=predictions)
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    
    return model
    

root = Path(__file__).parent
model_path = root / "model.keras"
if not model_path.exists():
    print("Making model")
    model = make_model()
    print(model.summary())
else:
    print("Loading model")
    model = load_model(model_path)

window_name = "Main"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
camera = cv2.VideoCapture(0)

while camera.isOpened():
    _, image = camera.read()
    x = cv2.resize(image, (224, 224))
    x = x.reshape((1, ) + x.shape)
    
    
    key = cv2.waitKey(1)
    x_cls = -1
    if key == ord('p'):
        x_cls = 0
    if key == ord('n'):
        x_cls = 1
    
    if x_cls != -1:
        y = to_categorical(np.array([x_cls]), 2)
        model.fit(x, y)
    if key in [ord('q'), 27]:
        break
    
    prediction = model.predict(x)
    isMe = np.argmax(prediction[0]) == 0
    
    cv2.putText(image,
                f"{'Me' if isMe else 'Not me'}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0) if isMe else (0, 0, 255),
                2,
                cv2.LINE_AA)
    cv2.imshow(window_name, image)

model.save(model_path)
cv2.destroyAllWindows()
camera.release()
