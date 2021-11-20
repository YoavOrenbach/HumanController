import numpy as np
import tensorflow as tf
from cv2 import cv2
from movenetUtils import load_movenet_model, movenet_inference
import time


model_choice = "movenet"

if model_choice == "mobilenet":
    model = tf.keras.models.load_model("mobilenet_saved_model")
else:
    model = tf.keras.models.load_model("movenet_saved_model")
    model_name = "movenet_thunder"
    movenet, input_size = load_movenet_model(model_name)

class_names = ["a","d","enter","normal", "s", "w", "x"]
cv2.namedWindow("preview")
camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not camera.isOpened():
    print("The Camera is not Opened....Exiting")
    exit()
start = time.time()
num_frames = 0
while True:
    status, frame = camera.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if model_choice == "mobilenet":
        predict_frame = cv2.resize(image, (160, 160))
    else:
        predict_frame = movenet_inference(image, movenet, input_size)
    predict_frame = np.expand_dims(predict_frame, axis=0)
    prediction = model.predict(predict_frame)
    predicted_class = np.argmax(prediction)
    if np.max(prediction) > 0.5:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, class_names[predicted_class],
                    (200, 250), font,
                    7, (0, 255, 255),
                    4, cv2.LINE_AA)
    #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    num_frames = num_frames + 1
    cv2.imshow("preview", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
end = time.time()
camera.release()
cv2.destroyAllWindows()
seconds = end - start
fps = num_frames / seconds
print("Estimated frames per second : {0}".format(fps))


