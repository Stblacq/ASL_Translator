import cv2
import numpy as np
from tensorflow import keras

class_names = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
               12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W',
               23: 'X', 24: 'Y', 25: 'Z', 26: 'del', 27: 'NOTHING', 28: 'SPACE'}

default_model = keras.models.load_model('./felix_model')


def predict_asl(cap, model=default_model):
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            font = cv2.FONT_HERSHEY_SIMPLEX
            class_pred = get_class_name(frame, model)
            frame = cv2.putText(frame, class_pred, (10, 100), font, 3, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


def get_class_name(frame, model):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (300, 300))
    frame = np.expand_dims(frame, 0)
    classes_x = np.argmax(model.predict(frame), axis=1)
    class_pred = class_names[int(classes_x)]
    return class_pred
