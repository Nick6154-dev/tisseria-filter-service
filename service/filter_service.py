from model.response_model import ResponseModel
import tensorflow as tf
import numpy as np
import os


class CrochetDetector:

    def __init__(self):
        path_to_filer_model = os.getenv('PATH_TO_FILTER_MODEL', './filter_model')
        self.tensor_model = tf.keras.models.load_model(path_to_filer_model)

    def is_a_crochet(self, img):
        img = np.array(img).astype(float) / 255
        img = np.expand_dims(img, axis=0)
        prediction = self.tensor_model.predict(img)
        is_crochet = (np.argmax(prediction, axis=-1))[0] == 0
        response = ResponseModel(is_crochet=is_crochet, message="Should return pattern if it is a crochet")
        return response
