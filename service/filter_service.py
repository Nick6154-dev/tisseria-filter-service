from model.response_model import ResponseModel
from transformers import pipeline
import tensorflow as tf
import numpy as np
import cv2
import os


class CrochetDetector:

    def __init__(self):
        self.pipe = pipeline("image-classification",
                             model="Nick6154-dev/vit-base-patch16-224-finetuned-crochets-clothes")
        path_to_filer_model = os.getenv('PATH_TO_FILTER_MODEL', './filter_model')
        self.tensor_model = tf.keras.models.load_model(path_to_filer_model)

    def is_a_crochet(self, img):
        img = np.array(img).astype(float) / 255
        img = cv2.resize(img, (224, 224))
        img = np.expand_dims(img, axis=0)
        prediction = self.tensor_model.predict(img)
        is_crochet = (np.argmax(prediction, axis=-1))[0] == 0
        response = ResponseModel(is_crochet=is_crochet, message="Should return pattern if it is a crochet")
        return response

    def is_a_crochet_transformers(self, img):
        prediction = self.pipe(img)[0]
        label = prediction['label']
        score = prediction['score']
        return label, score
