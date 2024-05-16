import tensorflow as tf


class ModelClient:
    def __init__(self, model):
        self.model = model

    def predict(self, req):
        x = ModelClient.convert_to_tensor(req)
        return self.model(x)

    @staticmethod
    def convert_to_tensor(req):
        return tf.constant([req['reviewText']], dtype=tf.string)
