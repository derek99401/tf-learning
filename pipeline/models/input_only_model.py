from .base_model import BaseModel
import tensorflow as tf

class InputOnlyModel(BaseModel):

    def __init__(self):
        pass

    def inference(self, batch):
        with tf.name_scope('inference'):
            return tf.constant([0, 0])

    def loss(self, logit, label):
        with tf.name_scope('loss'):
            return tf.constant(0.0)

    def predict(self, logit):
        with tf.name_scope('predict'):
            return tf.constant([0, 1])

