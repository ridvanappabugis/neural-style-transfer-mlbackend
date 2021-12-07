import tensorflow as tf


class PreprocessInput(tf.keras.layers.Layer):

    def __init__(self):
        super(PreprocessInput, self).__init__()
        self.trainable = False

    def call(self, inputs, training=False):
        """Expects float input in [0,1]"""
        inputs = inputs * 255.0
        return tf.keras.applications.vgg19.preprocess_input(inputs)
