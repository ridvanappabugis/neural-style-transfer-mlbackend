import tensorflow as tf

from model.layers.PreprocessInput import PreprocessInput

content_layers_const = ['block5_conv2']


class ContentExtractor(tf.keras.layers.Layer):
    def __init__(self):
        super(ContentExtractor, self).__init__()

        self.preprocess_input = PreprocessInput()
        self.vgg = self._vgg_layers(content_layers_const)
        self.trainable = False

    @staticmethod
    def _vgg_layers(layer_names):
        """ Creates a vgg model that returns a list of intermediate output values."""
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False

        outputs = [vgg.get_layer(name).output for name in layer_names]

        model = tf.keras.Model([vgg.input], outputs)
        return model

    def call(self, inputs):

        output = tf.expand_dims(self.vgg(self.preprocess_input.call(inputs)), axis=0)
        return {content_name: value
                for content_name, value
                in zip(content_layers_const, output)}
