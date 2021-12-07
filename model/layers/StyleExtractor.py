import tensorflow as tf

from model.layers.PreprocessInput import PreprocessInput

style_layers_const = ['block1_conv1',
                      'block2_conv1',
                      'block3_conv1',
                      'block4_conv1',
                      'block5_conv1']


class StyleExtractor(tf.keras.layers.Layer):
    def __init__(self):
        super(StyleExtractor, self).__init__()

        self.preprocess_input = PreprocessInput()
        self.vgg = self._vgg_layers(style_layers_const)
        self.trainable = False

    @staticmethod
    def _vgg_layers(layer_names):
        """ Creates a vgg model that returns a list of intermediate output values."""
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False

        outputs = [vgg.get_layer(name).output for name in layer_names]

        model = tf.keras.Model([vgg.input], outputs)
        return model

    @staticmethod
    def _gram_matrix(input_tensor):
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
        return result / num_locations

    def call(self, inputs, training=False):
        outputs = self.vgg(self.preprocess_input.call(inputs))

        style_outputs = [self._gram_matrix(style_output)
                         for style_output in outputs]

        return {style_name: value
                for style_name, value
                in zip(style_layers_const, style_outputs)}
