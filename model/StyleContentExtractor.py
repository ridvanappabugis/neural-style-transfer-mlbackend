import tensorflow as tf

style_layers_const = ['block1_conv1',
                      'block2_conv1',
                      'block3_conv1',
                      'block4_conv1',
                      'block5_conv1']

content_layers_const = ['block5_conv2']


class StyleContentExtractor(tf.keras.models.Model):
    def __init__(self, input_shape):
        super(StyleContentExtractor, self).__init__(input_shape)
        self.vgg = self._vgg_layers(style_layers_const + content_layers_const)
        self.style_layers = style_layers_const
        self.content_layers = content_layers_const
        self.num_style_layers = len(style_layers_const)
        self.trainable = False
        self.build(input_shape)

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

    @tf.function(input_signature=[tf.TensorSpec((None, None, None, 3), tf.float32)])
    def call(self, inputs):
        """Expects float input in [0,1]"""
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        style_outputs = [self._gram_matrix(style_output)
                         for style_output in style_outputs]

        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}
