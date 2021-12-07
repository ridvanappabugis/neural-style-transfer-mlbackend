import tensorflow as tf

"""
Wrapping class, acts as a Modularised model that can be used for exporting and serving.

As our actual model needs to be trained over the user input, we cannot directly build, and serve our model for usage.
This class defines a Module that builds a new model, over the input, every time it is called.
"""


class ModelExportWrapper(tf.keras.Model):
    def __init__(self, model):
        super(ModelExportWrapper, self).__init__()
        self.model = model
        self.outputs = None

    @tf.function(input_signature=[tf.TensorSpec((None, None, None, 3), tf.float32),
                                  tf.TensorSpec((None, None, None, 3), tf.float32)])
    def __call__(self, img_input):
        model = self.model.build(img_input[0], img_input[1])
        self.outputs = {"result": model.run()}
        return self.outputs
