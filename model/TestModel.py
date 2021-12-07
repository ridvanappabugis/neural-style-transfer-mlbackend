import tensorflow as tf
from model.StyleContentExtractor import StyleContentExtractor


class NeuralTransferModel(tf.keras.Model):

    def __init__(self,
                 style_weight=1e-2,
                 content_weight=1e4,
                 total_variation_weight=30,
                 name="NeuralTransferModel",
                 **kwargs):
        super(NeuralTransferModel, self).__init__(name=name, **kwargs)
        self.extractor = StyleContentExtractor((None, None, None, 3))
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.total_variation_weight = total_variation_weight
        self.image = None

    def build(self, input_shape):
        super(NeuralTransferModel, self).build(input_shape)

    @tf.function(input_signature=([tf.TensorSpec((None, None, None, 3), tf.float32)]))
    def call(self, inputs):
        print(inputs)
        style_target = self.extractor(tf.constant(inputs[0]))['style']
        content_target = self.extractor(tf.constant(inputs[1]))['content']

        trained_image = inputs[1]

        with tf.GradientTape() as tape:
            outputs = self.extractor.call(trained_image)

            loss = self.style_content_loss(outputs, style_target, content_target)
            print(loss)
            loss += self.total_variation_weight * tf.image.total_variation(trained_image)
        self.add_loss(loss)

        grad = tape.gradient(loss, trained_image)

        self.optimizer.apply_gradients([(grad, trained_image)])

        return trained_image

    def style_content_loss(self, outputs, style_target, content_target):
        style_outputs = outputs['style']
        content_outputs = outputs['content']

        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_target[name]) ** 2)
                               for name in style_outputs.keys()])
        style_loss *= self.style_weight / 5

        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_target[name]) ** 2)
                                 for name in content_outputs.keys()])
        content_loss *= self.content_weight / 5
        loss = style_loss + content_loss
        print(loss.eval())
        return loss
