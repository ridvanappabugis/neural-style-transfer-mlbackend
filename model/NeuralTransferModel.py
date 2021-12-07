import tensorflow as tf

from model.layers.ContentExtractor import ContentExtractor
from model.layers.StyleContentLoss import StyleContentLoss
from model.layers.StyleExtractor import StyleExtractor


class NeuralTransferModel(tf.keras.Model):

    def __init__(self,
                 total_variation_weight=30,
                 name="NeuralTransferModel",
                 **kwargs):
        super(NeuralTransferModel, self).__init__(name=name, **kwargs)
        self.style_extract = StyleExtractor()
        self.content_extract = ContentExtractor()
        self.resStyleContentLoss = StyleContentLoss()

        self.opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

        self.total_variation_weight = total_variation_weight
        self.result = None

    @staticmethod
    def clip_0_1(image):
        return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

    def train_step(self, data, **kwargs):
        style_target, content_target = data

        with tf.GradientTape() as tape:
            loss = self.resStyleContentLoss.call(
                self.style_extract.call(self.result),
                self.content_extract.call(self.result),
                style_target,
                content_target
            )
            loss += self.total_variation_weight * tf.image.total_variation(self.result)
        grad = tape.gradient(loss, self.result)
        self.opt.apply_gradients([(grad, self.result)])
        self.result.assign(self.clip_0_1(self.result))

    def call(self, data, **kwargs):
        style, content, epochs, steps_per_epoch = data
        self.result = tf.Variable(content)

        style_target = self.style_extract.call(style)
        content_target = self.content_extract.call(content)

        step = 0
        for n in range(epochs):
            for m in range(steps_per_epoch):
                step += 1
                self.train_step((style_target, content_target))
                print(".", end='', flush=True)

        return self.result
