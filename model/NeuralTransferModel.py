import time

import tensorflow as tf
from model.StyleContentExtractor import StyleContentExtractor


class NeuralTransferModel:

    def __init__(self):
        super().__init__()
        self.extractor = StyleContentExtractor()
        self.content_image = None
        self.style_image = None
        self.style_target = None
        self.content_target = None
        self.generated_image = None
        self.style_weight = 1e-2
        self.content_weight = 1e4
        self.total_variation_weight = 30
        self.opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    def build(self, content_image, style_image):
        self.content_image = content_image
        self.style_image = style_image
        self.style_target = self.extractor.call(content_image)['style']
        self.content_target = self.extractor.call(style_image)['content']
        return self

    @tf.function()
    def train_step(self, image):
        with tf.GradientTape() as tape:
            outputs = self.extractor(image)
            loss = self.style_content_loss(outputs)
            loss += self.total_variation_weight * tf.image.total_variation(image)
        grad = tape.gradient(loss, image)
        self.opt.apply_gradients([(grad, image)])
        return image

    def style_content_loss(self, outputs):
        style_outputs = outputs['style']
        content_outputs = outputs['content']
        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - self.style_target[name]) ** 2)
                               for name in style_outputs.keys()])
        style_loss *= self.style_weight / 5

        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - self.content_target[name]) ** 2)
                                 for name in content_outputs.keys()])
        content_loss *= self.content_weight / 5
        loss = style_loss + content_loss
        return loss

    def run(self):
        start = time.time()
        epochs = 10
        steps_per_epoch = 100

        trained_image = tf.Variable(self.content_image)

        step = 0
        for n in range(epochs):
            for m in range(steps_per_epoch):
                step += 1
                self.train_step(trained_image)
                print(".", end='', flush=True)
            print("Train step: {}".format(step))

        end = time.time()
        print("Total time: {:.1f}".format(end - start))
        return trained_image
