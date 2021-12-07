import tensorflow as tf


class StyleContentLoss(tf.keras.layers.Layer):
    def __init__(self,
                 style_weight=1e-2,
                 content_weight=1e4):
        super(StyleContentLoss, self).__init__()
        self.style_weight = style_weight
        self.content_weight = content_weight

    def call(self, res_style, res_content, style_target, content_target):
        style_loss = tf.add_n([tf.reduce_mean((res_style[name] - style_target[name]) ** 2)
                               for name in res_style.keys()])
        style_loss *= self.style_weight / 5

        content_loss = tf.add_n([tf.reduce_mean((res_content[name] - content_target[name]) ** 2)
                                 for name in res_content.keys()])
        content_loss *= self.content_weight / 5
        loss = style_loss + content_loss
        return loss
