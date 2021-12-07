import argparse
import os
import tensorflow as tf

from model.TestModelFunctional import NeuralTransferModel
from ImgUtil import load_img, imshow, tensor_to_image

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys

from model.StyleContentExtractor import StyleContentExtractor

"""
Exporter class, exports the defined model wrapper into a SavedModel, which is able to be served and run independently.
"""


class ModelExporter(object):
    def __init__(self, export_path, version):
        self.export_dir = export_path
        self.version = version

    def export(self):
        export_path = os.path.join(self.export_dir, str(self.version))
        print('export_path = {}\n'.format(export_path))

        content_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
        style_path = tf.keras.utils.get_file('test2.jpg', 'https://uploads2.wikiart.org/images/edvard-munch/the-scream-1893(2).jpg!Large.jpg')

        content_image = load_img(content_path)
        style_image = load_img(style_path)

        model = NeuralTransferModel()
        model.build(input_shape=[(None, None, None, 3), (None, None, None, 3)])

        print(model.summary(line_length=100))
        #
        # tf.keras.models.save_model(
        #     model,
        #     export_path,
        #     overwrite=True,
        #     include_optimizer=True,
        #     save_format=None,
        #     signatures=None,
        #     options=None
        # )

        print('\nSaved model to: {}'.format(export_path))


""" Define console argument parser """
parser = argparse.ArgumentParser(description='Neural style transfer model exporter.')
parser.add_argument(
    'export_dir',
    metavar='path',
    type=str,
    help='Path to which the model will be exported. Value example: "C:/Users/admin"'
)
parser.add_argument('--version', dest='version', type=int, default=1, help='Version of the model.')

""" Run exporter either trough IDE or by console cmd python ModelExporter.py -h """
if __name__ == '__main__':
    args = parser.parse_args()

    if not os.path.isdir(args.export_dir):
        print(os.path.dirname(os.path.realpath(__file__)))
        print('The path {} does not exist'.format(args.export_dir))
        sys.exit()

    ModelExporter(args.export_dir, args.version).export()
