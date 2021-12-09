import argparse
import os
import sys

"""
Exporter class, exports the defined model wrapper into a SavedModel, 
which is able to be served and run independently.
"""


class ModelExporter(object):
    def __init__(self, export_path, version):
        self.export_dir = export_path
        self.version = version

    def export(self):
        export_path = os.path.join(self.export_dir, str(self.version))
        print('export_path = {}\n'.format(export_path))

        # Initial and build Model here

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
