import base64

from flask import Flask, request
from flask_restful import Api

from model.NeuralTransferModel import NeuralTransferModel
from util.ImgUtil import load_img, tensor_to_b64

app = Flask(__name__)
app.logger.setLevel('INFO')

api = Api(app)


@app.route('/stylize', methods=['POST'])
def post():
    content_img = request.get_json()['content_img']
    style_img = request.get_json()['style_img']
    epochs = request.get_json()['epochs']
    steps_per_epoch = request.get_json()['steps_per_epoch']

    content_data = load_img(content_img)
    style_data = load_img(style_img)

    model = NeuralTransferModel()

    result = model.call((style_data, content_data, epochs, steps_per_epoch))

    return {"result": tensor_to_b64(result)}


if __name__ == '__main__':
    app.run(debug=True)