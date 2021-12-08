from flask import Flask, request
from flask_cors import CORS

from model.NeuralTransferModel import NeuralTransferModel
from util.ImgUtil import load_img, tensor_to_b64

app = Flask(__name__)
app.logger.setLevel('INFO')

api = CORS(app)


@app.route('/stylize', methods=['POST'])
def post():
    content_img = request.get_json()['content_img'].split("base64,")[1]
    style_img = request.get_json()['style_img'].split("base64,")[1]
    epochs = request.get_json()['epochs']
    steps_per_epoch = request.get_json()['steps_per_epoch']

    content_data = load_img(content_img)
    style_data = load_img(style_img)

    model = NeuralTransferModel()

    result = model.call((style_data, content_data, epochs, steps_per_epoch))

    return {"result": "data:image/jpeg;base64," + tensor_to_b64(result)}, 200


if __name__ == '__main__':
    app.run(debug=True)
