from flask import Flask, request
from flask_cors import CORS
from flasgger import Swagger
import os
import tensorflow as tf
from src.model_client import ModelClient
from src.swagger import flask_swagger_ui
from src import config


MODEL_NAME = os.environ.get('MODEL_NAME')

app = Flask(__name__)
CORS(app)
Swagger(app)
app.register_blueprint(flask_swagger_ui, url_prefix='/swagger')
model = tf.keras.layers.TFSMLayer(config.MODELS_PATH, call_endpoint=config.SIG_NAME)
model_client = ModelClient(model)


@app.route('/health', methods=['GET'])
def health():
    return {'status': 'OK'}


@app.route('/predict', methods=["POST"])
def predict():
    req = request.get_json()
    pred = model_client.predict(req)
    return {key: value.numpy().tolist()[0] for key, value in pred.items()}


if __name__ == '__main__':
    app.run()
