from sentence_transformers import SentenceTransformer
from language_modeling.Project import Project
from language_modeling.util import load_hparams
from flask import Flask, request, make_response, jsonify
from flask_cors import CORS
from language_modeling.util import get_default_logger
from Apis.helpers import make_error_object
import json
import numpy as np
hparams = load_hparams()
model = SentenceTransformer(Project.export_dir / hparams.api_model_dir)
logger = get_default_logger()
logger.info("Starting API Service")
app = Flask(__name__)
CORS(app)
cors = CORS(app, resource={r"/*": {"origins": "*"}})
# Our sentences we like to encode

# Dummy route
@app.route("/", methods=["GET"])
def index():
    return {"status": "SUCCESS"}


@app.route("/embed", methods=["POST"])
def get_embeddings():
    required_params = [
        "sentence",
    ]

    for p in required_params:
        if p not in request.form:
            return make_response(
                jsonify(make_error_object(f"'{p}'parameter missing")), 400
            )
    sentence = request.form["sentence"]
    sentence = sentence.strip()

    vec = model.encode(sentence)
    vec = vec.astype(np.float64)
    return make_response(jsonify({"embedding": list(vec)}), 200)


app.run()
