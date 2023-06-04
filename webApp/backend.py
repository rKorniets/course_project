import os

import torch
import numpy as np
import plotly.express as px
from flask import *
from werkzeug.utils import secure_filename
from models.modelHelper import ModelHelper

from models.Autoencoder import Autoencoder

app = Flask(__name__, template_folder="./templates")
model = Autoencoder(100000)
model_helper = None
PATH_TO_STATE = "./models/saved/better_graphic_but_terrible_sound_2000elem_16batch_1hour.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UPLOAD_FOLDER = "./webApp/temp/"
ALLOWED_EXTENSIONS = {'.wav', '.mp3'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
def allowed_file(filename):
    return True
@app.post('/api/demo/upload')
def demo():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'result': 'error'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'result': 'error'}, 400)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            response = model_helper.demo_filtering(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return jsonify({'result': 'ok', 'response': response}, 200)
    return jsonify({'result': 'error'}, 400)


@app.get('/')
def index():
    return render_template("index.html")


def run_app(debug=False):
    global model_helper, model
    model.load_state_dict(torch.load(PATH_TO_STATE))
    model.to(device)
    model.eval()
    model_helper = ModelHelper(model)
    app.run(debug=debug)