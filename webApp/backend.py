import os

import uuid
import torch
import numpy as np
import plotly.express as px
from flask import *
from werkzeug.utils import secure_filename
from models.modelHelper import ModelHelper

from models.Autoencoder import Autoencoder
from models.AudioDenoisingCNN import AudioDenoisingCNN

app = Flask(__name__, template_folder="./templates")
model = AudioDenoisingCNN()
model_helper = None
PATH_TO_STATE = "./models/saved/new_model_updated_weights.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UPLOAD_FOLDER = "./webApp/audiofile/"
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
            filename = str(uuid.uuid4()) + "." + file.filename.split('.')[-1]
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            audio_arrays = model_helper.demo_filtering(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            noised_filename = model_helper.update_filename(filename, "noised")
            filtered_filename = model_helper.update_filename(filename, "filtered")
            model_helper.save_to_disk(audio_arrays[0], os.path.join(app.config['UPLOAD_FOLDER'], filename))
            model_helper.save_to_disk(audio_arrays[1], os.path.join(app.config['UPLOAD_FOLDER'], noised_filename))
            model_helper.save_to_disk(audio_arrays[2], os.path.join(app.config['UPLOAD_FOLDER'], filtered_filename))
            #return list of files, not filenames
            return jsonify({'result': 'ok', 'filenames': [filename, noised_filename, filtered_filename]}), 200
    return jsonify({'result': 'error'}, 400)

@app.post('/api/real/upload')
def real():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'result': 'error'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'result': 'error'}, 400)
        if file and allowed_file(file.filename):
            filename = str(uuid.uuid4()) + "." + file.filename.split('.')[-1]
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            audio_arrays = model_helper.real_filtering(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filtered_filename = model_helper.update_filename(filename, "filtered")
            model_helper.save_to_disk(audio_arrays[0], os.path.join(app.config['UPLOAD_FOLDER'], filename))
            model_helper.save_to_disk(audio_arrays[1], os.path.join(app.config['UPLOAD_FOLDER'], filtered_filename))
            #return list of files, not filenames
            return jsonify({'result': 'ok', 'filenames': [filename, filtered_filename]}), 200
    return jsonify({'result': 'error'}, 400)
@app.get('/api/audio/download/<filename>')
def download(filename):
    if os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'], filename)):
        #hardcoded path to audiofile, for some reason it doesn't work with app.config['UPLOAD_FOLDER']
        return send_file(os.path.join("./audiofile/", filename))
    return jsonify({'result': 'error'}), 400

@app.post('/api/audio/plot')
def get_plot():
    """
    Receives json with audio names array and returns plotly figure
    {
    filenames: [filename1, filename2, ...]
    display_names: [display_name1, display_name2, ...]
    }
    :return:
    """
    data = request.get_json()
    if data is None:
        return jsonify({'result': 'error'}), 400
    filenames = data['filenames']
    display_names = data['display_names']
    if len(filenames) != len(display_names):
        return jsonify({'result': 'error'}), 400

    fig = px.scatter(title="Audio comparison")
    for i in range(len(filenames)):
        if os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'], filenames[i])):
            audio_array = model_helper.load_from_disk(os.path.join(app.config['UPLOAD_FOLDER'], filenames[i]))
            fig.add_scatter(x=np.arange(10**5), y=audio_array[:10**5], name=display_names[i])
        else:
            return jsonify({'result': 'error'}), 400
    return jsonify({"plot" : fig.to_html()}) , 200

@app.get('/')
def index():
    return render_template("index.html")


def run_app(clear_files_on_start=True, debug=False):
    global model_helper, model
    model.load_state_dict(torch.load(PATH_TO_STATE))
    model.to(device)
    model.eval()
    model_helper = ModelHelper(model)

    if clear_files_on_start:
        print("Clearing files on start. Disable this in app.py")
        for file in os.listdir(UPLOAD_FOLDER):
            os.remove(os.path.join(UPLOAD_FOLDER, file))
    app.run(debug=debug)