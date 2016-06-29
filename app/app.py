#----------------------------------------------------------------------------#
# Imports
#----------------------------------------------------------------------------#
import flask
from flask import (Flask, render_template, request, url_for, 
                   redirect, send_from_directory)
from flask_multistatic import MultiStaticFlask
from werkzeug.utils import secure_filename

import logging
from logging import Formatter, FileHandler
from forms import *
import os
import requests
import cPickle as pickle
import numpy as np
import pandas as pd
import json

import sys


#----------------------------------------------------------------------------#
# App Config.
#----------------------------------------------------------------------------#
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'tiff'])

app = MultiStaticFlask(__name__)
#app.static_folder = [
#        os.path.join(app.root_path, '../images', app.config['CUSTOM_STATIC_PATH']), 
#        os.path.join(app.root_path, 'static', 'default')]
app.config.from_object('config')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CUSTOM_STATIC_PATH'] = '../images/'
sys.path.append('/Users/reallimoges/projects/transfer_learning')

from src.model.run_neural_net import score_one_photo, build_model
from src.clustering.ANN import get_tree_index

with open('../data/fc1_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

MODEL = build_model('fc1')
TREE, INDEXES = get_tree_index('euclidean')


#----------------------------------------------------------------------------#
# Miscellaneous Functions 
#----------------------------------------------------------------------------#

def getitem(obj, item, default):
    if item not in obj:
        return default
    else:
        return obj[item]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in  \
                    ALLOWED_EXTENSIONS


#----------------------------------------------------------------------------#
# Controllers 
#----------------------------------------------------------------------------#

@app.route('/')
def home():
    return render_template('pages/placeholder.home.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_photo():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            return redirect(url_for('neighbors', filename=filename)) 

    return render_template('pages/placeholder.upload.html')


@app.route('/neighbors', methods=['GET', 'POST'])
@app.route('/neighbors/<filename>', methods=['GET', 'POST'])
def neighbors(filename):
    path = os.path.abspath('static/uploads/' + filename)
    score = score_one_photo(MODEL, path)
    kneighs = TREE.get_nns_by_vector(score, 5)
    filenames = []
    for item in kneighs: filenames.append(INDEXES[item])

    return render_template('pages/placeholder.neighbors.html', filename=filename, filenames=filenames)
    #return """<h1> I made it here {} </h1> """.format(filename)

@app.route('/about')
def about():
    return render_template('pages/placeholder.about.html')


@app.route('/gallery')
def gallery():
    return render_template('pages/placeholder.gallery.html')

# Error handlers.

@app.errorhandler(404)
def not_found_error(error):
    return render_template('errors/404.html'), 404

#----------------------------------------------------------------------------#
# Launch.
#----------------------------------------------------------------------------#

# Default port:
if __name__ == '__main__':
    app.run(debug=True)
