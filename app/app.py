#----------------------------------------------------------------------------#
# Imports
#----------------------------------------------------------------------------#
from __future__ import print_function
import flask
from flask import (Flask, render_template, request, url_for,
                   redirect, send_from_directory)
from werkzeug.utils import secure_filename

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

app = Flask(__name__)
app.config.from_object('config')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CUSTOM_STATIC_PATH'] = '../images/'
sys.path.append('/Users/reallimoges/projects/transfer_learning')

from src.modeling.run_neural_net import score_one_photo, build_model
from src.modeling.ANN import get_tree_index

with open('../data/fc1_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

MODEL = build_model('fc1')
TREE, INDEXES = get_tree_index('angular')


#----------------------------------------------------------------------------#
# Miscellaneous Functions
#----------------------------------------------------------------------------#

def allowed_file(filename):
    '''
    INPUT: Filename (string)
    OUTPUTS: Boolean. True if the file conforms to typical photo
             conventions.
    '''
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in  \
        ALLOWED_EXTENSIONS


#----------------------------------------------------------------------------#
# Controllers
#----------------------------------------------------------------------------#

@app.route('/')
def home():
    '''
    INPUTS: None
    OUTPUTS: Side Effect Only (Displays base HTML page)
    '''
    return render_template('pages/placeholder.home.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_photo():
    '''
    INPUTS: If POST, user uploaded file
    OUTPUTS: If GET: placeholder.upload.html,
             If POST: Redirects to '/neighbors'

    Allows user to upload image to site to be scored and shown similar
    images.
    '''
    if request.method == 'POST':
        if 'uploaded' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        f = request.files['uploaded']
        if f.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if f and allowed_file(f.filename):
            filename = secure_filename(f.filename)
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            return redirect(url_for('neighbors', filename=filename))

    return render_template('pages/placeholder.upload.html')


@app.route('/neighbors', methods=['GET', 'POST'])
@app.route('/neighbors/<filename>', methods=['GET', 'POST'])
def neighbors(filename):
    '''
    INPUTS: Filename of file to be processed
    OUTPUTS: placeholder.neighbors.html template; pointer to user 


    Scores the user uploaded image, finds the nearest neighbors to it,
    and returns the placeholder.neighbors.html template with the original
    image, the closest neighbor, and the remaining as a list.
    '''
    path = os.path.abspath('static/uploads/' + filename)
    
    score = score_one_photo(MODEL, path)
    kneighs = TREE.get_nns_by_vector(score, 10)
    
    # Creates the list of approximate nearest neighbors    
    filenames = []
    for item in kneighs:
        filenames.append(INDEXES[item])
    first_file = filenames.pop(0)
    original = filename
    return render_template('pages/placeholder.neighbors.html',
                           original=original, active_file=first_file, 
                           filenames=filenames)

@app.route('/about')
def about():
    '''
    INPUTS: None
    OUTPUTS: Side Effect Only (shows placeholder.about.html)
    '''
    return render_template('pages/placeholder.about.html')


@app.route('/gallery')
def gallery():
    '''
    INPUTS: None
    OUTPUTS: Side Effects Only (shows placeholder.gallery.html)
    '''
    return render_template('pages/placeholder.gallery.html')


@app.errorhandler(404)
def not_found_error(error):
    '''
    INPUTS: Error
    OUTPUTS: Side Effects Only (shows 404.html)
    '''
    return render_template('errors/404.html'), 404

#----------------------------------------------------------------------------#
# Launch.
#----------------------------------------------------------------------------#

if __name__ == '__main__':
    app.run(debug=True)
