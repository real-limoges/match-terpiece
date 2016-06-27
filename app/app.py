#----------------------------------------------------------------------------#
# Imports
#----------------------------------------------------------------------------#
import flask
from flask import Flask, render_template, request, url_for, redirect
from werkzeug.utils import secure_filename

import logging
from logging import Formatter, FileHandler
from forms import *
import os
import requests

import numpy as np
import pandas as pd
from annoy import AnnoyIndex


#----------------------------------------------------------------------------#
# App Config.
#----------------------------------------------------------------------------#

UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'tiff'])

app = Flask(__name__)
app.config.from_object('config')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            print "bad file"
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            print "no file"
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print "confirmed good file"
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print "saved file"
            return redirect(url_for('about'))

    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''
    #return render_template('pages/placeholder.home.html')

@app.route('/about')
def about():
    return render_template('pages/placeholder.about.html')



# Error handlers.

@app.errorhandler(500)
def internal_error(error):
    #db_session.rollback()
    return render_template('errors/500.html'), 500


@app.errorhandler(404)
def not_found_error(error):
    return render_template('errors/404.html'), 404

#----------------------------------------------------------------------------#
# Launch.
#----------------------------------------------------------------------------#

# Default port:
if __name__ == '__main__':
    print app.config['UPLOAD_FOLDER']
    app.run(debug=True)
