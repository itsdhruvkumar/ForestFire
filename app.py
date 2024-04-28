import pickle
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import logging

app = Flask(__name__)
model1 = pickle.load(open('ForestModel.pickle', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict')
def index():
    return render_template('predict.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/data_predict', methods=['GET', 'POST'])
def predict():
    latitude = float(request.form['latitude'])
    longitude = float(request.form['longitude'])
    brightness = float(request.form['brightness'])
    track = float(request.form['track'])

    satellite = str(request.form['satellite'])
    if satellite.lower()=='terra':
        satellite=1
    else:
        satellite=0

    frp = float(request.form['frp'])
    daynight = str(request.form['daynight'])
    if daynight.lower()=='night':
        daynight=0
    else:
        daynight=1

    type_ = int(request.form['type'])
    if type_=='0':
        type_2='0'
        type_3='0'
    elif type_=='2':
        type_2='1'
        type_3='0'
    else:
        type_2='0'
        type_3='1'

    scan = float(request.form['scan'])
    year = int(request.form['year'])
    month = int(request.form['month'])
    day = int(request.form['day'])
    

    prediction = model1.predict(pd.DataFrame([[latitude,longitude,brightness,track,satellite,frp,daynight,type_2,type_3,scan,year,month,day]],columns = ['latitude','longitude','brightness','track','satellite','frp','daynight','type_2','type_3','scan_binned','year','month','day']))
    logging.info(prediction)
    prediction = round(prediction.tolist()[0], 2)

    return render_template('productivity.html', y=prediction)

if __name__ == "__main__":
    app.run(debug=True)
