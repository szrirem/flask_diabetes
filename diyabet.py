# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 20:08:39 2022

@author: iremsezer
"""


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__) 
model = pickle.load(open('diabetes.pkl', 'rb')) 

@app.route('/') 
def home():
    return render_template('index_diyabet.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    input_features = [float(x) for x in request.form.values()]

    features_value = [np.array(input_features)]
    prediction = model.predict_proba(features_value)

    output='{0:.{1}f}'.format(prediction[0][1], 2)
    



    return render_template('index_diyabet.html', prediction_text='Kişinin diyabet olma yüzdesi % {}'.format(output)) 

if __name__ == "__main__":
    app.run(debug=True)