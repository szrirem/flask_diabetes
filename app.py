# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 20:08:39 2022

@author: iremsezer
"""


import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__) 
model = pickle.load(open('diabetes.pkl', 'rb')) 

@app.route('/') 
def home():
    return render_template('index_diabetes.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    Sonuçları HTML GUI'de işlemek için
    
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    print(final_features)
    prediction = model.predict(final_features)
    prob= model.predict_proba(final_features)
    diabetes_prob=prob[0][1]


    output = round(prediction[0], 2)
    if output == 1 :
        x="diyabet hastası"
    else:
       x="diyabet hastası değil"
    print(output)

    return render_template('index_diabetes.html', prediction_text='Hastanın Durumu: {}'.format(x),probability='Hastanın diyabet olma olasılığı: {}'.format(diabetes_prob))




if __name__ == "__main__":
    app.run(debug=True)