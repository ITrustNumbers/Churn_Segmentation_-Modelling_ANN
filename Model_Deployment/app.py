from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
from Transformer import transform, batch_transform
import json

#loading Model
Classifier = load_model('../Trained_Model/Ann_Classifier.h5')

#Model and Output Details
positive = 'The customer with the given profile will leave the bank' #prediction Statement
negative = 'The customer with the given profile will not leave the bank' #prediction Statement
threshold = 0.61 #Prediction threshold
warning = 'There seems to be error in your API call, look at /help route for more information about the API call' #Warning

#Defining App
app = Flask(__name__)

#Index Route
@app.route('/')

def give_index():
    return render_template('index.html')

#Help Route
@app.route('/help',methods=['GET','POST'])
def give_help():
    fh = open('Help_Information.txt')
    content = fh.read()
    req_type_str = request.method
    if req_type_str == 'GET':
        return content
    if req_type_str == 'POST':
        return jsonify({'information':content})

#Predict Route
@app.route('/predict',methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        data = transform(data['data'])
        probs = Classifier.predict(data)
        probs = probs[0].tolist()
        preds = [positive if prob > threshold else negative for prob in probs]
        response = {'prob' : probs,'pred':preds}
        return jsonify(response)
    except:
        return jsonify({'warning':warning})
    
#Batch_Predict Route
@app.route('/batch_predict',methods=['POST'])
def batch_predict():
    try:
        data = request.get_json(force=True)
        data = batch_transform(data['data'])
        probs = Classifier.predict(data).reshape(1,-1)
        probs = probs[0].tolist()
        preds = [positive if prob > threshold else negative for prob in probs]
        response = {'probs' : probs,'preds':preds}
        return jsonify(response)
    except:
        return jsonify({'warning':warning})

if __name__ == "__main__":
    app.run()