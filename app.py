import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('california.pkl','rb'))
scalar = pickle.load(open('scaling.pkl','rb'))

'''decorator'''
@app.route('/')
def home():
    return render_template('home.html')

'''path'''
@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.get_json('data')
    print(data)
    print(np.array(list(data.value())).reshape(1,-1))
    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/home')
def home1():
    return ('my cute boyfriend is hell cheesy thats why I is dramebaz and emotional blackmailer else Im innocent and its important to be a blackmailer aab kaam bhi to nikalwana na')

if __name__=="__main__":
    app.run(debug=True)
