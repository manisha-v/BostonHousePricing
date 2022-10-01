import pickle
from flask import Flask, request,render_template,app,jsonify,url_for, redirect
import numpy as np
import pandas as pd

app = Flask(__name__)
regmodel = pickle.load(open('regmodel.pkl','rb'))
scalar = pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output = regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_data = scalar.transform(np.array(list(data)).reshape(1,-1))
    output = regmodel.predict(final_data)[0]
    return render_template('home.html',prediction_text='Predicted Price is {}'.format(round(output,6)))

if __name__ == "__main__":
    app.run(debug=True)
