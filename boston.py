import requests
from joblib import dump, load
import numpy as np
from flask import Flask, request,render_template
import sys
app = Flask("boston")
app.config['DEBUG'] = True

model = load('Boston.joblib')

@app.route('/')
def home():
    return render_template('mainpage.html')
    
@app.route('/predict',methods=['POST','GET'])
def predict():
    features = [int(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('mainpage.html', prediction_text='Median value of owner-occupied homes in $1000 is {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)