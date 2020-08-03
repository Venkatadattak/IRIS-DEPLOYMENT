from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model = pickle.load(open('classifiersvc.pickle', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        SepalLengthCm = float(request.form['SepalLengthCm'])
        SepalWidthCm = float(request.form['SepalWidthCm'])

        PetalLengthCm = float(request.form['PetalLengthCm'])

        PetalWidthCm = float(request.form['PetalWidthCm'])

        
        prediction=model.predict([[SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm]])
        output=round(prediction[0],2)
        
        return render_template('index.html',prediction_text="The flower is {}".format(output))
    else:
        return render_template('index.html')
    

if __name__=="__main__":
    app.run(debug=True)



