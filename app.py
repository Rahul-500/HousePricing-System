
import numpy as np
import pandas as pd
from flask import Flask, render_template,request
import pickle

app = Flask(__name__)
data=pd.read_csv('Cleaned_data.csv')
pipe=pickle.load(open('Linear_Regression_Model.pkl','rb'))

@app.route("/")
def index():

    locations=sorted(data['location'].unique())

    return render_template('index.html',locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    locations= request.form.get('location')
    sqft= request.form.get('total_sqft')
    bhk= float(request.form.get('bhk'))

    input=pd.DataFrame([[locations,sqft,bhk]],columns=['location','total_sqft','bhk'])
    prediction=np.round(pipe.predict(input)[0]*1e5,2)
    return str(prediction)

if __name__=='__main__':
    app.run(debug=True,port=5000)