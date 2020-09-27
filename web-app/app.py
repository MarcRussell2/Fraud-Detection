from flask import Flask, request, render_template
import pickle
import json
import requests
import socket
import time
from datetime import datetime
from src.data_clean import get_data
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

with open('static/model.pkl', 'rb') as f_un:
    model_upkl = pickle.load(f_un)

@app.route('/', methods=['GET'])
def index():
    """ Render a simple splash page"""
    return render_template('index.html')

@app.route('/score', methods=['POST'])
def score():
    df = get_data()
    df.drop_duplicates(inplace=True)
    df_use = df[['has_logo','listed','num_payouts','user_age','user_type','description']]
    print("Predicting")
    X = df_use.values
    df['fraud'] = model_upkl.predict(X)
    return df.to_string()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8808, debug=True)
