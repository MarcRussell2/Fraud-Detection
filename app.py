from flask import Flask, request, render_template
#import cPickle as pickle
import pickle
import json
import requests
import socket
import time
from datetime import datetime
from src.data_clean import get_data

app = Flask(__name__)
PORT = 5353
#REGISTER_URL = "http://10.3.0.79:5000/register"
REGISTER_URL = "http://galvanize-case-study-on-fraud.herokuapp.com/data_point"
DATA = []
TIMESTAMP = []

# with open('static/model.pkl', 'rb') as f_un:
#     model_upkl = pickle.load(f_un)

@app.route('/', methods=['GET'])
def index():
    """ Render a simple splash page"""
    return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     """ Receive the article to be classified from the input and use the model to predict
#     the class."""
#     # will need to use the two commented lines if switching back to use "form" to obtain user input text
#     #data = str(request.form['article_body'])
#     data = str(request.json['article_body'])
#     pred = str(model.predict([data])[0])
#     #return render_template('predict.html', article=data, predicted=pred)
#     return jsonify({'prediction': pred})

@app.route('/score', methods=['POST'])
def score():
    print("this is a ttest!")
    df = get_data()
    return df.to_string()
    # DATA.append(json.dumps(request.json, sort_keys=True, indent=4, separators=(',', ': ')))
    # TIMESTAMP.append(time.time())
    #return render_template("score.html")

@app.route('/check')
def check():
    line1 = "Number of data points: {0}".format(len(DATA))
    if DATA and TIMESTAMP:
        dt = datetime.fromtimestamp(TIMESTAMP[-1])
        data_time = dt.strftime('%Y-%m-%d %H:%M:%S')
        line2 = "Latest datapoint received at: {0}".format(data_time)
        line3 = DATA[-1]
        output = "{0}\n\n{1}\n\n{2}".format(line1, line2, line3)
    else:
        output = line1
    return output, 200, {'Content-Type': 'text/css; charset=utf-8'}

def register_for_ping(ip, port):
    registration_data = {'ip': ip, 'port': port}
    requests.post(REGISTER_URL, data=registration_data)

if __name__ == '__main__':
    # Register for pinging service
    # ip_address = socket.gethostbyname(socket.gethostname())
    # print("attempting to register {}:{}".format(ip_address, PORT))
    # register_for_ping(ip_address, str(PORT))

    # Start Flask app
    #app.run(host='0.0.0.0', port=PORT, debug=True)
    app.run(host='0.0.0.0', port=8808, debug=True)
