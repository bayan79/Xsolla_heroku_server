import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import pickle



from flask import Flask
from flask import request
import requests
from flask import jsonify

import os
import json
from ast import literal_eval
import traceback

from preprocessor import Preprocessor
from libherokuserver import clear_message


application = Flask(__name__)


#загружаем модели из файла
preproc = pickle.load(open("./models/final_preproc.pickle", "rb"))
model = pickle.load(open("./models/final_model.pickle", "rb"))


# тестовый вывод
@application.route("/")  
def hello():
    resp = {'message':"Hello World!"}
    
    response = jsonify(resp)
    
    return response


def class_predict(message, preproc, model):
    """Make prediction of message with ordering
    message: string
    preproc: Preprocessor
    model: ML model with predict_proba func
    """
    _order=['afs', 'other', 'ps']
    vec = preproc.message_prepare(message)
    pred = model.predict_proba(vec)[0]    
    
    results = dict(enumerate(pred))  
    return [results[preproc.label_map[_class]] for _class in _order]



# предикт категории
@application.route("/categoryPrediction" , methods=['GET', 'POST'])  
def registration():
    resp = {'message':'ok'
           ,'category': -1
           }

    try:
        getData = request.get_data()
        json_params = json.loads(getData) 
        message = json_params['user_message']
        
        if not clear_message(message):
            return {
                'message':'ok',
                'category': 'afs',
                'prediction': [1, 0, 0]
            }
            
        pred = class_predict(message, preproc, model)
        resp['prediction'] = pred
        resp['category'] = ['afs', 'other', 'ps'][np.argmax(pred)]

    except Exception as e: 
        resp['message'] = "Sorry, we are already working on error!!! Take a rest and drink tea :)"
      
    response = jsonify(resp)
    
    return response

        

if __name__ == "__main__":
    port = int(os.getenv('PORT', 5000))
    application.run(debug=False, port=port, host='0.0.0.0' , threaded=True)



