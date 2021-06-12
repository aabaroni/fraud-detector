import pandas as pd
from joblib import dump
import uvicorn
from fastapi import FastAPI

import numpy as np
import pickle
import pandas as pd


import sys
sys.path.insert(0, '../fraud-detector/')
sys.path.insert(0, '../data')
sys.path.insert(0, '../')

import numpy as np
import conf.scoring_conf as conf
from utils import load_data_local
from feateng import FeatPipeline
import joblib
import requests
from typing import List
from datetime import datetime


###############################
# 0. Variable Assignment
###############################

inpath = conf.toscore_inpath # input path of data to be scored
intype = conf.toscore_intype # input path type of data to be scored
model_chosen = conf.model_chosen
outpath = conf.output_location_api_tagged

###############################
# 1. Data read
###############################
df = load_data_local(inpath, intype).drop(columns = ["isFlaggedFraud"])

###############################
# 2. Feature engineering pipeline
###############################
df2 = FeatPipeline.fit_transform(df)

###############################
# 3. Scoring
###############################
model = joblib.load(model_chosen+ ".pkl")

# Convert data to json
target_feature = ["isFraud"]
id_features = ['nameDest', 'nameOrig']
all_features = df.columns.difference(target_feature).difference(id_features)
data = pd.DataFrame.to_json(df2[all_features])

from pydantic import BaseModel

# TODO: make configurable
class Transaction(BaseModel):
    amount: float
    isDestBalanceNewZero: int
    isDestBalanceOldZero: int
    isDestMerchant: int
    isOrigBalanceNewZero : int
    isOrigBalanceOldZero : int
    newbalanceDest : float
    newbalanceOrig : float
    oldbalanceDest : float
    oldbalanceOrg : float
    step: int
    stepDay: int
    stepHour: int
    stepWeekDay: int
    type: str

app = FastAPI()
#pickle_in = open(model_chosen+ ".pkl","rb")
#classifier=pickle.load(pickle_in)
classifier = model

@app.get('/')
def index():
    return {'message': 'Hello, World'}


@app.post('/predict')
def predict_fraud(input_data: List[Transaction]):

    list_results = []
    df = pd.DataFrame(input_data)
    df.columns = ['amount','isDestBalanceNewZero','isDestBalanceOldZero','isDestMerchant','isOrigBalanceNewZero','isOrigBalanceOldZero','newbalanceDest','newbalanceOrig','oldbalanceDest','oldbalanceOrg','step','stepDay','stepHour','stepWeekDay','type']
    df["apiTagged"] = ""

    for i in range(len(input_data)):
        data = input_data[i].dict()

        amount = data['amount']
        isDestBalanceNewZero = data['isDestBalanceNewZero']
        isDestBalanceOldZero = data['isDestBalanceOldZero']
        isDestMerchant = data['isDestMerchant']
        isOrigBalanceNewZero  = data['isOrigBalanceNewZero']
        isOrigBalanceOldZero = data['isOrigBalanceOldZero']
        newbalanceDest = data['newbalanceDest']
        newbalanceOrig = data['newbalanceOrig']
        oldbalanceDest = data['oldbalanceDest']
        oldbalanceOrg = data['oldbalanceOrg']
        step = data['step']
        stepDay = data['stepDay']
        stepHour = data['stepHour']
        stepWeekDay = data['stepWeekDay']
        type = data['type']

        prediction = classifier.predict([[amount
                                        , isDestBalanceNewZero
                                        , isDestBalanceOldZero
                                        , isDestMerchant
                                        , isOrigBalanceNewZero
                                        , isOrigBalanceOldZero
                                        , newbalanceDest
                                        , newbalanceOrig
                                        , oldbalanceDest
                                        , oldbalanceOrg
                                        , step
                                        , stepDay
                                        , stepHour
                                        , stepWeekDay
                                        , type]])

        if (prediction == 1):
            prediction = "Fraud"
        else:
            prediction = "Not Fraud"

        list_results.append(prediction)
        df["apiTagged"][i] = prediction

    now = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    df.to_csv(outpath + "/api_tagged_" + now + ".csv")
    return {
        'Predictions (also saved in data folder)': list_results
    }


############
## METHOD 2
############
#class RequestBody(BaseModel):
#    samples: List[str]

#    def to_array(self):
#        return [sample.text for sample in self.samples]


##@app.post("/predict2")
##def predict(body: RequestBody):
##    data = np.array(body.to_array())
##
##    probas = classifier.predict_proba(data)
##    predictions = classifier.predict(data)
##
##    return {
##        "predictions": (
##            np.tile(classifier.classes_, (len(predictions), 1))[
##                np.arange(len(predictions)), predictions
##            ].tolist()
##        ),
##        "probabilities": probas[np.arange(len(predictions)), predictions].tolist(),
##    }
##

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

# uvicorn app:app --reload will point to the app.py file
# if port already taken, check what processes: lsof -i:8000