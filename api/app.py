from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder

from api.model import CensusData
from ml.process import process_data
from ml.train_model import inference
from api.utils import update_dvc, get_labels, get_production_model

import logging
import joblib
import os
import pandas as pd
import yaml
import uvicorn

app = FastAPI()

APP_VARIABLES = {}
WELCOME_MESSAGE = "Welcome to the census inference application. Please use the 'infer' endpoint to request an inference."

@app.on_event("startup")
async def startup_event():

    cat, target = get_labels()
    APP_VARIABLES['categorical_features'] = cat
    APP_VARIABLES['target'] = target

    model, encoder, lb = get_production_model()
    APP_VARIABLES['model'] = model
    APP_VARIABLES['encoder'] = encoder
    APP_VARIABLES['lb'] = lb

    if "DYNO" in os.environ and os.path.isdir(".dvc"):
        update_dvc()

    logging.info('Application Loaded')

@app.on_event("shutdown")
async def shutdown_event():
    logging.info('Application Shutdown')

@app.get('/')
def root():
    return {"message": WELCOME_MESSAGE}

@app.post('/infer')
def predict(data: CensusData):

    df = pd.DataFrame.from_dict([jsonable_encoder(data)])

    X_input, _, _, _ = process_data(
        df, categorical_features=APP_VARIABLES['categorical_features'],
            training=False, encoder=APP_VARIABLES['encoder'],
            lb=APP_VARIABLES['lb'])

    y_hat = inference(APP_VARIABLES['model'], X_input)
    label = APP_VARIABLES['lb'].inverse_transform(y_hat)[0]

    return label

if __name__ == "__main__":
    uvicorn.run(app=app)