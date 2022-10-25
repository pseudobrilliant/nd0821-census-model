"""Inference API"""

import logging
import os

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder

import pandas as pd
import uvicorn

from api.model import CensusData
from api.utils import production_update_dvc, get_labels, get_production_model

from ml.process import process_data
from ml.train_model import inference

app = FastAPI()

APP_VARIABLES = {}
WELCOME_MESSAGE = "Welcome to the census inference application. \
    Please use the 'infer' endpoint to request an inference."


@app.on_event("startup")
def startup_event():
    """ "Startup function to bootstrap required app variables"""
    production_update_dvc()

    cat, target = get_labels()
    APP_VARIABLES["categorical_features"] = cat
    APP_VARIABLES["target"] = target

    model, encoder, lb = get_production_model()
    APP_VARIABLES["model"] = model
    APP_VARIABLES["encoder"] = encoder
    APP_VARIABLES["lb"] = lb

    logging.info("Application Loaded")


@app.on_event("shutdown")
def shutdown_event():
    """Shutdown function that logs a shutdown event"""
    logging.info("Application Shutdown")


@app.get("/")
def root():
    """Provides a greeting message for the main index function"""
    return {"message": WELCOME_MESSAGE}


@app.post("/infer")
def predict(data: CensusData):
    """Predicts appropriate salary labels based on census data provided"""
    df = pd.DataFrame.from_dict([jsonable_encoder(data)])

    x_input, _, _, _ = process_data(
        df,
        categorical_features=APP_VARIABLES["categorical_features"],
        training=False,
        encoder=APP_VARIABLES["encoder"],
        lb=APP_VARIABLES["lb"],
    )

    y_hat = inference(APP_VARIABLES["model"], x_input)
    label = APP_VARIABLES["lb"].inverse_transform(y_hat)[0]

    return label


if __name__ == "__main__":
    uvicorn.run(app=app)
