"""Utility Module"""

import logging
import os

import joblib
import yaml


def production_update_dvc():
    """Utility function used to load / reload DVC stored data"""
    logging.info("Loading / Reloading DVC Stored Data")
    os.system("touch lock")
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull -f") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


def get_labels():
    """Utility function used to load the categorical label settings"""
    with open("conf/labels/census.yaml", "r", encoding="utf8") as fh:
        data_loaded = yaml.safe_load(fh)
        return data_loaded["categorical"], data_loaded["target"]


def get_production_model(
    model_path="model/random_forest_model.pkl",
    encoder_path="model/encoder.pkl",
    lb_path="model/lb.pkl",
):
    """Utility function used to get the production model"""

    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    lb = joblib.load(lb_path)

    return model, encoder, lb
