"""Utility Module"""
import joblib
import yaml

from dvc.repo import Repo


def production_update_dvc():
    """Utility function used to load / reload DVC stored data"""
    repo = Repo()
    repo.pull()


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
