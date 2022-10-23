import joblib
import pytest
import pandas as pd

from fastapi.testclient import TestClient
from sklearn.model_selection import train_test_split

from api.app import app
from api.utils import update_dvc, get_labels, get_production_model

@pytest.fixture(scope="session",autouse=True)
def get_dvc_data():
   # update_dvc()
   pass


@pytest.fixture
def api_client():
    with TestClient(app) as client:
        yield client

@pytest.fixture
def testing_data(scope="session"):
    df = pd.read_csv('./data/cleaned/census_clean.csv')
    _, test = train_test_split(df)
    return test

@pytest.fixture(scope="session")
def production_model():
    return get_production_model()

@pytest.fixture(scope="session")
def input_labels():
    return get_labels()