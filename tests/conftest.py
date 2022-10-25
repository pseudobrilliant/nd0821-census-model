"""Testing Configuration"""

import pytest
import pandas as pd

from dvc.repo import Repo
from fastapi.testclient import TestClient
from sklearn.model_selection import train_test_split

from api.app import app
from api.utils import get_labels, get_production_model


TEST_DATA_PATH = "./data/cleaned/census_clean.csv"


@pytest.fixture(scope="session", autouse=True)
def get_dvc_data():
    """Test fixture that retrieves DVC data at beginning of test session"""
    repo = Repo()
    repo.pull()


@pytest.fixture
def api_client():
    """Test fixture that generates a API test client"""
    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="session")
def testing_data():
    """Test fixture that returns a subset of test data from training data"""
    df = pd.read_csv(TEST_DATA_PATH)
    _, test = train_test_split(df)
    return test


@pytest.fixture(scope="session")
def production_model():
    """Test fixture that reutrns a production model and all it's dependencies"""
    return get_production_model()


@pytest.fixture(scope="session")
def input_labels():
    """Test fixture that returns input data labels for categorical and target variables"""
    return get_labels()
