"""API Tests"""
from http import HTTPStatus

from api.app import WELCOME_MESSAGE
from api.model import CensusData

LESS_THAN_EQUAL_50_SAMPLE = CensusData(
    age=46,
    education="HS-grad",
    education_num=9,
    fnlgt=51618,
    marital_status="Married-civ-spouse",
    occupation="Other-service",
    relationship="Wife",
    race="White",
    sex="Female",
    capital_gain=0,
    capital_loss=0,
    hours_per_week=40,
    native_country="United-States",
    workclass="Private",
)

GREATER_THAN_50_SAMPLE = CensusData(
    age=31,
    education="Masters",
    education_num=14,
    fnlgt=45781,
    marital_status="Never-married",
    occupation="Prof-specialty",
    relationship="Not-in-family",
    race="White",
    sex="Female",
    capital_gain=14084,
    capital_loss=0,
    hours_per_week=50,
    native_country="United-States",
    workclass="Private",
)


def test_welcome(api_client):
    """Test a successful welcome message from the main index page"""
    ret = api_client.get("/")
    assert ret.status_code == HTTPStatus.OK
    assert ret.json()["message"] == WELCOME_MESSAGE


def test_less_than_inference(api_client):
    """Test a successful welcome message from the main index page"""
    ret = api_client.post("/infer", json=LESS_THAN_EQUAL_50_SAMPLE.dict())
    assert ret.status_code == HTTPStatus.OK
    assert ret.json() == "<=50K"


def test_greater_than_inference(api_client):
    """Test a successful welcome message from the main index page"""
    ret = api_client.post("/infer", json=GREATER_THAN_50_SAMPLE.dict())
    assert ret.status_code == HTTPStatus.OK
    assert ret.json() == ">50K"
