"""Example Inference"""

import requests
from api.model import CensusData
from http import HTTPStatus

SAMPLE = CensusData(
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

APP_URI = "https://cblythe-census.herokuapp.com/infer"


def example_inference():
    """Test a inference request on the production API"""

    ret = requests.post(url=APP_URI, data=SAMPLE.json(), headers={'content-type': 'application/json'}, verify=True)

    if ret.status_code == HTTPStatus.OK:
        print("Inference request succesfully received")
        print(ret.json())
    else:
        print(f"Inference failed with error code {ret.status_code}")


if __name__ == "__main__":
    example_inference()
