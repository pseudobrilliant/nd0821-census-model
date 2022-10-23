from pydantic import BaseModel

class CensusData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        schema_extra = {
            "example": {
                "age": 27,
                "capital_gain": 100,
                "capital_loss": 10,
                "education": "Graduates",
                "education_num": 10,
                "fnlgt": 145612,
                "marital_status": "Married-civ-spouse",
                "native_country": "United-States",
                "occupation": "Handlers-cleaners",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "hours_per_week": 40
            }
        }
