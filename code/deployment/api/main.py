import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = os.environ.get("MODEL_PATH", "models/best_model.joblib")
bundle = joblib.load(MODEL_PATH)
pipeline = bundle["pipeline"]
NUMERIC_COLS = bundle["numeric_cols"]
CATEGORICAL_COLS = bundle["categorical_cols"]

app = FastAPI(title="Adult Income Predictor", version="1.0")


class AdultInput(BaseModel):
    age: int = Field(..., ge=17, le=90)
    workclass: str = "Private"
    fnlwgt: int = 100000
    education: str = "Bachelors"
    education_num: int = Field(13, alias="education-num")
    marital_status: str = Field("Never-married", alias="marital-status")
    occupation: str = "Prof-specialty"
    relationship: str = "Not-in-family"
    race: str = "White"
    sex: str = "Male"
    capital_gain: int = Field(0, alias="capital-gain")
    capital_loss: int = Field(0, alias="capital-loss")
    hours_per_week: int = Field(40, alias="hours-per-week")
    native_country: str = Field("United-States", alias="native-country")

    class Config:
        populate_by_name = True


@app.get("/health")
def health():
    return {"status": "ok", "model": bundle["model_name"]}


@app.post("/predict")
def predict(payload: AdultInput):
    row = {
        "age": payload.age,
        "workclass": payload.workclass,
        "fnlwgt": payload.fnlwgt,
        "education": payload.education,
        "education-num": payload.education_num,
        "marital-status": payload.marital_status,
        "occupation": payload.occupation,
        "relationship": payload.relationship,
        "race": payload.race,
        "sex": payload.sex,
        "capital-gain": payload.capital_gain,
        "capital-loss": payload.capital_loss,
        "hours-per-week": payload.hours_per_week,
        "native-country": payload.native_country,
    }
    df = pd.DataFrame([row])
    proba = float(pipeline.predict_proba(df)[0, 1])
    pred = int(proba >= 0.5)
    return {
        "prediction": pred,
        "label": ">50K" if pred == 1 else "<=50K",
        "probability": proba,
    }