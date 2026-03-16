import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from app.schema import HousingInput

NAME = "Pratyush Nanda"
ROLL_NO = "2022BCS0168"

app = FastAPI(title="Housing Price Prediction API")

model = joblib.load("model/model.joblib")

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs", status_code=301)

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: HousingInput):
    input_df = pd.DataFrame([{
        "longitude": data.longitude,
        "latitude": data.latitude,
        "housing_median_age": data.housing_median_age,
        "total_rooms": data.total_rooms,
        "total_bedrooms": data.total_bedrooms,
        "population": data.population,
        "households": data.households,
        "median_income": data.median_income,
        "ocean_proximity": data.ocean_proximity,
    }])

    # Handle missing total_bedrooms
    if input_df["total_bedrooms"].isnull().any():
        input_df["total_bedrooms"] = input_df["total_bedrooms"].fillna(0)

    # One-hot encode ocean_proximity
    input_df = pd.get_dummies(input_df, columns=["ocean_proximity"], drop_first=True)

    # Align with model's expected features (fill missing dummy columns with 0)
    expected_cols = model.feature_names_in_
    input_df = input_df.reindex(columns=expected_cols, fill_value=0)

    prediction = model.predict(input_df)[0]

    return {
        "name": NAME,
        "roll_no": ROLL_NO,
        "predicted_median_house_value": float(prediction)
    }

