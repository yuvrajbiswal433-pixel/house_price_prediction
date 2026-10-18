from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Import custom classes BEFORE loading the model
from preprocessing import (
    dropper,
    numimputer,
    ordinalencoder,
    OHEEncoder,
    RareCategoryGrouper,
    MissingImputer,
    LogTransformer,
    CatImputer,
)

app = FastAPI(title="House Price Prediction API")

MODEL_PATH = Path(__file__).parent / "house_price_model.pkl"

model = joblib.load(MODEL_PATH)


class HouseFeatures(BaseModel):
    features: dict[str, object] = Field(...)


@app.get("/")
def home():
    return {"message": "House Price Prediction API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(data: HouseFeatures):
    try:
        input_df = pd.DataFrame([data.features])
        prediction = float(model.predict(input_df)[0])

        # Use this only if your model predicts log1p(SalePrice).
        price = float(np.expm1(prediction))

        if not np.isfinite(price):
            raise ValueError("Invalid prediction")

        return {
            "predicted_price": round(price, 2),
            "currency": "USD",
        }

    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc