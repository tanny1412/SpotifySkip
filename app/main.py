"""
FastAPI application to serve Spotify skip predictions.
"""
import pickle

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Spotify Skip Predictor API",
    description="Predict whether a Spotify track will be skipped based on playback features.",
    version="0.1",
)

# Global placeholders for loaded model and expected feature order
model = None
feature_names = None

# Pydantic model for input features
class TrackFeatures(BaseModel):
    # Features available at playback start
    hour: int                # Hour of day (0-23)
    month: int               # Month of year (1-12)
    weekday: int             # Day of week (0=Mon..6=Sun)
    platform: int            # Encoded platform index
    reason_start: int        # Encoded reason the track started
    shuffle: int             # 0 or 1 for shuffle mode

@app.on_event("startup")
def load_model():
    """
    Load the trained pipeline at startup.
    """
    global model, feature_names
    # Load trained pipeline
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    # Capture the expected feature order from the scaler (if available)
    try:
        scaler = model.named_steps.get('scaler', None)
        if scaler is not None and hasattr(scaler, 'feature_names_in_'):
            feature_names = list(scaler.feature_names_in_)
        else:
            feature_names = None
    except Exception:
        feature_names = None

@app.get("/", tags=["health"])
def read_root():
    return {"message": "Spotify Skip Prediction API is running"}

@app.post("/predict", tags=["prediction"])
def predict_skip(data: TrackFeatures):
    """
    Predict whether a track will be skipped (0/1) and return probability.
    """
    # Convert input data to DataFrame
    # Convert input to DataFrame and reorder columns if needed
    df = pd.DataFrame([data.dict()])
    if feature_names:
        df = df[feature_names]
    # Run prediction
    pred = int(model.predict(df)[0])
    prob = float(model.predict_proba(df)[0][1])
    return {"skipped": pred, "probability": prob}