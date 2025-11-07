from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from joblib import load
from typing import Optional
from ..pipeline import load_model
from ..config import API_HOST, API_PORT, REDIS_URL, MODEL_DIR
import redis
import json
import pandas as pd
from prometheus_client import Summary, Counter, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from contextlib import asynccontextmanager

# Metrics
REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')
PREDICTION_COUNTER = Counter('predictions_total', 'Total number of predictions')

# Redis client (optional)
try:
    redis_client = redis.from_url(REDIS_URL)
except Exception:
    redis_client = None

# Pydantic model for input
class CarIn(BaseModel):
    company: str
    model: str
    fuel: str
    seller_type: str
    transmission: str
    km_driven: float
    mileage_mpg: float
    engine_cc: float
    max_power_bhp: float
    seats: int
    age: Optional[int] = None

# Use lifespan instead of deprecated startup event
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        model = load_model()
        print(f"Model loaded successfully from {MODEL_DIR / 'rf_model.joblib'}")
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {MODEL_DIR / 'rf_model.joblib'}")
        raise
    except Exception as e:
        print(f"ERROR: Failed to load model: {str(e)}")
        raise
    yield
    # cleanup if needed

app = FastAPI(title="CarPriceML API", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get('/metrics')
def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

@app.post('/predict')
@REQUEST_TIME.time()
def predict(payload: CarIn):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
    
    data = payload.dict()

    # compute age if not provided
    if data.get('age') is None:
        data['age'] = 0  # default age if missing

    # Redis caching
    key = json.dumps(data, sort_keys=True)
    if redis_client:
        try:
            cached = redis_client.get(key)
            if cached:
                PREDICTION_COUNTER.inc()
                return json.loads(cached)
        except Exception:
            pass  # Redis connection issues, continue

    # Convert input to DataFrame (fixes 500 error)
    X = pd.DataFrame([{
        'company': data['company'],
        'model': data['model'],
        'fuel': data['fuel'],
        'seller_type': data['seller_type'],
        'transmission': data['transmission'],
        'km_driven': data['km_driven'],
        'mileage_mpg': data['mileage_mpg'],
        'engine_cc': data['engine_cc'],
        'max_power_bhp': data['max_power_bhp'],
        'seats': data['seats'],
        'age': data['age']
    }])

    try:
        pred = float(model.predict(X)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    result = {"predicted_price_mad": pred}

    if redis_client:
        try:
            redis_client.set(key, json.dumps(result), ex=60*60)
        except Exception:
            pass

    PREDICTION_COUNTER.inc()
    return result

# To run: uvicorn src.api.main:app --host 0.0.0.0 --port 8000
