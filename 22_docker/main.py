from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from forecasting import forecast, AIRLINE

app = FastAPI(title="Time Series Forecasting API")


class ForecastRequest(BaseModel):
    values: list[float]
    steps: int = 12
    seasonal_periods: Optional[int] = None
    method: str = "auto"


@app.get("/")
def root():
    return {"service": "time series forecasting api", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/forecast")
def predict(req: ForecastRequest):
    try:
        preds, method = forecast(req.values, req.steps, req.seasonal_periods, req.method)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {"forecast": preds, "method_used": method, "steps": req.steps}


@app.get("/forecast/sample")
def sample(steps: int = 12):
    preds, method = forecast(AIRLINE, steps=steps, seasonal_periods=12)
    return {"forecast": preds, "method_used": method, "steps": steps}
