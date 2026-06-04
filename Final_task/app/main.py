import os
from typing import List
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

from app.schemas import ApartmentFeatures, PredictionResult, DISTRICTS
from app.model import predict_price, load_model

app = FastAPI(title="Apartment Price Predictor", version="1.0.0")

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(_APP_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(_APP_DIR, "static")), name="static")

_history: List[dict] = []


@app.on_event("startup")
def startup_event():
    load_model()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"districts": DISTRICTS, "history": list(reversed(_history[-20:]))},
    )


@app.post("/predict", response_model=PredictionResult)
def predict(features: ApartmentFeatures):
    if features.floor > features.max_floor:
        raise HTTPException(status_code=422, detail="Этаж не может быть больше этажности дома")
    if features.life_sq >= features.full_sq:
        raise HTTPException(status_code=422, detail="Жилая площадь должна быть меньше общей")

    price = predict_price(features)
    price_fmt = f"{price:,.0f}".replace(",", " ") + " руб."

    _history.append({
        "timestamp": datetime.now().strftime("%d.%m.%Y %H:%M"),
        "full_sq": features.full_sq,
        "life_sq": features.life_sq,
        "floor": features.floor,
        "max_floor": features.max_floor,
        "build_year": features.build_year,
        "num_room": features.num_room,
        "kitch_sq": features.kitch_sq,
        "sub_area": features.sub_area,
        "price": price,
        "price_formatted": price_fmt,
    })

    return PredictionResult(price=price, price_formatted=price_fmt, features=features)


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8004, reload=True)