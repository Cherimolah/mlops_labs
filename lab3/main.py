import pickle

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI()

# Загружаем модель
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)


@app.get('/predict')
async def predict(
    area_sqm: float = Query(
        ...,
        ge=20,
        le=200,
        description='Площадь от 20 до 200 кв.м.'
    ),
    distance_to_center_km: float = Query(
        ...,
        ge=0,
        le=15,
        description='Расстояние от Площади 1905 года (от 0 до 15 км)'
    )
):
    features = [[area_sqm, distance_to_center_km]]
    prediction = model.predict(features)
    # Возвращаем предсказание в виде числа или списка
    return JSONResponse(content={'prediction': prediction.tolist()[0]})


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
