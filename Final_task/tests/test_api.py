"""
Модульные тесты HTTP-слоя (app/main.py) через FastAPI TestClient.
Фикстура client (conftest.py) поднимает приложение с игрушечной моделью.
"""
from app.schemas import DISTRICTS


def _payload(**overrides):
    base = dict(
        full_sq=65.0, life_sq=40.0, floor=5, max_floor=9,
        build_year=2005, num_room=2, kitch_sq=10.0, sub_area=DISTRICTS[0],
    )
    base.update(overrides)
    return base


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_index_page_renders(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_predict_ok(client):
    r = client.post("/predict", json=_payload())
    assert r.status_code == 200
    body = r.json()
    assert "price" in body and isinstance(body["price"], (int, float))
    assert body["price_formatted"].endswith("руб.")


def test_predict_floor_gt_max_floor(client):
    r = client.post("/predict", json=_payload(floor=10, max_floor=9))
    assert r.status_code == 422
    assert "Этаж" in r.json()["detail"]


def test_predict_life_sq_ge_full_sq(client):
    r = client.post("/predict", json=_payload(full_sq=40.0, life_sq=40.0))
    assert r.status_code == 422
    assert "Жилая площадь" in r.json()["detail"]


def test_predict_validation_error_on_bad_field(client):
    # full_sq <= 0 отсекается Pydantic-схемой ещё до обработчика.
    r = client.post("/predict", json=_payload(full_sq=0))
    assert r.status_code == 422
