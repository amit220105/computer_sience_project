from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health():
    assert client.get("/health").json() == {"ok": True}

def test_exhibits_shape():
    data = client.get("/exhibits").json()
    assert isinstance(data, list) and len(data) >= 1
    assert set(data[0].keys()) >= {"id","name","type","pos","score","avg_view_time_min"}

def test_plan_returns_path():
    body = {"time_budget_min": 30, "entrance": [0,0]}
    res = client.post("/route/plan", json=body).json()
    assert "path" in res and isinstance(res["path"], list)