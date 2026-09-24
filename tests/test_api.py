from fastapi.testclient import TestClient
from app.main import app
client=TestClient(app)
def test_health(): assert client.get("/healthz").status_code==200
def test_model_status(): assert client.get("/v1/model/status").json()["loaded"] is False
