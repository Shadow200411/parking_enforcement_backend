from fastapi.testclient import TestClient
from app.main import app

# Create a fake browser client that connects to your FastAPI app
client = TestClient(app)

def test_unauthorized_patch_is_rejected():
    payload = {
        "is_valid_violation": True,
        "notes": "Hacker trying to bypass security"
    }
    
    response = client.patch("/api/v1/flags/999/verify", json=payload)
    
    assert response.status_code == 401
    
    assert response.json()["detail"] == "Not authenticated"