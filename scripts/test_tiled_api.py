import torch
from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


data = torch.randn(
    1,
    8,
    16,
    16,
).flatten().tolist()


payload = {
    "data": data,
    "channels": 1,
    "depth": 8,
    "height": 16,
    "width": 16,
}


response = client.post(
    "/predict/tiles",
    json=payload,
)


print("Status code:")
print(response.status_code)

print("\nResponse:")
print(response.json())