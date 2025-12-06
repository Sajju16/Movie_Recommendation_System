import requests, json
resp = requests.post("http://127.0.0.1:5000/recommend", json={"user_id": 1, "topk": 5})
print(json.dumps(resp.json(), indent=2))
