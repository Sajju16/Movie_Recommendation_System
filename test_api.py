from __future__ import annotations

import json

import requests


def main() -> None:
    resp = requests.post("http://127.0.0.1:5000/recommend", json={"user_id": 1, "topk": 5}, timeout=10)
    resp.raise_for_status()
    print(json.dumps(resp.json(), indent=2))


if __name__ == "__main__":
    main()
