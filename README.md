# Movie_Recom (Movie Recommender)

A simple movie recommendation project using the MovieLens dataset.

## Project structure

- `src/movie_recom/`: Python package (API server + recommenders)
- `dataset/`: CSV data (MovieLens + generated `user_item.csv`)
- `pyproject.toml`: package metadata + dependencies
- `requirements.txt`: convenience dependencies list
- `code/`: legacy scripts (kept for reference)

## Setup

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -e .
```

## Generate `user_item.csv` (if needed)

This project expects `dataset/user_item.csv`. If it’s missing, generate it:

```bash
python -m movie_recom.create_user_item
```

## Run the API server

```bash
python -m movie_recom.server
```

Then open:
- `http://127.0.0.1:5000/` (simple browser UI)
- `POST http://127.0.0.1:5000/recommend` with JSON `{"user_id": 1, "topk": 5}`

## Quick test (client)

```bash
python -m movie_recom.test_api
```

## Notes

- The server uses an SVD-based recommender (`SVDRecommender`). On first run it trains and saves a model file (`svd_model.joblib`) in the project root.
