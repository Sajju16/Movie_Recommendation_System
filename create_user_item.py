from __future__ import annotations

import pandas as pd

from .paths import dataset_dir


def rating_to_play(r: float) -> int:
    if r >= 4.0:
        return 3
    if r >= 3.0:
        return 1
    return 0


def main() -> None:
    ratings_path = dataset_dir() / "ratings.csv"
    out_path = dataset_dir() / "user_item.csv"

    ratings = pd.read_csv(ratings_path)
    ratings["plays"] = ratings["rating"].apply(rating_to_play)
    ratings = ratings[ratings["plays"] > 0]

    user_item = ratings[["userId", "movieId", "plays"]]
    user_item.to_csv(out_path, index=False)

    print("user_item.csv created successfully at:", out_path)
    print(user_item.head())


if __name__ == "__main__":
    main()
