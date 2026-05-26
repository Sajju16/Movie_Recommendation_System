from __future__ import annotations

import pandas as pd

from .paths import dataset_dir


def load_movie_map(movies_csv: str | None = None) -> dict[int, str]:
    path = movies_csv or str(dataset_dir() / "movies.csv")
    df = pd.read_csv(path)
    df = df.rename(columns=lambda c: c.strip())
    return dict(zip(df["movieId"], df["title"]))


if __name__ == "__main__":
    m = load_movie_map()
    print(m.get(318, "Unknown"))
