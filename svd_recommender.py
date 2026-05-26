from __future__ import annotations

import os
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD

from .paths import dataset_dir, models_dir


def load_user_item(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV not found at {csv_path}")
        print("Make sure dataset/user_item.csv exists (run: python -m movie_recom.create_user_item).")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    col_map: dict[str, str] = {}

    if "user_id" in df.columns:
        col_map["user"] = "user_id"
    elif "userId" in df.columns:
        col_map["user"] = "userId"
    elif "user" in df.columns:
        col_map["user"] = "user"
    else:
        print("ERROR: Could not find a user id column. Expected one of: user_id, userId, user")
        sys.exit(1)

    if "movie_id" in df.columns:
        col_map["item"] = "movie_id"
    elif "movieId" in df.columns:
        col_map["item"] = "movieId"
    elif "movie" in df.columns:
        col_map["item"] = "movie"
    else:
        print("ERROR: Could not find a movie id column. Expected one of: movie_id, movieId, movie")
        sys.exit(1)

    if "plays" in df.columns:
        col_map["plays"] = "plays"
    elif "play" in df.columns:
        col_map["plays"] = "play"
    elif "rating" in df.columns:
        df["plays"] = df["rating"].apply(lambda r: 3 if r >= 4.0 else (1 if r >= 3.0 else 0))
        df = df[df["plays"] > 0]
        col_map["plays"] = "plays"
    else:
        print("ERROR: Could not find 'plays' or 'rating' column.")
        sys.exit(1)

    df2 = df[[col_map["user"], col_map["item"], col_map["plays"]]].copy()
    df2.columns = ["user_id", "movie_id", "plays"]
    return df2


class SVDRecommender:
    def __init__(
        self,
        user_item_csv: str | None = None,
        model_path: str | None = None,
        n_factors: int = 64,
    ):
        user_item_csv = user_item_csv or str(dataset_dir() / "user_item.csv")
        model_path = model_path or str(models_dir() / "svd_model.joblib")

        self.df = load_user_item(user_item_csv)
        self.user_ids = list(pd.Categorical(self.df["user_id"]).categories)
        self.item_ids = list(pd.Categorical(self.df["movie_id"]).categories)

        user_codes = pd.Categorical(self.df["user_id"]).codes
        item_codes = pd.Categorical(self.df["movie_id"]).codes
        data = self.df["plays"].astype(float).values

        self.num_users = len(self.user_ids)
        self.num_items = len(self.item_ids)
        self.user_item = coo_matrix(
            (data, (user_codes, item_codes)),
            shape=(self.num_users, self.num_items),
        ).tocsr()

        self.model_path = model_path
        self.n_factors = n_factors

        if os.path.exists(model_path):
            self.user_factors, self.item_factors = joblib.load(model_path)
        else:
            n_comp = min(self.n_factors, max(1, self.num_items - 1))
            item_user = self.user_item.T.tocsc()
            svd = TruncatedSVD(n_components=n_comp, n_iter=10, random_state=42)
            item_factors = svd.fit_transform(item_user)
            user_factors = self.user_item.dot(item_factors)
            self.user_factors = user_factors
            self.item_factors = item_factors
            joblib.dump((self.user_factors, self.item_factors), model_path)

    def recommend(self, user_id, N: int = 10):
        if user_id not in self.user_ids:
            return []

        uidx = self.user_ids.index(user_id)
        uvec = self.user_factors[uidx]
        scores = self.item_factors.dot(uvec)

        user_row = self.user_item[uidx].toarray().flatten()
        seen_idx = user_row.nonzero()[0]
        if seen_idx.size > 0:
            scores[seen_idx] = -np.inf

        k = min(N, max(0, len(scores)))
        if k == 0:
            return []

        top_idx = np.argpartition(-scores, range(min(k, len(scores) - 1)))[:k]
        top_idx_sorted = top_idx[np.argsort(-scores[top_idx])]
        return [(self.item_ids[i], float(scores[i])) for i in top_idx_sorted]


if __name__ == "__main__":
    rec = SVDRecommender()
    sample_user = rec.user_ids[0]
    print("Sample user:", sample_user)
    print(rec.recommend(sample_user, N=10))
