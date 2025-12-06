# als_fallback_svd.py  (robust version)
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD
import joblib
import os
import sys

CSV_PATH = "../dataset/user_item.csv"
MODEL_PATH = "svd_model.joblib"
N_FACTORS = 64

def load_user_item(csv_path):
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV not found at {csv_path}")
        print("Make sure dataset/user_item.csv exists (run create_user_item.py if not).")
        sys.exit(1)
    df = pd.read_csv(csv_path)
    print("Loaded user_item.csv — columns:", list(df.columns))
    # Accept multiple possible column name variants
    col_map = {}
    if 'user_id' in df.columns:
        col_map['user'] = 'user_id'
    elif 'userId' in df.columns:
        col_map['user'] = 'userId'
    elif 'user' in df.columns:
        col_map['user'] = 'user'
    else:
        print("ERROR: Could not find a user id column. Expected one of: user_id, userId, user")
        sys.exit(1)

    if 'movie_id' in df.columns:
        col_map['item'] = 'movie_id'
    elif 'movieId' in df.columns:
        col_map['item'] = 'movieId'
    elif 'movie' in df.columns:
        col_map['item'] = 'movie'
    else:
        print("ERROR: Could not find a movie id column. Expected one of: movie_id, movieId, movie")
        sys.exit(1)

    if 'plays' in df.columns:
        col_map['plays'] = 'plays'
    elif 'play' in df.columns:
        col_map['plays'] = 'play'
    elif 'rating' in df.columns:
        # fallback: convert rating -> plays (simple heuristic)
        print("Note: 'plays' column not found. Falling back to using 'rating' -> plays mapping.")
        col_map['plays'] = 'rating'
        df['plays'] = df['rating'].apply(lambda r: 3 if r >= 4.0 else (1 if r >= 3.0 else 0))
        df = df[df['plays'] > 0]
    else:
        print("ERROR: Could not find 'plays' or 'rating' column.")
        sys.exit(1)

    # keep only necessary columns and rename
    df2 = df[[col_map['user'], col_map['item'], col_map['plays']]].copy()
    df2.columns = ['user_id', 'movie_id', 'plays']
    return df2

class SVDRecommender:
    def __init__(self, user_item_csv=CSV_PATH, model_path=MODEL_PATH, n_factors=N_FACTORS):
        print("Initializing SVDRecommender...")
        self.df = load_user_item(user_item_csv)
        # map ids to indices
        self.user_ids = list(pd.Categorical(self.df['user_id']).categories)
        self.item_ids = list(pd.Categorical(self.df['movie_id']).categories)
        user_codes = pd.Categorical(self.df['user_id']).codes
        item_codes = pd.Categorical(self.df['movie_id']).codes
        data = self.df['plays'].astype(float).values
        self.num_users = len(self.user_ids)
        self.num_items = len(self.item_ids)
        # build sparse matrix
        self.user_item = coo_matrix((data, (user_codes, item_codes)), shape=(self.num_users, self.num_items)).tocsr()

        self.model_path = model_path
        self.n_factors = n_factors
        if os.path.exists(model_path):
            print("Loading existing model:", model_path)
            self.user_factors, self.item_factors = joblib.load(model_path)
        else:
            print("Training SVD model... (this may take a moment)")
            # safe n_components
            n_comp = min(self.n_factors, max(1, self.num_items - 1))
            item_user = self.user_item.T.tocsc()
            svd = TruncatedSVD(n_components=n_comp, n_iter=10, random_state=42)
            item_factors = svd.fit_transform(item_user)  # (num_items, n_comp)
            user_factors = (self.user_item.dot(item_factors))  # shape (num_users, n_comp)
            self.user_factors = user_factors
            self.item_factors = item_factors
            joblib.dump((self.user_factors, self.item_factors), model_path)
            print("Model saved to", model_path)

    def recommend(self, user_id, N=10):
        # user_id must match the dtype stored in CSV (int/str)
        if user_id not in self.user_ids:
            print(f"Unknown user_id: {user_id} — available sample users: {self.user_ids[:5]}")
            return []
        uidx = self.user_ids.index(user_id)
        uvec = self.user_factors[uidx]  # (n_factors,)
        scores = self.item_factors.dot(uvec)
        # mask already seen
        user_row = self.user_item[uidx].toarray().flatten()
        seen_idx = user_row.nonzero()[0]
        if seen_idx.size > 0:
            scores[seen_idx] = -np.inf
        top_idx = np.argpartition(-scores, range(min(N, len(scores)-1)))[:N]
        top_idx_sorted = top_idx[np.argsort(-scores[top_idx])]
        recs = [(self.item_ids[i], float(scores[i])) for i in top_idx_sorted]
        return recs

if __name__ == "__main__":
    rec = SVDRecommender()
    # show first 5 users
    print("Sample users (first 5):", rec.user_ids[:5])
    # try recommend for first user
    sample_user = rec.user_ids[0]
    print("Generating sample recommendations for user:", sample_user)
    print(rec.recommend(sample_user, N=10))
