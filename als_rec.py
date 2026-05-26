# als_rec.py
import pandas as pd
from scipy.sparse import coo_matrix
import implicit
import joblib
import numpy as np

class ALSRecommender:
    def __init__(self, user_item_csv="user_item.csv", model_path="als_model.joblib"):
        self.df = pd.read_csv(user_item_csv)
        # map categories
        self.user_cat = pd.Categorical(self.df['user_id'])
        self.item_cat = pd.Categorical(self.df['movie_id'])
        rows = self.user_cat.codes
        cols = self.item_cat.codes
        data = self.df['plays'].astype(float).values
        self.user_item = coo_matrix((data, (rows, cols)),
                                   shape=(len(self.user_cat.categories), len(self.item_cat.categories)))
        self.model_path = model_path
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
        else:
            self.model = implicit.als.AlternatingLeastSquares(factors=64, regularization=0.1, iterations=20)
            # implicit expects item-user
            item_user = self.user_item.T.tocsr()
            # apply confidence scaling externally if desired
            self.model.fit(item_user)
            joblib.dump(self.model, model_path)

    def recommend(self, user_id, N=10):
        try:
            uidx = list(self.user_cat.categories).index(user_id)
        except ValueError:
            return []  # cold-start
        recs = self.model.recommend(userid=uidx, user_items=self.user_item.tocsr(), N=N)
        movie_ids = [self.item_cat.categories[i] for i,score in recs]
        scores = [float(s) for i,s in recs]
        return list(zip(movie_ids, scores))
