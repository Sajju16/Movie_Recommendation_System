# content_rec.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import joblib
import os

class ContentReRanker:
    def __init__(self, movies_csv="movies.csv", model_path="content_tfidf.joblib"):
        self.movies = pd.read_csv(movies_csv)
        self.movies['meta'] = self.movies['title'].fillna('') + ' ' + self.movies['genres'].fillna('')
        self.model_path = model_path
        if os.path.exists(model_path):
            self.tfidf = joblib.load(model_path)
            self.tfidf_matrix = joblib.load(model_path + ".mat")
        else:
            self.tfidf = TfidfVectorizer(stop_words='english', max_features=15000)
            self.tfidf_matrix = self.tfidf.fit_transform(self.movies['meta'])
            joblib.dump(self.tfidf, model_path)
            joblib.dump(self.tfidf_matrix, model_path + ".mat")

    def score_candidates(self, candidate_movie_ids, liked_movie_ids):
        # returns dict movieId -> score (higher better)
        idx_map = {m: i for i,m in enumerate(self.movies['movieId'].tolist())}
        liked_idxs = [idx_map[m] for m in liked_movie_ids if m in idx_map]
        cand_idxs = [idx_map[m] for m in candidate_movie_ids if m in idx_map]
        if not liked_idxs or not cand_idxs:
            return {m: 0.0 for m in candidate_movie_ids}
        liked_vec = self.tfidf_matrix[liked_idxs].mean(axis=0)
        sims = linear_kernel(liked_vec, self.tfidf_matrix[cand_idxs]).flatten()
        return {candidate_movie_ids[i]: float(sims[i]) for i in range(len(cand_idxs))}
