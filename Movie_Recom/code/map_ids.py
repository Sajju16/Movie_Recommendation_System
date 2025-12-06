# map_ids.py
import pandas as pd

def load_movie_map(movies_csv="../dataset/movies.csv"):
    df = pd.read_csv(movies_csv)
    # ensure movieId column exists (MovieLens uses movieId)
    df = df.rename(columns=lambda c: c.strip())
    id_to_title = dict(zip(df['movieId'], df['title']))
    return id_to_title

if __name__ == "__main__":
    m = load_movie_map()
    # example
    print(m.get(318, "Unknown"))
