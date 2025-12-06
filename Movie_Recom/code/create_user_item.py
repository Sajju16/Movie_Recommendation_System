import pandas as pd

# Load ratings.csv from the MovieLens dataset folder
ratings = pd.read_csv("../dataset/ratings.csv")

# Convert explicit rating → implicit plays score
# Simple mapping:
# rating >= 4.0 → 3 plays
# rating >= 3.0 → 1 play
# else → 0 plays (ignore)
def rating_to_play(r):
    if r >= 4.0:
        return 3
    elif r >= 3.0:
        return 1
    else:
        return 0

ratings['plays'] = ratings['rating'].apply(rating_to_play)

# Remove rows where plays = 0
ratings = ratings[ratings['plays'] > 0]

# Keep only required columns
user_item = ratings[['userId', 'movieId', 'plays']]

# Save file
user_item.to_csv("../dataset/user_item.csv", index=False)

print("user_item.csv created successfully!")
print(user_item.head())
