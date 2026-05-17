# ============================================
# MOVIE RECOMMENDATION SYSTEM
# Dataset: MovieLens 100K (Downloads automatically)
# ============================================

import pandas as pd
import numpy as np
import urllib.request
import zipfile
import os
from sklearn.metrics.pairwise import cosine_similarity

print("=" * 60)
print("MOVIE RECOMMENDATION SYSTEM")
print("=" * 60)

# ============================================
# STEP 1: DOWNLOAD AND EXTRACT DATA (AUTOMATIC)
# ============================================

print("\n[STEP 1] Getting MovieLens dataset...")

if not os.path.exists('ml-100k'):
    print("  Downloading dataset (may take 1 minute)...")
    url = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
    urllib.request.urlretrieve(url, 'ml-100k.zip')
    
    print("  Extracting...")
    with zipfile.ZipFile('ml-100k.zip', 'r') as zip_ref:
        zip_ref.extractall('.')
    print("  Done!")
else:
    print("  Dataset already downloaded!")

# ============================================
# STEP 2: LOAD RATINGS AND MOVIES
# ============================================

print("\n[STEP 2] Loading ratings and movies...")

# Load ratings
ratings = pd.read_csv('ml-100k/u.data', sep='\t',
                       names=['user_id', 'movie_id', 'rating', 'timestamp'])

# Load movie titles
movies = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1',
                      names=['movie_id', 'title', 'release_date', 'video', 'imdb'],
                      usecols=range(5))

print(f"  Total ratings: {len(ratings):,}")
print(f"  Total users: {ratings['user_id'].nunique()}")
print(f"  Total movies: {len(movies)}")
print(f"  Rating range: {ratings['rating'].min()} to {ratings['rating'].max()}")

# ============================================
# STEP 3: CREATE USER-ITEM MATRIX
# ============================================

print("\n[STEP 3] Creating user-item matrix...")

# Create matrix (users x movies)
user_movie_matrix = ratings.pivot_table(
    index='user_id',
    columns='movie_id',
    values='rating'
).fillna(0)

print(f"  Matrix: {user_movie_matrix.shape[0]} users x {user_movie_matrix.shape[1]} movies")

# ============================================
# STEP 4: CALCULATE USER SIMILARITY
# ============================================

print("\n[STEP 4] Calculating user similarities...")

# Calculate cosine similarity between users
user_similarity = cosine_similarity(user_movie_matrix)
user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_movie_matrix.index,
    columns=user_movie_matrix.index
)

print(f"  Similarity matrix: {user_similarity_df.shape}")

# ============================================
# STEP 5: RECOMMENDATION FUNCTION
# ============================================

def recommend_movies(user_id, num_recommendations=10):
    """
    Recommend movies for a given user
    """
    # Get movies this user already rated
    user_rated = ratings[ratings['user_id'] == user_id]['movie_id'].tolist()
    
    # Find similar users
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)
    similar_users = similar_users[similar_users.index != user_id].head(20)
    
    # Get movies rated highly by similar users
    movie_scores = {}
    
    for sim_user in similar_users.index:
        sim_score = similar_users[sim_user]
        
        # Get this similar user's high ratings (4 or 5)
        sim_ratings = ratings[(ratings['user_id'] == sim_user) & (ratings['rating'] >= 4)]
        
        for _, row in sim_ratings.iterrows():
            movie_id = row['movie_id']
            
            # Only recommend movies the user hasn't seen
            if movie_id not in user_rated:
                if movie_id not in movie_scores:
                    movie_scores[movie_id] = 0
                movie_scores[movie_id] += sim_score
    
    # Sort by score and get top movies
    sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
    top_movies = sorted_movies[:num_recommendations]
    
    # Get movie titles
    recommendations = []
    for movie_id, score in top_movies:
        title = movies[movies['movie_id'] == movie_id]['title'].values
        if len(title) > 0:
            recommendations.append((movie_id, title[0], score))
    
    return recommendations

# ============================================
# STEP 6: TEST WITH SAMPLE USERS
# ============================================

print("\n[STEP 5] Testing recommendations...")
print("-" * 50)

# Test user 1
test_user = 1
print(f"\nUser {test_user} - What they rated:")

# Show what this user rated
user_ratings = ratings[ratings['user_id'] == test_user].sort_values('rating', ascending=False)
for i, (idx, row) in enumerate(user_ratings.head(5).iterrows()):
    movie_title = movies[movies['movie_id'] == row['movie_id']]['title'].values
    if len(movie_title) > 0:
        print(f"  {i+1}. {movie_title[0]} (Rating: {row['rating']})")

# Get recommendations
print(f"\nRecommended movies for User {test_user}:")
recommendations = recommend_movies(test_user, 10)

for i, (movie_id, title, score) in enumerate(recommendations, 1):
    print(f"  {i}. {title}")

# ============================================
# STEP 7: EVALUATE WITH PRECISION@K
# ============================================

print("\n[STEP 6] Evaluating recommendation quality...")
print("-" * 50)

def precision_at_k(user_id, k=10):
    """
    Calculate Precision@K for a user
    """
    # Split user's ratings into train (80%) and test (20%)
    user_ratings = ratings[ratings['user_id'] == user_id]
    
    if len(user_ratings) < 10:
        return None
    
    # Randomly split
    test_size = max(1, int(len(user_ratings) * 0.2))
    test_movies = user_ratings.sample(n=test_size, random_state=42)['movie_id'].tolist()
    
    # Get recommendations (excluding training movies)
    # We want to allow recommending test movies
    user_rated_train = [m for m in user_ratings['movie_id'].tolist() if m not in test_movies]
    
    # Find similar users
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)
    similar_users = similar_users[similar_users.index != user_id].head(20)
    
    movie_scores = {}
    for sim_user in similar_users.index:
        sim_score = similar_users[sim_user]
        sim_high_rated = ratings[(ratings['user_id'] == sim_user) & (ratings['rating'] >= 4)]
        
        for _, row in sim_high_rated.iterrows():
            movie_id = row['movie_id']
            if movie_id not in user_rated_train:
                if movie_id not in movie_scores:
                    movie_scores[movie_id] = 0
                movie_scores[movie_id] += sim_score
    
    # Top K recommendations
    top_k = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)[:k]
    recommended_movies = [m[0] for m in top_k]
    
    # Calculate precision
    relevant = len([m for m in recommended_movies if m in test_movies])
    return relevant / k if k > 0 else 0

# Calculate average precision
print("Calculating Precision@10 for 10 users...")
precisions = []

for user_id in range(1, 11):
    prec = precision_at_k(user_id, k=10)
    if prec is not None:
        precisions.append(prec)
        print(f"  User {user_id}: Precision@10 = {prec:.4f} ({prec*100:.1f}%)")

avg_precision = np.mean(precisions) if precisions else 0
print(f"\n  AVERAGE PRECISION@10: {avg_precision:.4f} ({avg_precision*100:.2f}%)")

# ============================================
# STEP 8: INTERACTIVE MODE
# ============================================

print("\n" + "=" * 60)
print("INTERACTIVE RECOMMENDATION MODE")
print("=" * 60)

try:
    choice = input("\nDo you want to get recommendations for a specific user? (y/n): ").strip().lower()
    
    if choice == 'y':
        user_id = int(input("Enter user ID (1-943): "))
        
        if user_id in user_movie_matrix.index:
            # Show user stats
            user_rating_count = len(ratings[ratings['user_id'] == user_id])
            user_avg_rating = ratings[ratings['user_id'] == user_id]['rating'].mean()
            
            print(f"\nUser {user_id} Statistics:")
            print(f"  Movies rated: {user_rating_count}")
            print(f"  Average rating: {user_avg_rating:.2f}")
            
            # Get recommendations
            recs = recommend_movies(user_id, 10)
            
            print(f"\nTop 10 Recommended Movies for User {user_id}:")
            for i, (movie_id, title, score) in enumerate(recs, 1):
                print(f"  {i}. {title}")
        else:
            print(f"User {user_id} not found. Try 1-943")
except:
    pass

# ============================================
# FINAL SUMMARY
# ============================================

print("\n" + "=" * 60)
print("PROJECT COMPLETION SUMMARY")
print("=" * 60)

print(f"""
[TASK 5 COMPLETE]: Movie Recommendation System

[DATASET]
  Name: MovieLens 100K
  Source: GroupLens (downloaded automatically)
  Ratings: {len(ratings):,}
  Users: {ratings['user_id'].nunique()}
  Movies: {len(movies)}

[ALGORITHM]
  Type: User-based Collaborative Filtering
  Similarity: Cosine Similarity

[PERFORMANCE]
  Average Precision@10: {avg_precision:.4f} ({avg_precision*100:.2f}%)

[HOW TO USE]
  - The code automatically downloads the dataset
  - No manual data entry needed
  - Recommendations based on similar users' ratings
""")

print("=" * 60)
print("READY! Run 'python movie_recommender.py'")
print("=" * 60)
