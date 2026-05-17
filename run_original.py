from flask import Flask, jsonify, render_template, request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os
import urllib.request
import zipfile

app = Flask(__name__)

if not os.path.exists('ml-100k'):
    url = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
    urllib.request.urlretrieve(url, 'ml-100k.zip')
    with zipfile.ZipFile('ml-100k.zip', 'r') as zf:
        zf.extractall('.')

ratings = pd.read_csv('ml-100k/u.data', sep='\t',
                      names=['user_id', 'movie_id', 'rating', 'timestamp'])

movies = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1',
                     names=['movie_id', 'title', 'release_date', 'video', 'imdb'],
                     usecols=range(5))

genre_cols = ['unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
              'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
              'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

movies_full = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1', header=None)
movies_full.columns = ['movie_id', 'title', 'release_date', 'video', 'imdb'] + genre_cols

user_movie_matrix = ratings.pivot_table(
    index='user_id', columns='movie_id', values='rating'
).fillna(0)

user_similarity = cosine_similarity(user_movie_matrix)
user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_movie_matrix.index,
    columns=user_movie_matrix.index
)

print("Data loaded. Starting server...")


def get_movie_genres(movie_id):
    row = movies_full[movies_full['movie_id'] == movie_id]
    if row.empty:
        return []
    return [g for g in genre_cols if row.iloc[0][g] == 1]


def recommend_movies(user_id, n=10):
    user_rated = ratings[ratings['user_id'] == user_id]['movie_id'].tolist()
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)
    similar_users = similar_users[similar_users.index != user_id].head(20)

    movie_scores = {}
    for sim_user in similar_users.index:
        sim_score = similar_users[sim_user]
        sim_ratings = ratings[(ratings['user_id'] == sim_user) & (ratings['rating'] >= 4)]
        for _, row in sim_ratings.iterrows():
            mid = row['movie_id']
            if mid not in user_rated:
                movie_scores[mid] = movie_scores.get(mid, 0) + sim_score

    top = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)[:n]
    result = []
    for mid, score in top:
        title_row = movies[movies['movie_id'] == mid]
        if not title_row.empty:
            result.append({
                'movie_id': int(mid),
                'title': title_row.iloc[0]['title'],
                'score': round(float(score), 3),
                'genres': get_movie_genres(mid),
            })
    return result


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/recommend/<int:user_id>')
def api_recommend(user_id):
    if user_id not in user_movie_matrix.index:
        return jsonify({'error': f'User {user_id} not found. Try 1-943.'}), 404

    n = request.args.get('n', 10, type=int)
    user_ratings = ratings[ratings['user_id'] == user_id]
    avg = round(float(user_ratings['rating'].mean()), 2)
    count = int(len(user_ratings))

    top_rated = []
    for _, row in user_ratings.sort_values('rating', ascending=False).head(5).iterrows():
        t = movies[movies['movie_id'] == row['movie_id']]
        if not t.empty:
            top_rated.append({'title': t.iloc[0]['title'], 'rating': int(row['rating'])})

    recs = recommend_movies(user_id, n)
    return jsonify({
        'user_id': user_id,
        'stats': {'rated': count, 'avg_rating': avg},
        'top_rated': top_rated,
        'recommendations': recs,
    })


@app.route('/api/stats')
def api_stats():
    return jsonify({
        'total_ratings': int(len(ratings)),
        'total_users': int(ratings['user_id'].nunique()),
        'total_movies': int(len(movies)),
    })


if __name__ == '__main__':
    app.run(debug=False, use_reloader=False, port=5000)
