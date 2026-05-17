from flask import Flask, jsonify, render_template, request
import pandas as pd
import os
import re
import urllib.request
import zipfile

app = Flask(__name__)

# ── Load data once at startup ──────────────────────────────────────────────────

if not os.path.exists('ml-100k'):
    url = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
    urllib.request.urlretrieve(url, 'ml-100k.zip')
    with zipfile.ZipFile('ml-100k.zip', 'r') as zf:
        zf.extractall('.')

ratings = pd.read_csv('ml-100k/u.data', sep='\t',
                      names=['user_id', 'movie_id', 'rating', 'timestamp'])

GENRE_COLS = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
              'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
              'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

movies_full = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1', header=None)
movies_full.columns = ['movie_id', 'title', 'release_date', 'video', 'imdb', 'unknown'] + GENRE_COLS

movies_full['year'] = movies_full['release_date'].apply(
    lambda d: int(re.search(r'\d{4}', str(d)).group()) if pd.notna(d) and re.search(r'\d{4}', str(d)) else None
)
movies_full['decade'] = movies_full['year'].apply(
    lambda y: f"{int(y) // 10 * 10}s" if pd.notna(y) else None
)

# Per-movie stats
movie_stats = ratings.groupby('movie_id').agg(
    avg_rating=('rating', 'mean'),
    num_ratings=('rating', 'count')
).reset_index()
movie_stats['avg_rating'] = movie_stats['avg_rating'].round(2)

movies_merged = movies_full.merge(movie_stats, on='movie_id', how='left')
movies_merged['avg_rating'] = movies_merged['avg_rating'].fillna(0)
movies_merged['num_ratings'] = movies_merged['num_ratings'].fillna(0).astype(int)

print("Data loaded. Starting server at http://127.0.0.1:5000")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/discover')
def api_discover():
    genres = request.args.getlist('genres')
    decades = request.args.getlist('decades')
    min_stars = request.args.get('min_stars', 3, type=float)
    n = request.args.get('n', 24, type=int)

    df = movies_merged.copy()
    df = df[df['avg_rating'] >= min_stars]
    df = df[df['num_ratings'] >= 10]

    if decades:
        df = df[df['decade'].isin(decades)]

    for g in genres:
        if g in df.columns:
            df = df[df[g] == 1]

    if df.empty:
        return jsonify({'movies': [], 'total': 0})

    C = movies_merged['avg_rating'].mean()
    m_votes = 25
    df = df.copy()
    df['score'] = (
        (df['num_ratings'] / (df['num_ratings'] + m_votes)) * df['avg_rating'] +
        (m_votes / (df['num_ratings'] + m_votes)) * C
    )
    df = df.sort_values('score', ascending=False).head(n)

    result = []
    for _, row in df.iterrows():
        genres_list = [g for g in GENRE_COLS if row.get(g, 0) == 1]
        result.append({
            'movie_id': int(row['movie_id']),
            'title': row['title'],
            'year': int(row['year']) if pd.notna(row['year']) else None,
            'decade': row['decade'],
            'genres': genres_list,
            'avg_rating': float(row['avg_rating']),
            'num_ratings': int(row['num_ratings']),
            'score': round(float(row['score']), 3),
        })

    return jsonify({'movies': result, 'total': len(result)})


@app.route('/api/meta')
def api_meta():
    decades = sorted(
        [d for d in movies_full['decade'].dropna().unique() if d],
        key=lambda x: int(x[:-1])
    )
    return jsonify({
        'genres': GENRE_COLS,
        'decades': decades,
        'total_ratings': int(len(ratings)),
        'total_users': int(ratings['user_id'].nunique()),
        'total_movies': int(len(movies_full)),
    })


if __name__ == '__main__':
    app.run(debug=False, use_reloader=False, port=5000)
