from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import requests
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import transformers
import sentence_transformers
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# New imports for fuzzy searching
from rapidfuzz import process, fuzz

app = Flask(__name__)

data = pd.read_csv('imdb_movies.csv')

movies = data[['names', 'date_x', 'score', 'genre', 'overview', 'crew', 'orig_lang']]

def create_tags(row):
    tags = []
    tags.append(str(row['genre']))
    tags.append(str(row['overview']))
    tags.append(str(row['crew']))
    return " ".join(tags)

movies['tags'] = movies.apply(create_tags, axis=1)
movies['tags'] = movies['tags'].str.lower().str.replace('[^\w\s]', '', regex=True)
movies = movies.dropna(subset=['tags']).reset_index(drop=True)
movies['tags'] = movies['tags'].astype(str)

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess_text(text):
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

movies['tags'] = movies['tags'].apply(preprocess_text)

# Create a lowercase column for movie names for case-insensitive matching
movies['names_lower'] = movies['names'].str.lower()

# Load the Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

tags_list = movies['tags'].tolist()
embeddings = model.encode(tags_list, convert_to_numpy=True, show_progress_bar=True)

orig_lang_encoded = pd.get_dummies(movies['orig_lang'], prefix='lang')
orig_lang_array = orig_lang_encoded.values

embeddings_combined = np.hstack((embeddings, orig_lang_array))

cosine_sim_matrix = cosine_similarity(embeddings_combined)

API_KEY = 'ad91a5a135df4c09d36b5739aaa31242'
BASE_URL = 'https://api.themoviedb.org/3'
IMAGE_BASE_URL = 'https://image.tmdb.org/t/p/w500'


def search_movie(movie_name, year=None):
    url = f"{BASE_URL}/search/movie"
    params = {
        'api_key': API_KEY,
        'query': movie_name
    }
    if year:
        params['year'] = year
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if data.get('results'):
            # Try exact match first
            for result in data['results']:
                if result['title'].lower() == movie_name.lower():
                    return result
            # If no exact match, return the first result
            return data['results'][0]
        else:
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching movie data: {e}")
        return None


def get_poster_url(poster_path):
    if poster_path:
        return f"{IMAGE_BASE_URL}{poster_path}"
    else:
        return None


def recommend_movies_by_index(idx, top_n=5):
    sim_scores = list(enumerate(cosine_sim_matrix[idx]))
    sim_scores = [(i, float(score)) for i, score in sim_scores]
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices]


def recommend_movies_with_posters(idx, top_n=5):
    recommended_movies = recommend_movies_by_index(idx, top_n)
    poster_urls = []
    for _, row in recommended_movies.iterrows():
        movie_name = row['names']
        release_date = row['date_x']
        if pd.isnull(release_date) or release_date == '':
            year = None
        else:
            try:
                year = str(pd.to_datetime(release_date).year)
            except:
                year = None
        movie_info = search_movie(movie_name, year)
        if movie_info:
            poster_path = movie_info.get('poster_path')
            poster_url = get_poster_url(poster_path)
            poster_urls.append(poster_url)
        else:
            poster_urls.append(None)
    recommended_movies = recommended_movies.copy()
    recommended_movies['poster_url'] = poster_urls
    return recommended_movies


def get_recommendations(movie_name, top_n=5):
    movie_name_lower = movie_name.lower().strip()

    # Use fuzzy search to find the closest title match in movies['names_lower']
    best_match = process.extractOne(movie_name_lower, movies['names_lower'].values, scorer=fuzz.WRatio)

    # Check if we got a good match
    # 'best_match' is a tuple: (matched_string, score, index)
    # 'score' ranges from 0 to 100; adjust threshold as needed
    if best_match and best_match[1] > 70:
        matched_name = best_match[0]
        # Find index of the matched movie
        idx = movies.index[movies['names_lower'] == matched_name][0]
        recommendations = recommend_movies_with_posters(idx, top_n)
        result = []
        for _, row in recommendations.iterrows():
            result.append({
                'name': row['names'],
                'poster_url': row['poster_url']
            })
        return result
    else:
        return None


@app.route('/recommend', methods=['GET'])
def recommend():
    movie_name = request.args.get('movie')
    if movie_name:
        recommendations = get_recommendations(movie_name)
        if recommendations:
            return jsonify(recommendations)
        else:
            return jsonify({'error': 'Movie not found in the database.'}), 404
    else:
        return jsonify({'error': 'No movie title provided.'}), 400


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    # Remember to run: pip install rapidfuzz before starting
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
