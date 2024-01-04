# app.py

from flask import Flask, render_template, request
import requests
import pandas as pd
import numpy as np
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)


def load_movie_data():
    movie = pd.read_csv("movies.csv")
    return movie


def preprocess_data(movie):
    col = ['genres', 'keywords', 'director', 'tagline', 'cast']
    for i in col:
        movie[i].fillna("", inplace=True)
    return movie


def calculate_similarity(movie):
    combined_data = movie['genres'] + movie['keywords'] + movie['director'] + movie['tagline'] + movie['cast']
    vectorizer = TfidfVectorizer()
    features_vectors = vectorizer.fit_transform(combined_data)
    similarity = cosine_similarity(features_vectors)
    return similarity


def find_recommendations(user_movie, movie, similarity):
    all_movies = movie['title'].to_list()
    close_match_movie = difflib.get_close_matches(user_movie, all_movies)
    if not close_match_movie:
        return []

    matched_movie = close_match_movie[0]
    matched_movie_index = all_movies.index(matched_movie)
    similar_score = list(enumerate(similarity[matched_movie_index]))

    sorted_similar_score = sorted(similar_score, key=lambda x: x[1], reverse=True)
    movie['release_date'] = movie['release_date'].str[:4]
    recommended_movies = []

    for i in range(6):
        m = movie[movie["title"] == all_movies[sorted_similar_score[i][0]]]['release_date'].values[0]
        recommended_movies.append((m, all_movies[sorted_similar_score[i][0]]))

    return recommended_movies


def get_poster_urls(recommendations):
    poster_url = []
    for i in range(len(recommendations)):
        response = requests.get('https://www.omdbapi.com/?s=' + recommendations[i][1] + '&apikey=855c3c5f')
        if response.status_code == 200:
            movie_json = response.json()

            if 'Search' in movie_json and len(movie_json['Search']) > 0:
                poster_url.append(movie_json['Search'][0]['Poster'])
            else:
                poster_url.append("No poster available")
        else:
            poster_url.append(f"Error: {response.status_code}")

    return poster_url


@app.route('/')
def index():
    return render_template('home.html')


@app.route('/recommend', methods=['POST'])
def process_input():
    user_input = request.form['movie']
    movie_data = load_movie_data()
    preprocess_data(movie_data)
    similarity_matrix = calculate_similarity(movie_data)

    recommendations = find_recommendations(user_input, movie_data, similarity_matrix)

    if not recommendations:
        return render_template('no_recommendations.html')

    poster_urls = get_poster_urls(recommendations)
    return render_template('result.html', data=recommendations, poster_url=poster_urls)


if __name__ == '__main__':
    app.run(debug=True)
