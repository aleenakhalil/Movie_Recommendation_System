import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import requests


# TMDB API setup
TMDB_API_KEY = '1771e9f470d23c78c648f17730ba672f'  # Replace with your key

# Caching loaders
@st.cache_data
def load_data():
    ratings = pd.read_csv("ratings.dat", sep="::", names=["UserID", "MovieID", "Rating", "Timestamp"], engine='python', encoding='latin-1')
    movies = pd.read_csv("movies.dat", sep="::", names=["MovieID", "Title", "Genres"], engine='python', encoding='latin-1')
    users = pd.read_csv("users.dat", sep="::", names=["UserID", "Gender", "Age", "Occupation", "Zip-code"], engine='python', encoding='latin-1')
    return ratings, movies, users

@st.cache_data
def preprocess_content_based(movies):
    tfidf = TfidfVectorizer(token_pattern='[a-zA-Z0-9\-]+')
    tfidf_matrix = tfidf.fit_transform(movies['Genres'])
    similarity = cosine_similarity(tfidf_matrix)
    return similarity

@st.cache_data
def train_cf_model(ratings):
    user_item_matrix = ratings.pivot(index='UserID', columns='MovieID', values='Rating').fillna(0)
    svd = TruncatedSVD(n_components=20)
    reduced_matrix = svd.fit_transform(user_item_matrix)
    sim = cosine_similarity(reduced_matrix)
    return user_item_matrix, sim

# Fetch poster from TMDB
@st.cache_data
def get_movie_poster(title):
    query = title.split(' (')[0]  # Remove year for better match
    url = f'https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={query}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data['results']:
            poster_path = data['results'][0].get('poster_path')
            if poster_path:
                return f'https://image.tmdb.org/t/p/w500{poster_path}'
    return "https://via.placeholder.com/300x450?text=No+Image"

def recommend_content(movie_id, similarity, movies):
    idx = movies[movies['MovieID'] == movie_id].index[0]
    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    recommendations = movies.iloc[[i[0] for i in sim_scores]]
    return recommendations

def recommend_cf(user_id, ratings, user_item_matrix, sim_matrix, movies):
    if user_id not in user_item_matrix.index:
        return pd.DataFrame(columns=["MovieID", "Title"])
    user_idx = user_item_matrix.index.get_loc(user_id)
    sim_scores = list(enumerate(sim_matrix[user_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:4]
    
    top_users = [user_item_matrix.index[i[0]] for i in sim_scores]
    user_ratings = ratings[ratings['UserID'].isin(top_users)]
    top_movies = user_ratings.groupby('MovieID').mean()['Rating'].sort_values(ascending=False).head(5)
    return movies[movies['MovieID'].isin(top_movies.index)]

def recommend_hybrid(user_id, movie_id, content_sim, ratings, user_item_matrix, sim_matrix, movies, alpha=0.6):
    # Content-based similarity scores
    idx = movies[movies['MovieID'] == movie_id].index[0]
    cbf_scores = list(enumerate(content_sim[idx]))
    
    # Normalize
    cbf_df = pd.DataFrame(cbf_scores, columns=['index', 'cbf_score'])
    cbf_df['MovieID'] = movies.iloc[cbf_df['index']]['MovieID'].values
    cbf_df = cbf_df[['MovieID', 'cbf_score']]

    # Collaborative filtering scores (average ratings from similar users)
    if user_id not in user_item_matrix.index:
        cf_df = pd.DataFrame(columns=['MovieID', 'cf_score'])
    else:
        user_idx = user_item_matrix.index.get_loc(user_id)
        sim_scores = list(enumerate(sim_matrix[user_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:4]
        top_users = [user_item_matrix.index[i[0]] for i in sim_scores]

        user_ratings = ratings[ratings['UserID'].isin(top_users)]
        cf_scores = user_ratings.groupby('MovieID')['Rating'].mean().reset_index(name='cf_score')

        cf_df = cf_scores

    # Merge CF + CBF
    hybrid_df = pd.merge(cbf_df, cf_df, on='MovieID', how='outer').fillna(0)
    hybrid_df['score'] = alpha * hybrid_df['cf_score'] + (1 - alpha) * hybrid_df['cbf_score']
    hybrid_df = hybrid_df.sort_values(by='score', ascending=False).head(5)

    return movies[movies['MovieID'].isin(hybrid_df['MovieID'])]

# App Interface
st.set_page_config()
st.markdown(
    """
    <style>
        .stApp {
            background-color: #f8e8f4;
        }
        .stRadio > div {
            justify-content: center;
        }
    </style>
    """,
    unsafe_allow_html=True
)
st.title("🎬 Hybrid Movie Recommendation System")

ratings, movies, users = load_data()
content_sim = preprocess_content_based(movies)
user_item_matrix, sim_matrix = train_cf_model(ratings)

mode = st.radio("Select Recommendation Type", ["Content-Based", "Collaborative Filtering", "Hybrid"], horizontal=True)

def display_movies(movie_df):
    for i in range(0, len(movie_df), 3):
        cols = st.columns(3)
        for j in range(3):
            if i + j < len(movie_df):
                with cols[j]:
                    row = movie_df.iloc[i + j]
                    st.image(get_movie_poster(row['Title']), width=180)
                    st.caption(f"**{row['Title']}**")

if mode == "Content-Based":
    selected_movie = st.selectbox("Choose a Movie", movies['Title'].values)
    movie_id = movies[movies['Title'] == selected_movie]['MovieID'].values[0]
    recs = recommend_content(movie_id, content_sim, movies)
    st.subheader("Similar Movies:")
    display_movies(recs)

elif mode == "Collaborative Filtering":
    user_id = st.number_input("Enter your User ID (1-6040)", min_value=1, max_value=6040, step=1)
    recs = recommend_cf(user_id, ratings, user_item_matrix, sim_matrix, movies)
    st.subheader("Recommended Movies:")
    display_movies(recs)

elif mode == "Hybrid":
    user_id = st.number_input("Enter your User ID (1-6040)", min_value=1, max_value=6040, step=1)
    selected_movie = st.selectbox("Choose a Movie You Liked", movies['Title'].values)
    movie_id = movies[movies['Title'] == selected_movie]['MovieID'].values[0]
    recs = recommend_hybrid(user_id, movie_id, content_sim, ratings, user_item_matrix, sim_matrix, movies)
    st.subheader("Hybrid Recommendations:")
    display_movies(recs)
