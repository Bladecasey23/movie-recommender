import streamlit as st
st.set_page_config(page_title="Movie Recommender 🎬", page_icon="🎥", layout="centered")

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import requests

# Load API key from Streamlit secrets
TMDB_API_KEY = st.secrets["TMDB_API_KEY"]

@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    movies['title'] = movies['title'].str.strip()  # Clean titles
    df = pd.merge(ratings, movies, on="movieId")
    user_movie_matrix = df.pivot_table(index="userId", columns="title", values="rating").fillna(0)
    similarity = cosine_similarity(user_movie_matrix.T)
    similarity_df = pd.DataFrame(similarity, index=user_movie_matrix.columns, columns=user_movie_matrix.columns)
    return movies, similarity_df

def fetch_poster(movie_title):
    try:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_title}"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data.get("results"):
            poster_path = data["results"][0].get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except Exception as e:
        print(f"Poster fetch error for '{movie_title}': {e}")
    return "https://via.placeholder.com/150x225?text=No+Image"

def recommend(movie, similarity_df):
    if movie not in similarity_df.index:
        st.warning(f"No recommendations found for '{movie}'. Please choose another movie.")
        return []
    sim_scores = similarity_df[movie].sort_values(ascending=False)[1:6]
    return sim_scores.index.tolist()

# Load data once
movies, similarity_df = load_data()

st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🎬 Movie Recommendation System</h1>", unsafe_allow_html=True)

selected_movie = st.selectbox("Select a movie you like:", movies["title"].unique())

if st.button("Recommend"):
    with st.spinner("Fetching recommendations..."):
        try:
            recommendations = recommend(selected_movie, similarity_df)
            if not recommendations:
                st.error("No recommendations found.")
            else:
                cols = st.columns(5)
                for i, movie in enumerate(recommendations):
                    with cols[i]:
                        st.image(fetch_poster(movie), width=150)
                        st.caption(movie)
        except Exception as e:
            st.error(f"An error occurred: {e}")
