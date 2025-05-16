import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import requests

st.set_page_config(page_title="Movie Recommender 🎬", page_icon="🎥", layout="centered")

# ====== Config ======
TMDB_API_KEY = "182950a3164a9f4ae4e52052982bc88e"  # <-- Put your TMDB API key here

# ====== Functions ======

def fetch_poster(movie_title):
    """Fetch movie poster URL from TMDB API"""
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_title}"
    response = requests.get(url)
    data = response.json()
    if data.get("results"):
        poster_path = data["results"][0].get("poster_path")
        if poster_path:
            full_path = f"https://image.tmdb.org/t/p/w500{poster_path}"
            return full_path
    return None

def recommend(movie_name, similarity_df):
    if movie_name in similarity_df:
        # Get top 5 similar movies excluding the movie itself
        return similarity_df[movie_name].sort_values(ascending=False)[1:6].index.tolist()
    else:
        return []

# ====== Load Data ======

@st.cache_data
def load_data():
    movies = pd.read_csv('movies.csv')
    ratings = pd.read_csv('ratings.csv')
    df = pd.merge(ratings, movies, on='movieId')
    user_movie_matrix = df.pivot_table(index='userId', columns='title', values='rating').fillna(0)
    similarity = cosine_similarity(user_movie_matrix.T)
    similarity_df = pd.DataFrame(similarity, index=user_movie_matrix.columns, columns=user_movie_matrix.columns)
    return movies, similarity_df

movies, similarity_df = load_data()

# ====== Streamlit UI ======


st.markdown("""
    <style>
    .title {
        font-size:40px;
        color:#00b4d8;
        text-align:center;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .recommend-box {
        background-color: #0077b6;
        padding: 12px 20px;
        border-radius: 12px;
        margin: 12px 0;
        font-size: 18px;
        color: white;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🎬 Movie Recommendation System</div>', unsafe_allow_html=True)
st.write("### Choose a movie you like and get similar suggestions:")

movie_list = sorted(movies['title'].unique())
selected_movie = st.selectbox("🎞️ Select a Movie", movie_list)

if st.button("🔍 Recommend"):
    recommendations = recommend(selected_movie, similarity_df)
    if recommendations:
        st.success("Here are the top 5 movies you might like:")
        cols = st.columns(5)
        for i, rec in enumerate(recommendations):
            poster_url = fetch_poster(rec)
            with cols[i]:
                if poster_url:
                    st.image(poster_url, width=150, caption=rec)
                else:
                    st.markdown(f'<div class="recommend-box">🎬 {rec}</div>', unsafe_allow_html=True)
    else:
        st.error("Movie not found or insufficient data to recommend.")

st.markdown("---")
st.markdown("Made with ❤️ using Streamlit", unsafe_allow_html=True)
