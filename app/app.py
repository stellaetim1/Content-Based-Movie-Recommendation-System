import streamlit as st
import pickle
import pandas as pd
from fuzzywuzzy import process
from sklearn.metrics.pairwise import cosine_similarity

# Load the selected model
with open('count vectorizer', 'rb') as f:
    count_vectorizer = pickle.load(f)

with open('similar', 'rb') as f:
    cosine_sim_matrix = pickle.load(f)

with open('movies', 'rb') as f:
    se_cleaned_movie = pickle.load(f)

# creating a function to obtain top 10 movie recommendations based on a movie title
def get_top_n_recommendations_fuzzy(movie_title, cosine_sim, df, n=10):
    movie_title = movie_title.strip().lower()
    best_match = process.extractOne(movie_title, df['movie title'].values)
    if best_match is None or best_match[1] < 70:
        return pd.DataFrame(columns=['movie title', 'similarity score'])

    best_match_title = best_match[0]
    idx = df[df['movie title'] == best_match_title].index[0]
    #get the pairwise similarity scores of all movies
    se_simil_scores = list(enumerate(cosine_sim[idx]))
    se_simil_scores = sorted(se_simil_scores, key=lambda x: x[1], reverse=True)
    se_simil_scores = se_simil_scores[1:n+1]
    movie_indices = [i[0] for i in se_simil_scores]
    similarity_scores = [i[1] for i in se_simil_scores]
    recommendations = df.iloc[movie_indices][['movie title']].copy()
    recommendations['similarity score'] = similarity_scores
    return recommendations

# Streamlit web-based application
st.title("Stella's Movie Recommendation System")

# Steps for  new user profile creation
st.header("Step 1: Create a New User Profile")

# inputs for creating a  new user profile
se_user_name = st.text_input('Enter your name')
se_user_email = st.text_input('Enter your email')

if st.button('Create Profile'):
    if se_user_name and se_user_email:
        st.success(f'Profile created for {se_user_name}')
    else:
        st.error('Please enter both name and email to create a profile.')

# Tabs to and get movie recommendations
tab1, tab2 = st.tabs(["Search by Movie", "User Preferences"])

with tab1:
    st.header("Search Movies and Get Similar Recommendations")
    movie_title = st.text_input('Enter a movie title')
    if st.button('Get Recommendations', key='search'):
        if movie_title:
            recommendations = get_top_n_recommendations_fuzzy(movie_title, cosine_sim_matrix, se_cleaned_movie, n=10)
            if recommendations.empty:
                st.write(f"No recommendations found for '{movie_title}'.")
            else:
                st.write(f"Top 10 recommendations for '{movie_title}':")
                st.table(recommendations)
        else:
            st.write("Please enter a movie title.")

with tab2:
    st.header("User Preferences and Recommendations")
    st.write("Select your preferences to get movie recommendations.")

    # Extract all genres from the dataset
    genres = set()
    for genre_list in se_cleaned_movie['genres']:
        genres.update(genre_list)
    genres = list(genres)
    
    selected_genres = st.multiselect('Select genres you like', genres)

    directors = se_cleaned_movie['director name'].unique()
    selected_directors = st.multiselect('Select favorite directors', directors)

    actors = pd.concat([se_cleaned_movie['actor 1 name'], se_cleaned_movie['actor 2 name']]).unique()
    selected_actors = st.multiselect('Select favorite actors', actors)

    if st.button('Get Personalized Recommendations', key='profile'):
        if selected_genres or selected_directors or selected_actors:
            user_profile = " ".join(selected_genres + list(selected_directors) + list(selected_actors))
            user_vector = count_vectorizer.transform([user_profile])
            cosine_sim = cosine_similarity(user_vector, count_vectorizer.transform(se_cleaned_movie['joint features']))
            user_se_simil_scores = cosine_sim.flatten()
            se_simil_scores = list(enumerate(user_se_simil_scores))
            se_simil_scores = sorted(se_simil_scores, key=lambda x: x[1], reverse=True)
            se_simil_scores = se_simil_scores[:10]
            movie_indices = [i[0] for i in se_simil_scores]
            similarity_scores = [i[1] for i in se_simil_scores]
            recommendations = se_cleaned_movie.iloc[movie_indices][['movie title']].copy()
            recommendations['similarity score'] = similarity_scores
            st.write("Top 10 personalized recommendations:")
            st.table(recommendations)
        else:
            st.write("Please select at least one genre, director, or actor.")
