import streamlit as st
import pandas as pd
import joblib

new_df = pd.read_csv(r"../data/preprocessed_data")

similarities = joblib.load("similarity_data.pkl")

def recommend(movie):
    movie_index = new_df[new_df['title']==movie].index[0]
    distances = similarities[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key = lambda x:x[1])[1:6]

    recommended_movies = []
    
    for i in movies_list:
        recommended_movies.append(new_df.iloc[i[0]].title)
    
    return recommended_movies

st.title("Movie Recommender System")

selected_movie_name = st.selectbox(
'Movies List',
new_df['title'].values
)

if st.button('Recommend'):
    recommendations = recommend(selected_movie_name)
    for i in recommendations:
        st.write(i)