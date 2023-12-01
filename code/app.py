import streamlit as st

from ml import recommend
from preprocessing import new_df

st.title("Movie Recommender System")

selected_movie_name = st.selectbox(
'Movies List',
new_df['title'].values
)

if st.button('Recommend'):
    recommendations = recommend(selected_movie_name)
    for i in recommendations:
        st.write(i)