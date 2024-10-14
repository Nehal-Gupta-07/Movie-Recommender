import pandas as pd
import joblib

new_df = pd.read_csv(r"../data/preprocessed_data")

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')

vectors = cv.fit_transform(new_df['tags']).toarray()

similarities = cosine_similarity(vectors)

joblib.dump(similarities, 'similarity_data.pkl')