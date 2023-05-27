import pandas as pd
import difflib
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(page_title="Movie Recommender", page_icon=":movie_camera:", layout="wide")
st.title("Movie Recommender System")

# Add the input text widget to your Streamlit app
movie_name = st.text_input("Enter Movie's name")


@st.cache_resource
def load_data():
    # Load the movies data from CSV
    movies_data = pd.read_csv("https://github.com/abdullah-khaled0/My-Projects/raw/main/Streamlit-Apps/movie-recommendation-system/movies.csv", on_bad_lines='skip')
    
    # Select the relevant features for recommendation
    selected_features = ['genres','keywords','tagline','cast','director']

    # Replace the null values with null string
    for feature in selected_features:
        movies_data[feature] = movies_data[feature].fillna('')

    # Combine all the selected features into one string
    combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']

    # Convert the text data to feature vectors
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(combined_features)

    # Calculate the cosine similarity matrix
    similarity = cosine_similarity(feature_vectors)

    return movies_data, similarity

# Load the data using the cache function
movies_data, similarity = load_data()




# creating a list with all the movie names given in the dataset
list_of_all_titles = movies_data['title'].tolist()

# finding the close match for the movie name given by the user
find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
if len(find_close_match) != 0:
    close_match = find_close_match[0]

    # finding the index of the movie with title
    index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]

    # getting a list of similar movies
    similarity_score = list(enumerate(similarity[index_of_the_movie]))

    # sorting the movies based on their similarity score
    sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) 

    st.subheader("Top 10 Recommended Movies:")

    i = 0
    movie_titles = []
    for movie in sorted_similar_movies:
        index = movie[0]
        title = movies_data[movies_data.index==index]['title'].values[0]
        if (i<10):
            movie_titles.append(title)
            i+=1

    df = pd.DataFrame({'Movie Titles': movie_titles})
    st.write(df)
