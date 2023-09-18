import os
import time
import requests
import csv
from bs4 import BeautifulSoup
import pandas as pd
import re
import nltk
from nltk.corpus import movie_reviews, stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from textblob import TextBlob
from gensim import corpora


# Example corpus (list of text documents)
corpus = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
    "is this the first document in this document?"
]

# Tokenize the documents
tokenized_corpus = [doc.split() for doc in corpus]

# Create a dictionary mapping words to unique ids
dictionary = corpora.Dictionary(tokenized_corpus)

# Create a bag of words representation
bag_of_words = [dictionary.doc2bow(doc) for doc in tokenized_corpus]

print("Bag of Words Representation:")
for doc in bag_of_words:
    print(doc)

# Bag of Words
count_vectorizer = CountVectorizer()
bag_of_words_matrix = count_vectorizer.fit_transform(corpus)

# TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

print("Bag of Words Matrix:")
print(bag_of_words_matrix.toarray())

print("\nTF-IDF Matrix:")
print(tfidf_matrix.toarray())

# Example target variable (sentiments)
sentiments = ["positive", "negative", "neutral", "happy", "sad", "neutral", "excited", "angry", "calm"]

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the sentiments to numeric labels
numeric_labels = label_encoder.fit_transform(sentiments)

# Inverse transform to see the original labels
original_sentiments = label_encoder.inverse_transform(numeric_labels)

print("Original Sentiments:", sentiments)
print("Numeric Labels:", numeric_labels)
print("Decoded Sentiments:", original_sentiments)

# Example sentence
sentence = "This is an example sentence with some stop words."

os.getcwd()
os.chdir('Desktop/python-playground')

nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Example review
review = "This is an example review containing some words."

# Tokenize the review
words = word_tokenize(review)

# Remove punctuation and convert to lowercase
words = [word.lower() for word in words if word.isalpha()]

# Remove stop words
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word not in stop_words]

# Perform stemming
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in filtered_words]

# Perform lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]

print("Original Words:", words)
print("Stemmed Words:", stemmed_words)
print("Lemmatized Words:", lemmatized_words)

# Create a TextBlob object
blob = TextBlob(review)

# Tokenize the review
words = blob.words

# Perform lemmatization
lemmatized_words = [word.lemmatize() for word in words]

print("Original Words:", words)
print("Lemmatized Words:", lemmatized_words)

# Example sentence
sentence = "This is an example sentence with some stop words."

# Tokenize the sentence
words = word_tokenize(sentence)

# Remove stop words
stop_words = set(stopwords.words('english'))
filtered_words_nltk = [word for word in words if word.lower() not in stop_words]
filtered_words_sklearn = [word for word in sentence.split() if word.lower() not in ENGLISH_STOP_WORDS]
filtered_sentence_nltk = ' '.join(filtered_words_nltk)
filtered_sentence_sklearn = ' '.join(filtered_words_sklearn)

# Join the words back into a sentence
print(filtered_sentence_nltk)
print(filtered_sentence_sklearn)

# List of file IDs for positive reviews
positive_fileids = movie_reviews.fileids('pos')

# List of file IDs for negative reviews
negative_fileids = movie_reviews.fileids('neg')

# Example of accessing a positive review text
positive_review = movie_reviews.raw(positive_fileids[0])

# Example of accessing a negative review text
negative_review = movie_reviews.raw(negative_fileids[0])

# Print the first few characters of each review
print("Positive Review:")
print(positive_review[:200])  # Print first 200 characters
print("\nNegative Review:")
print(negative_review[:200])  # Print first 200 characters

def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation using regular expression
    text = re.sub(r'[^\w\s]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Replace 'movie_id' with the actual IMDb movie ID or URL
movie_id = "tt2301451"  # Example: The Shawshank Redemption
url = f"https://www.imdb.com/title/{movie_id}/reviews/"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')
review_elements = soup.find_all("div", class_="text show-more__control")
for review in review_elements:
    review_text = review.get_text()  # Extract text from the element
    cleaned_review = clean_text(review_text)  # Clean the extracted text
    print(cleaned_review)
    print("-" * 50)

# Initialize an empty dictionary to store reviews
reviews_dict = {}

# Read the CSV file
reader = pd.read_csv("movie-review-data.csv", encoding="utf-8")
#reader = pd.read_csv("movie-review-data.csv", encoding="utf-8").iloc[0:100,:]

start_time = time.time()

for index, row in reader.iterrows():
    movie_id = row["movie_id"]
    url = f"https://www.imdb.com/title/{movie_id}/reviews/"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract movie reviews
    review_elements = soup.find_all("div", class_="text show-more__control")
    reviews = [review.get_text() for review in review_elements]

    # Initialize the "reviews" key if the movie_id is not in the dictionary
    if movie_id not in reviews_dict:
        reviews_dict[movie_id] = {"reviews": []}

    # Clean and add each review to the dictionary
    cleaned_reviews = [clean_text(review) for review in reviews]
    reviews_dict[movie_id]["reviews"].extend(cleaned_reviews)

    
# Stop the timer
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"Time taken: {elapsed_time:.2f} seconds") # 9700 seconds, 2.7 hours

# Print the reviews for a specific movie ID
movie_id_to_lookup = "tt1683087"  # Replace with the movie ID you're interested in
if movie_id_to_lookup in reviews_dict:
    movie_reviews = reviews_dict[movie_id_to_lookup]["reviews"]
    for review in movie_reviews:
        print(review)
else:
    print("Movie ID not found in the dataset.")
    
for movie_id, data in reviews_dict.items():
    num_reviews = len(data["reviews"])
    print(f"Movie ID: {movie_id}")
    print(f"Number of Reviews: {num_reviews}")
    print("-" * 50)
