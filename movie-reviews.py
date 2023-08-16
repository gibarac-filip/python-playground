import nltk
nltk.download('movie_reviews')

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

import requests
from bs4 import BeautifulSoup
# Replace 'movie_id' with the actual IMDb movie ID or URL
movie_id = "tt2301451"  # Example: The Shawshank Redemption
url = f"https://www.imdb.com/title/{movie_id}/reviews/"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')
review_elements = soup.find_all("div", class_="text show-more__control")
for review in review_elements:
    print(review.get_text())
    print("-" * 50)

