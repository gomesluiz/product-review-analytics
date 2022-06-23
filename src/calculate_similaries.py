"""Calculate similaries between sellers and buyers."""

# Import third-part packages.
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def read_data(path):
    """Read the read data from csv."""

# Read the sellers and buyers data
sellers_corpus = pd.Series({'1':'2 a 15'})
buyers_corpus  = pd.Series({'1':'2 a 10'})

# Create TfidfVectorizer object
vectorizer = TfidfVectorizer()

# Generate matrix of word vectors.
sellers_matrix = vectorizer.fit_transform(sellers_corpus)
buyers_matrix = vectorizer.transform(buyers_corpus)
similarity = cosine_similarity(sellers_matrix, buyers_matrix)

# Print the shape of matrices
print(f"Sellers matrix: {sellers_matrix.shape}")
print(f"Buyers  matrix: {buyers_matrix.shape}")
print(f"Sellers and buyers  similarity: {similarity}")
