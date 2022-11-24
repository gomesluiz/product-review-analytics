# %% [markdown]
# # Exploratory Data Analysis
# In order to better understanding of the data we will be looking at throughout that project.

# %%
import os
import re
import string

import nltk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from collections import Counter
from nltk import word_tokenize

nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('portuguese')

# %matplotlib inline

# %% [markdown]
# When we getting started an Exploratory Data Analisys (EDA), we should answer the next questions:
# 
# 1. Look at the number of rows and columns in the dataset. 
# 2. Check if there are missing values in any of the rows or columns.
# 3. Check if any columns are of different data types than you would expect(e.g., numbers showing as strings)
# 4. Check that each column is a variable, and each row is an individual.
# 5. Build exploratory plots like bar charts, histograms, and scatterplots to better understand the data.

# %%
def get_data(file):
    """Get the data from csv file
    
    """
    return pd.read_csv(file)

# %%
reviews = get_data('../data/raw/buscape.csv')
reviews.head()

# %% [markdown]
# ### Question 1.
# How many rows and columns the dataset has?

# %%
num_rows = reviews.shape[0]
num_cols = reviews.shape[1]
print(f"The dataset has {num_rows} rows and {num_cols} columns.")

# %% [markdown]
# ### Question 2.
# 1. Which columns had missing values?

# %%
missing_cols = set(reviews.columns[reviews.isnull().sum() != 0])
print(missing_cols)

# %% [markdown]
# 2. Which columns have more than 20% of missing values?

# %%
most_missing_cols = set(reviews.columns[reviews.isnull().mean() > 0.20])
print(most_missing_cols)

# %% [markdown]
# ### Question 3.
# Check if any columns are of different data types than you would expect(e.g., numbers showing as strings)
# 

# %%
reviews.info()

# %%
plt.figure(figsize=(20, 10))
polarity_counts = reviews['polarity'].value_counts()
polarity_counts.plot(kind="bar")

# %% [markdown]
# ### Question 4.
# Text statistics on **review_text**.

# %%
def word_counter(text):
    """ Word counter.
    """
    return len(text.split())

def clean_text(text):
    """ Make text lowercase, remove text in square brackets, remove punctuation and 
        remove words containing numbers.
    
    Args:
        text(str): string text to be cleaned.

    Returns:
        A cleaned text

    """
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('[``""...]', '', text)
    text = re.sub('\n', ' ', text)

    return text

# %%
reviews.dropna(subset=['review_text'], inplace=True)
reviews.loc[:, ['review_text_cleaned']] = reviews['review_text'].apply(lambda x: clean_text(x))
reviews.loc[:, ['review_text_cleaned_len']] = reviews['review_text_cleaned'].apply(word_counter)
reviews.loc[:, ['review_text_cleaned_no_stopwords']] = reviews['review_text_cleaned'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))
reviews.loc[:, ['review_text_cleaned_len_no_stopwords']] = reviews['review_text_cleaned_no_stopwords'].apply(word_counter)

# %%
fig, ax = plt.subplots(figsize=(20, 10))
ax.set_title('Number of Words')
_ = ax.boxplot([reviews['review_text_cleaned_len'], reviews['review_text_cleaned_len_no_stopwords']], 
    showfliers=True)

# %%
DATA_PROCESSED_FOLDER = "../data/processed" 
if not os.path.exists(DATA_PROCESSED_FOLDER):
    os.mkdir(DATA_PROCESSED_FOLDER)

reviews[['original_index', 'review_text', 'review_text_cleaned', 'review_text_cleaned_len',
         'review_text_cleaned_no_stopwords', 'review_text_cleaned_len_no_stopwords', 'polarity']].to_csv(f"{DATA_PROCESSED_FOLDER}/buscape_reviews_full_dataset.csv", index=False)


