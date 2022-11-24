# %% [markdown]
# # Data Preparation

# %%
import re
import string

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd

from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split 

nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('portuguese')

RANDOM_SEED = 19730115
rng = np.random.RandomState(RANDOM_SEED)

print("Required packages installed.")

# %%
def load_dataset(path):
    """Get the data from csv file
    
    Args:
        path(str): the file complete path. 

    Returns:
        A pandas dataframe.
    """

    return pd.read_csv(path)


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
# Load the reviews dataset.
reviews = load_dataset('../data/processed/buscape_reviews_full_dataset.csv')

# Replace the original polarity to -1 from 0, nan to 0.  
reviews_cleaned = reviews.copy() 
reviews_cleaned['polarity'] = reviews_cleaned['polarity'].replace({0:-1, np.nan: 0})
reviews_cleaned['polarity'] = reviews_cleaned['polarity'].astype(int)
# 
reviews_cleaned.dropna(subset=['review_text_cleaned_no_stopwords'], inplace=True)
reviews_cleaned.head()

# %%
# plt.figure(figsize=(20, 10))
# plt.title('Polarity Distribution')
# reviews_cleaned['polarity'].value_counts().plot(kind='bar')

# %% [markdown]
# ### Split in train and test datasets

# %%
reviews_cleaned_train, reviews_cleaned_test = train_test_split(reviews_cleaned, stratify=reviews_cleaned['polarity'], test_size=.20, random_state=rng)

# %%
reviews_cleaned_train.to_csv("../data/processed/buscape_reviews_train_dataset.csv", index=False)
reviews_cleaned_test.to_csv("../data/processed/buscape_reviews_test_dataset.csv", index=False)


