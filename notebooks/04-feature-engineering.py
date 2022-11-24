# %% [markdown]
# # Feature Engineering

# %%
import logging
import os

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

nltk.download("stopwords")
stopwords = nltk.corpus.stopwords.words("portuguese")

RANDOM_SEED = 19730115
NUMBER_OF_WORDS = 50
rng = np.random.RandomState(RANDOM_SEED)

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
logging.info("Required packages installed.")


# %%
def load_dataset(path, stratify=False):
    """Get the data from csv file

    Args:
        path(str): the file complete path. 

    Returns:
        dataframe: A pandas dataframe.
    """
    dataset = pd.read_csv(path)

    if stratify:
        dataset = dataset.groupby('polarity', group_keys=False).apply(
            lambda x: x.sample(frac=0.4))
        dataset.reset_index(drop=True, inplace=True)

    return dataset


# %%
# Load the reviews datasets.
logging.info("Load the reviews datasets.")
reviews_train_dataset = load_dataset(
    "../data/processed/buscape_reviews_train_dataset.csv", True)
reviews_test_dataset = load_dataset(
    "../data/processed/buscape_reviews_test_dataset.csv", True)


# %%
reviews_train_dataset.head()


# %%
plt.figure(figsize=(20, 10))
plt.title('Polarity Distribution in Train')
_ = reviews_train_dataset['polarity'].value_counts().plot(kind='bar')

# %%
reviews_test_dataset.info()


# %%
plt.figure(figsize=(20, 10))
plt.title('Polarity Distribution in Test')
_ = reviews_test_dataset['polarity'].value_counts().plot(kind='bar')

# %% [markdown]
# ### Counter Vectorizer

# %%
cv = CountVectorizer(stop_words=stopwords, max_features=NUMBER_OF_WORDS)
reviews_train_cv = cv.fit_transform(
    reviews_train_dataset['review_text_cleaned_no_stopwords'])
reviews_train_dtm_cv = pd.DataFrame(
    reviews_train_cv.toarray(), columns=cv.get_feature_names_out())
reviews_train_dtm_cv.index = reviews_train_dataset.index
reviews_train_processed_cv = pd.concat([reviews_train_dataset[[
                                       'original_index']], reviews_train_dtm_cv, reviews_train_dataset[['polarity']]], axis=1)
logging.info(
    f"The counter vectorizer train matrix has {reviews_train_processed_cv.shape[0]} rows and {reviews_train_processed_cv.shape[1]} columns")

reviews_test_cv = cv.transform(
    reviews_test_dataset['review_text_cleaned_no_stopwords'])
reviews_test_dtm_cv = pd.DataFrame(
    reviews_test_cv.toarray(), columns=cv.get_feature_names_out())
reviews_test_dtm_cv.index = reviews_test_dataset.index
reviews_test_processed_cv = pd.concat([reviews_test_dataset[[
                                      'original_index']], reviews_test_dtm_cv, reviews_test_dataset[['polarity']]], axis=1)
logging.info(
    f"The counter vectorizer test matrix has {reviews_test_processed_cv.shape[0]} rows and {reviews_test_processed_cv.shape[1]} columns")


# %%
reviews_train_processed_cv.head(5)


# %%
reviews_test_processed_cv.head(5)


# %%
reviews_train_processed_cv.to_pickle(
    f'../data/processed/buscape_reviews_train_dataset_cv_s{NUMBER_OF_WORDS}.pkl')
reviews_test_processed_cv.to_pickle(
    f'../data/processed/buscape_reviews_test_dataset_cv_s{NUMBER_OF_WORDS}.pkl')


# %% [markdown]
# ### TF-IDF Vectorizer

# %%

tv = TfidfVectorizer(stop_words=stopwords, max_features=50)
reviews_train_tv = tv.fit_transform(reviews_train_dataset['review_text'])
reviews_train_dtm_tv = pd.DataFrame(
    reviews_train_tv.toarray(), columns=tv.get_feature_names_out())
reviews_train_dtm_tv.index = reviews_train_dataset.index
reviews_train_processed_tv = pd.concat([reviews_train_dataset[[
                                       'original_index']], reviews_train_dtm_tv, reviews_train_dataset[['polarity']]], axis=1)
logging.info(
    f"The tf-idf vectorizer train matrix has {reviews_train_processed_tv.shape[0]} rows and {reviews_train_processed_tv.shape[1]} columns")

reviews_test_tv = tv.transform(reviews_test_dataset['review_text'])
reviews_test_dtm_tv = pd.DataFrame(
    reviews_test_tv.toarray(), columns=tv.get_feature_names_out())
reviews_test_dtm_tv.index = reviews_test_dataset.index
reviews_test_processed_tv = pd.concat([reviews_test_dataset[[
                                      'original_index']], reviews_test_dtm_tv, reviews_test_dataset[['polarity']]], axis=1)
logging.info(
    f"The tf-idf vectorizer test matrix has {reviews_test_processed_tv.shape[0]} rows and {reviews_test_processed_tv.shape[1]} columns")


# %%
reviews_train_processed_tv.head(5)

# %%
reviews_train_processed_tv.to_pickle(
    f'../data/processed/buscape_reviews_train_dataset_tv_s{NUMBER_OF_WORDS}.pkl')
reviews_test_processed_tv.to_pickle(
    f'../data/processed/buscape_reviews_test_dataset_tv_s{NUMBER_OF_WORDS}.pkl')


# %% [markdown]
# ### Embedding Vectorizer

# %%
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile


EMBEDDING_NAMES = [
    ["word2vec", "cbow_s50"],
    ["word2vec", "skip_s50"],
    ["fasttext", "cbow_s50"],
    ["fasttext", "skip_s50"],
    ["glove", "glove_s50"],
    ["wang2vec", "cbow_s50"],
    ["wang2vec", "skip_s50"],
]


def download_extract(model, architecture):
    """
    """
    url = f"http://143.107.183.175:22980/download.php?file=embeddings/{model}/{architecture}.zip"
    out_folder_path = os.path.join("../data/embeddings/", model)
    out_file_path = os.path.join(out_folder_path, architecture)
    logging.info(f"Downloading: {model}_{architecture}")
    if not os.path.exists(out_file_path):
        with urlopen(url) as response:
            with ZipFile(BytesIO(response.read())) as in_file_zip:
                in_file_zip.extractall(out_folder_path)


for model, architecture in EMBEDDING_NAMES:
    download_extract(model, architecture)


# %%
# Load the pre-trainned fast text embedding.
logging.info("Load fast text embeddings.")
fasttext_cbow_s50 = KeyedVectors.load_word2vec_format(
    "../data/embeddings/fasttext/cbow_s50.txt"
)
fasttext_skip_s50 = KeyedVectors.load_word2vec_format(
    "../data/embeddings/fasttext/skip_s50.txt"
)


# %%
# Load the pre-trainned glove embedding.
logging.info("Load glove embeddings.")
glove_s50 = KeyedVectors.load_word2vec_format("../data/embeddings/glove/glove_s50.txt")

# %%
# Load the pre-trainned wang2vec embedding.
logging.info("Load wang2vec embeddings.")
wang2vec_cbow_s50 = KeyedVectors.load_word2vec_format(
    "../data/embeddings/wang2vec/cbow_s50.txt"
)
wang2vec_skip_s50 = KeyedVectors.load_word2vec_format(
    "../data/embeddings/wang2vec/skip_s50.txt"
)

# %%
# Load the pre-trainned word2vec embedding.
logging.info("Load word2vec embeddings.")
word2vec_cbow_s50 = KeyedVectors.load_word2vec_format(
    '../data/embeddings/word2vec_cbow_s50/cbow_s50.txt')
word2vec_skip_s50 = KeyedVectors.load_word2vec_format(
    '../data/embeddings/word2vec_skip_s50/skip_s50.txt')


# %%
# def text_to_bert(text)
def text_to_embedding(text, model, vectorizer=None, vocab=None, size=50):
    if not vectorizer:
        raise Exception("The vectorizer parameter must not be None")

    transformed = vectorizer.transform(text)
    vectorized = pd.DataFrame(transformed.toarray(
    ), columns=vectorizer.get_feature_names_out())

    embeedings = pd.DataFrame()
    for i in range(vectorized.shape[0]):
        sentence = np.zeros(size)
        for word in vocab[vectorized.iloc[i, :] > 0]:
            if model.get_index(word, default=-1) != -1:
                sentence = sentence + model.get_vector(word)
            else:
                print("Out of Vocabulary")

        embeedings = pd.concat([embeedings, pd.DataFrame([sentence])])

    return embeedings


# %%
embedding_models = [fasttext_cbow_s50, fasttext_skip_s50, glove_s50,
                    wang2vec_cbow_s50, wang2vec_skip_s50, word2vec_cbow_s50, word2vec_skip_s50]

for name, model in zip(EMBEDDING_NAMES, embedding_models):
    reviews_train_dtm = text_to_embedding(
        reviews_train_dataset['review_text'], model, tv, reviews_test_processed_tv.columns[1:-1], 50)
    reviews_train_processed = pd.concat([reviews_train_dataset.reset_index()[['original_index']], reviews_train_dtm.reset_index(
        drop=True), reviews_train_dataset.reset_index()[['polarity']]], axis=1, ignore_index=True)
    reviews_train_processed.to_pickle(
        f"../data/processed/buscape_reviews_train_dataset_{name[0]}_{name[1]}.pkl")
    print(
        f"The {name} vectorized train dataframe has {reviews_train_processed.shape[0]} rows and {reviews_train_processed.shape[1]} columns")

    reviews_test_dtm = text_to_embedding(
        reviews_test_dataset['review_text'], model, tv, reviews_test_processed_tv.columns[1:-1], 50)
    reviews_test_processed = pd.concat([reviews_test_dataset.reset_index()[['original_index']], reviews_test_dtm.reset_index(
        drop=True), reviews_test_dataset.reset_index()[['polarity']]], axis=1, ignore_index=True)
    reviews_test_processed.to_pickle(
        f"../data/processed/buscape_reviews_test_dataset_{name[0]}_{name[1]}.pkl")
    print(
        f"The {name} vectorized test dataframe has {reviews_test_processed.shape[0]} rows and {reviews_test_processed.shape[1]} columns")


# %%
import torch

from transformers import AutoTokenizer
from transformers import AutoModel

model = AutoModel.from_pretrained("neuralmind/bert-base-portuguese-cased")
logging.info(f"Transformers model class model: {type(model)}")
tokenizer = AutoTokenizer.from_pretrained(
    "neuralmind/bert-base-portuguese-cased", do_lower_case=True
)
logging.info(f"Transformers tokenizer class: {type(tokenizer)}")


# Create a function to tokenize a set of texts
def preprocessing_for_bert(data):
    """Perform required preprocessing steps for pretrained BERT.
    @param    data (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """
    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in data:
        # `encode_plus` will:
        #    (1) Tokenize the sentence
        #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
        #    (3) Truncate/Pad sentence to max length
        #    (4) Map tokens to their IDs
        #    (5) Create attention mask
        #    (6) Return a dictionary of outputs
        encoded_sent = tokenizer.encode_plus(
            text=sent,  # Preprocess sentence
            add_special_tokens=True,     # Add `[CLS]` and `[SEP]`
            max_length=64,               # Max length to truncate/pad
            padding='max_length',        # Pad sentence to max length
            truncation='only_first',     # Truncate sentence to max length
            return_attention_mask=True,  # Return attention mask
        )

        # Add the outputs to the lists
        input_ids.append(encoded_sent.get("input_ids"))
        attention_masks.append(encoded_sent.get("attention_mask"))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks


review_train_inputs, review_train_masks = preprocessing_for_bert(
    list(reviews_train_dataset["review_text"])
)
with torch.no_grad():
    outs = model(review_train_inputs, review_train_masks)
    review_train_bert_encoded = outs[0][:, 0, :]

review_test_inputs, review_test_masks = preprocessing_for_bert(
    list(reviews_test_dataset["review_text"])
)
with torch.no_grad():
    outs = model(review_test_inputs, review_test_masks)
    review_test_bert_encoded = outs[0][:, 0, :]

# %%
reviews_train_processed_bert = pd.concat(
    [
        reviews_train_dataset[["original_index"]],
        review_train_bert_encoded,
        reviews_train_dataset[["polarity"]],
    ],
    axis=1,
)
reviews_train_processed_bert.to_pickle(f"../data/processed/buscape_reviews_train_dataset_bert.pkl")   


