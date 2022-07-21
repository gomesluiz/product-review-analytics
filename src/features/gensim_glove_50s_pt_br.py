""" 

"""
from gensim.models import KeyedVectors

model=KeyedVectors.load_word2vec_format('../../data/embeedings/glove_s50.txt')

# Retrieve the vocabulary of a model.
print('The 10 firts words in vocabulary of model.')
for index, word in enumerate(model.index_to_key):
    if index == 10:
        break

    print(f"word #{index}/{len(model.index_to_key)} is {word}")

pairs = [
    ('carro', 'minivan'),
    ('carro', 'bicicleta'),
    ('carro', 'aeroplano'),
    ('carro', 'cereal'),
    ('carro', 'democracia'),
]

print('The similarities between words.')
for w1, w2 in pairs:
    print(f'{w1}\t{w2}\t{model.similarity(w1, w2)}')

print('The 5 most similar words to carro or bicicleta')
print(model.most_similar(positive=['carro', 'bicicleta'], topn=5))

