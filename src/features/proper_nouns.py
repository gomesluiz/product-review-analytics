import spacy

nlp = spacy.load("en_core_web_sm")
def proper_nouns(text, mode):
    # Create doc object.
    doc = model(text)

    # Generate list of POS tags.
    pos = [token.pos_ for token in doc]

    return pos.count("PROPN")