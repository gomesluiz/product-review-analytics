import spacy

nlp = spacy.load("en_core_web_sm")
def nouns(text, model=nlp):
    # Create doc object.
    doc = model(text)

    # Generate list POS tags.
    pos = [token.pos_ for token in doc]

    # Return number of other nouns.
    return pos.count("NOUN")
