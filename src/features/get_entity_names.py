import spacy


def get_entity_names(text, model="en_core_web_sm"):
    # Load the required model.
    nlp = spacy.load(model)

    # Create a Doc instance.
    doc = nlp(text)

    # Return all named entities and theirs labels.
    return [(ent.text, ent.label_) for ent in doc.ents]
