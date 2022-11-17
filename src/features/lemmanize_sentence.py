import spacy

def lemmanize_text(text, model='en_core_web_sm')
	"""
	"""
	nlp = spacy.load(model)
	doc = nlp(text)
	lemmas = [token.lemma_ for token in doc]
	return lemmas