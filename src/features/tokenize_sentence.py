import spacy

def tokenize_text(text, model='en_core_web_sm')
	"""
	"""
	nlp = spacy.load(model)
	doc = nlp(text)
	tokens = [token.text for token in doc]
	return tokens