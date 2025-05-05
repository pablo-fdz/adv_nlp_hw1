import re

def tokenize(text, rm_stopwords = False, stopword_set = None):
    
    """
    Preprocess text by:
       - Converting to lowercase.
       - Removing punctuation.
       - Tokenizing.
       - Removing stopwords (optional).
    
    Returns:
        list: A list of tokens lowercased and without punctuation.
    """
    
    # Convert to lowercase
    text_lower = text.lower()
    tokens = re.findall(r"\w+", text_lower)  # Match all alphanumeric characters
    # Remove stopwords if desired
    if rm_stopwords == True:
        tokens_nostop = [token for token in tokens if token not in stopword_set]
    # Return the tokens without stopwords (if set)
    return tokens_nostop